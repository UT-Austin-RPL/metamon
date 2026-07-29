"""Phase 2: paired + mirrored end-to-end evaluation (skill §23).

The scientific question (skill §23): does correct exhaustive oracle search
improve game outcomes over the frozen baseline, measured with a paired
statistical design that resolves the practical effect?

Design (skill §23 "Paired battle unit"):

For each seed ``s`` in a seed manifest and each physical side ``side in {0, 1}``:

  * **search-on**: run the frozen policy + test-time search on ``side`` vs the
    plain frozen policy on the other side, seed ``s``.
  * **search-off (control)**: run the plain frozen policy on ``side`` vs the
    plain frozen policy on the other side, seed ``s``.

Because ``run_search_eval(seed=s)`` fixes all stochastic sources (Showdown
battle PRNG, team draws, frozen-policy sampling stream), the search-on and
search-off battles for the same ``(s, side)`` start from identical initial
conditions and take identical actions at non-searched decisions; the only
difference is the search-selected action at searched decisions. Each battle
index ``i`` therefore gives a **paired** binary outcome
``(search_win_i, baseline_win_i)`` (skill §23: "pair a search-enabled game
with a baseline control using the same initial battle setup").

The two sides form a **mirrored pair** (skill §23 "swap sides"): with a fixed
seed the ``competitive`` TeamSet draws the same two teams, and
``eval_player_side`` swaps which physical side gets which team (verified in
``team_adapter.coupled_player_specs`` -- the "both plain" branch draws
``eval_team`` then ``opp_team`` and assigns by ``eval_side``). So side 0 and
side 1 with the same seed are the same matchup with sides swapped.

Statistics (skill §23 "Primary statistics" + §32):

  * search / baseline win rates;
  * paired win-rate delta (search - baseline);
  * McNemar test on discordant pairs (b = search-win & baseline-loss,
    c = search-loss & baseline-win);
  * paired bootstrap CI on the delta (resample pairs with replacement);
  * per-side breakdown;
  * draw / incomplete-battle accounting.

This module is research-mode: ``search_error_policy=raise`` is enforced -- a
silent base_fallback would invalidate the paired comparison (skill §19).

Usage::

    uv run python -m metamon.rl.experimental.test_time_search.paired_eval \\
        --agent MiniOnlinePsroV1_4 --checkpoint 740 --format gen1ou \\
        --team_set competitive --rollouts_per_action 64 --search_depth 0 \\
        --search_beta 1.0 --value_scale_mode global_standardized \\
        --global_advantage_scale 300.0 \\
        --num_seeds 3 --battles_per_seed 100 --num_parallel 4 \\
        --seed_base 1000 --output_dir /tmp/tts_phase2_screen
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import SearchConfig
from .eval_search import run_search_eval, build_config_from_args, add_cli

# ---------------------------------------------------------------------------
# Paired statistics (skill §23 / §32)
# ---------------------------------------------------------------------------


def _binom_two_sided_p(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided p-value for ``P(X = k)`` under ``X ~ Binomial(n, p)``.

    Two-sided = 2 * min(P(X >= k), P(X <= k)), clamped to 1.0. Computed via
    the regularized incomplete beta function (``math.comb`` is exact but
    blows up for large n; the beta relation is stable). For ``n_disc < 25``
    (where we use the exact test) ``math.comb`` is perfectly fine.
    """
    if n == 0:
        return 1.0
    # P(X >= k) = I_p(k, n-k+1) (regularized incomplete beta); P(X <= k) = 1 - P(X >= k+1)
    # For small n use the direct PMF sum (stable, exact).
    from math import comb, pow as _pow

    pmf = [comb(n, i) * (p**i) * ((1.0 - p) ** (n - i)) for i in range(n + 1)]
    p_ge_k = sum(pmf[i] for i in range(k, n + 1))
    p_le_k = sum(pmf[i] for i in range(0, k + 1))
    return min(1.0, 2.0 * min(p_ge_k, p_le_k))


def _chi2_sf_df1(stat: float) -> float:
    """Survival function P(X > stat) for chi-squared with df=1.

    chi2(df=1) survival = erfc(sqrt(stat / 2)). Uses ``math.erfc`` (no scipy).
    """
    from math import erfc, sqrt

    if stat <= 0.0:
        return 1.0
    return float(erfc(sqrt(stat / 2.0)))


def mcnemar_test(b: int, c: int, exact: bool = True) -> Dict[str, Any]:
    """McNemar test on discordant pairs.

    ``b`` = pairs where search won and baseline lost (search-better).
    ``c`` = pairs where search lost and baseline won (search-worse).

    For small discordant counts (< 25) use the exact binomial two-sided test;
    otherwise use the chi-squared continuity-corrected approximation.
    """
    n_disc = b + c
    if n_disc == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b": b, "c": c, "method": "none"}
    if exact and n_disc < 25:
        p = _binom_two_sided_p(b, n_disc, 0.5)
        return {
            "statistic": float(b),
            "p_value": p,
            "b": b,
            "c": c,
            "method": "exact_binomial",
        }
    # chi-squared with continuity correction
    stat = (abs(b - c) - 1) ** 2 / max(b + c, 1)
    p = _chi2_sf_df1(float(stat))
    return {
        "statistic": float(stat),
        "p_value": p,
        "b": b,
        "c": c,
        "method": "chi2_continuity",
    }


def paired_bootstrap_ci(
    pairs: List[Tuple[float, float]],
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> Dict[str, Any]:
    """Paired bootstrap CI on the win-rate delta (search - baseline).

    Resamples pairs with replacement (skill §32: "bootstrap over battles or
    pairs, not over individual search roots"). Returns the delta point estimate
    and the CI bounds.
    """
    if not pairs:
        return {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_boot": 0}
    rng = np.random.default_rng(seed)
    arr = np.asarray(pairs, dtype=np.float64)  # (n, 2): (search_win, baseline_win)
    deltas = arr[:, 0] - arr[:, 1]
    point = float(deltas.mean())
    n = len(deltas)
    boot_deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_deltas[i] = deltas[idx].mean()
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(boot_deltas, alpha))
    hi = float(np.quantile(boot_deltas, 1.0 - alpha))
    return {
        "delta": point,
        "ci_low": lo,
        "ci_high": hi,
        "ci_level": ci,
        "n_boot": n_boot,
        "n_pairs": n,
    }


def wilson_interval(wins: int, n: int, ci: float = 0.95) -> Tuple[float, float]:
    """Wilson score interval for a single proportion."""
    if n == 0:
        return (0.0, 0.0)
    # z for the CI level (no scipy): 0.95 -> 1.959964, 0.99 -> 2.575829.
    _Z = {0.90: 1.644854, 0.95: 1.959964, 0.99: 2.575829}
    z = _Z.get(ci, 1.959964)
    p = wins / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * np.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n)) / denom
    return (float(center - spread), float(center + spread))


def analyze_pairs(pairs: List[Tuple[float, float]], sides: List[int]) -> Dict[str, Any]:
    """Full paired analysis from (search_win, baseline_win) pairs.

    ``sides[i]`` is the physical side (0 or 1) for pair ``i``.
    """
    pairs_arr = np.asarray(pairs, dtype=np.float64)
    n = len(pairs)
    search_wins = int(pairs_arr[:, 0].sum())
    baseline_wins = int(pairs_arr[:, 1].sum())
    # discordant
    b = int(((pairs_arr[:, 0] == 1) & (pairs_arr[:, 1] == 0)).sum())
    c = int(((pairs_arr[:, 0] == 0) & (pairs_arr[:, 1] == 1)).sum())
    # ties (both win or both lose) -- both-win is impossible in a non-draw game;
    # both-lose means a draw or an incomplete battle recorded as 0 for both.
    both_lose = int(((pairs_arr[:, 0] == 0) & (pairs_arr[:, 1] == 0)).sum())
    both_win = int(((pairs_arr[:, 0] == 1) & (pairs_arr[:, 1] == 1)).sum())

    search_wr = search_wins / n if n else 0.0
    baseline_wr = baseline_wins / n if n else 0.0
    delta = search_wr - baseline_wr

    mcnemar = mcnemar_test(b, c)
    boot = paired_bootstrap_ci(pairs, n_boot=10000, ci=0.95, seed=0)
    s_lo, s_hi = wilson_interval(search_wins, n)
    b_lo, b_hi = wilson_interval(baseline_wins, n)

    # per-side breakdown
    by_side: Dict[int, Dict[str, Any]] = {}
    for side_val in sorted(set(sides)):
        mask = np.asarray(sides) == side_val
        sp = [pairs[i] for i in range(n) if mask[i]]
        if not sp:
            continue
        sp_arr = np.asarray(sp)
        sw = int(sp_arr[:, 0].sum())
        bw = int(sp_arr[:, 1].sum())
        nn = len(sp)
        bb = int(((sp_arr[:, 0] == 1) & (sp_arr[:, 1] == 0)).sum())
        cc = int(((sp_arr[:, 0] == 0) & (sp_arr[:, 1] == 1)).sum())
        by_side[side_val] = {
            "n": nn,
            "search_win_rate": sw / nn,
            "baseline_win_rate": bw / nn,
            "delta": sw / nn - bw / nn,
            "discordant_b": bb,
            "discordant_c": cc,
        }

    return {
        "n_pairs": n,
        "search_wins": search_wins,
        "baseline_wins": baseline_wins,
        "search_win_rate": search_wr,
        "baseline_win_rate": baseline_wr,
        "paired_delta": delta,
        "discordant_b": b,
        "discordant_c": c,
        "both_lose": both_lose,
        "both_win": both_win,
        "wilson_search": [s_lo, s_hi],
        "wilson_baseline": [b_lo, b_hi],
        "mcnemar": mcnemar,
        "bootstrap_ci": boot,
        "by_side": by_side,
    }


# ---------------------------------------------------------------------------
# Paired eval runner
# ---------------------------------------------------------------------------


def run_paired_eval(
    agent_name: str,
    checkpoint: int,
    battle_format: str,
    team_set_name: str,
    search_config: SearchConfig,
    num_seeds: int,
    battles_per_seed: int,
    num_parallel: int = 4,
    device: str = "cuda",
    opponent_agent: Optional[str] = None,
    opponent_checkpoint: Optional[int] = None,
    seed_base: int = 1000,
    sides: Tuple[int, ...] = (0, 1),
    progress_fn=None,
) -> Dict[str, Any]:
    """Run the paired + mirrored eval (skill §23).

    For each seed and side, runs search-on and search-off (control) with the
    same seed, pairing battles by index. Returns the full paired analysis +
    per-run raw results.

    ``search_config`` must have ``search_error_policy='raise'`` (enforced).
    """
    if search_config.search_error_policy != "raise":
        raise ValueError(
            "paired_eval requires search_error_policy='raise' (skill §19: "
            "silent base_fallback invalidates the paired comparison)."
        )

    # The baseline config is the search config with search disabled.
    baseline_config = SearchConfig(**search_config.__dict__)
    baseline_config.search_mode = "none"

    all_pairs: List[Tuple[float, float]] = []
    pair_sides: List[int] = []
    pair_seeds: List[int] = []
    pair_battle_idx: List[int] = []
    runs: List[Dict[str, Any]] = []

    total_runs = num_seeds * len(sides) * 2  # on/off per (seed, side)
    run_idx = 0
    t0 = time.perf_counter()

    # Stream per-run results to disk for crash safety (a 4-hour screen should
    # not lose all data if the process dies at run 10/12). The analysis is
    # recomputed from these at the end, but the raw per-run + per-pair data
    # survives a crash.
    _runs_fh = None
    _pairs_fh = None
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        _runs_fh = open(os.path.join(output_dir, "paired_runs.jsonl"), "w")
        _pairs_fh = open(os.path.join(output_dir, "paired_pairs.jsonl"), "w")

    for si in range(num_seeds):
        seed = seed_base + si
        for side in sides:
            # --- search-on ---
            run_idx += 1
            if progress_fn:
                progress_fn(
                    f"[{run_idx}/{total_runs}] seed={seed} side={side} search=ON"
                )
            res_on = run_search_eval(
                agent_name=agent_name,
                checkpoint=checkpoint,
                battle_format=battle_format,
                team_set_name=team_set_name,
                config=search_config,
                total_battles=battles_per_seed,
                num_parallel=num_parallel,
                device=device,
                opponent_agent=opponent_agent,
                opponent_checkpoint=opponent_checkpoint,
                seed=seed,
                eval_player_side=side,
            )
            runs.append({"seed": seed, "side": side, "search": "on", **res_on})
            if _runs_fh is not None:
                _runs_fh.write(
                    json.dumps(
                        {"seed": seed, "side": side, "search": "on", **res_on},
                        default=lambda o: str(o),
                    )
                    + "\n"
                )
                _runs_fh.flush()
            # Inter-run GPU cleanup: each run_search_eval creates+destroys an
            # env and runner; without an explicit cache clear, CUDA memory can
            # accumulate over many runs (a likely cause of silent death at run
            # 8/12 in the first screen attempt).
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # --- search-off (control) ---
            run_idx += 1
            if progress_fn:
                progress_fn(
                    f"[{run_idx}/{total_runs}] seed={seed} side={side} search=OFF"
                )
            res_off = run_search_eval(
                agent_name=agent_name,
                checkpoint=checkpoint,
                battle_format=battle_format,
                team_set_name=team_set_name,
                config=baseline_config,
                total_battles=battles_per_seed,
                num_parallel=num_parallel,
                device=device,
                opponent_agent=opponent_agent,
                opponent_checkpoint=opponent_checkpoint,
                seed=seed,
                eval_player_side=side,
            )
            runs.append({"seed": seed, "side": side, "search": "off", **res_off})
            if _runs_fh is not None:
                _runs_fh.write(
                    json.dumps(
                        {"seed": seed, "side": side, "search": "off", **res_off},
                        default=lambda o: str(o),
                    )
                    + "\n"
                )
                _runs_fh.flush()
            # Inter-run GPU cleanup (same as above).
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # pair by battle index
            wins_on = res_on.get("per_battle_wins", [])
            wins_off = res_off.get("per_battle_wins", [])
            n_pair = min(len(wins_on), len(wins_off))
            for bi in range(n_pair):
                all_pairs.append((float(wins_on[bi]), float(wins_off[bi])))
                pair_sides.append(side)
                pair_seeds.append(seed)
                pair_battle_idx.append(bi)
                if _pairs_fh is not None:
                    _pairs_fh.write(
                        json.dumps(
                            {
                                "seed": seed,
                                "side": side,
                                "battle_idx": bi,
                                "search_win": float(wins_on[bi]),
                                "baseline_win": float(wins_off[bi]),
                            }
                        )
                        + "\n"
                    )
                    _pairs_fh.flush()

    elapsed = time.perf_counter() - t0
    analysis = analyze_pairs(all_pairs, pair_sides)
    analysis["elapsed_sec"] = elapsed
    analysis["num_seeds"] = num_seeds
    analysis["battles_per_seed"] = battles_per_seed
    analysis["sides"] = list(sides)
    analysis["seed_base"] = seed_base

    if _runs_fh is not None:
        _runs_fh.close()
    if _pairs_fh is not None:
        _pairs_fh.close()

    return {
        "analysis": analysis,
        "pairs": {
            "all": all_pairs,
            "sides": pair_sides,
            "seeds": pair_seeds,
            "battle_idx": pair_battle_idx,
        },
        "runs": runs,
        "search_config": search_config.__dict__,
    }


def write_paired_results(
    result: Dict[str, Any],
    output_dir: str,
) -> Dict[str, str]:
    """Write the paired-eval artifacts (analysis JSON + per-run JSONL + report)."""
    os.makedirs(output_dir, exist_ok=True)
    analysis_path = os.path.join(output_dir, "paired_analysis.json")
    runs_path = os.path.join(output_dir, "paired_runs.jsonl")
    pairs_path = os.path.join(output_dir, "paired_pairs.jsonl")
    report_path = os.path.join(output_dir, "REPORT.md")

    with open(analysis_path, "w") as f:
        json.dump(result["analysis"], f, indent=2, default=lambda o: str(o))
    with open(runs_path, "w") as f:
        for r in result["runs"]:
            f.write(json.dumps(r, default=lambda o: str(o)) + "\n")
    with open(pairs_path, "w") as f:
        for (sw, bw), side, seed, bi in zip(
            result["pairs"]["all"],
            result["pairs"]["sides"],
            result["pairs"]["seeds"],
            result["pairs"]["battle_idx"],
        ):
            f.write(
                json.dumps(
                    {
                        "seed": seed,
                        "side": side,
                        "battle_idx": bi,
                        "search_win": sw,
                        "baseline_win": bw,
                    }
                )
                + "\n"
            )

    a = result["analysis"]
    cfg = result["search_config"]
    md = [
        "# Test-Time Search — Phase 2 Paired + Mirrored Evaluation (skill §23)",
        "",
        f"- search config: K={cfg.get('search_rollouts_per_action')}, "
        f"D={cfg.get('search_depth')}, "
        f"operator={cfg.get('search_ablation')}, "
        f"chance={cfg.get('search_chance_mode')}, "
        f"leaf={cfg.get('search_leaf_value_mode')}, "
        f"beta={cfg.get('search_beta')}, "
        f"scale={cfg.get('search_value_scale_mode')}",
        f"- seeds: {a['num_seeds']} (base {a['seed_base']}) × "
        f"sides {a['sides']} × {a['battles_per_seed']} battles/seed",
        f"- **{a['n_pairs']} paired battles** ({a['discordant_b'] + a['discordant_c']} discordant)",
        f"- elapsed: {a['elapsed_sec']:.0f}s ({a['elapsed_sec'] / 60:.1f} min)",
        "",
        "## Headline result",
        "",
        f"| metric | value |",
        f"|---|---|",
        f"| search win rate | {a['search_win_rate']:.4f} "
        f"(Wilson 95%: [{a['wilson_search'][0]:.4f}, {a['wilson_search'][1]:.4f}]) |",
        f"| baseline win rate | {a['baseline_win_rate']:.4f} "
        f"(Wilson 95%: [{a['wilson_baseline'][0]:.4f}, {a['wilson_baseline'][1]:.4f}]) |",
        f"| **paired delta (search - baseline)** | **{a['paired_delta']:+.4f}** |",
        f"| 95% paired bootstrap CI | "
        f"[{a['bootstrap_ci']['ci_low']:+.4f}, {a['bootstrap_ci']['ci_high']:+.4f}] |",
        f"| McNemar b (search>better) / c (search>worse) | "
        f"{a['discordant_b']} / {a['discordant_c']} |",
        f"| McNemar p-value ({a['mcnemar']['method']}) | {a['mcnemar']['p_value']:.4f} |",
        f"| both-lose (draw/incomplete) | {a['both_lose']} |",
        "",
        "## Per-side breakdown",
        "",
        "| side | n | search WR | baseline WR | delta | discordant b/c |",
        "|---|---|---|---|---|---|",
    ]
    for side_val, sd in sorted(a["by_side"].items()):
        md.append(
            f"| {side_val} | {sd['n']} | {sd['search_win_rate']:.4f} | "
            f"{sd['baseline_win_rate']:.4f} | {sd['delta']:+.4f} | "
            f"{sd['discordant_b']}/{sd['discordant_c']} |"
        )
    md += [
        "",
        "## Verdict (skill §23 go/no-go gate)",
        "",
        _verdict_text(a),
        "",
        "## Full analysis",
        "",
        "```json",
        json.dumps(a, indent=2, default=lambda o: str(o)),
        "```",
    ]
    with open(report_path, "w") as f:
        f.write("\n".join(md))

    return {
        "analysis": analysis_path,
        "runs": runs_path,
        "pairs": pairs_path,
        "report": report_path,
    }


def _verdict_text(a: Dict[str, Any]) -> str:
    """Skill §23 Phase 2 go/no-go gate verdict text."""
    ci_lo = a["bootstrap_ci"]["ci_low"]
    ci_hi = a["bootstrap_ci"]["ci_high"]
    delta = a["paired_delta"]
    b = a["discordant_b"]
    c = a["discordant_c"]
    n = a["n_pairs"]
    if n < 50:
        return (
            f"INCONCLUSIVE (n={n} < 50): too few pairs for a credible "
            "conclusion; this is a smoke check, not a result."
        )
    if a["both_lose"] > 0.1 * n:
        return (
            f"WARNING: {a['both_lose']} both-lose pairs ({a['both_lose'] / n:.1%} "
            "of n) -- draws or incomplete battles may be inflating noise; "
            "investigate before concluding."
        )
    if ci_lo > 0.0:
        return (
            f"POSITIVE: paired delta {delta:+.4f}, 95% CI [{ci_lo:+.4f}, "
            f"{ci_hi:+.4f}] excludes zero. Discordant pairs b={b} > c={c}. "
            "Corrected oracle search shows a held-out paired gain (skill §23)."
        )
    if ci_hi < 0.0:
        return (
            f"NEGATIVE: paired delta {delta:+.4f}, 95% CI [{ci_lo:+.4f}, "
            f"{ci_hi:+.4f}] excludes zero (on the negative side). Search is "
            "harmful at this configuration; do not deploy."
        )
    return (
        f"INCONCLUSIVE: paired delta {delta:+.4f}, 95% CI [{ci_lo:+.4f}, "
        f"{ci_hi:+.4f}] includes zero. The uncertainty is too large for a "
        f"conclusion; b={b}, c={c}, n={n}. Increase pairs or investigate "
        "estimator/opponent mismatch (skill §40)."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 paired + mirrored eval (skill §23)"
    )
    add_cli(parser)
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=3,
        help="number of seeds; each seed × sides × battles_per_seed = pairs",
    )
    parser.add_argument(
        "--battles_per_seed",
        type=int,
        default=100,
        help="battles per (seed, side, on/off) run; paired by battle index",
    )
    parser.add_argument(
        "--seed_base",
        type=int,
        default=1000,
        help="first seed; seeds are seed_base, seed_base+1, ... (distinct from "
        "the Phase 1 dev corpus to keep the held-out split honest)",
    )
    parser.add_argument(
        "--sides",
        type=int,
        nargs="+",
        default=[0, 1],
        help="physical sides to run (0 and 1 = mirrored; [0] = side-0 only)",
    )
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    # enforce research mode
    args.error_policy = "raise"
    args.search_mode = "oracle-root-mc"
    config = build_config_from_args(args)

    def progress(msg):
        print(msg, flush=True)

    result = run_paired_eval(
        agent_name=args.agent,
        checkpoint=args.checkpoint,
        battle_format=args.format,
        team_set_name=args.team_set,
        search_config=config,
        num_seeds=args.num_seeds,
        battles_per_seed=args.battles_per_seed,
        num_parallel=args.num_parallel,
        device=args.device,
        opponent_agent=args.opponent_agent,
        opponent_checkpoint=args.opponent_checkpoint,
        seed_base=args.seed_base,
        sides=tuple(args.sides),
        progress_fn=progress,
    )
    paths = write_paired_results(result, args.output_dir)
    a = result["analysis"]
    print(
        json.dumps(
            {
                "n_pairs": a["n_pairs"],
                "paired_delta": a["paired_delta"],
                "bootstrap_ci": a["bootstrap_ci"],
                "discordant_b": a["discordant_b"],
                "discordant_c": a["discordant_c"],
                "verdict": _verdict_text(a).split(":")[0],
                "outputs": paths,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
