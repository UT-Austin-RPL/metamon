"""Fixed-root estimator convergence benchmark (skill §22 Phase 1).

The first scientific question after the Phase 0 correctness gate is **not**
"does search win more games?" but:

> Does increasing rollout compute produce a more stable and more accurate root
> action ranking?

This module answers it. For each fixed root it runs the rollout estimator once
at a high rollout count ``K_ref`` (the reference) and **derives** every lower-K
estimate from that single run by prefix / block averaging. Derivation is valid
because the per-rollout-index ``k`` branch seed is K-independent
(``rng.RootSeedBank`` keys on ``(root, k)``, not on ``K``): the first ``k``
rollouts of a K=256 run are identical to a K=4 run's 4 rollouts at the same
root. So ``Q_K'(a) = mean(R[a, :K'])`` is exactly what a standalone K' run would
have produced with those chance streams, and non-overlapping blocks of size
``K'`` give an independent sample of the K' estimator's sampling distribution
(block means) -- enough to measure top-action agreement, rank correlation,
simple regret, and SE calibration as ``K`` grows, without re-running the
simulator for every ``K``.

Configs compared per root (skill §22 "Candidate configurations" / §39):

* ``root_critic_only`` -- no rollout; ``Q_root(a) = frozen critic Q(h_root, a)``;
* ``D=0`` at ``K_ref`` -- settle the root exchange then exact-``V_pi`` bootstrap;
* ``D=1`` at ``K_ref`` -- one additional policy-guided settled decision;
* (optional) ``inherited_trunk_rng`` D=0 -- the future-chance oracle diagnostic.

The reference for D=0 is the full-``K_ref`` D=0 mean; for D=1 the full-``K_ref``
D=1 mean. Convergence is judged by whether K={4,16,64} prefix estimates move
toward the reference as K grows (skill §22 "Expected convergence pattern" /
"Phase 1 go/no-go gate").

Output: a per-root JSONL (one record per root with the full K/depth grid of
per-action Q + the derived metrics), a manifest JSONL (skill §31), a run-level
manifest JSON (skill §20), and a summary report (JSON + Markdown).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import SearchConfig
from .root_dataset import (
    RootManifestEntry,
    make_manifest_entry,
    fill_reference_fields,
    write_manifest,
    run_manifest,
    entropy_band,
    top2_gap_band,
    ref_gap_band,
    phase_band,
)

# ---------------------------------------------------------------------------
# Rank correlations (numpy-only, no scipy dependency)
# ---------------------------------------------------------------------------


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), tie-aware (matches scipy's 'average' method)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    sx = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (Pearson of ranks). NaN if degenerate."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2:
        return float("nan")
    ra, rb = _rankdata(a), _rankdata(b)
    ca = ra - ra.mean()
    cb = rb - rb.mean()
    denom = np.sqrt((ca * ca).sum() * (cb * cb).sum())
    return float(np.dot(ca, cb) / denom) if denom > 0 else float("nan")


def kendall_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Kendall tau-a: (concordant - discordant) / (n choose 2). O(n^2); n<=~13."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = a.size
    if n < 2:
        return float("nan")
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    return float((conc - disc) / (n * (n - 1) / 2))


# ---------------------------------------------------------------------------
# Per-branch return matrix + derivation (skill §31 "Reference uncertainty")
# ---------------------------------------------------------------------------


def branch_return_matrix(estimate) -> np.ndarray:
    """Reshape a :class:`RootEstimate`'s per-branch returns into ``R (A, K)``.

    ``R[a, k]`` is the return of the branch that forced legal action ``a`` with
    rollout-index ``k``'s chance stream. ``NaN`` only if a (a, k) pair is
    missing (should not happen for rollout modes). Returns an empty array for
    ``root_critic_only`` (no per-branch returns).
    """
    if estimate.root_action is None or estimate.q_per_branch is None:
        return np.empty((0, 0))
    legal_arr = np.asarray(estimate.legal_arr)
    A = legal_arr.size
    K = int(estimate.K)
    R = np.full((A, K), np.nan, dtype=np.float64)
    for ai, a in enumerate(legal_arr):
        idxs = np.where(estimate.root_action == a)[0]
        ks = estimate.rollout_index[idxs]
        R[ai, ks] = estimate.q_per_branch[idxs]
    return R


def reference_q(R: np.ndarray) -> np.ndarray:
    """Full-``K`` mean return per action: the high-K reference estimate."""
    return np.nanmean(R, axis=1)


def prefix_q(R: np.ndarray, k_prime: int) -> np.ndarray:
    """Q estimate using the first ``k_prime`` chance streams (a single K' run)."""
    k = min(int(k_prime), R.shape[1])
    return np.nanmean(R[:, :k], axis=1)


def block_means(R: np.ndarray, block_size: int) -> np.ndarray:
    """Non-overlapping block means of size ``block_size`` -> ``(n_blocks, A)``.

    Each row is an independent K'=``block_size`` estimate from a disjoint set of
    chance streams. Used for SE calibration (does the block spread match the
    theoretical ``std/sqrt(K')``?) and for the distribution of top-action
    selections across chance-stream draws.
    """
    A, K = R.shape
    n = K // int(block_size)
    if n == 0:
        return np.empty((0, A))
    Rtrim = R[:, : n * int(block_size)]
    return Rtrim.reshape(A, n, int(block_size)).mean(axis=2).T  # (n, A)


def split_half_top1_agreement(R: np.ndarray) -> float:
    """Fraction-style {0,1}: do the two halves of K agree on the top action?

    A stable reference should agree with itself across split halves (skill §22
    "Check split-half stability of the reference itself").
    """
    A, K = R.shape
    half = K // 2
    if half < 1:
        return float("nan")
    q1 = np.nanmean(R[:, :half], axis=1)
    q2 = np.nanmean(R[:, half : 2 * half], axis=1)
    return float(int(np.argmax(q1) == np.argmax(q2)))


# ---------------------------------------------------------------------------
# Per-root convergence metrics (skill §22 "Estimator metrics")
# ---------------------------------------------------------------------------


@dataclass
class RootConvergenceMetrics:
    """Convergence of the K' prefix estimate toward the high-K reference."""

    k_prime: int
    depth: int
    leaf_mode: str
    # ranking vs reference
    top1_agree: int  # 1 if argmax(Q_K') == argmax(Q_ref)
    kp_argmax: int
    ref_argmax: int
    regret: float  # Q_ref(a*_ref) - Q_ref(a*_K')  (skill §31 root-level regret)
    mae: float  # mean |Q_K' - Q_ref|
    spearman: float
    kendall: float
    # SE calibration (block means)
    theo_se_mean: float  # mean over actions of std(R[a,:])/sqrt(K')
    block_std_mean: float  # mean over actions of std(block_means)
    se_ratio: float  # block_std_mean / theo_se_mean (~1 if calibrated)
    block_top1_agree: float  # frac of K'-blocks whose argmax == ref argmax
    n_blocks: int

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def root_convergence_metrics(
    R: np.ndarray,
    k_prime: int,
    depth: int,
    leaf_mode: str,
    legal_arr: np.ndarray,
    ref_q: Optional[np.ndarray] = None,
) -> Optional[RootConvergenceMetrics]:
    """Compute convergence metrics for one (depth, leaf_mode, K') cell.

    Returns ``None`` for degenerate roots (``A < 2`` -- no ranking to compare).
    """
    A, K = R.shape
    if A < 2:
        return None
    if ref_q is None:
        ref_q = reference_q(R)
    ref_argmax = int(np.argmax(ref_q))
    q_kp = prefix_q(R, k_prime)
    kp_argmax = int(np.argmax(q_kp))
    regret = float(ref_q[ref_argmax] - ref_q[kp_argmax])
    mae = float(np.mean(np.abs(q_kp - ref_q)))
    sp = spearman_corr(q_kp, ref_q)
    kd = kendall_corr(q_kp, ref_q)

    blocks = block_means(R, k_prime)  # (n_blocks, A)
    n_blocks = int(blocks.shape[0])
    if n_blocks >= 2:
        block_std = blocks.std(axis=0, ddof=1)  # (A,)
        theo_se = np.nanstd(R, axis=1, ddof=1) / np.sqrt(k_prime)  # (A,)
        block_std_mean = float(np.nanmean(block_std))
        theo_se_mean = float(np.nanmean(theo_se))
        se_ratio = float(
            block_std_mean / theo_se_mean if theo_se_mean > 1e-9 else float("nan")
        )
        block_top1 = float(np.mean([int(np.argmax(b)) == ref_argmax for b in blocks]))
    else:
        block_std_mean = theo_se_mean = se_ratio = block_top1 = float("nan")
        n_blocks = 0

    return RootConvergenceMetrics(
        k_prime=int(k_prime),
        depth=int(depth),
        leaf_mode=str(leaf_mode),
        top1_agree=int(kp_argmax == ref_argmax),
        kp_argmax=int(legal_arr[kp_argmax]),
        ref_argmax=int(legal_arr[ref_argmax]),
        regret=regret,
        mae=mae,
        spearman=float(sp),
        kendall=float(kd),
        theo_se_mean=theo_se_mean,
        block_std_mean=block_std_mean,
        se_ratio=se_ratio,
        block_top1_agree=block_top1,
        n_blocks=n_blocks,
    )


# ---------------------------------------------------------------------------
# Per-root result record (JSONL)
# ---------------------------------------------------------------------------


@dataclass
class RootResultRecord:
    """One root's full estimator grid + derived convergence metrics (JSONL)."""

    root_id: str
    battle_id: str
    lane: int
    decision: int
    legal_actions: List[int]
    base_probs: List[float]
    base_argmax: int
    base_entropy: float
    base_top2_gap: float
    entropy_band: str
    top2_gap_band: str
    phase_band: str
    n_legal: int
    # per-config per-action Q (over legal_actions) + argmax + latency
    configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # derived convergence metrics (key: f"{leaf_mode}:D{depth}:K{k_prime}")
    convergence: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # reference self-stability
    split_half_top1_d0: Optional[float] = None
    split_half_top1_d1: Optional[float] = None
    # per-branch return matrices are NOT stored by default (large); the
    # per-action Q means + the reference are enough for the aggregate report.
    # Set ``store_branch_matrices=True`` to keep R (A,K) per rollout config.

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


def _config_summary(estimate, legal_arr: np.ndarray) -> Dict[str, Any]:
    """Per-config summary (per-action Q, argmax, std, latency, diag)."""
    q = np.asarray(estimate.q_mean, dtype=np.float64)
    return {
        "K": int(estimate.K),
        "depth": int(estimate.search_depth),
        "leaf_mode": estimate.leaf_value_mode,
        "chance_mode": estimate.chance_mode,
        "q_mean": [float(x) for x in q],
        "q_std": [float(x) for x in estimate.q_std],
        "q_argmax": int(legal_arr[int(np.argmax(q))]) if q.size else None,
        "term_frac": [float(x) for x in estimate.term_frac],
        "counts": [int(x) for x in estimate.counts],
        "critic_disagreement": float(estimate.diag.get("critic_disagreement", 0.0)),
        "intermediate_reward_mean": float(
            estimate.diag.get("intermediate_reward_mean", 0.0)
        ),
        "bootstrap_mean": float(estimate.diag.get("bootstrap_mean", 0.0)),
        "n_settled_mean": float(estimate.diag.get("n_settled_mean", 0.0)),
        "latency_ms": float(estimate.latency_ms),
    }


# ---------------------------------------------------------------------------
# Config grid
# ---------------------------------------------------------------------------


def _base_search_config(**overrides) -> SearchConfig:
    """Research-safe defaults for the benchmark (skill §15)."""
    cfg = SearchConfig(
        search_mode="oracle-root-mc",
        search_root_candidate_mode="all_legal",
        search_chance_mode="resample_crn",
        search_root_opponent_coupling=True,
        search_leaf_value_mode="policy_expectation",
        search_value_normalization=False,
        search_ablation="single_anchor_kl",
        search_error_policy="raise",
        search_include_intermediate_rewards=True,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def build_grid_configs(
    k_ref: int,
    depths: List[int],
    include_inherited_rng: bool = False,
    search_seed: int = 0,
) -> Dict[str, SearchConfig]:
    """Build the per-root estimator grid configs.

    Returns a dict ``name -> SearchConfig``. The ``root_critic_only`` config has
    no rollout; each ``d{depth}`` config runs K=``k_ref`` rollouts at that depth.
    ``inherited_d0`` is the future-chance oracle diagnostic (skill §7).
    """
    grid: Dict[str, SearchConfig] = {}
    grid["root_critic_only"] = _base_search_config(
        search_leaf_value_mode="root_critic_only", search_seed=search_seed
    )
    for d in depths:
        grid[f"d{d}"] = _base_search_config(
            search_rollouts_per_action=k_ref,
            search_depth=d,
            search_seed=search_seed,
        )
    if include_inherited_rng:
        grid["inherited_d0"] = _base_search_config(
            search_rollouts_per_action=k_ref,
            search_depth=0,
            search_chance_mode="inherited_trunk_rng",
            search_seed=search_seed,
        )
    return grid


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def _last_committed(env, lane: int) -> Tuple[int, int]:
    """(eval_action, opp_action) committed on ``lane`` in the last env.step."""
    eval_a = int(getattr(env, "_step_eval_actions", [0])[lane])
    opp_a = getattr(env, "_committed_side_actions", {}).get((lane, env.opp_side))
    return eval_a, (int(opp_a) if opp_a is not None else -1)


def benchmark_roots(
    *,
    bundle,
    config_grid: Dict[str, SearchConfig],
    k_ref: int,
    derived_ks: List[int],
    depths: List[int],
    max_roots: int = 64,
    max_battles: int = 20,
    root_stride: int = 1,
    decision_stride: int = 1,
    min_decision: int = 0,
    max_decision: Optional[int] = None,
    store_branch_matrices: bool = False,
    progress_every: int = 5,
    env_seed: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> Tuple[List[RootResultRecord], List[RootManifestEntry]]:
    """Run the in-battle fixed-root benchmark (skill §22).

    Drives the vectorized env with the **baseline** frozen policy (so the root
    corpus is natural self-play), and at each settled eval-side decision runs
    the full estimator grid via ``runner.estimate_root`` at the *same* trunk
    state (search forks never advance the trunk), then takes the baseline action
    to continue. Stops once ``max_roots`` roots are captured or ``max_battles``
    battles complete.

    Phase coverage (skill §22 "Stratify across early/middle/late battle"): the
    ``decision_stride`` / ``min_decision`` / ``max_decision`` filters select
    *which* eval-decisions per battle are benchmarked, based on the **true
    per-battle decision index** (``lane_total_dec``). ``decision_stride=3``
    captures every 3rd decision per battle, spreading the corpus across
    early/mid/late instead of clustering on the opening (the Phase 1 pilot's
    ``root_stride=1`` + small ``max_roots`` produced an all-early corpus).

    Args:
        bundle: a ``FrozenBundle``-like object (env, eval_driver, opponent,
            eval_policy, opponent_policy, model, opp_model, action_dim, device,
            reward_multiplier, eval/opponent_action_space, battle_format).
        config_grid: ``name -> SearchConfig`` from :func:`build_grid_configs`.
        k_ref: the reference rollout count (the high-K runs).
        derived_ks: low-K values derived from each high-K run via prefix/block
            averaging (e.g. ``[4, 16, 64]``).
        depths: rollout depths with a high-K run (e.g. ``[0, 1]``).
        max_roots / max_battles: corpus caps.
        root_stride: global capture cadence (every N-th *encountered* eval-
            decision across all lanes); use for a quick smoke. Prefer
            ``decision_stride`` for phase coverage.
        decision_stride: per-battle capture cadence (capture when
            ``lane_total_dec % decision_stride == 0``); spreads across phases.
        min_decision / max_decision: optional per-battle decision-index window
            (e.g. ``min_decision=20`` to skip the opening and target mid/late).
        store_branch_matrices: keep the per-branch ``R (A,K)`` matrices in each
            record (verbose; needed only for deep post-hoc analysis).
        output_dir: if set, stream each root record + manifest entry to
            ``<output_dir>/root_results.jsonl`` and ``root_manifest.jsonl`` as
            they are produced (flushed per record). A crash or signal kill mid-
            run then still leaves the first N roots on disk for analysis; the
            summary/comparison/report are written at the end by ``write_results``
            (or regenerated from the streamed JSONL by a recovery tool).

    Returns ``(records, manifest_entries)``.
    """
    import torch
    from metamon.env.vectorized.obs_utils import unstack_obs_dicts
    from .search_driver import SearchEvalRunner

    env = bundle.env
    eval_driver = bundle.eval_driver
    runner = SearchEvalRunner(
        env=env,
        eval_driver=eval_driver,
        opponent=bundle.opponent,
        eval_policy=bundle.eval_policy,
        opponent_policy=bundle.opponent_policy,
        eval_action_space=bundle.eval_action_space,
        opponent_action_space=bundle.opponent_action_space,
        eval_reward_function=bundle.model.reward_function,
        opponent_reward_function=bundle.opp_model.reward_function,
        config=next(iter(config_grid.values())),  # placeholder; estimate_root swaps
        device=bundle.device,
        action_dim=bundle.action_dim,
        battle_format=bundle.battle_format,
        reward_multiplier=bundle.reward_multiplier,
    )

    n = env.batched_envs
    obs, info = env.reset()
    # per-lane battle / decision counters (stable, distinct root identities for
    # the seed bank: (battle_id, side, decision_idx) is unique per root).
    lane_battle = [0] * n
    lane_decision = [0] * n  # benchmarked count (legacy; kept for compat)
    lane_total_dec = [0] * n  # true per-battle decision index (phase filter)
    lane_history: List[List[List[int]]] = [[] for _ in range(n)]

    records: List[RootResultRecord] = []
    manifest: List[RootManifestEntry] = []
    # Incremental streaming (skill §35 robustness): a high-K phase-spanning run
    # can be killed mid-benchmark by a signal (e.g. the Node host hitting its
    # V8 heap limit under ~A*K concurrent battles) with no Python traceback,
    # losing all results if they are only written at the end. Stream each record
    # to JSONL as it is produced so the first N roots survive a crash.
    _roots_fh = None
    _manifest_fh = None
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        _roots_fh = open(os.path.join(output_dir, "root_results.jsonl"), "w")
        _manifest_fh = open(os.path.join(output_dir, "root_manifest.jsonl"), "w")
    battles_done = 0
    steps = 0
    max_steps = max(max_battles * 400 // n + 200, 400)
    root_capture_idx = 0  # only every root_stride-th eval decision is benchmarked

    try:
        while (
            len(records) < max_roots
            and battles_done < max_battles
            and steps < max_steps
        ):
            steps += 1
            obs_list = unstack_obs_dicts(obs)
            actions = np.zeros(n, dtype=np.int64)
            lane_root_this_step = [None] * n  # (record, manifest_entry) per lane

            for i in range(n):
                lane = env.lanes[i]
                if lane.ended or not lane.needs_agent_decision(env.eval_side):
                    actions[i] = 0
                    continue
                legal = info["legal_actions"][i]
                root_capture_idx += 1
                lane_total_dec[i] += 1
                tdec = lane_total_dec[i]
                # phase-spreading capture filter: per-battle decision stride +
                # optional window (skill §22 phase stratification). The global
                # root_stride is an additional cadence (kept for smoke runs).
                in_window = (
                    (tdec % max(decision_stride, 1) == 0)
                    and (tdec >= min_decision)
                    and (max_decision is None or tdec <= max_decision)
                )
                run_grid = (
                    (root_capture_idx % max(root_stride, 1) == 0)
                    and in_window
                    and (len(records) < max_roots)
                )
                if run_grid:
                    battle_id = f"b{i}_{lane_battle[i]}"
                    decision_idx = tdec  # true per-battle decision index
                    # stable identity for the seed bank across all grid configs
                    runner._battle_id = battle_id
                    runner._decision_counter = decision_idx
                    try:
                        rec, mentry = _benchmark_one_root(
                            runner=runner,
                            bundle=bundle,
                            lane_idx=i,
                            obs=obs_list[i],
                            legal=legal,
                            config_grid=config_grid,
                            k_ref=k_ref,
                            derived_ks=derived_ks,
                            depths=depths,
                            battle_id=battle_id,
                            decision_idx=decision_idx,
                            battle_seed=env_seed,
                            action_history=list(lane_history[i]),
                            store_branch_matrices=store_branch_matrices,
                        )
                        lane_root_this_step[i] = (rec, mentry)
                    except Exception as exc:  # noqa: BLE001
                        # Research runs fail loudly (skill §19). estimate_root
                        # cleans up its branches before re-raising, so the trunk
                        # is safe; surface the failing root for diagnosis.
                        raise RuntimeError(
                            f"benchmark failed at root {battle_id}:d{decision_idx}: {exc!r}"
                        ) from exc
                # advance the trunk with the baseline action (natural self-play)
                active = np.zeros(n, dtype=bool)
                active[i] = True
                actions[i] = int(eval_driver.act(active, obs_list)[i])

            obs, rewards, terminated, truncated, info = env.step(actions)
            for i in range(n):
                eval_driver.observe(i, float(rewards[i]), int(actions[i]))

            # record committed actions for the manifest replay history
            for i in range(n):
                if lane_root_this_step[i] is not None:
                    rec, mentry = lane_root_this_step[i]
                    records.append(rec)
                    manifest.append(mentry)
                    if _roots_fh is not None:
                        try:
                            _roots_fh.write(rec.to_json() + "\n")
                            _roots_fh.flush()
                        except OSError:
                            # Disk-full etc: the record is in `records` in
                            # memory; streaming is best-effort crash safety, not
                            # a hard requirement. Keep benchmarking.
                            _roots_fh = None
                    if _manifest_fh is not None:
                        try:
                            _manifest_fh.write(mentry.to_json() + "\n")
                            _manifest_fh.flush()
                        except OSError:
                            _manifest_fh = None
                    lane_decision[i] += 1
                    if (len(records) % progress_every) == 0:
                        print(
                            f"  [benchmark] {len(records)}/{max_roots} roots, "
                            f"{battles_done} battles, step {steps}"
                        )
                    if len(records) >= max_roots:
                        break
                if (
                    env.lanes[i].needs_agent_decision(env.eval_side)
                    or env.lanes[i].ended
                ):
                    # this lane just had a decision cycle -> record its action
                    ea, oa = _last_committed(env, i)
                    lane_history[i].append([ea, oa])

            done = terminated | truncated
            if done.any():
                for i in np.where(done)[0]:
                    battles_done += 1
                    eval_driver.reset_lanes(
                        np.array([i == j for j in range(n)], dtype=bool)
                    )
                    env.opponent.reset_lanes(
                        np.array([i == j for j in range(n)], dtype=bool)
                    )
                    lane_battle[i] += 1
                    lane_decision[i] = 0
                    lane_total_dec[i] = 0
                    lane_history[i] = []
    finally:
        runner.close()
        if _roots_fh is not None:
            _roots_fh.close()
        if _manifest_fh is not None:
            _manifest_fh.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return records, manifest


def _benchmark_one_root(
    *,
    runner,
    bundle,
    lane_idx: int,
    obs: dict,
    legal: List[int],
    config_grid: Dict[str, SearchConfig],
    k_ref: int,
    derived_ks: List[int],
    depths: List[int],
    battle_id: str,
    decision_idx: int,
    battle_seed: Optional[int],
    action_history: List[List[int]],
    store_branch_matrices: bool,
) -> Tuple[RootResultRecord, RootManifestEntry]:
    """Run the full estimator grid at one fixed root; return (record, manifest)."""
    estimates: Dict[str, Any] = {}
    legal_arr = None
    base_probs = None
    base_argmax = None

    # Run the grid. root_critic_only first (cheap; gives base distribution for
    # the manifest); then the rollout configs. All see the SAME trunk state.
    order = ["root_critic_only"] + [f"d{d}" for d in depths]
    if "inherited_d0" in config_grid:
        order.append("inherited_d0")
    for name in order:
        if name not in config_grid:
            continue
        est = runner.estimate_root(lane_idx, obs, legal, config_grid[name])
        estimates[name] = est
        if legal_arr is None:
            legal_arr = np.asarray(est.legal_arr)
            base_probs = est.base_probs
            base_argmax = est.base_argmax

    assert legal_arr is not None
    # manifest entry (pre-estimate features from the base distribution)
    mentry = make_manifest_entry(
        battle_id=battle_id,
        lane=lane_idx,
        decision=decision_idx,
        battle_seed=battle_seed,
        legal=legal,
        base_probs=base_probs,
        legal_arr=legal_arr,
        base_argmax=base_argmax,
        action_history=action_history,
    )

    # per-config summaries + convergence metrics
    config_summaries: Dict[str, Dict[str, Any]] = {}
    convergence: Dict[str, Dict[str, Any]] = {}
    split_d0 = split_d1 = None

    for name, est in estimates.items():
        config_summaries[name] = _config_summary(est, legal_arr)
        R = branch_return_matrix(est)
        if store_branch_matrices and R.size:
            config_summaries[name]["R"] = R.tolist()
        # convergence metrics only for the rollout configs (have per-branch R)
        if est.leaf_value_mode == "root_critic_only" or R.size == 0:
            continue
        ref_q = reference_q(R)
        d = est.search_depth
        for kp in derived_ks:
            if kp > k_ref:
                continue
            m = root_convergence_metrics(
                R,
                kp,
                depth=d,
                leaf_mode=est.leaf_value_mode,
                legal_arr=legal_arr,
                ref_q=ref_q,
            )
            if m is not None:
                convergence[f"{est.leaf_value_mode}:D{d}:K{kp}"] = m.to_dict()
        # reference self-stability (split-half) per depth
        if d == 0:
            split_d0 = split_half_top1_agreement(R)
        elif d == 1:
            split_d1 = split_half_top1_agreement(R)

    # fill the manifest's reference fields from the D=0 high-K estimate
    d0_est = estimates.get("d0")
    if d0_est is not None and d0_est.q_mean.size:
        R0 = branch_return_matrix(d0_est)
        ref_q0 = reference_q(R0) if R0.size else d0_est.q_mean
        fill_reference_fields(
            mentry,
            ref_q=ref_q0,
            legal_arr=legal_arr,
            critic_disagreement=float(d0_est.diag.get("critic_disagreement", 0.0)),
            terminal_frac_d0=float(
                np.mean(d0_est.term_frac) if d0_est.term_frac.size else 0.0
            ),
        )

    record = RootResultRecord(
        root_id=mentry.root_id,
        battle_id=battle_id,
        lane=lane_idx,
        decision=decision_idx,
        legal_actions=[int(x) for x in legal_arr],
        base_probs=[float(x) for x in base_probs[legal_arr]],
        base_argmax=int(base_argmax),
        base_entropy=mentry.base_entropy,
        base_top2_gap=mentry.base_top2_gap,
        entropy_band=mentry.entropy_band,
        top2_gap_band=mentry.top2_gap_band,
        phase_band=mentry.phase_band,
        n_legal=int(legal_arr.size),
        configs=config_summaries,
        convergence=convergence,
        split_half_top1_d0=split_d0,
        split_half_top1_d1=split_d1,
    )
    return record, mentry


# ---------------------------------------------------------------------------
# Aggregate analysis
# ---------------------------------------------------------------------------


def aggregate_convergence(
    records: List[RootResultRecord],
) -> Dict[str, Any]:
    """Aggregate per-root convergence metrics across the corpus (skill §22).

    Reports, per ``(leaf_mode, depth, K')`` cell and stratified by entropy /
    top-2-gap / phase / reference-gap band:
    * mean top-1 agreement with the reference (should rise with K);
    * mean block top-1 agreement (chance-stream draw distribution);
    * mean simple regret (should fall with K);
    * mean MAE and rank correlations (should improve with K);
    * SE calibration ratio (block spread / theoretical SE; ~1);
    * reference split-half top-1 agreement (reference stability).
    """
    # collect per-cell lists
    cells: Dict[str, Dict[str, List[float]]] = {}
    split_d0: List[float] = []
    split_d1: List[float] = []

    def cell(key: str) -> Dict[str, List[float]]:
        return cells.setdefault(key, _empty_metric_lists())

    for r in records:
        if r.split_half_top1_d0 is not None and not np.isnan(r.split_half_top1_d0):
            split_d0.append(float(r.split_half_top1_d0))
        if r.split_half_top1_d1 is not None and not np.isnan(r.split_half_top1_d1):
            split_d1.append(float(r.split_half_top1_d1))
        for key, m in r.convergence.items():
            c = cell(key)
            c["top1_agree"].append(float(m["top1_agree"]))
            c["block_top1_agree"].append(float(m["block_top1_agree"]))
            c["regret"].append(float(m["regret"]))
            c["mae"].append(float(m["mae"]))
            c["spearman"].append(float(m["spearman"]))
            c["kendall"].append(float(m["kendall"]))
            c["se_ratio"].append(float(m["se_ratio"]))
            c["theo_se_mean"].append(float(m["theo_se_mean"]))
            c["block_std_mean"].append(float(m["block_std_mean"]))
            # stratification tags (same for all cells of one root)
            c["_entropy_band"].append(r.entropy_band)
            c["_top2_gap_band"].append(r.top2_gap_band)
            c["_phase_band"].append(r.phase_band)

    summary: Dict[str, Any] = {}
    for key, c in cells.items():
        summary[key] = _summarize_cell(c)
    summary["_reference_stability"] = {
        "split_half_top1_d0": _nanmean(split_d0),
        "split_half_top1_d1": _nanmean(split_d1),
        "n_d0": len(split_d0),
        "n_d1": len(split_d1),
    }
    summary["_n_roots"] = len(records)
    return summary


def _empty_metric_lists() -> Dict[str, List]:
    return {
        "top1_agree": [],
        "block_top1_agree": [],
        "regret": [],
        "mae": [],
        "spearman": [],
        "kendall": [],
        "se_ratio": [],
        "theo_se_mean": [],
        "block_std_mean": [],
        "_entropy_band": [],
        "_top2_gap_band": [],
        "_phase_band": [],
    }


def _summarize_cell(c: Dict[str, List]) -> Dict[str, Any]:
    def nm(x):
        return float(np.nanmean(x)) if len(x) else float("nan")

    out = {
        "n": len(c["top1_agree"]),
        "top1_agree": nm(c["top1_agree"]),
        "block_top1_agree": nm(c["block_top1_agree"]),
        "regret_mean": nm(c["regret"]),
        "regret_p90": (
            float(np.nanpercentile(c["regret"], 90))
            if len(c["regret"])
            else float("nan")
        ),
        "mae_mean": nm(c["mae"]),
        "spearman_mean": nm(c["spearman"]),
        "kendall_mean": nm(c["kendall"]),
        "se_ratio_mean": nm(c["se_ratio"]),
        "theo_se_mean": nm(c["theo_se_mean"]),
        "block_std_mean": nm(c["block_std_mean"]),
    }
    # stratified top-1 agreement
    for bandkey, tagkey in [
        ("by_entropy", "_entropy_band"),
        ("by_top2_gap", "_top2_gap_band"),
        ("by_phase", "_phase_band"),
    ]:
        bands: Dict[str, List[float]] = {}
        for v, tag in zip(c["top1_agree"], c[tagkey]):
            bands.setdefault(tag, []).append(float(v))
        out[bandkey] = {b: float(np.nanmean(v)) for b, v in bands.items() if v}
    return out


def _nanmean(x: List[float]) -> float:
    x = [v for v in x if not np.isnan(v)]
    return float(np.mean(x)) if x else float("nan")


def convergence_table(summary: Dict[str, Any], derived_ks: List[int]) -> str:
    """Render the headline K-convergence table (Markdown) for the report."""
    lines = [
        "## K-convergence (top-1 agreement with high-K reference)",
        "",
        "| leaf | D | K | n | top1_agree | block_top1 | regret_mean | MAE | spearman | se_ratio |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for d in (0, 1):
        for kp in derived_ks:
            key = f"policy_expectation:D{d}:K{kp}"
            if key not in summary:
                continue
            s = summary[key]
            lines.append(
                f"| policy_expectation | {d} | {kp} | {s['n']} | "
                f"{s['top1_agree']:.3f} | {s['block_top1_agree']:.3f} | "
                f"{s['regret_mean']:.1f} | {s['mae_mean']:.1f} | "
                f"{s['spearman_mean']:.3f} | {s['se_ratio_mean']:.3f} |"
            )
    # root critic only baseline
    lines.append("")
    lines.append("Reference self-stability (split-half top-1 agreement):")
    rs = summary.get("_reference_stability", {})
    lines.append(
        f"- D=0: {rs.get('split_half_top1_d0', float('nan')):.3f} "
        f"(n={rs.get('n_d0', 0)})"
    )
    lines.append(
        f"- D=1: {rs.get('split_half_top1_d1', float('nan')):.3f} "
        f"(n={rs.get('n_d1', 0)})"
    )
    return "\n".join(lines)


def go_no_go_assessment(
    summary: Dict[str, Any], derived_ks: List[int]
) -> Dict[str, Any]:
    """Phase 1 go/no-go gate (skill §22 "Phase 1 go/no-go gate").

    Returns a dict with per-criterion pass/fail and an overall verdict. This is
    a *structured* assessment of the convergence direction, not a win-rate
    claim; a fail means "stop and debug the estimator", not "search is useless".
    """
    criteria: Dict[str, Any] = {}

    # 1. high-K reference is itself stable (split-half top-1 agreement > 0.8)
    rs = summary.get("_reference_stability", {})
    ref_stab = rs.get("split_half_top1_d0", float("nan"))
    criteria["reference_stable_d0"] = {
        "value": ref_stab,
        "pass": bool(not np.isnan(ref_stab) and ref_stab > 0.80),
    }

    # 2. increasing K improves top-1 agreement (monotone non-decreasing)
    agrees = []
    for kp in sorted(derived_ks):
        key = f"policy_expectation:D0:K{kp}"
        if key in summary:
            agrees.append((kp, summary[key]["top1_agree"]))
    mono = all(agrees[i][1] <= agrees[i + 1][1] + 1e-9 for i in range(len(agrees) - 1))
    improved = bool(len(agrees) >= 2 and agrees[-1][1] > agrees[0][1])
    criteria["top1_agreement_rises_with_K_D0"] = {
        "values": agrees,
        "monotone": mono,
        "improved": improved,
        "pass": bool(mono and improved),
    }

    # 3. regret falls with K
    regrets = []
    for kp in sorted(derived_ks):
        key = f"policy_expectation:D0:K{kp}"
        if key in summary:
            regrets.append((kp, summary[key]["regret_mean"]))
    regret_falls = bool(len(regrets) >= 2 and regrets[-1][1] <= regrets[0][1] + 1e-9)
    criteria["regret_falls_with_K_D0"] = {
        "values": regrets,
        "pass": regret_falls,
    }

    # 4. SE calibration: block spread ~ theoretical SE (ratio in [0.7, 1.4])
    se_ok = True
    for kp in derived_ks:
        key = f"policy_expectation:D0:K{kp}"
        if key in summary:
            r = summary[key]["se_ratio_mean"]
            if not np.isnan(r) and not (0.7 <= r <= 1.4):
                se_ok = False
    criteria["se_calibrated_D0"] = {"pass": se_ok}

    # 5. root critic only / D0 / D1 interpretable differences (sanity: recorded)
    criteria["configs_present"] = {
        "pass": bool(
            "root_critic_only" in summary or any("D0:K" in k for k in summary)
        ),
    }

    passed = sum(1 for c in criteria.values() if c.get("pass"))
    total = len(criteria)
    verdict = (
        "PASS" if passed == total else ("INCONCLUSIVE" if passed == 0 else "PARTIAL")
    )
    return {
        "verdict": verdict,
        "passed": passed,
        "total": total,
        "criteria": criteria,
        "note": (
            "PASS = the rollout estimator converges in the expected direction as K "
            "grows (skill §22 gate). A non-PASS means stop and debug the estimator, "
            "not that search is useless."
        ),
    }


# ---------------------------------------------------------------------------
# Estimator head-to-head: actor vs root_critic_only vs D=0 vs D=1 (skill §40)
# ---------------------------------------------------------------------------


def _critic_disagreement_band(value: float, threshold: float) -> str:
    """Low/high critic-disagreement band (threshold frozen from the dev corpus)."""
    return "high" if value >= threshold else "low"


def estimator_comparison(
    records: List[RootResultRecord],
    reference_name: str = "d0",
    legacy_prune_threshold: float = 0.05,
) -> Dict[str, Any]:
    """Head-to-head: actor vs root_critic_only vs D=0 (ref) vs D=1 (skill §40).

    This is the decision-relevant analysis that the convergence gate does NOT
    cover: it asks whether the estimators *disagree with the actor* (addressable
    opportunity) and whether the real simulator transition *adds anything over a
    cheap critic rerank* (D=0 vs root_critic_only). The reference is the high-K
    D=0 estimate (our best estimator); D=0 is the reference by construction, so
    its disagreement is 0 and it is not in the regret table.

    Per root we read the argmax of each config from ``record.configs[name][
    "q_argmax"]`` and the reference Q from ``record.configs[reference_name][
    "q_mean"]`` (over ``record.legal_actions``). We also compute the §39 Q6
    pruning diagnostic: would the legacy 5%-of-max pruning rule have dropped the
    reference-best action?

    Returns a dict with:
    * ``n_roots``;
    * disagreement-with-reference rates for actor / critic_only / d1 (frac of
      roots whose argmax != ref argmax);
    * pairwise agreement rates (actor-vs-critic, critic-vs-d0, d0-vs-d1);
    * mean regret vs reference for actor / critic_only / d1
      (``ref_q[ref_argmax] - ref_q[their_argmax]`` -- skill §31 root regret);
    * all of the above stratified by entropy / phase / critic-disagreement /
      reference-gap band;
    * the pruning diagnostic (frac where the legacy rule would drop ref-best).
    """
    if not records:
        return {"n_roots": 0}

    # critic-disagreement threshold: median split over the corpus (frozen here
    # for this run; a production gate would freeze it from a dev set).
    cd_vals = []
    for r in records:
        cd = r.configs.get(reference_name, {}).get("critic_disagreement")
        if cd is not None:
            cd_vals.append(float(cd))
    cd_threshold = float(np.median(cd_vals)) if cd_vals else 0.0

    est_names = ["root_critic_only", "d1"]
    # collectors
    disagree = {n: [] for n in est_names}  # 1.0 if argmax != ref
    regret = {n: [] for n in est_names}
    actor_disagree = []
    actor_regret = []
    pairwise = {"actor_vs_critic": [], "critic_vs_d0": [], "d0_vs_d1": []}
    prune_drop = []
    # stratification tags (collected per-root, same for all estimators)
    tags = {"entropy": [], "phase": [], "critic_disagree": [], "ref_gap": []}

    for r in records:
        legal = np.asarray(r.legal_actions)
        ref_cfg = r.configs.get(reference_name)
        if ref_cfg is None or ref_cfg.get("q_argmax") is None:
            continue
        ref_q = np.asarray(ref_cfg["q_mean"], dtype=np.float64)
        ref_argmax_idx = int(np.argmax(ref_q))  # index into legal_actions
        ref_argmax = int(legal[ref_argmax_idx])
        actor_argmax = int(r.base_argmax)
        actor_disagree.append(float(actor_argmax != ref_argmax))
        actor_regret.append(
            float(
                ref_q[ref_argmax_idx]
                - ref_q[int(np.argmin(np.abs(legal - actor_argmax)))]
            )
        )

        # critic_disagreement band for this root
        cd = ref_cfg.get("critic_disagreement", 0.0)
        cd_band = _critic_disagreement_band(float(cd), cd_threshold)
        # reference-gap band (recompute from ref_q)
        order = np.argsort(-ref_q)
        top1 = float(ref_q[order[0]])
        top2 = float(ref_q[order[1]]) if ref_q.size > 1 else 0.0
        rg_band = ref_gap_band(top1 - top2)

        tags["entropy"].append(r.entropy_band)
        tags["phase"].append(r.phase_band)
        tags["critic_disagree"].append(cd_band)
        tags["ref_gap"].append(rg_band)

        # pairwise agreements
        critic_cfg = r.configs.get("root_critic_only", {})
        critic_argmax = critic_cfg.get("q_argmax")
        d1_cfg = r.configs.get("d1", {})
        d1_argmax = d1_cfg.get("q_argmax")
        if critic_argmax is not None:
            pairwise["actor_vs_critic"].append(float(actor_argmax == critic_argmax))
            pairwise["critic_vs_d0"].append(float(critic_argmax == ref_argmax))
            # regret for critic_only
            ci_idx = int(np.argmin(np.abs(legal - int(critic_argmax))))
            regret["root_critic_only"].append(
                float(ref_q[ref_argmax_idx] - ref_q[ci_idx])
            )
            disagree["root_critic_only"].append(float(critic_argmax != ref_argmax))
        if d1_argmax is not None:
            pairwise["d0_vs_d1"].append(float(ref_argmax == d1_argmax))
            d1_idx = int(np.argmin(np.abs(legal - int(d1_argmax))))
            regret["d1"].append(float(ref_q[ref_argmax_idx] - ref_q[d1_idx]))
            disagree["d1"].append(float(d1_argmax != ref_argmax))

        # §39 Q6: would legacy 5%-of-max pruning drop the ref-best action?
        bp = np.asarray(r.base_probs, dtype=np.float64)
        if bp.size:
            max_p = float(bp.max())
            ref_best_prob = float(bp[ref_argmax_idx])
            prune_drop.append(float(ref_best_prob < legacy_prune_threshold * max_p))

    def nm(x):
        return float(np.nanmean(x)) if len(x) else float("nan")

    def stratified(values, tag_list):
        bands: Dict[str, List[float]] = {}
        for v, t in zip(values, tag_list):
            bands.setdefault(t, []).append(float(v))
        return {b: float(np.nanmean(vv)) for b, vv in bands.items() if vv}

    out: Dict[str, Any] = {
        "n_roots": len(records),
        "reference": reference_name,
        "critic_disagreement_threshold": cd_threshold,
        "actor_disagrees_with_ref": nm(actor_disagree),
        "actor_mean_regret_vs_ref": nm(actor_regret),
        "root_critic_only_disagrees_with_ref": nm(disagree["root_critic_only"]),
        "root_critic_only_mean_regret_vs_ref": nm(regret["root_critic_only"]),
        "d1_disagrees_with_ref": nm(disagree["d1"]),
        "d1_mean_regret_vs_ref": nm(regret["d1"]),
        "actor_vs_critic_agree": nm(pairwise["actor_vs_critic"]),
        "critic_vs_d0_agree": nm(pairwise["critic_vs_d0"]),
        "d0_vs_d1_agree": nm(pairwise["d0_vs_d1"]),
        "legacy_prune_would_drop_ref_best": nm(prune_drop),
    }
    # stratified actor disagreement (the addressable-opportunity signal)
    out["actor_disagrees_by_entropy"] = stratified(actor_disagree, tags["entropy"])
    out["actor_disagrees_by_phase"] = stratified(actor_disagree, tags["phase"])
    out["actor_disagrees_by_critic_disagree"] = stratified(
        actor_disagree, tags["critic_disagree"]
    )
    out["actor_disagrees_by_ref_gap"] = stratified(actor_disagree, tags["ref_gap"])
    # stratified critic_vs_d0 disagreement (does the transition add anything?)
    cvd = [1.0 - x for x in pairwise["critic_vs_d0"]]
    out["critic_disagrees_with_d0_by_phase"] = stratified(cvd, tags["phase"])
    out["critic_disagrees_with_d0_by_entropy"] = stratified(cvd, tags["entropy"])
    # phase distribution of the corpus (for honest reporting)
    out["phase_distribution"] = {
        b: int(tags["phase"].count(b)) for b in ("early", "mid", "late")
    }
    return out


def comparison_table(comp: Dict[str, Any]) -> str:
    """Render the head-to-head comparison table (Markdown) for the report."""
    if comp.get("n_roots", 0) == 0:
        return "## Estimator head-to-head (no roots)\n"
    lines = [
        "## Estimator head-to-head (actor vs critic-only vs D=0-ref vs D=1)",
        "",
        f"Reference = high-K D=0. n = {comp['n_roots']} roots.",
        f"Critic-disagreement threshold (median split) = {comp.get('critic_disagreement_threshold', 0):.3f}.",
        "",
        "| estimator | disagrees w/ ref | mean regret vs ref |",
        "|---|---|---|",
        f"| actor (frozen) | {comp['actor_disagrees_with_ref']:.3f} | {comp['actor_mean_regret_vs_ref']:.1f} |",
        f"| root_critic_only | {comp['root_critic_only_disagrees_with_ref']:.3f} | {comp['root_critic_only_mean_regret_vs_ref']:.1f} |",
        f"| D=1 (vs D=0 ref) | {comp['d1_disagrees_with_ref']:.3f} | {comp['d1_mean_regret_vs_ref']:.1f} |",
        "",
        "| pair | agreement rate |",
        "|---|---|",
        f"| actor vs critic-only | {comp['actor_vs_critic_agree']:.3f} |",
        f"| critic-only vs D=0 | {comp['critic_vs_d0_agree']:.3f} |",
        f"| D=0 vs D=1 | {comp['d0_vs_d1_agree']:.3f} |",
        "",
        f"Legacy 5%-of-max pruning would drop the reference-best action on "
        f"{comp['legacy_prune_would_drop_ref_best']:.1%} of roots (skill §39 Q6).",
        "",
        "Actor disagreement with reference (addressable opportunity), stratified:",
        f"- by entropy band: {comp.get('actor_disagrees_by_entropy', {})}",
        f"- by phase band: {comp.get('actor_disagrees_by_phase', {})}",
        f"- by critic disagreement: {comp.get('actor_disagrees_by_critic_disagree', {})}",
        f"- by reference gap: {comp.get('actor_disagrees_by_ref_gap', {})}",
        "",
        f"Phase distribution: {comp.get('phase_distribution', {})}",
        "",
        "Critic-only vs D=0 disagreement (does the real transition add anything?), stratified:",
        f"- by phase: {comp.get('critic_disagrees_with_d0_by_phase', {})}",
        f"- by entropy: {comp.get('critic_disagrees_with_d0_by_entropy', {})}",
    ]
    return "\n".join(lines)


def search_justification_gate(comp: Dict[str, Any]) -> Dict[str, Any]:
    """The §23-precondition gate: is there enough signal to justify a paired
    win-rate eval? (skill §40 interpretation + §39 Q3/Q6/Q7).

    This is separate from the §22 convergence gate (which passed): convergence
    proves the *estimator is sound*; this gate asks whether *search has a
    target*. The criteria are intentionally conservative -- a non-PASS means
    "the estimator is valid but there is no evidence search will change
    outcomes; do not spend the §23 budget yet."

    Criteria:
    * **addressable_opportunity**: the actor disagrees with the best estimator
      (D=0 ref) on a non-trivial fraction of roots (> 10%). If the actor nearly
      always matches the reference, search cannot change actions by definition.
    * **search_adds_over_critic**: D=0 disagrees with root_critic_only on a
      non-trivial fraction (> 10%), OR (if not) root_critic_only itself
      disagrees with the actor -- i.e. either the simulator transition adds
      information, or a cheap critic rerank already captures the gain (which
      still merits a rerank experiment, just not full search). FAIL only if both
      the critic AND D=0 match the actor (nothing to gain).
    * **concentrated_not_random**: actor disagreement is higher at high-entropy
      or high-critic-disagreement roots than at low (the changes are where the
      actor is uncertain, not random noise).
    * **pruning_safe**: the legacy 5%-of-max rule does NOT drop the reference-
      best action on most roots (search can actually find these under
      exhaustive ``all_legal``).
    """
    n = comp.get("n_roots", 0)
    if n == 0:
        return {
            "verdict": "INCONCLUSIVE",
            "passed": 0,
            "total": 4,
            "criteria": {},
            "note": "no roots",
        }

    actor_dis = comp.get("actor_disagrees_with_ref", 0.0)
    critic_vs_d0_dis = 1.0 - comp.get("critic_vs_d0_agree", 1.0)
    critic_vs_actor_dis = 1.0 - comp.get("actor_vs_critic_agree", 1.0)

    crit_opportunity = {
        "actor_disagrees_with_ref": float(actor_dis),
        "pass": bool(actor_dis > 0.10),
    }

    # search adds over critic: D0 != critic_only, OR critic_only != actor
    search_adds = bool(critic_vs_d0_dis > 0.10 or critic_vs_actor_dis > 0.10)
    crit_search_adds = {
        "d0_disagrees_with_critic_only": float(critic_vs_d0_dis),
        "critic_only_disagrees_with_actor": float(critic_vs_actor_dis),
        "pass": search_adds,
    }

    # concentrated at high-entropy / high-critic-disagreement
    by_ent = comp.get("actor_disagrees_by_entropy", {})
    by_cd = comp.get("actor_disagrees_by_critic_disagree", {})
    high_ent = by_ent.get("high", by_ent.get("medium", 0.0))
    low_ent = by_ent.get("low", 0.0)
    high_cd = by_cd.get("high", 0.0)
    low_cd = by_cd.get("low", 0.0)
    concentrated = bool((high_ent > low_ent) or (high_cd > low_cd))
    crit_concentrated = {
        "high_entropy_disagree": float(high_ent),
        "low_entropy_disagree": float(low_ent),
        "high_critic_disagree": float(high_cd),
        "low_critic_disagree": float(low_cd),
        "pass": concentrated,
    }

    prune = comp.get("legacy_prune_would_drop_ref_best", 1.0)
    crit_prune = {"value": float(prune), "pass": bool(prune < 0.50)}

    criteria = {
        "addressable_opportunity": crit_opportunity,
        "search_adds_over_critic": crit_search_adds,
        "concentrated_not_random": crit_concentrated,
        "pruning_safe": crit_prune,
    }
    passed = sum(1 for c in criteria.values() if c.get("pass"))
    total = len(criteria)
    verdict = (
        "PASS" if passed == total else ("INCONCLUSIVE" if passed == 0 else "PARTIAL")
    )
    return {
        "verdict": verdict,
        "passed": passed,
        "total": total,
        "criteria": criteria,
        "note": (
            "PASS = there is a non-trivial, located signal that search could "
            "change actions where the actor is uncertain, justifying the §23 "
            "paired win-rate investment. A non-PASS means the estimator is valid "
            "(§22 gate passed) but there is no evidence search will change "
            "outcomes -- do not spend the §23 budget yet. Per skill §40: if "
            "root_critic_only ~= D=0 >> actor, the win is a cheap critic rerank, "
            "not simulator search."
        ),
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_results(
    records: List[RootResultRecord],
    manifest: List[RootManifestEntry],
    summary: Dict[str, Any],
    assessment: Dict[str, Any],
    comparison: Dict[str, Any],
    justification: Dict[str, Any],
    run_manifest_dict: Dict[str, Any],
    output_dir: str,
    derived_ks: List[int],
) -> Dict[str, str]:
    """Write per-root JSONL + manifest + summary + run manifest + MD report."""
    os.makedirs(output_dir, exist_ok=True)
    roots_path = os.path.join(output_dir, "root_results.jsonl")
    manifest_path = os.path.join(output_dir, "root_manifest.jsonl")
    summary_path = os.path.join(output_dir, "summary.json")
    comparison_path = os.path.join(output_dir, "comparison.json")
    run_path = os.path.join(output_dir, "run_manifest.json")
    report_path = os.path.join(output_dir, "REPORT.md")

    with open(roots_path, "w") as f:
        for r in records:
            f.write(r.to_json() + "\n")
    write_manifest(manifest, manifest_path)
    summary_with = dict(summary)
    summary_with["_comparison"] = comparison
    summary_with["_justification_gate"] = justification
    with open(summary_path, "w") as f:
        json.dump(summary_with, f, indent=2, default=_json_default)
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2, default=_json_default)
    with open(run_path, "w") as f:
        json.dump(run_manifest_dict, f, indent=2, default=_json_default)

    md = [
        f"# Test-Time Search — Phase 1 Fixed-Root Estimator Benchmark",
        "",
        f"- roots: {summary.get('_n_roots', 0)}",
        f"- K_ref: {run_manifest_dict.get('k_ref')}",
        f"- derived K: {run_manifest_dict.get('derived_ks')}",
        f"- depths: {run_manifest_dict.get('depths')}",
        f"- chance mode: {run_manifest_dict.get('chance_mode')}",
        f"- §22 convergence verdict: **{assessment['verdict']}** ({assessment['passed']}/{assessment['total']} criteria)",
        f"- §23-precondition (search-justification) verdict: **{justification['verdict']}** ({justification['passed']}/{justification['total']} criteria)",
        "",
        convergence_table(summary, derived_ks),
        "",
        comparison_table(comparison),
        "",
        "## §22 convergence gate (estimator validity)",
        "",
        "```json",
        json.dumps(assessment, indent=2, default=_json_default),
        "```",
        "",
        "## §23-precondition gate (is there a search signal?)",
        "",
        "```json",
        json.dumps(justification, indent=2, default=_json_default),
        "```",
        "",
        "## Aggregate per-cell convergence metrics",
        "",
        "```json",
        json.dumps(summary, indent=2, default=_json_default),
        "```",
    ]
    with open(report_path, "w") as f:
        f.write("\n".join(md))

    return {
        "roots": roots_path,
        "manifest": manifest_path,
        "summary": summary_path,
        "comparison": comparison_path,
        "run_manifest": run_path,
        "report": report_path,
    }


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class _CliBundle:
    """Bundle of frozen-policy + env handles for the CLI (mirrors the test
    ``FrozenBundle``; kept here so the CLI does not import from ``tests/``)."""

    env: Any
    eval_driver: Any
    opponent: Any
    eval_policy: Any
    opponent_policy: Any
    model: Any
    opp_model: Any
    agent: Any
    action_dim: int
    device: Any
    reward_multiplier: float
    eval_action_space: Any
    opponent_action_space: Any
    battle_format: str


def _build_bundle(args):
    """Build the bundle (mirrors conftest.frozen_env_bundle / eval_search)."""
    import torch
    import metamon.env
    from metamon.env.vectorized.amago_policy import AmagoLadderPolicyDriver
    from metamon.env.vectorized.vector_env import BattleAgainstMetamon
    from metamon.rl.pretrained import get_pretrained_model

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed or 0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed or 0)
    model = get_pretrained_model(args.agent)
    agent = model.initialize_agent(
        checkpoint=args.checkpoint, log=False, action_temperature=1.0
    )
    policy = agent.policy.to(dev)
    policy.eval()
    action_dim = model.action_space.gym_space.n
    opp_model = model
    opp_policy = agent.policy.to(dev)
    opp_policy.eval()
    team_set = metamon.env.get_metamon_teams(args.format, args.team_set)
    env = BattleAgainstMetamon(
        battle_format=args.format,
        observation_space=model.observation_space,
        action_space=model.action_space,
        reward_function=model.reward_function,
        team_set=team_set,
        opponent_model=opp_model,
        opponent_checkpoint=args.checkpoint,
        opponent_sample=True,
        batched_envs=args.num_parallel,
        n_workers=1,
        eval_player_side=0,
        seed=args.seed,
        device=str(dev),
    )
    eval_driver = AmagoLadderPolicyDriver(
        policy=policy,
        device=dev,
        num_lanes=env.batched_envs,
        action_dim=action_dim,
        sample=True,
    )
    return (
        _CliBundle(
            env=env,
            eval_driver=eval_driver,
            opponent=env.opponent,
            eval_policy=policy,
            opponent_policy=opp_policy,
            model=model,
            opp_model=opp_model,
            agent=agent,
            action_dim=action_dim,
            device=dev,
            reward_multiplier=float(getattr(agent.policy, "reward_multiplier", 10.0)),
            eval_action_space=model.action_space,
            opponent_action_space=opp_model.action_space,
            battle_format=args.format,
        ),
        env,
        agent,
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Phase 1 fixed-root estimator convergence benchmark (skill §22)"
    )
    p.add_argument("--agent", default="MiniOnlinePsroV1_4")
    p.add_argument("--checkpoint", type=int, default=740)
    p.add_argument("--format", default="gen1ou")
    p.add_argument("--team_set", default="competitive")
    p.add_argument("--num_parallel", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--search_seed", type=int, default=0)
    p.add_argument("--k_ref", type=int, default=128)
    p.add_argument(
        "--derived_ks",
        type=int,
        nargs="+",
        default=[4, 16, 64],
        help="low-K estimates derived from the K_ref run via prefix/block averaging",
    )
    p.add_argument("--depths", type=int, nargs="+", default=[0, 1])
    p.add_argument("--max_roots", type=int, default=64)
    p.add_argument("--max_battles", type=int, default=20)
    p.add_argument("--root_stride", type=int, default=1)
    p.add_argument(
        "--decision_stride",
        type=int,
        default=1,
        help="per-battle capture cadence: capture when (decision_idx %% stride)==0; "
        "spreads the corpus across early/mid/late (skill §22 phase stratification)",
    )
    p.add_argument(
        "--min_decision",
        type=int,
        default=0,
        help="skip per-battle decisions below this index (target mid/late)",
    )
    p.add_argument(
        "--max_decision",
        type=int,
        default=None,
        help="skip per-battle decisions above this index",
    )
    p.add_argument(
        "--include_inherited_rng",
        action="store_true",
        help="also run the inherited-trunk-RNG future-chance oracle diagnostic (D=0)",
    )
    p.add_argument("--store_branch_matrices", action="store_true")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--progress_every", type=int, default=5)
    args = p.parse_args()

    derived_ks = sorted(k for k in args.derived_ks if k <= args.k_ref)
    if not derived_ks:
        derived_ks = [4, 16, 64]
    config_grid = build_grid_configs(
        k_ref=args.k_ref,
        depths=args.depths,
        include_inherited_rng=args.include_inherited_rng,
        search_seed=args.search_seed,
    )

    bundle, env, agent = _build_bundle(args)
    try:
        t0 = time.perf_counter()
        records, manifest = benchmark_roots(
            bundle=bundle,
            config_grid=config_grid,
            k_ref=args.k_ref,
            derived_ks=derived_ks,
            depths=args.depths,
            max_roots=args.max_roots,
            max_battles=args.max_battles,
            root_stride=args.root_stride,
            decision_stride=args.decision_stride,
            min_decision=args.min_decision,
            max_decision=args.max_decision,
            store_branch_matrices=args.store_branch_matrices,
            progress_every=args.progress_every,
            env_seed=args.seed,
            output_dir=args.output_dir,
        )
        elapsed = time.perf_counter() - t0
        summary = aggregate_convergence(records)
        assessment = go_no_go_assessment(summary, derived_ks)
        comparison = estimator_comparison(records)
        justification = search_justification_gate(comparison)

        import subprocess

        git_sha = None
        try:
            git_sha = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)
                )
                .decode()
                .strip()
            )
        except Exception:
            pass
        ckpt = os.path.expanduser(
            f"~/metamon_runs/mini_online_psro_v1.4/mini_online_psro_v1.4/"
            f"ckpts/policy_weights/policy_epoch_{args.checkpoint}.pt"
        )
        rm = run_manifest(
            agent=args.agent,
            checkpoint=args.checkpoint,
            ckpt_path=ckpt,
            battle_format=args.format,
            team_set=args.team_set,
            env_seed=args.seed,
            search_seed=args.search_seed,
            k_ref=args.k_ref,
            derived_ks=derived_ks,
            depths=args.depths,
            leaf_modes=["root_critic_only", "policy_expectation"]
            + (["inherited_trunk_rng"] if args.include_inherited_rng else []),
            chance_mode="resample_crn",
            n_roots=len(records),
            n_battles=args.max_battles,
            git_sha=git_sha,
            extra={
                "elapsed_sec": elapsed,
                "include_inherited_rng": args.include_inherited_rng,
            },
        )
        paths = write_results(
            records,
            manifest,
            summary,
            assessment,
            comparison,
            justification,
            rm,
            args.output_dir,
            derived_ks,
        )
        print(
            json.dumps(
                {
                    "convergence_verdict": assessment["verdict"],
                    "search_justification_verdict": justification["verdict"],
                    "n_roots": len(records),
                    "elapsed_sec": elapsed,
                    "outputs": paths,
                },
                indent=2,
            )
        )
    finally:
        try:
            env.close()
        except Exception:
            pass
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
