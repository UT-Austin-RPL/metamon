"""Phase A: terminal-win fixed-root benchmark (the central go/no-go gate).

The §22 Phase 1 benchmark proved the *estimator converges* (K stabilizes the
shaped-Q ranking) and the §23-precondition gate proved there *is* a search
signal (the actor is wrong 63% of the time vs the D=0 reference). But the §23
paired win-rate eval was **estimator-positive, game-negative**: the shaped-Q
improvement did not translate to more wins.

The expert diagnosis is that the central unmeasured question is objective
alignment:

> Does the frozen critic's preference after an exact oracle transition predict
> which action actually increases **terminal win probability**?

The shaped critic is trained on ``AggressiveShapedReward`` (damage + HP +
200*victory), not win probability. A locally higher shaped Q may not correspond
to a higher win probability. Until that is measured, larger K, deeper search,
more opponents, and thousands of additional battles are premature.

This module answers it. For each fixed root it:

1. records the **shaped-Q predictors** -- ``root_critic_only``, ``D=0`` at
   ``K_ref`` (with the per-branch return matrix ``R (A, K_ref)`` so every lower-K
   is derived by prefix averaging), and optionally ``D=1`` -- via the existing
   :meth:`SearchEvalRunner.estimate_root` (skill §22 infrastructure);
2. for **each legal action**, plays ``G = K_ref`` coupled continuations to a
   **terminal state** with the frozen policy on both sides
   (:meth:`SearchEvalRunner.terminal_continuations`), recording the actual
   win/loss outcome per branch;
3. derives every lower-``G'`` terminal-win estimate by prefix averaging (the
   per-``k`` chance stream is K-independent -- skill §7 ``rng.py``), so one
   ``G=K_ref`` run per action reconstructs every ``G'``;
4. reports the central outputs:

* **Spearman correlation** between each shaped-Q predictor and terminal win
  probability (per-root + aggregated);
* **pairwise action-ordering accuracy** (does the predictor's top-1 / pairwise
  order match the terminal-win order?);
* **terminal-win regret** of each selector (actor argmax, root-critic argmax,
  D=0 K=16 argmax, D=0 K=128 argmax, D=1 argmax) vs the terminal-win-best action;
* **frequency with which a shaped-search argmax decreases terminal win
  probability** relative to the actor;
* results **stratified by game phase and tactical category** (imminent KO,
  status, forced switch, endgame).

CRN pairing (skill §7): the terminal continuation for action ``a`` at rollout
index ``k`` uses the *same* branch seed and the *same* coupled opponent root
action as the shaped-Q estimate's branch ``k`` for action ``a`` (the seed bank
keys on ``(root, k)``, not action identity). So ``wins[a, k]`` pairs with
``R[a, k]`` on the same chance stream -- the exact counterfactual pairing.

Gate A (skill §37 "Gate A: Is the critic suitable for search?"): proceed with
the existing shaped-critic evaluator only when the K=128 shaped-Q action ranking
is positively correlated with terminal win probability AND K=128 search-selected
actions improve terminal win probability over the actor on held-out roots AND
catastrophic search errors are acceptably rare. Otherwise the recommendation is
to train a terminal-outcome value head.
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
    phase_band,
    compute_root_features,
)
from .benchmark_roots import (
    branch_return_matrix,
    reference_q,
    prefix_q,
    spearman_corr,
    kendall_corr,
    _base_search_config,
    _build_bundle,
    _CliBundle,
)

# ---------------------------------------------------------------------------
# Tactical-feature extraction (skill §22 stratification + expert "tactical
# situations such as imminent KOs, sleep, paralysis, ... endgames")
# ---------------------------------------------------------------------------


def _extract_tactical_features(lane, eval_side: str) -> Dict[str, Any]:
    """Read battle-state tactical tags from the root's universal state.

    Stratifies the corpus by the tactical categories the expert asked for
    (imminent KOs, status, forced switch, endgames) in addition to the
    phase / entropy / top-2-gap bands from :mod:`root_dataset`.
    """
    state = lane.universal_state(eval_side)
    pa = state.player_active_pokemon  # eval-side active
    oa = state.opponent_active_pokemon  # opponent active
    forced = bool(getattr(state, "forced_switch", False))
    eval_hp = float(pa.hp_pct) if pa is not None else 1.0
    opp_hp = float(oa.hp_pct) if oa is not None else 1.0
    eval_status = str(pa.status) if pa is not None else "nostatus"
    opp_status = str(oa.status) if oa is not None else "nostatus"
    status_kinds = {"sleep", "paralysis", "poison", "burn", "freeze", "toxic"}
    eval_status_present = eval_status in status_kinds
    opp_status_present = opp_status in status_kinds
    opponents_remaining = int(getattr(state, "opponents_remaining", 6))
    eval_low_hp = eval_hp < 0.25
    opp_low_hp = opp_hp < 0.25
    cats: List[str] = ["forceswitch" if forced else "move"]
    if opp_low_hp:
        cats.append("imminent_ko")
    if eval_low_hp:
        cats.append("at_risk")
    if eval_status_present or opp_status_present:
        cats.append("status")
    if opponents_remaining <= 2:
        cats.append("endgame")
    return {
        "forced_switch": forced,
        "request_kind": "forceswitch" if forced else "move",
        "eval_hp_pct": eval_hp,
        "opp_hp_pct": opp_hp,
        "eval_low_hp": eval_low_hp,
        "opp_low_hp": opp_low_hp,
        "eval_status": eval_status,
        "opp_status": opp_status,
        "status_present": eval_status_present or opp_status_present,
        "opponents_remaining": opponents_remaining,
        "tactical_category": "+".join(cats),
    }


# ---------------------------------------------------------------------------
# Per-root record
# ---------------------------------------------------------------------------


@dataclass
class TerminalWinRootRecord:
    """One fixed root with shaped-Q predictors + terminal-win ground truth."""

    root_id: str
    battle_id: str
    lane: int
    decision: int
    legal_actions: List[int]
    base_probs: List[float]  # over legal_arr
    base_argmax: int
    # stratification (pre-estimate)
    n_legal: int
    base_entropy: float
    base_top2_gap: float
    entropy_band: str
    top2_gap_band: str
    phase_band: str
    # tactical features
    forced_switch: bool
    request_kind: str
    eval_low_hp: bool
    opp_low_hp: bool
    status_present: bool
    opponents_remaining: int
    tactical_category: str
    # shaped-Q predictors (per legal action):
    root_critic_q: List[float]
    d0_q: List[float]  # D=0 K_ref mean per action
    d0_q_sem: List[float]  # D=0 K_ref SEM per action
    d1_q: Optional[List[float]]  # D=1 K_ref mean per action (None if skipped)
    # derived-K shaped Q (prefix averages of the K_ref per-branch R):
    derived_shaped_q: Dict[str, List[float]]  # "D0:K16" -> [q per action]
    # terminal-win ground truth (per legal action):
    terminal_win: List[float]  # win rate at G=K_ref (draws=0.5, excl. truncated)
    terminal_win_sem: List[float]  # binomial SEM at G
    n_truncated: List[int]  # per action
    n_draws: List[int]  # per action
    # derived-G' terminal win (prefix averages of the G per-branch wins):
    derived_terminal_win: Dict[str, List[float]]  # "G16" -> [win per action]
    # per-branch matrices (verbose; for paired diagnostics + re-derivation):
    per_action_wins: Optional[List[List[float]]]  # [a][k] win outcome
    per_action_shaped_r: Optional[List[List[float]]]  # [a][k] D=0 shaped return
    # diagnostics
    G: int
    mean_steps_to_terminal: float
    latency_ms_shaped: float
    latency_ms_terminal: float
    # kimi-search M1: extra gamma-head predictors (e.g. "root_critic_g4" /
    # "d0_g5" -> per-action Q at that critic horizon):
    gamma_predictor_q: Optional[Dict[str, List[float]]] = None
    # kimi-search M3: trained win-head per-action Q(s,a) at the root:
    win_head_q: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


def load_records(path: str) -> List["TerminalWinRootRecord"]:
    """Load streamed ``terminal_win_roots.jsonl`` records (crash-recovery).

    Each line is a ``TerminalWinRootRecord.to_json()``; all fields are JSON
    types (lists / floats / ints / bools / str / dict), so reconstruction is a
    direct dataclass unfurl. Used by ``recover_terminal_win`` and
    ``audit_terminal_win`` so a long G=128 run is crash-proof: the streamed
    JSONL is the source of truth and the summary / gate / report / audit can be
    regenerated from it at any point.
    """
    records: List[TerminalWinRootRecord] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(TerminalWinRootRecord(**json.loads(line)))
    return records


# ---------------------------------------------------------------------------
# Derived-G' terminal win (prefix averaging -- the per-k chance stream is
# G-independent, so the first G' continuations of a G=K_ref run ARE a G'
# estimate, exactly mirroring benchmark_roots.prefix_q for shaped Q).
# ---------------------------------------------------------------------------


def prefix_win_rate(wins: np.ndarray, g_prime: int) -> np.ndarray:
    """Terminal-win rate using the first ``g_prime`` continuations per action.

    ``wins`` is ``(A, G)``; returns ``(A,)`` mean over the first ``g_prime``
    columns (draws count as 0.5; truncated branches keep their 0.5 placeholder
    but are flagged separately in ``n_truncated``).
    """
    g = min(int(g_prime), wins.shape[1])
    return np.nanmean(wins[:, :g], axis=1)


def win_rate_sem(p: np.ndarray, g: int) -> np.ndarray:
    """Binomial standard error of a win-rate estimate over ``g`` continuations."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.sqrt(p * (1.0 - p) / max(int(g), 1))


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def _terminal_config(
    g: int, search_seed: int, chance_mode: str = "resample_crn"
) -> SearchConfig:
    """Config for a to-terminal continuation run (G branches, one action)."""
    return _base_search_config(
        search_rollouts_per_action=g,
        search_depth=0,  # unused (terminal_continuations loops to terminal)
        search_chance_mode=chance_mode,
        search_seed=search_seed,
    )


def _shaped_q_configs(
    k_ref: int,
    depths: List[int],
    search_seed: int,
    gamma_indices: Optional[List[int]] = None,
) -> Dict[str, SearchConfig]:
    """Shaped-Q estimator grid (the predictors). Reuses build_grid_configs shape.

    ``gamma_indices`` (kimi-search M1): additional critic gamma-head indices to
    evaluate as predictors, e.g. ``[4, 5]`` adds ``root_critic_g4`` /
    ``root_critic_g5`` (root-critic at those horizons) and ``d0_g4`` / ``d0_g5``
    (D=0 K_ref leaf value at those horizons). The primary head (None ->
    ``critic_horizon_index = -1``) is always included as ``root_critic_only`` /
    ``d{d}``. This measures whether the higher-gamma heads are better aligned
    with terminal win than the primary shaped head (H2 of RESEARCH_PLAN_KIMI).
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
    for g in gamma_indices or []:
        grid[f"root_critic_g{g}"] = _base_search_config(
            search_leaf_value_mode="root_critic_only",
            search_seed=search_seed,
            search_critic_horizon=int(g),
        )
        for d in depths:
            grid[f"d{d}_g{g}"] = _base_search_config(
                search_rollouts_per_action=k_ref,
                search_depth=d,
                search_seed=search_seed,
                search_critic_horizon=int(g),
            )
    return grid


def benchmark_terminal_win(
    *,
    bundle,
    k_ref: int,
    derived_ks: List[int],
    depths: List[int],
    gamma_indices: Optional[List[int]] = None,
    win_head_path: Optional[str] = None,
    save_distill: bool = False,
    max_roots: int = 64,
    max_battles: int = 40,
    root_stride: int = 1,
    decision_stride: int = 3,
    min_decision: int = 0,
    max_decision: Optional[int] = None,
    store_per_branch: bool = False,
    progress_every: int = 2,
    env_seed: Optional[int] = None,
    search_seed: int = 0,
    max_steps_to_terminal: int = 250,
    output_dir: Optional[str] = None,
) -> Tuple[List[TerminalWinRootRecord], List[RootManifestEntry]]:
    """Run the Phase A terminal-win fixed-root benchmark.

    Drives the vectorized env with the **baseline** frozen policy (natural
    self-play root corpus), and at each captured (phase-stratified) root runs
    the shaped-Q estimator grid + a to-terminal continuation for every legal
    action, then takes the baseline action to continue. Stops at ``max_roots``
    or ``max_battles``.

    Args:
        bundle: ``_CliBundle`` (or test ``FrozenBundle``).
        k_ref: the reference rollout/continuation count (shaped-Q K_ref AND
            terminal-win G -- they share the CRN seed bank so per-branch shaped
            Q and terminal win pair on the same chance stream k).
        derived_ks: low-K shaped-Q + low-G terminal-win values derived by prefix
            averaging (e.g. ``[4, 16, 64]``).
        depths: shaped-Q rollout depths with a K_ref run (e.g. ``[0]`` or
            ``[0, 1]``). D=0 is the primary predictor; D=1 is optional.
        max_roots / max_battles: corpus caps.
        decision_stride: per-battle capture cadence (spreads across phases).
        store_per_branch: keep the per-branch ``wins (A,G)`` and ``R (A,K)``
            matrices (verbose; needed for paired diagnostics / re-derivation).
        max_steps_to_terminal: safety cap on continuation length (a non-ended
            branch at this cap is flagged ``truncated`` and excluded from the
            win rate).
        output_dir: stream each root record + manifest entry to JSONL as it is
            produced (crash safety; the summary/report are written at the end).
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
        config=_terminal_config(k_ref, search_seed),  # placeholder; swapped per call
        device=bundle.device,
        action_dim=bundle.action_dim,
        battle_format=bundle.battle_format,
        reward_multiplier=bundle.reward_multiplier,
    )

    shaped_cfgs = _shaped_q_configs(k_ref, depths, search_seed, gamma_indices)
    win_head = None
    if win_head_path:
        from .win_head import load_win_head

        win_head = load_win_head(win_head_path, device=str(bundle.device))
    term_cfg = _terminal_config(k_ref, search_seed)
    derived_ks_sorted = sorted(k for k in derived_ks if k <= k_ref)
    if not derived_ks_sorted:
        derived_ks_sorted = [4, 16, 64]

    n = env.batched_envs
    obs, info = env.reset()
    lane_battle = [0] * n
    lane_total_dec = [0] * n
    lane_history: List[List[List[int]]] = [[] for _ in range(n)]

    records: List[TerminalWinRootRecord] = []
    manifest: List[RootManifestEntry] = []
    distill_rows: List[Dict[str, Any]] = []
    _roots_fh = None
    _manifest_fh = None
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        _roots_fh = open(os.path.join(output_dir, "terminal_win_roots.jsonl"), "w")
        _manifest_fh = open(os.path.join(output_dir, "root_manifest.jsonl"), "w")
    battles_done = 0
    steps = 0
    max_steps = max(max_battles * 400 // n + 200, 400)
    root_capture_idx = 0

    try:
        while (
            len(records) < max_roots
            and battles_done < max_battles
            and steps < max_steps
        ):
            steps += 1
            obs_list = unstack_obs_dicts(obs)
            actions = np.zeros(n, dtype=np.int64)
            lane_root_this_step = [None] * n

            for i in range(n):
                lane = env.lanes[i]
                if lane.ended or not lane.needs_agent_decision(env.eval_side):
                    actions[i] = 0
                    continue
                legal = info["legal_actions"][i]
                root_capture_idx += 1
                lane_total_dec[i] += 1
                tdec = lane_total_dec[i]
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
                    decision_idx = tdec
                    runner._battle_id = battle_id
                    runner._decision_counter = decision_idx
                    try:
                        rec, mentry = _benchmark_one_root_terminal(
                            runner=runner,
                            bundle=bundle,
                            lane_idx=i,
                            obs=obs_list[i],
                            legal=legal,
                            shaped_cfgs=shaped_cfgs,
                            term_cfg=term_cfg,
                            k_ref=k_ref,
                            derived_ks=derived_ks_sorted,
                            depths=depths,
                            battle_id=battle_id,
                            decision_idx=decision_idx,
                            battle_seed=env_seed,
                            action_history=list(lane_history[i]),
                            store_per_branch=store_per_branch,
                            max_steps_to_terminal=max_steps_to_terminal,
                            gamma_indices=gamma_indices,
                            win_head=win_head,
                        )
                        lane_root_this_step[i] = (rec, mentry)
                    except Exception as exc:  # noqa: BLE001
                        raise RuntimeError(
                            f"terminal-win benchmark failed at root "
                            f"{battle_id}:d{decision_idx}: {exc!r}"
                        ) from exc
                active = np.zeros(n, dtype=bool)
                active[i] = True
                actions[i] = int(eval_driver.act(active, obs_list)[i])

            obs, rewards, terminated, truncated, info = env.step(actions)
            for i in range(n):
                eval_driver.observe(i, float(rewards[i]), int(actions[i]))

            for i in range(n):
                if lane_root_this_step[i] is not None:
                    rec, mentry = lane_root_this_step[i]
                    records.append(rec)
                    manifest.append(mentry)
                    if save_distill:
                        _obs_i = obs_list[i]
                        distill_rows.append(
                            {
                                "obs": {
                                    k: np.asarray(v).tolist() for k, v in _obs_i.items()
                                },
                                "legal_actions": [int(x) for x in rec.legal_actions],
                                "label": int(
                                    rec.legal_actions[
                                        int(
                                            np.nanargmax(
                                                np.asarray(rec.terminal_win, float)
                                            )
                                        )
                                    ]
                                ),
                                "terminal_win": [float(x) for x in rec.terminal_win],
                            }
                        )
                    if _roots_fh is not None:
                        try:
                            _roots_fh.write(rec.to_json() + "\n")
                            _roots_fh.flush()
                        except OSError:
                            _roots_fh = None
                    if _manifest_fh is not None:
                        try:
                            _manifest_fh.write(mentry.to_json() + "\n")
                            _manifest_fh.flush()
                        except OSError:
                            _manifest_fh = None
                    if (len(records) % progress_every) == 0 or len(records) <= 3:
                        print(
                            f"  [terminal-win] {len(records)}/{max_roots} roots, "
                            f"{battles_done} battles, step {steps}",
                            flush=True,
                        )
                    if len(records) >= max_roots:
                        break
                if (
                    env.lanes[i].needs_agent_decision(env.eval_side)
                    or env.lanes[i].ended
                ):
                    from .benchmark_roots import _last_committed

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
                    lane_total_dec[i] = 0
                    lane_history[i] = []
    finally:
        runner.close()
        if _roots_fh is not None:
            _roots_fh.close()
        if _manifest_fh is not None:
            _manifest_fh.close()
        if save_distill and output_dir is not None:
            with open(os.path.join(output_dir, "distill_labels.jsonl"), "w") as _fh:
                for row in distill_rows:
                    _fh.write(json.dumps(row) + "\n")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return records, manifest


def _benchmark_one_root_terminal(
    *,
    runner,
    bundle,
    lane_idx: int,
    obs: dict,
    legal: List[int],
    shaped_cfgs: Dict[str, SearchConfig],
    term_cfg: SearchConfig,
    k_ref: int,
    derived_ks: List[int],
    depths: List[int],
    battle_id: str,
    decision_idx: int,
    battle_seed: Optional[int],
    action_history: List[List[int]],
    store_per_branch: bool,
    max_steps_to_terminal: int,
    gamma_indices: Optional[List[int]] = None,
    win_head=None,
) -> Tuple[TerminalWinRootRecord, RootManifestEntry]:
    """Run shaped-Q predictors + terminal-win ground truth at one fixed root."""
    env = bundle.env
    eval_side = env.eval_side

    # --- tactical features (from the root's universal state) ---
    tac = _extract_tactical_features(env.lanes[lane_idx], eval_side)

    # --- shaped-Q predictors (root_critic_only first -> base distribution) ---
    estimates: Dict[str, Any] = {}
    legal_arr = None
    base_probs = None
    base_argmax = None
    t_shaped = time.perf_counter()
    gamma_indices = gamma_indices or []
    order = ["root_critic_only"] + [f"d{d}" for d in depths]
    order += [f"root_critic_g{g}" for g in gamma_indices]
    order += [f"d{d}_g{g}" for g in gamma_indices for d in depths]
    for name in order:
        if name not in shaped_cfgs:
            continue
        est = runner.estimate_root(lane_idx, obs, legal, shaped_cfgs[name])
        estimates[name] = est
        if legal_arr is None:
            legal_arr = np.asarray(est.legal_arr)
            base_probs = est.base_probs
            base_argmax = est.base_argmax
    latency_ms_shaped = (time.perf_counter() - t_shaped) * 1000.0
    assert legal_arr is not None
    A = int(legal_arr.size)

    # shaped-Q per-action arrays
    rc_est = estimates["root_critic_only"]
    root_critic_q = np.asarray(rc_est.q_mean, dtype=np.float64)  # (A,)
    d0_est = estimates.get("d0")
    if d0_est is not None:
        R0 = branch_return_matrix(d0_est)  # (A, K_ref)
        d0_q = reference_q(R0) if R0.size else np.asarray(d0_est.q_mean)
        d0_sem = (
            np.asarray(d0_est.q_std, dtype=np.float64)
            / np.sqrt(np.maximum(np.asarray(d0_est.counts, dtype=np.float64), 1))
            if d0_est.counts.size
            else np.zeros(A)
        )
    else:
        R0 = np.empty((A, 0))
        d0_q = np.zeros(A)
        d0_sem = np.zeros(A)
    d1_est = estimates.get("d1")
    d1_q = np.asarray(d1_est.q_mean, dtype=np.float64) if d1_est is not None else None

    # derived-K shaped Q (prefix averages of the K_ref per-branch R)
    derived_shaped_q: Dict[str, List[float]] = {}
    if R0.size:
        for kp in derived_ks:
            if kp <= k_ref:
                derived_shaped_q[f"D0:K{kp}"] = prefix_q(R0, kp).tolist()

    # kimi-search M3: trained win-head per-action Q(s, a) at the root. Uses the
    # runner's _root_distribution to get the root embedding + base policy, then
    # evaluates the frozen win head on the legal actions.
    win_head_q: Optional[List[float]] = None
    if win_head is not None:
        import torch as _torch

        _bp, _la, _emb, _il, _ba = runner._root_distribution(lane_idx, obs, legal)
        with _torch.no_grad():
            _e = _emb.squeeze(1) if _emb.ndim == 3 else _emb
            _q = win_head.q_win(_e)[0].cpu().numpy()  # (A_full,)
        win_head_q = [float(_q[int(a)]) for a in legal_arr]

    # kimi-search M1: per-action Q for the extra gamma-head predictors
    gamma_predictor_q: Dict[str, List[float]] = {}
    for name, est in estimates.items():
        if name in ("root_critic_only",) or name in {f"d{d}" for d in depths}:
            continue
        if est.q_mean.size:
            gamma_predictor_q[name] = np.asarray(est.q_mean, dtype=np.float64).tolist()

    # --- terminal-win ground truth: one to-terminal continuation per action ---
    t_term = time.perf_counter()
    wins_matrix = np.full((A, k_ref), 0.5, dtype=np.float64)  # (A, G)
    n_trunc = np.zeros(A, dtype=np.int64)
    n_draws = np.zeros(A, dtype=np.int64)
    steps_list: List[float] = []
    for ai, a in enumerate(legal_arr):
        res = runner.terminal_continuations(
            lane_idx,
            obs,
            legal,
            forced_action=int(a),
            config=term_cfg,
            max_steps_to_terminal=max_steps_to_terminal,
        )
        wins_matrix[ai, :] = res.wins  # (G,) ordered by rollout_index k
        n_trunc[ai] = int(res.n_truncated)
        n_draws[ai] = int(res.n_draws)
        non_trunc = ~res.truncated
        if non_trunc.any():
            steps_list.append(float(res.steps_to_terminal[non_trunc].mean()))
    latency_ms_terminal = (time.perf_counter() - t_term) * 1000.0
    mean_steps = float(np.mean(steps_list)) if steps_list else 0.0

    # terminal-win per action (draws = 0.5; truncated branches excluded)
    terminal_win = np.zeros(A, dtype=np.float64)
    for ai in range(A):
        n_valid = k_ref - int(n_trunc[ai])
        if n_valid > 0:
            # wins_matrix holds 0.5 placeholders for truncated branches; subtract
            # their contribution so the mean is over non-truncated branches only.
            total = float(wins_matrix[ai].sum()) - 0.5 * int(n_trunc[ai])
            terminal_win[ai] = total / n_valid
        else:
            terminal_win[ai] = float("nan")
    terminal_win_sem = win_rate_sem(terminal_win, k_ref)

    # derived-G' terminal win (prefix averages)
    derived_terminal_win: Dict[str, List[float]] = {}
    for gp in derived_ks:
        if gp <= k_ref:
            derived_terminal_win[f"G{gp}"] = prefix_win_rate(wins_matrix, gp).tolist()

    # manifest entry (pre-estimate features from the base distribution)
    feats = compute_root_features(base_probs, legal_arr, int(base_argmax))
    mentry = make_manifest_entry(
        battle_id=battle_id,
        lane=lane_idx,
        decision=decision_idx,
        battle_seed=battle_seed,
        legal=legal,
        base_probs=base_probs,
        legal_arr=legal_arr,
        base_argmax=int(base_argmax),
        action_history=action_history,
    )
    # fill reference fields from D=0 high-K (the existing shaped reference)
    if d0_est is not None and d0_est.q_mean.size:
        R0r = branch_return_matrix(d0_est)
        ref_q0 = reference_q(R0r) if R0r.size else d0_est.q_mean
        fill_reference_fields(
            mentry,
            ref_q=ref_q0,
            legal_arr=legal_arr,
            critic_disagreement=float(d0_est.diag.get("critic_disagreement", 0.0)),
            terminal_frac_d0=float(
                np.mean(d0_est.term_frac) if d0_est.term_frac.size else 0.0
            ),
        )

    record = TerminalWinRootRecord(
        root_id=mentry.root_id,
        battle_id=battle_id,
        lane=lane_idx,
        decision=decision_idx,
        legal_actions=[int(x) for x in legal_arr],
        base_probs=[float(x) for x in base_probs[legal_arr]],
        base_argmax=int(base_argmax),
        n_legal=int(A),
        base_entropy=feats["base_entropy"],
        base_top2_gap=feats["base_top2_gap"],
        entropy_band=entropy_band(feats["base_entropy"]),
        top2_gap_band=top2_gap_band(feats["base_top2_gap"]),
        phase_band=phase_band(decision_idx),
        forced_switch=tac["forced_switch"],
        request_kind=tac["request_kind"],
        eval_low_hp=tac["eval_low_hp"],
        opp_low_hp=tac["opp_low_hp"],
        status_present=tac["status_present"],
        opponents_remaining=tac["opponents_remaining"],
        tactical_category=tac["tactical_category"],
        root_critic_q=root_critic_q.tolist(),
        d0_q=d0_q.tolist(),
        d0_q_sem=d0_sem.tolist(),
        d1_q=(d1_q.tolist() if d1_q is not None else None),
        derived_shaped_q=derived_shaped_q,
        gamma_predictor_q=gamma_predictor_q or None,
        win_head_q=win_head_q,
        terminal_win=terminal_win.tolist(),
        terminal_win_sem=terminal_win_sem.tolist(),
        n_truncated=n_trunc.tolist(),
        n_draws=n_draws.tolist(),
        derived_terminal_win=derived_terminal_win,
        per_action_wins=(wins_matrix.tolist() if store_per_branch else None),
        per_action_shaped_r=(R0.tolist() if store_per_branch and R0.size else None),
        G=int(k_ref),
        mean_steps_to_terminal=mean_steps,
        latency_ms_shaped=float(latency_ms_shaped),
        latency_ms_terminal=float(latency_ms_terminal),
    )
    return record, mentry


# ---------------------------------------------------------------------------
# Aggregate analysis: does shaped Q predict terminal win?
# ---------------------------------------------------------------------------


def _safe(x) -> float:
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return float("nan")
        return v
    except Exception:
        return float("nan")


def _argmax_over_legal(values: List[float], legal: List[int]) -> int:
    """Argmax of ``values`` (per legal action) -> the legal action index."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return int(legal[0]) if legal else 0
    return int(legal[int(np.nanargmax(arr))])


def _regret(terminal_win: np.ndarray, best_idx: int, sel_idx: int) -> float:
    """terminal-win regret = win[best] - win[sel] (>=0; higher is worse)."""
    if sel_idx < 0 or sel_idx >= terminal_win.size:
        return float("nan")
    if best_idx < 0 or best_idx >= terminal_win.size:
        return float("nan")
    return float(terminal_win[best_idx] - terminal_win[sel_idx])


def aggregate_terminal_win(
    records: List[TerminalWinRootRecord],
    derived_ks: List[int],
) -> Dict[str, Any]:
    """Compute the central Phase A outputs across the corpus.

    For each root and each predictor (root_critic, D=0 at each K, D=1), measures
    how well the shaped-Q action ranking predicts the terminal-win action
    ranking, plus the terminal-win regret of each selector. Stratified by phase,
    entropy, top-2 gap, request kind, and tactical category.
    """
    # predictor name -> list of per-root values
    spearman_by_pred: Dict[str, List[float]] = {
        "root_critic": [],
        "d0_k_ref": [],
        "d1": [],
    }
    kendall_by_pred: Dict[str, List[float]] = dict(spearman_by_pred)
    top1_match_by_pred: Dict[str, List[float]] = dict(spearman_by_pred)
    for kp in derived_ks:
        spearman_by_pred[f"d0_K{kp}"] = []
        kendall_by_pred[f"d0_K{kp}"] = []
        top1_match_by_pred[f"d0_K{kp}"] = []
        spearman_by_pred[f"term_G{kp}"] = []  # self-consistency: derived-G' vs G_ref
        kendall_by_pred[f"term_G{kp}"] = []
        top1_match_by_pred[f"term_G{kp}"] = []

    # regret of each selector vs the terminal-win-best action
    regret_by_sel: Dict[str, List[float]] = {
        "actor": [],
        "root_critic": [],
        "d0_k_ref": [],
        "d1": [],
    }
    for kp in derived_ks:
        regret_by_sel[f"d0_K{kp}"] = []
        regret_by_sel[f"term_G{kp}"] = []

    # does the shaped-search argmax DECREASE terminal win vs the actor?
    decrease_freq_by_pred: Dict[str, List[float]] = {
        "root_critic": [],
        "d0_k_ref": [],
        "d1": [],
    }
    for kp in derived_ks:
        decrease_freq_by_pred[f"d0_K{kp}"] = []

    # stratification tags (parallel lists to the per-root metrics)
    tags_phase: List[str] = []
    tags_entropy: List[str] = []
    tags_top2gap: List[str] = []
    tags_request: List[str] = []
    tags_tactical: List[str] = []
    # per-root terminal-win best/actor gap (the addressable opportunity)
    actor_vs_best_gap: List[float] = []
    n_truncated_total = 0
    n_draws_total = 0
    n_branches_total = 0

    # "meaningful" subset: roots where the action actually changes the terminal
    # win (spread > threshold). All-tied roots (e.g. already-won positions where
    # every action wins 100%) contribute zero regret (good -- search can't hurt)
    # but noise to the Spearman correlation (a flat target makes any ranking
    # look "anti-correlated"). The shaped-Q alignment should be judged on the
    # subset where the action matters; the no-opportunity roots are counted
    # separately.
    SPREAD_THRESHOLD = 0.02  # terminal-win (max - min) above which the action matters
    meaningful_spearman_d0: List[float] = []
    meaningful_regret_actor: List[float] = []
    meaningful_regret_d0: List[float] = []
    meaningful_actor_gap: List[float] = []
    n_no_opportunity = 0  # all-tied terminal-win (action doesn't matter)
    n_meaningful = 0

    # kimi-search M1: collect the union of extra gamma-head predictor names
    gamma_pred_names: List[str] = sorted(
        {name for r in records for name in (r.gamma_predictor_q or {})}
    )
    # kimi-search M3: the trained win head, when present, is an extra predictor
    if any(r.win_head_q is not None for r in records):
        gamma_pred_names = gamma_pred_names + ["win_head"]
    for gname in gamma_pred_names:
        spearman_by_pred[gname] = []
        kendall_by_pred[gname] = []
        top1_match_by_pred[gname] = []
        regret_by_sel[gname] = []
        decrease_freq_by_pred[gname] = []

    for r in records:
        legal = r.legal_actions
        A = r.n_legal
        if A < 2:
            continue
        tw = np.asarray(r.terminal_win, dtype=np.float64)
        if np.all(np.isnan(tw)):
            continue
        best_idx = int(np.nanargmax(tw))
        best_win = float(tw[best_idx])
        actor_idx_in_legal = int(
            np.nanargmax(np.asarray(r.base_probs, dtype=np.float64))
        )
        actor_win = float(tw[actor_idx_in_legal])
        actor_vs_best_gap.append(best_win - actor_win)

        # classify the root: does the action actually matter?
        tw_spread = float(np.nanmax(tw) - np.nanmin(tw))
        is_meaningful = tw_spread > SPREAD_THRESHOLD
        if is_meaningful:
            n_meaningful += 1
        else:
            n_no_opportunity += 1

        rc = np.asarray(r.root_critic_q, dtype=np.float64)
        d0 = np.asarray(r.d0_q, dtype=np.float64)
        d1 = np.asarray(r.d1_q, dtype=np.float64) if r.d1_q is not None else None

        pred_arrays = {"root_critic": rc, "d0_k_ref": d0}
        if d1 is not None:
            pred_arrays["d1"] = d1
        for gname in gamma_pred_names:
            if gname == "win_head":
                if r.win_head_q is not None:
                    pred_arrays["win_head"] = np.asarray(r.win_head_q, dtype=np.float64)
                continue
            gq = (r.gamma_predictor_q or {}).get(gname)
            if gq is not None:
                pred_arrays[gname] = np.asarray(gq, dtype=np.float64)
        for kp in derived_ks:
            dq = r.derived_shaped_q.get(f"D0:K{kp}")
            if dq is not None:
                pred_arrays[f"d0_K{kp}"] = np.asarray(dq, dtype=np.float64)
            dw = r.derived_terminal_win.get(f"G{kp}")
            if dw is not None:
                pred_arrays[f"term_G{kp}"] = np.asarray(dw, dtype=np.float64)

        for pname, pq in pred_arrays.items():
            if pname.startswith("term_"):
                # self-consistency: does derived-G' ranking match G_ref ranking?
                sp = spearman_corr(pq, tw)
                tm = (
                    float(np.nanargmax(pq) == best_idx)
                    if not np.all(np.isnan(pq))
                    else float("nan")
                )
            else:
                sp = spearman_corr(pq, tw)
                tm = (
                    float(np.nanargmax(pq) == best_idx)
                    if not np.all(np.isnan(pq))
                    else float("nan")
                )
            if not np.isnan(sp):
                spearman_by_pred[pname].append(sp)
                kendall_by_pred[pname].append(kendall_corr(pq, tw))
            if not np.isnan(tm):
                top1_match_by_pred[pname].append(tm)

        # regret of each selector
        sel_idx = {
            "actor": actor_idx_in_legal,
            "root_critic": int(np.nanargmax(rc)),
            "d0_k_ref": int(np.nanargmax(d0)),
        }
        if d1 is not None:
            sel_idx["d1"] = int(np.nanargmax(d1))
        for gname in gamma_pred_names:
            if gname in pred_arrays:
                sel_idx[gname] = int(np.nanargmax(pred_arrays[gname]))
        for kp in derived_ks:
            dq = r.derived_shaped_q.get(f"D0:K{kp}")
            if dq is not None:
                sel_idx[f"d0_K{kp}"] = int(
                    np.nanargmax(np.asarray(dq, dtype=np.float64))
                )
            dw = r.derived_terminal_win.get(f"G{kp}")
            if dw is not None:
                sel_idx[f"term_G{kp}"] = int(
                    np.nanargmax(np.asarray(dw, dtype=np.float64))
                )
        for sname, sidx in sel_idx.items():
            regret_by_sel[sname].append(_regret(tw, best_idx, sidx))

        # meaningful-subset tracking (roots where the action changes the outcome)
        if is_meaningful:
            sp_d0 = spearman_corr(d0, tw)
            if not np.isnan(sp_d0):
                meaningful_spearman_d0.append(sp_d0)
            meaningful_regret_actor.append(_regret(tw, best_idx, actor_idx_in_legal))
            meaningful_regret_d0.append(_regret(tw, best_idx, int(np.nanargmax(d0))))
            meaningful_actor_gap.append(best_win - actor_win)

        # does shaped-search argmax decrease terminal win vs actor?
        actor_sel = actor_idx_in_legal
        for pname in (
            ("root_critic", "d0_k_ref", "d1")
            + tuple(f"d0_K{kp}" for kp in derived_ks)
            + tuple(gamma_pred_names)
        ):
            if pname not in pred_arrays:
                continue
            pred = pred_arrays[pname]
            pidx = int(np.nanargmax(pred))
            decrease_freq_by_pred[pname].append(float(tw[pidx] < actor_win - 1e-9))

        # stratification tags -- phase recomputed from the ACTUAL battle-length
        # estimate (decision + mean_steps_to_terminal), not the static
        # typical_battle_len=120 heuristic in root_dataset.phase_band (which
        # mislabels late-game roots as "early" in short ~35-decision self-play
        # games). The record's stored phase_band is kept as phase_band_static
        # for audit; the analysis uses the calibrated band.
        est_battle_len = r.decision + max(r.mean_steps_to_terminal, 1.0)
        frac = r.decision / max(est_battle_len, 1.0)
        if frac < 0.33:
            phase_cal = "early"
        elif frac < 0.66:
            phase_cal = "mid"
        else:
            phase_cal = "late"
        tags_phase.append(phase_cal)
        tags_entropy.append(r.entropy_band)
        tags_top2gap.append(r.top2_gap_band)
        tags_request.append(r.request_kind)
        tags_tactical.append(r.tactical_category)
        n_truncated_total += int(sum(r.n_truncated))
        n_draws_total += int(sum(r.n_draws))
        n_branches_total += A * r.G

    def nm(x):
        return float(np.nanmean(x)) if len(x) else float("nan")

    def stratified(values: List[float], tag_list: List[str]) -> Dict[str, float]:
        bands: Dict[str, List[float]] = {}
        for v, t in zip(values, tag_list):
            bands.setdefault(t, []).append(float(v))
        return {b: float(np.nanmean(vv)) for b, vv in bands.items() if vv}

    out: Dict[str, Any] = {
        "n_roots": len(records),
        "n_roots_used": len(actor_vs_best_gap),
        "n_branches_total": n_branches_total,
        "n_truncated_total": n_truncated_total,
        "truncation_rate": (
            float(n_truncated_total / n_branches_total) if n_branches_total else 0.0
        ),
        "n_draws_total": n_draws_total,
        "draw_rate": (
            float(n_draws_total / n_branches_total) if n_branches_total else 0.0
        ),
        "derived_ks": derived_ks,
        # central outputs: correlation of shaped Q with terminal win
        "spearman_shaped_vs_terminal": {
            p: {
                "mean": nm(v),
                "median": float(np.nanmedian(v)) if v else float("nan"),
                "n": len(v),
            }
            for p, v in spearman_by_pred.items()
        },
        "kendall_shaped_vs_terminal": {
            p: {"mean": nm(v), "n": len(v)} for p, v in kendall_by_pred.items()
        },
        "top1_match_vs_terminal": {
            p: {"mean": nm(v), "n": len(v)} for p, v in top1_match_by_pred.items()
        },
        # terminal-win regret of each selector (higher = worse)
        "terminal_win_regret": {
            s: {
                "mean": nm(v),
                "median": float(np.nanmedian(v)) if v else float("nan"),
                "n": len(v),
            }
            for s, v in regret_by_sel.items()
        },
        # frequency a shaped-search argmax DECREASES terminal win vs actor
        "decrease_freq_vs_actor": {p: nm(v) for p, v in decrease_freq_by_pred.items()},
        # addressable opportunity: how much terminal win the actor leaves on the table
        "actor_vs_best_gap_mean": nm(actor_vs_best_gap),
        "actor_vs_best_gap_median": (
            float(np.nanmedian(actor_vs_best_gap))
            if actor_vs_best_gap
            else float("nan")
        ),
        # meaningful-subset: roots where the action actually changes the terminal
        # win (spread > 0.02). All-tied roots are "no-opportunity" (search can't
        # help, and their Spearman is noise against a flat target).
        "n_meaningful": n_meaningful,
        "n_no_opportunity": n_no_opportunity,
        "meaningful_spearman_d0_k_ref_mean": nm(meaningful_spearman_d0),
        "meaningful_spearman_d0_k_ref_median": (
            float(np.nanmedian(meaningful_spearman_d0))
            if meaningful_spearman_d0
            else float("nan")
        ),
        "meaningful_actor_regret_mean": nm(meaningful_regret_actor),
        "meaningful_d0_k_ref_regret_mean": nm(meaningful_regret_d0),
        "meaningful_actor_gap_mean": nm(meaningful_actor_gap),
    }

    # stratification of the primary correlation (D=0 K=ref) + actor regret
    out["stratified"] = {
        "spearman_d0_k_ref_by_phase": stratified(
            spearman_by_pred["d0_k_ref"], tags_phase
        ),
        "spearman_d0_k_ref_by_entropy": stratified(
            spearman_by_pred["d0_k_ref"], tags_entropy
        ),
        "spearman_d0_k_ref_by_top2gap": stratified(
            spearman_by_pred["d0_k_ref"], tags_top2gap
        ),
        "spearman_d0_k_ref_by_request": stratified(
            spearman_by_pred["d0_k_ref"], tags_request
        ),
        "spearman_d0_k_ref_by_tactical": stratified(
            spearman_by_pred["d0_k_ref"], tags_tactical
        ),
        "actor_regret_by_phase": stratified(regret_by_sel["actor"], tags_phase),
        "actor_regret_by_tactical": stratified(regret_by_sel["actor"], tags_tactical),
        "d0_k_ref_regret_by_phase": stratified(regret_by_sel["d0_k_ref"], tags_phase),
        "d0_k_ref_regret_by_tactical": stratified(
            regret_by_sel["d0_k_ref"], tags_tactical
        ),
    }
    out["phase_distribution"] = {
        b: int(tags_phase.count(b)) for b in ("early", "mid", "late")
    }
    # also report the static (heuristic) phase bands for comparison, so the
    # miscalibration is visible in the report
    static_phases = [r.phase_band for r in records if r.n_legal >= 2]
    out["phase_distribution_static_heuristic"] = {
        b: int(static_phases.count(b)) for b in ("early", "mid", "late")
    }
    out["request_kind_distribution"] = {
        b: int(tags_request.count(b)) for b in ("move", "forceswitch")
    }
    out["tactical_distribution"] = {
        b: int(tags_tactical.count(b)) for b in sorted(set(tags_tactical))
    }
    return out


# ---------------------------------------------------------------------------
# Gate A: is the shaped critic suitable for search?
# ---------------------------------------------------------------------------


def terminal_win_gate(summary: Dict[str, Any], derived_ks: List[int]) -> Dict[str, Any]:
    """The Phase A go/no-go gate (skill §37 "Gate A").

    Proceed with the existing shaped-critic evaluator only when:
    * **correlated**: the K_ref shaped-Q action ranking is positively correlated
      with terminal win probability (mean Spearman > 0; a useful heuristic is
      > 0.5, but the operational metric is the regret/improvement, not the
      correlation alone);
    * **improves_over_actor**: the K_ref shaped-Q argmax has lower mean
      terminal-win regret than the actor argmax (search-selected actions win
      more than the actor on held-out roots);
    * **not_catastrophic**: the K_ref shaped-Q argmax decreases terminal win
      vs the actor on a minority of roots (catastrophic errors are acceptably
      rare);
    * **converges_with_k**: higher-K shaped Q is at least as well-correlated
      with terminal win as lower-K (the signal is not pure noise that happens to
      correlate at one K).

    Failure -> the recommendation is to train a terminal-outcome value head
    (skill §37 "Failure outcome: train a terminal-outcome value head").
    """
    n = summary.get("n_roots_used", 0)
    if n == 0:
        return {
            "verdict": "INCONCLUSIVE",
            "passed": 0,
            "total": 4,
            "criteria": {},
            "note": "no usable roots",
        }
    sp = summary.get("spearman_shaped_vs_terminal", {})
    reg = summary.get("terminal_win_regret", {})
    dec = summary.get("decrease_freq_vs_actor", {})

    # the highest available K (k_ref or the largest derived K)
    k_ref_key = "d0_k_ref"
    sp_ref = sp.get(k_ref_key, {}).get("mean", float("nan"))
    reg_actor = reg.get("actor", {}).get("mean", float("nan"))
    reg_ref = reg.get(k_ref_key, {}).get("mean", float("nan"))
    dec_ref = dec.get(k_ref_key, float("nan"))

    crit_corr = {
        "d0_k_ref_spearman_mean": float(sp_ref),
        "pass": bool(np.isfinite(sp_ref) and sp_ref > 0.0),
    }
    crit_improves = {
        "actor_regret_mean": float(reg_actor),
        "d0_k_ref_regret_mean": float(reg_ref),
        "regret_delta": float(reg_ref - reg_actor),
        "pass": bool(
            np.isfinite(reg_ref) and np.isfinite(reg_actor) and reg_ref < reg_actor
        ),
    }
    crit_not_cat = {
        "d0_k_ref_decrease_freq_vs_actor": float(dec_ref),
        "pass": bool(np.isfinite(dec_ref) and dec_ref < 0.50),
    }
    # converges with K: spearman should be non-decreasing in K (or at least the
    # highest K is >= the lowest K)
    sp_low = (
        sp.get(f"d0_K{derived_ks[0]}", {}).get("mean", float("nan"))
        if derived_ks
        else float("nan")
    )
    sp_high = (
        sp.get(f"d0_K{derived_ks[-1]}", {}).get("mean", float("nan"))
        if derived_ks
        else float("nan")
    )
    crit_converges = {
        "spearman_d0_K_low": float(sp_low),
        "spearman_d0_K_high": float(sp_high),
        "pass": bool(
            np.isfinite(sp_low) and np.isfinite(sp_high) and sp_high >= sp_low - 0.02
        ),
    }
    criteria = {
        "correlated": crit_corr,
        "improves_over_actor": crit_improves,
        "not_catastrophic": crit_not_cat,
        "converges_with_k": crit_converges,
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
            "PASS = the shaped critic's preference after an exact oracle transition "
            "predicts which action increases terminal win probability, and selecting "
            "by it improves over the actor -- proceed with the existing evaluator "
            "(then Phase B/C/D). PARTIAL/FAIL = the shaped objective is not aligned "
            "with winning; the recommendation (skill §37) is to train a terminal-"
            "outcome value head before further K/depth/opponent tuning."
        ),
    }


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def _fmt(x: float, prec: int = 3) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "nan"
    return f"{x:.{prec}f}"


def report_markdown(
    summary: Dict[str, Any],
    gate: Dict[str, Any],
    derived_ks: List[int],
) -> str:
    """Render the Phase A report (Markdown)."""
    sp = summary.get("spearman_shaped_vs_terminal", {})
    reg = summary.get("terminal_win_regret", {})
    tm = summary.get("top1_match_vs_terminal", {})
    dec = summary.get("decrease_freq_vs_actor", {})
    lines = [
        "# Phase A: Terminal-Win Fixed-Root Benchmark",
        "",
        f"**Gate verdict: {gate['verdict']}** ({gate['passed']}/{gate['total']} criteria).",
        "",
        f"n_roots = {summary.get('n_roots_used', 0)} "
        f"(of {summary.get('n_roots', 0)} captured), "
        f"n_branches = {summary.get('n_branches_total', 0)}, "
        f"truncation_rate = {_fmt(summary.get('truncation_rate', 0))}, "
        f"draw_rate = {_fmt(summary.get('draw_rate', 0))}.",
        "",
        "## Central question",
        "",
        "> Does the frozen critic's preference after an exact oracle transition",
        "> predict which action actually increases terminal win probability?",
        "",
        "## Spearman correlation: shaped Q vs terminal win probability",
        "",
        "| predictor | mean Spearman | median | top-1 match vs terminal | n |",
        "|---|---|---|---|---|",
    ]
    gamma_preds = sorted(
        p for p in sp if p.startswith("root_critic_g") or p.startswith("d0_g")
    ) + (["win_head"] if "win_head" in sp else [])
    for p in (
        ["root_critic", "d0_k_ref"]
        + [f"d0_K{k}" for k in derived_ks]
        + (["d1"] if "d1" in sp else [])
        + gamma_preds
        + [f"term_G{k}" for k in derived_ks]
    ):
        if p not in sp:
            continue
        s = sp[p]
        t = tm.get(p, {})
        lines.append(
            f"| {p} | {_fmt(s.get('mean'))} | {_fmt(s.get('median'))} | "
            f"{_fmt(t.get('mean'))} | {s.get('n', 0)} |"
        )
    lines += [
        "",
        "## Terminal-win regret of each selector (vs terminal-win-best action)",
        "",
        "| selector | mean regret | median regret | n |",
        "|---|---|---|---|",
    ]
    for s in (
        ["actor", "root_critic", "d0_k_ref"]
        + [f"d0_K{k}" for k in derived_ks]
        + (["d1"] if "d1" in reg else [])
        + gamma_preds
        + [f"term_G{k}" for k in derived_ks]
    ):
        if s not in reg:
            continue
        r = reg[s]
        lines.append(
            f"| {s} | {_fmt(r.get('mean'))} | {_fmt(r.get('median'))} | {r.get('n', 0)} |"
        )
    lines += [
        "",
        "## Frequency a shaped-search argmax DECREASES terminal win vs the actor",
        "",
        "| predictor | decrease frequency |",
        "|---|---|",
    ]
    for p in (
        ["root_critic", "d0_k_ref"]
        + [f"d0_K{k}" for k in derived_ks]
        + (["d1"] if "d1" in dec else [])
        + gamma_preds
    ):
        if p not in dec:
            continue
        lines.append(f"| {p} | {_fmt(dec.get(p))} |")
    lines += [
        "",
        f"Actor vs terminal-win-best gap: mean = {_fmt(summary.get('actor_vs_best_gap_mean'))}, "
        f"median = {_fmt(summary.get('actor_vs_best_gap_median'))} "
        "(the terminal-win the actor leaves on the table -- the addressable opportunity).",
        "",
        f"**Meaningful subset** (roots where the action changes the terminal win, "
        f"spread > 0.02): n = {summary.get('n_meaningful', 0)} of "
        f"{summary.get('n_roots_used', 0)} ({_fmt(summary.get('n_meaningful', 0) / max(summary.get('n_roots_used', 1), 1))} of roots). "
        f"{summary.get('n_no_opportunity', 0)} roots are no-opportunity (all actions tie -- search can't help, and their Spearman is noise).",
        "",
        f"On the meaningful subset: D=0 K=ref Spearman = {_fmt(summary.get('meaningful_spearman_d0_k_ref_mean'))} "
        f"(median {_fmt(summary.get('meaningful_spearman_d0_k_ref_median'))}), "
        f"actor regret = {_fmt(summary.get('meaningful_actor_regret_mean'))}, "
        f"D=0 K=ref regret = {_fmt(summary.get('meaningful_d0_k_ref_regret_mean'))}, "
        f"actor-vs-best gap = {_fmt(summary.get('meaningful_actor_gap_mean'))}.",
        "",
        "## Stratified (D=0 K_ref Spearman + regret)",
        "",
        f"- by phase: {summary.get('stratified', {}).get('spearman_d0_k_ref_by_phase', {})}",
        f"- by entropy: {summary.get('stratified', {}).get('spearman_d0_k_ref_by_entropy', {})}",
        f"- by top-2 gap: {summary.get('stratified', {}).get('spearman_d0_k_ref_by_top2gap', {})}",
        f"- by request kind: {summary.get('stratified', {}).get('spearman_d0_k_ref_by_request', {})}",
        f"- by tactical category: {summary.get('stratified', {}).get('spearman_d0_k_ref_by_tactical', {})}",
        "",
        f"Actor regret by phase: {summary.get('stratified', {}).get('actor_regret_by_phase', {})}",
        f"D=0 K_ref regret by phase: {summary.get('stratified', {}).get('d0_k_ref_regret_by_phase', {})}",
        "",
        "## Distributions",
        "",
        f"- phase: {summary.get('phase_distribution', {})}",
        f"- request kind: {summary.get('request_kind_distribution', {})}",
        f"- tactical: {summary.get('tactical_distribution', {})}",
        "",
        "## Gate criteria",
        "",
    ]
    for cname, c in gate.get("criteria", {}).items():
        lines.append(f"### {cname}: {'PASS' if c.get('pass') else 'FAIL'}")
        for k, v in c.items():
            if k != "pass":
                lines.append(f"- {k}: {v}")
        lines.append("")
    lines.append(gate.get("note", ""))
    return "\n".join(lines)


def write_results(
    records: List[TerminalWinRootRecord],
    manifest: List[RootManifestEntry],
    summary: Dict[str, Any],
    gate: Dict[str, Any],
    run_manifest_obj: Dict[str, Any],
    output_dir: str,
    derived_ks: List[int],
) -> Dict[str, str]:
    """Write all Phase A artifacts to ``output_dir`` (JSON + Markdown report)."""
    os.makedirs(output_dir, exist_ok=True)
    paths: Dict[str, str] = {}
    p = os.path.join(output_dir, "terminal_win_summary.json")
    with open(p, "w") as f:
        json.dump(
            {"summary": summary, "gate": gate, "run_manifest": run_manifest_obj},
            f,
            indent=2,
        )
    paths["summary"] = p
    p = os.path.join(output_dir, "terminal_win_REPORT.md")
    with open(p, "w") as f:
        f.write(report_markdown(summary, gate, derived_ks))
    paths["report"] = p
    p = os.path.join(output_dir, "run_manifest.json")
    with open(p, "w") as f:
        json.dump(run_manifest_obj, f, indent=2)
    paths["run_manifest"] = p
    # (root_results.jsonl + root_manifest.jsonl are streamed during the run)
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Phase A terminal-win fixed-root benchmark (the central go/no-go gate)"
    )
    p.add_argument("--agent", default="MiniOnlinePsroV1_4")
    p.add_argument("--checkpoint", type=int, default=740)
    p.add_argument("--format", default="gen1ou")
    p.add_argument("--team_set", default="competitive")
    p.add_argument("--num_parallel", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--search_seed", type=int, default=0)
    p.add_argument(
        "--k_ref",
        type=int,
        default=128,
        help="reference rollout/continuation count (shaped-Q K_ref AND terminal-win G)",
    )
    p.add_argument(
        "--derived_ks",
        type=int,
        nargs="+",
        default=[4, 16, 64],
        help="low-K shaped-Q + low-G terminal-win values derived by prefix averaging",
    )
    p.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=[0],
        help="shaped-Q depths (D=0 primary)",
    )
    p.add_argument("--max_roots", type=int, default=40)
    p.add_argument("--max_battles", type=int, default=40)
    p.add_argument("--root_stride", type=int, default=1)
    p.add_argument(
        "--decision_stride",
        type=int,
        default=3,
        help="per-battle capture cadence (spreads corpus across early/mid/late)",
    )
    p.add_argument("--min_decision", type=int, default=0)
    p.add_argument("--max_decision", type=int, default=None)
    p.add_argument(
        "--gamma_indices",
        type=int,
        nargs="*",
        default=None,
        help="kimi-search M1: extra critic gamma-head indices to evaluate as "
        "predictors (e.g. --gamma_indices 4 5 adds root_critic_g4 / d0_g4 / "
        "root_critic_g5 / d0_g5 alongside the primary head). Measures whether "
        "higher-gamma heads are better aligned with terminal win (H2).",
    )
    p.add_argument(
        "--win_head_path",
        default=None,
        help="kimi-search M3: path to a trained WinHead checkpoint; adds a "
        "'win_head' predictor (per-action P(win) at the root) to the benchmark.",
    )
    p.add_argument(
        "--save_distill",
        action="store_true",
        help="kimi-search M4: also write distill_labels.jsonl (per-root obs + "
        "terminal-win-best action label) for policy distillation.",
    )
    p.add_argument("--store_per_branch", action="store_true")
    p.add_argument("--max_steps_to_terminal", type=int, default=250)
    p.add_argument("--progress_every", type=int, default=2)
    p.add_argument("--output_dir", required=True)

    # accept the same bundle args as benchmark_roots via _build_bundle
    class _Args:
        pass

    args = p.parse_args()
    # _build_bundle reads args.agent/checkpoint/format/team_set/num_parallel/seed
    bundle, env, agent = _build_bundle(args)
    try:
        t0 = time.perf_counter()
        derived_ks = sorted(k for k in args.derived_ks if k <= args.k_ref)
        if not derived_ks:
            derived_ks = [4, 16, 64]
        records, manifest = benchmark_terminal_win(
            bundle=bundle,
            k_ref=args.k_ref,
            derived_ks=derived_ks,
            depths=args.depths,
            max_roots=args.max_roots,
            max_battles=args.max_battles,
            root_stride=args.root_stride,
            decision_stride=args.decision_stride,
            min_decision=args.min_decision,
            max_decision=args.max_decision,
            store_per_branch=args.store_per_branch,
            gamma_indices=args.gamma_indices,
            win_head_path=args.win_head_path,
            save_distill=args.save_distill,
            progress_every=args.progress_every,
            env_seed=args.seed,
            search_seed=args.search_seed,
            max_steps_to_terminal=args.max_steps_to_terminal,
            output_dir=args.output_dir,
        )
        elapsed = time.perf_counter() - t0
        summary = aggregate_terminal_win(records, derived_ks)
        gate = terminal_win_gate(summary, derived_ks)

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
            leaf_modes=["root_critic_only", "policy_expectation"],
            chance_mode="resample_crn",
            n_roots=len(records),
            n_battles=args.max_battles,
            git_sha=git_sha,
            extra={
                "elapsed_sec": elapsed,
                "phase": "A_terminal_win",
                "max_steps_to_terminal": args.max_steps_to_terminal,
                "decision_stride": args.decision_stride,
            },
        )
        paths = write_results(
            records, manifest, summary, gate, rm, args.output_dir, derived_ks
        )
        print(
            json.dumps(
                {
                    "verdict": gate["verdict"],
                    "passed": gate["passed"],
                    "total": gate["total"],
                    "n_roots": len(records),
                    "elapsed_sec": elapsed,
                    "spearman_d0_k_ref": summary.get("spearman_shaped_vs_terminal", {})
                    .get("d0_k_ref", {})
                    .get("mean"),
                    "actor_regret": summary.get("terminal_win_regret", {})
                    .get("actor", {})
                    .get("mean"),
                    "d0_k_ref_regret": summary.get("terminal_win_regret", {})
                    .get("d0_k_ref", {})
                    .get("mean"),
                    "paths": paths,
                },
                indent=2,
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
