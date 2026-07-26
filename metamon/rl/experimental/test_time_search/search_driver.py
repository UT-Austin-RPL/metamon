"""Oracle root Monte Carlo test-time search over a frozen Metamon policy.

This module implements the eval-only search wrapper. At every searched settled
evaluated-player decision it:

  1. reads the frozen actor distribution over legal root actions and selects
     root candidates (all_legal / relative_threshold / cumulative_mass);
  2. snapshots the trunk simulator lane + forks the eval and opponent policy
     recurrent state into ``A * K`` branch lanes (``A`` retained actions, ``K``
     rollouts per action), reseeding each branch's Showdown PRNG from a
     deterministic common-random-number seed bank (skill §7);
  3. forces the eval player's candidate root action per branch and samples the
     opponent's simultaneous root action -- **one per rollout index ``k``,
     reused across candidate actions** (opponent root coupling, skill §7);
  4. lets the turn fully settle and continues policy-guided rollouts for
     ``search_depth`` settled decisions;
  5. estimates each leaf with the frozen critic -- exact policy expectation
     ``V_pi(h) = sum_a pi(a|h) Q(h,a)`` over all legal actions (skill §10);
  6. accumulates the return in the **same convention as the critic training
     target**: rewards scaled by ``reward_multiplier``, discounted per settled
     eval decision (gamma**k for the k-th settlement), terminal victory reward
     recorded once with zero bootstrap (skill §5);
  7. groups values by root action, builds the improved policy via the chosen
     operator (single_anchor_kl / magnetic_kl / ablations, skill §12) with a
     fixed global value scale (skill §11), and selects the live action;
  8. releases all branch lanes/snapshots and returns the selected action.

The trunk lane and the live eval/opponent drivers are never touched by phantom
rollouts (validated by the simulator-fork tests). Search is opt-in; when
disabled the runner reproduces the frozen baseline exactly. Research runs use
``search_error_policy="raise"`` so a broken config fails loudly (skill §19).
"""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .branch_state import make_branch_state
from .config import SearchConfig
from .improvement import improve_policy
from .rng import RootSeedBank, make_rng, policy_rng_key

# ---------------------------------------------------------------------------
# Policy helpers (frozen actor distribution + critic leaf value)
# ---------------------------------------------------------------------------


def _primary_probs(
    policy, traj_emb_t: torch.Tensor, illegal: torch.Tensor
) -> torch.Tensor:
    """Return the actor probabilities for the primary gamma, shape (B, A).

    ``illegal`` is a bool tensor (B, 1, A) where True == illegal (masked -inf),
    as produced by ``numpy_obs_to_torch`` (which unsqueezes the L=1 dim).
    """
    with torch.no_grad():
        dist = policy.actor(traj_emb_t, straight_from_obs={"illegal_actions": illegal})
        # dist.probs: (B, L=1, G, A); take primary gamma (-1) and squeeze L.
        probs = dist.probs[..., -1, :].squeeze(1)  # (B, A)
    return probs.detach()


def _critic_leaf_values(
    policy,
    traj_emb_t: torch.Tensor,
    action_idx: torch.Tensor,
    action_dim: int,
    horizon_index: int,
) -> torch.Tensor:
    """Denormalized expected Q(s, a) for a batch, averaged over the critic ensemble.

    Legacy single-action bootstrap (skill §10 ``sampled_action`` leaf mode).
    """
    B = traj_emb_t.shape[0]
    G = policy.gammas.numel()
    a_oh = F.one_hot(action_idx.long(), action_dim).float()  # (B, A)
    # tile the one-hot across the G (gamma) axis so the critic's (K,B,L,G,A)
    # action contract is satisfied: (B,A) -> (B,1,A) -> (B,G,A) -> (1,B,1,G,A).
    a_oh = a_oh.unsqueeze(1).expand(-1, G, -1).reshape(1, B, 1, G, action_dim)
    a_oh = a_oh.to(traj_emb_t.device)  # (K=1,B,L=1,G,A)
    bin_dist = policy.critics(traj_emb_t, a_oh)  # Categorical (K=1,B,L=1,C,G,bins)
    q_raw = policy.critics.bin_dist_to_raw_vals(bin_dist)  # (K=1,B,L=1,C,G,1)
    q_denorm = policy.popart(q_raw, normalized=False)  # (K=1,B,L=1,C,G,1)
    q = q_denorm[0, :, 0, :, horizon_index, 0].mean(dim=-1)  # (B,)
    return q.detach()


def _eager_critic_forward(
    policy, emb: torch.Tensor, a_oh: torch.Tensor
) -> torch.Tensor:
    """Run the critic forward eagerly (no torch.compile tracing of the caller).

    The critic head's ``critic_network_forward`` is ``@torch.compile``d; calling
    it with a varying batch dim ``B`` triggers recompilation storms. Decorating
    this helper with ``torch.compiler.disable`` makes the call eager (dynamo
    skips tracing the caller), so the compiled callee is invoked as a plain
    callable and never recompiles on dynamic ``B`` (skill §10). Correctness
    first; throughput is optimized later (skill §25).
    """
    bin_dist = policy.critics(emb, a_oh)
    q_raw = policy.critics.bin_dist_to_raw_vals(bin_dist)
    return policy.popart(q_raw, normalized=False)


_disable_compile = getattr(getattr(torch, "compiler", None), "disable", None)
if callable(_disable_compile):
    try:
        _eager_critic_forward = _disable_compile(_eager_critic_forward)
    except Exception:
        # Older torch versions: keep the undecorated (still correct, just not
        # compile-guarded) helper.
        pass


def _all_action_q(
    policy,
    emb: torch.Tensor,
    action_dim: int,
    horizon_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Q(h, a) for every action a in [0, action_dim), for a batch of leaves.

    Args:
        emb: (B, 1, d_model) state embedding.
        action_dim: A (fixed model output width; 13 for the frozen checkpoint).

    Returns:
        (q_mean (B, A) denormalized mean-over-critics Q at the horizon,
         q_per_head (B, A, C) per-critic-head Q for disagreement logging).

    Uses one critic call with K=action_dim (the critic tiles state along K
    internally; see the policy-driver audit). Illegal actions are NOT masked
    here -- the caller masks them.
    """
    B = emb.shape[0]
    G = policy.gammas.numel()
    A = int(action_dim)
    device = emb.device
    all_oh = torch.eye(A, device=device, dtype=torch.float32)  # (A, A)
    # (K=A, B, L=1, G, A): action k is the one-hot for action k.
    a_oh = all_oh.view(A, 1, 1, 1, A).expand(A, B, 1, G, A).contiguous()
    q_denorm = _eager_critic_forward(policy, emb, a_oh)  # (A, B, 1, C, G, 1)
    q_per_head = q_denorm[:, :, 0, :, horizon_index, 0]  # (A, B, C)
    q_mean = q_per_head.mean(dim=-1)  # (A, B)
    return q_mean.t().detach(), q_per_head.permute(1, 0, 2).detach()  # (B,A), (B,A,C)


def _exact_leaf_v_pi(
    policy,
    emb: torch.Tensor,
    illegal_3d: torch.Tensor,
    action_dim: int,
    horizon_index: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact ``V_pi(h) = sum_a pi(a|h) Q(h,a)`` over legal actions (skill §10).

    Args:
        emb: (B, 1, d_model).
        illegal_3d: (B, 1, A) bool, True == illegal.

    Returns:
        (v_pi (B,), q_mean (B, A), probs (B, A), q_per_head (B, A, C)).
    """
    probs = _primary_probs(policy, emb, illegal_3d)  # (B, A)
    q_mean, q_per_head = _all_action_q(
        policy, emb, action_dim, horizon_index
    )  # (B,A),(B,A,C)
    legal = ~illegal_3d[:, 0, :]  # (B, A) True == legal
    p = probs * legal.float()
    s = p.sum(-1, keepdim=True).clamp_min(1e-12)
    p = p / s
    q_legal = q_mean * legal.float()
    v_pi = (p * q_legal).sum(-1)  # (B,)
    return v_pi.detach(), q_mean.detach(), p.detach(), q_per_head.detach()


def _state_embedding(policy, obs, rl2s, time_idxs, hidden):
    """Wrapper around policy.get_state_embedding with no grad."""
    with torch.no_grad():
        emb, new_hidden = policy.get_state_embedding(
            obs=obs, rl2s=rl2s, time_idxs=time_idxs, hidden_state=hidden
        )
    return emb, new_hidden


# ---------------------------------------------------------------------------
# Search diagnostics record
# ---------------------------------------------------------------------------


@dataclass
class SearchRootRecord:
    """Structured per-search record (serialized to JSONL)."""

    battle_id: str
    decision: int
    legal_actions: List[int]
    base_probs: List[float]
    rollout_counts: List[int]
    search_q_mean: List[float]
    search_q_std: List[float]
    search_q_sem: List[float]
    terminal_frac: List[float]
    searched_probs: List[float]
    selected_action: int
    base_argmax: int
    changed_argmax: bool
    kl_to_base: float
    base_entropy: float
    searched_entropy: float
    critic_horizon: int
    search_depth: int
    n_rollouts: int
    latency_ms: float
    oracle: bool = True
    n_legal_before_prune: int = 0
    n_legal_after_prune: int = 0
    # --- research-mode fields (skill §11/§12/§15/§20) ---
    operator: str = "single_anchor_kl"
    alpha: float = 0.0
    beta: float = 1.0
    value_scale_mode: str = "raw"
    global_advantage_scale: Optional[float] = None
    chance_mode: str = "resample_crn"
    opp_root_coupling: bool = True
    leaf_value_mode: str = "policy_expectation"
    candidate_mode: str = "all_legal"
    reward_multiplier: float = 10.0
    intermediate_reward_mean: float = 0.0
    bootstrap_mean: float = 0.0
    critic_disagreement: float = 0.0
    n_settled_mean: float = 0.0
    env_seed_hashes: List[str] = field(default_factory=list)
    opp_root_actions: List[int] = field(default_factory=list)
    error: str = ""
    branch_details: Optional[List[dict]] = None

    def to_json(self) -> str:
        return json.dumps(self.__dict__)


# ---------------------------------------------------------------------------
# Rollout branch bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class _Branches:
    """All fork lanes for one trunk's search."""

    lane_ids: List[int]
    lanes: List[Any]
    root_action: np.ndarray  # (N,) eval root action index per branch
    rollout_index: np.ndarray  # (N,) rollout index k in [0, K) per branch
    active: np.ndarray  # (N,) bool: branch still rolling out
    depth_done: np.ndarray  # (N,) int: settled decisions completed (root + deeper)
    cum_reward: np.ndarray  # (N,) discounted, multiplier-scaled intermediate reward
    terminal: np.ndarray  # (N,) bool: branch reached a terminal state
    eval_hidden: Any
    eval_rl2s: np.ndarray
    eval_steps: np.ndarray
    opp_hidden: Any
    opp_rl2s: np.ndarray
    opp_steps: np.ndarray
    snap_id: int
    trunk_lane: int
    prev_eval_state: List[Any]
    gamma: float
    seed_bank: Optional[RootSeedBank]


# ---------------------------------------------------------------------------
# Search driver
# ---------------------------------------------------------------------------


class SearchEvalRunner:
    """Runs vectorized Showdown eval with optional oracle root MC search."""

    def __init__(
        self,
        env,
        eval_driver,
        opponent,
        eval_policy,
        opponent_policy,
        eval_action_space,
        opponent_action_space,
        eval_reward_function,
        config: SearchConfig,
        device: torch.device,
        action_dim: int,
        battle_format: str = "gen1ou",
        opponent_reward_function=None,
        reward_multiplier: float = 10.0,
    ):
        self.env = env
        self.eval_driver = eval_driver
        self.opponent = opponent
        self.eval_policy = eval_policy
        self.opponent_policy = opponent_policy
        self.eval_action_space = eval_action_space
        self.opponent_action_space = opponent_action_space
        self.eval_reward_function = eval_reward_function
        self.opponent_reward_function = opponent_reward_function
        self.config = config
        self.device = device
        self.action_dim = int(action_dim)
        self.battle_format = battle_format
        self.reward_multiplier = float(reward_multiplier)
        self._next_fork_lane = 10_000
        self._active_fork_lanes: List[int] = []
        self.root_records: List[SearchRootRecord] = []
        self._decision_counter = 0
        self._battle_counter = 0
        self._battle_id = ""
        self._log_file = None
        if config.search_log_roots:
            os.makedirs(os.path.dirname(config.search_log_roots) or ".", exist_ok=True)
            self._log_file = open(config.search_log_roots, "a")

    # ----- public API -----------------------------------------------------

    def close(self) -> None:
        self._release_all_fork_lanes()
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def _release_all_fork_lanes(self) -> None:
        proc = self.env.proc
        for lane_id in self._active_fork_lanes:
            try:
                proc.reset(lane_id)
            except Exception:
                pass
        self._active_fork_lanes.clear()

    # ----- lane/sim access ------------------------------------------------

    def _trunk_lane(self, i: int):
        return self.env.lanes[i]

    def _alloc_fork_lanes(self, n: int) -> List[int]:
        ids = [self._next_fork_lane + k for k in range(n)]
        self._next_fork_lane += n
        self._active_fork_lanes.extend(ids)
        return ids

    # ----- observation helpers -------------------------------------------

    def _build_obs(self, lane, side: str, action_space) -> Tuple[dict, List[int]]:
        state = lane.universal_state(side)
        if side == self.env.eval_side:
            obs = self.env.eval_obs_spaces[
                lane.lane_id % self.env.batched_envs
            ].state_to_obs(state)
            legal = lane.legal_action_indices(side, self.eval_action_space, state)
        else:
            obs = self.env.opponent_obs_spaces[
                lane.lane_id % self.env.batched_envs
            ].state_to_obs(state)
            legal = lane.legal_action_indices(side, self.opponent_action_space, state)
        n = action_space.gym_space.n
        mask = np.ones((n,), dtype=bool)
        for idx in legal:
            if 0 <= idx < n:
                mask[idx] = False
        obs["illegal_actions"] = mask
        return obs, legal

    # ----- value-scale resolution (skill §11) -----------------------------

    def _resolve_value_scale(self) -> Tuple[bool, Optional[float], str]:
        cfg = self.config
        mode = cfg.search_value_scale_mode
        if mode == "legacy_zscore":
            return True, None, "legacy_zscore"
        if mode == "environment_units":
            return False, float(self.reward_multiplier), "environment_units"
        if mode == "global_standardized":
            return (
                False,
                float(cfg.search_global_advantage_scale),
                "global_standardized",
            )
        # "raw": respect the legacy bool so old configs with normalization=True
        # but no explicit scale_mode still z-score.
        return (
            bool(cfg.search_value_normalization),
            cfg.search_global_advantage_scale,
            "raw",
        )

    # ----- root candidate selection (skill §9) ----------------------------

    @staticmethod
    def _select_candidates(
        legal_arr: np.ndarray,
        base_probs: np.ndarray,
        cfg: SearchConfig,
    ) -> np.ndarray:
        """Apply the configured root candidate mode; always keep the argmax."""
        if legal_arr.size <= 1:
            return legal_arr
        mode = cfg.search_root_candidate_mode
        if mode == "all_legal":
            return legal_arr
        legal_probs = base_probs[legal_arr]
        if mode == "relative_threshold":
            if cfg.search_root_prob_threshold <= 0.0:
                return legal_arr
            max_p = float(legal_probs.max())
            keep = legal_probs >= max_p * cfg.search_root_prob_threshold
            keep[int(np.argmax(legal_probs))] = True
            return legal_arr[keep]
        if mode == "cumulative_mass":
            order = np.argsort(-legal_probs)
            sorted_p = legal_probs[order]
            cum = np.cumsum(sorted_p)
            cutoff = int(np.searchsorted(cum, cfg.search_cumulative_mass_threshold) + 1)
            cutoff = max(cutoff, cfg.search_min_root_actions)
            cutoff = min(cutoff, legal_arr.size)
            kept = np.sort(legal_arr[order[:cutoff]])
            # always keep argmax
            kept = np.unique(np.append(kept, legal_arr[int(np.argmax(legal_probs))]))
            return kept
        return legal_arr

    # ----- the core search ------------------------------------------------

    def _root_distribution(
        self,
        trunk_lane_idx: int,
        obs: dict,
        legal: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor, int]:
        """Compute the base actor distribution and select root candidates.

        Returns (base_probs_full (A_full,), legal_arr (A_retained,), emb (1,1,d),
        illegal (1,1,A_full), base_argmax).
        """
        from metamon.env.vectorized.obs_utils import numpy_obs_to_torch, stack_obs_dicts
        from .branch_state import fork_hidden

        cfg = self.config
        legal_arr0 = np.array(legal, dtype=np.int64)
        obs_batch = stack_obs_dicts([obs])
        torch_obs = numpy_obs_to_torch(obs_batch, self.device)
        rl2 = (
            torch.from_numpy(self.eval_driver.rl2s[trunk_lane_idx : trunk_lane_idx + 1])
            .to(self.device)
            .unsqueeze(1)
        )
        tidx = (
            torch.from_numpy(
                self.eval_driver.step_counts[trunk_lane_idx : trunk_lane_idx + 1]
            )
            .to(self.device)
            .unsqueeze(1)
            .unsqueeze(1)
        )
        with torch.no_grad():
            tmp_hidden = fork_hidden(
                self.eval_driver.hidden_state, trunk_lane_idx, 1, self.device
            )
            emb, _ = self.eval_policy.get_state_embedding(
                obs=torch_obs, rl2s=rl2, time_idxs=tidx, hidden_state=tmp_hidden
            )
        illegal = torch_obs["illegal_actions"].to(self.device)  # (1, 1, A_full)
        base_probs = (
            _primary_probs(self.eval_policy, emb, illegal)[0].cpu().numpy()
        )  # (A_full,)
        base_probs = np.where(base_probs > 0, base_probs, 0.0)
        if base_probs.sum() <= 0:
            base_probs = np.zeros_like(base_probs)
            base_probs[legal_arr0] = 1.0 / len(legal_arr0)
        base_probs = base_probs / base_probs.sum()

        legal_arr = self._select_candidates(legal_arr0, base_probs, cfg)
        if (
            cfg.search_max_root_actions is not None
            and len(legal_arr) > cfg.search_max_root_actions
        ):
            lp = base_probs[legal_arr]
            order = np.argsort(-lp)[: cfg.search_max_root_actions]
            legal_arr = np.sort(legal_arr[order])
        base_argmax = int(legal_arr0[np.argmax(base_probs[legal_arr0])])
        return base_probs, legal_arr, emb, illegal, base_argmax

    def _sample_base_action(self, base_probs: np.ndarray, legal_arr: np.ndarray) -> int:
        p = base_probs[legal_arr]
        s = p.sum()
        if s <= 0:
            return int(legal_arr[0])
        rng = np.random.default_rng(self.config.search_seed + self._decision_counter)
        return int(rng.choice(legal_arr, p=p / s))

    def search_root(
        self, trunk_lane_idx: int, obs: dict, legal: List[int]
    ) -> Tuple[int, SearchRootRecord]:
        """Run oracle root MC search for one trunk lane; return (action, record)."""
        from metamon.env.vectorized.obs_utils import numpy_obs_to_torch, stack_obs_dicts

        cfg = self.config
        t0 = time.perf_counter()
        n_legal_before = int(np.array(legal, dtype=np.int64).size)

        # --- base distribution + candidate selection (cheap; outside the risky rollout) ---
        try:
            base_probs, legal_arr, emb, illegal, base_argmax = self._root_distribution(
                trunk_lane_idx, obs, legal
            )
        except Exception as exc:  # noqa: BLE001
            return self._finish_error(
                legal, n_legal_before, t0, exc, None, fallback=int(legal[0])
            )

        A = int(legal_arr.size)
        K = cfg.search_rollouts_per_action
        N = A * K
        norm_adv, g_scale, scale_mode = self._resolve_value_scale()

        br: Optional[_Branches] = None
        selected: Optional[int] = None
        res = None
        q_mean = np.zeros(A)
        q_std = np.zeros(A)
        counts = np.zeros(A, dtype=np.int64)
        term_frac = np.zeros(A)
        diag: Dict[str, Any] = {}
        error_msg = ""
        pending_raise: Optional[BaseException] = None
        seed_bank: Optional[RootSeedBank] = None
        opp_root_actions: List[int] = []

        try:
            if cfg.search_leaf_value_mode == "root_critic_only":
                # No simulator rollout: Q_root(a) = frozen critic Q(h_root, a).
                q_full, q_per_head = _all_action_q(
                    self.eval_policy, emb, self.action_dim, cfg.critic_horizon_index
                )
                q_full = q_full[0].cpu().numpy()  # (A_full,)
                q_mean = q_full[legal_arr]
                counts = np.ones(A, dtype=np.int64)
                diag["critic_disagreement"] = float(
                    q_per_head[0, legal_arr].std(dim=-1).mean().item()
                )
                diag["intermediate_reward_mean"] = 0.0
                diag["bootstrap_mean"] = float(q_mean.mean())
                diag["n_settled_mean"] = 0.0
            else:
                # --- snapshot + fork (validated deepcopy + no-replay path) ---
                proc = self.env.proc
                proc.drain()
                snap_id = proc.snapshot(trunk_lane_idx)
                proc.drain()

                # --- deterministic branch seed bank (skill §7) ---
                if cfg.search_chance_mode == "resample_crn":
                    seed_bank = RootSeedBank.build(
                        cfg.search_seed,
                        self._battle_id,
                        self.env.eval_side,
                        self._decision_counter,
                        K,
                    )
                    branch_seeds: Optional[List[Optional[List[int]]]] = [
                        seed_bank.env_seed_for_branch(b) for b in range(N)
                    ]
                else:  # inherited_trunk_rng (future-chance oracle DIAGNOSTIC)
                    branch_seeds = None

                branch_lane_ids = self._alloc_fork_lanes(N)
                branch_lanes = []
                for bid in branch_lane_ids:
                    fl = copy.deepcopy(self._trunk_lane(trunk_lane_idx))
                    fl.lane_id = bid
                    proc.register_lane(bid, fl)
                    branch_lanes.append(fl)
                proc.fork_batch(
                    snap_id, branch_lane_ids, replay_log=False, seeds=branch_seeds
                )

                eval_branch = make_branch_state(
                    self.eval_driver, trunk_lane_idx, N, self.device
                )
                opp_branch = make_branch_state(
                    self.opponent._driver, trunk_lane_idx, N, self.device
                )

                root_action = np.repeat(legal_arr, K)  # (N,): branch b = a*K + k
                rollout_index = np.tile(np.arange(K), A)  # (N,): k = b % K

                br = _Branches(
                    lane_ids=branch_lane_ids,
                    lanes=branch_lanes,
                    root_action=root_action,
                    rollout_index=rollout_index,
                    active=np.ones(N, dtype=bool),
                    depth_done=np.zeros(N, dtype=np.int64),
                    cum_reward=np.zeros(N, dtype=np.float64),
                    terminal=np.zeros(N, dtype=bool),
                    eval_hidden=eval_branch.hidden,
                    eval_rl2s=eval_branch.rl2s.copy(),
                    eval_steps=eval_branch.step_counts.copy(),
                    opp_hidden=opp_branch.hidden,
                    opp_rl2s=opp_branch.rl2s.copy(),
                    opp_steps=opp_branch.step_counts.copy(),
                    snap_id=snap_id,
                    trunk_lane=trunk_lane_idx,
                    prev_eval_state=[None] * N,
                    gamma=float(
                        self.eval_policy.gammas[cfg.critic_horizon_index].item()
                    ),
                    seed_bank=seed_bank,
                )

                # --- force root eval action + (coupled) opp root action; settle ---
                self._rollout_root(br, obs)
                opp_root_actions = self._last_opp_root_actions

                for d in range(cfg.search_depth):
                    if not br.active.any():
                        break
                    self._rollout_step(br)

                q_per_branch, leaf_diag = self._leaf_values(br)
                diag.update(leaf_diag)

                for ai, a in enumerate(legal_arr):
                    mask = br.root_action == a
                    vals = q_per_branch[mask]
                    if vals.size:
                        q_mean[ai] = float(vals.mean())
                        q_std[ai] = float(vals.std()) if vals.size > 1 else 0.0
                        counts[ai] = int(vals.size)
                        term_frac[ai] = float(br.terminal[mask].mean())

            # --- policy improvement (skill §12) ---
            full_q = np.full(self.action_dim, np.nan)
            full_q[legal_arr] = q_mean
            full_std = np.zeros(self.action_dim)
            full_std[legal_arr] = q_std
            full_counts = np.zeros(self.action_dim, dtype=np.int64)
            full_counts[legal_arr] = counts
            full_term = np.zeros(self.action_dim)
            full_term[legal_arr] = term_frac
            legal_mask = np.zeros(self.action_dim, bool)
            legal_mask[legal_arr] = True

            rng = np.random.default_rng(cfg.search_seed + self._decision_counter)
            res = improve_policy(
                base_probs=base_probs,
                search_q_mean=full_q,
                search_q_std=full_std,
                rollout_counts=full_counts,
                terminal_frac=full_term,
                legal_mask=legal_mask,
                beta=cfg.search_beta,
                prior_floor=cfg.search_policy_prior_floor,
                normalize_advantages=norm_adv,
                global_advantage_scale=g_scale,
                ablation=cfg.search_ablation,
                root_selection=cfg.search_root_selection,
                rng=rng,
                alpha=cfg.search_magnet_alpha,
            )
            selected = int(res.selected_action)

        except Exception as exc:  # noqa: BLE001
            error_msg = f"{type(exc).__name__}: {exc}"
            if cfg.search_error_policy == "raise":
                # Cleanup runs below; then re-raise so research runs fail loudly.
                pending_raise = exc
            selected = self._sample_base_action(base_probs, legal_arr)

        latency = (time.perf_counter() - t0) * 1000.0

        # --- diagnostics aggregation ---
        active_term = (
            (term_frac * counts).sum() / max(int(counts.sum()), 1)
            if counts.sum()
            else 0.0
        )
        record = SearchRootRecord(
            battle_id=self._battle_id,
            decision=self._decision_counter,
            legal_actions=legal_arr.tolist(),
            base_probs=[float(x) for x in base_probs[legal_arr]],
            rollout_counts=counts.tolist(),
            search_q_mean=[float(x) for x in q_mean],
            search_q_std=[float(x) for x in q_std],
            search_q_sem=[float(x) for x in (q_std / np.sqrt(np.maximum(counts, 1)))],
            terminal_frac=[float(x) for x in term_frac],
            searched_probs=(
                [float(x) for x in res.searched_probs[legal_arr]]
                if res is not None
                else []
            ),
            selected_action=(
                int(selected) if selected is not None else int(legal_arr[0])
            ),
            base_argmax=base_argmax,
            changed_argmax=bool(selected is not None and selected != base_argmax),
            kl_to_base=float(res.kl_to_base) if res is not None else 0.0,
            base_entropy=float(res.base_entropy) if res is not None else 0.0,
            searched_entropy=float(res.searched_entropy) if res is not None else 0.0,
            critic_horizon=cfg.critic_horizon_index,
            search_depth=cfg.search_depth,
            n_rollouts=N,
            latency_ms=float(latency),
            n_legal_before_prune=n_legal_before,
            n_legal_after_prune=A,
            operator=cfg.improvement_operator,
            alpha=float(cfg.search_magnet_alpha),
            beta=float(cfg.search_beta),
            value_scale_mode=scale_mode,
            global_advantage_scale=(float(g_scale) if g_scale is not None else None),
            chance_mode=cfg.search_chance_mode,
            opp_root_coupling=bool(cfg.search_root_opponent_coupling),
            leaf_value_mode=cfg.search_leaf_value_mode,
            candidate_mode=cfg.search_root_candidate_mode,
            reward_multiplier=self.reward_multiplier,
            intermediate_reward_mean=float(diag.get("intermediate_reward_mean", 0.0)),
            bootstrap_mean=float(diag.get("bootstrap_mean", 0.0)),
            critic_disagreement=float(diag.get("critic_disagreement", 0.0)),
            n_settled_mean=float(diag.get("n_settled_mean", 0.0)),
            env_seed_hashes=(
                seed_bank.env_seed_hashes if seed_bank is not None else []
            ),
            opp_root_actions=list(opp_root_actions),
            error=error_msg,
        )
        if self._log_file is not None:
            self._log_file.write(record.to_json() + "\n")
            self._log_file.flush()
        self.root_records.append(record)

        # --- cleanup (always) ---
        if br is not None:
            self._cleanup_branches(br)

        if pending_raise is not None:
            raise pending_raise
        return int(selected) if selected is not None else int(legal_arr[0]), record

    def _finish_error(
        self,
        legal,
        n_legal_before,
        t0,
        exc,
        record,
        fallback: int,
    ) -> Tuple[int, SearchRootRecord]:
        """Early-error path (before any branches exist). Respects error policy."""
        if self.config.search_error_policy == "raise":
            raise exc
        latency = (time.perf_counter() - t0) * 1000.0
        rec = SearchRootRecord(
            battle_id=self._battle_id,
            decision=self._decision_counter,
            legal_actions=list(legal),
            base_probs=[],
            rollout_counts=[],
            search_q_mean=[],
            search_q_std=[],
            search_q_sem=[],
            terminal_frac=[],
            searched_probs=[],
            selected_action=int(fallback),
            base_argmax=int(fallback),
            changed_argmax=False,
            kl_to_base=0.0,
            base_entropy=0.0,
            searched_entropy=0.0,
            critic_horizon=self.config.critic_horizon_index,
            search_depth=self.config.search_depth,
            n_rollouts=0,
            latency_ms=float(latency),
            n_legal_before_prune=n_legal_before,
            n_legal_after_prune=0,
            error=f"{type(exc).__name__}: {exc}",
        )
        if self._log_file is not None:
            self._log_file.write(rec.to_json() + "\n")
            self._log_file.flush()
        self.root_records.append(rec)
        return int(fallback), rec

    # ----- rollout steps --------------------------------------------------

    def _send_branch_choices(
        self, br: _Branches, eval_actions: np.ndarray, opp_actions: np.ndarray
    ) -> None:
        proc = self.env.proc
        entries = []
        for i, lane in enumerate(br.lanes):
            if not br.active[i]:
                continue
            eval_side = self.env.eval_side
            opp_side = self.env.opp_side
            ec = self._action_to_choice(
                eval_actions[i], lane, eval_side, self.eval_action_space
            )
            entries.append((lane.lane_id, eval_side, ec))
            if lane.needs_agent_decision(opp_side):
                oc = self._action_to_choice(
                    opp_actions[i], lane, opp_side, self.opponent_action_space
                )
                entries.append((lane.lane_id, opp_side, oc))
        if entries:
            proc.choose_batch(entries)

    def _action_to_choice(self, action_idx: int, lane, side: str, action_space) -> str:
        from metamon.env.vectorized.action_adapter import (
            DEFAULT_CHOICE,
            action_idx_to_choice,
        )

        legal = lane.legal_action_indices(side, action_space)
        idx = int(action_idx)
        if idx not in legal:
            rng = np.random.default_rng(
                self.config.search_seed + self._decision_counter * 9973 + len(legal)
            )
            idx = int(rng.choice(legal)) if legal else idx
        ch = action_idx_to_choice(idx, lane.battle(side), lane.last_request[side])
        return ch if ch is not None else DEFAULT_CHOICE

    def _branch_obs_batch(
        self, br: _Branches, side: str
    ) -> Tuple[torch.Tensor, List[dict], np.ndarray]:
        from metamon.env.vectorized.obs_utils import numpy_obs_to_torch, stack_obs_dicts

        obs_list = []
        active_idx = np.where(br.active)[0]
        for i in active_idx:
            lane = br.lanes[i]
            state = lane.universal_state(side)
            if side == self.env.eval_side:
                obs = self.env.eval_obs_spaces[br.trunk_lane].state_to_obs(state)
                legal = lane.legal_action_indices(side, self.eval_action_space, state)
                n = self.eval_action_space.gym_space.n
            else:
                obs = self.env.opponent_obs_spaces[br.trunk_lane].state_to_obs(state)
                legal = lane.legal_action_indices(
                    side, self.opponent_action_space, state
                )
                n = self.opponent_action_space.gym_space.n
            mask = np.ones((n,), dtype=bool)
            for lx in legal:
                if 0 <= lx < n:
                    mask[lx] = False
            obs["illegal_actions"] = mask
            obs_list.append(obs)
        if not obs_list:
            return None, [], active_idx
        batch = stack_obs_dicts(obs_list)
        torch_obs = numpy_obs_to_torch(batch, self.device)
        return torch_obs, obs_list, active_idx

    def _rollout_root(self, br: _Branches, trunk_obs: dict) -> None:
        """Force the eval root action; sample opponent root action; settle."""
        prev_active = br.active.copy()
        for lane in br.lanes:
            lane.mark_settled()
        eval_actions = br.root_action.copy()
        opp_actions = self._sample_opponent_root(br)
        self._last_opp_root_actions = self._condensed_opp_root_actions(br, opp_actions)
        for i in np.where(prev_active)[0]:
            br.prev_eval_state[i] = br.lanes[i].universal_state(self.env.eval_side)
        self._send_branch_choices(br, eval_actions, opp_actions)
        self._pump_branches(br)
        self._record_rollout_rewards(br, prev_active)
        br.depth_done[np.where(prev_active)[0]] += 1
        br.eval_steps[np.where(prev_active)[0]] += 1
        if os.environ.get("TTS_DEBUG") == "1":
            print(
                f"  [root] active={int(br.active.sum())} terminal={int(br.terminal.sum())} "
                f"cum={br.cum_reward[:4].tolist()}"
            )

    def _condensed_opp_root_actions(
        self, br: _Branches, opp_actions: np.ndarray
    ) -> List[int]:
        """One opponent root action per rollout index k (for logging)."""
        K = (
            br.seed_bank.K
            if br.seed_bank is not None
            else self.config.search_rollouts_per_action
        )
        out = []
        for k in range(K):
            idxs = np.where(br.rollout_index == k)[0]
            if idxs.size:
                out.append(int(opp_actions[idxs[0]]))
        return out

    def _sample_opponent_root(self, br: _Branches) -> np.ndarray:
        """Sample opponent root actions; couple per rollout index k (skill §7)."""
        N = len(br.lanes)
        opp_actions = np.zeros(N, dtype=np.int64)
        active_idx = np.where(br.active)[0]
        if active_idx.size == 0:
            return opp_actions
        torch_obs, _, active_idx = self._branch_obs_batch(br, self.env.opp_side)
        if torch_obs is None:
            return opp_actions
        rl2 = torch.from_numpy(br.opp_rl2s[active_idx]).to(self.device).unsqueeze(1)
        tidx = (
            torch.from_numpy(br.opp_steps[active_idx])
            .to(self.device)
            .unsqueeze(1)
            .unsqueeze(1)
        )
        emb, br.opp_hidden = _state_embedding(
            self.opponent_policy,
            torch_obs,
            rl2,
            tidx,
            _index_hidden(br.opp_hidden, active_idx, self.device),
        )
        illegal = torch_obs["illegal_actions"].to(self.device)
        probs = _primary_probs(self.opponent_policy, emb, illegal).cpu().numpy()
        # All branches are identical at the root (same snapshot) -> probs[0] is
        # the opponent's root distribution; use it for every rollout index.
        p0 = probs[0]
        legal0 = np.where(p0 > 0)[0]
        cfg = self.config
        K = (
            br.seed_bank.K
            if br.seed_bank is not None
            else cfg.search_rollouts_per_action
        )
        if legal0.size == 0:
            return opp_actions
        p0_legal = p0[legal0] / p0[legal0].sum()
        if cfg.search_root_opponent_coupling:
            for k in range(K):
                if br.seed_bank is not None:
                    r = make_rng(br.seed_bank.opp_root_keys[k])
                else:
                    r = np.random.default_rng(
                        cfg.search_seed + self._decision_counter * 1_000_003 + k
                    )
                opp_actions[br.rollout_index == k] = int(r.choice(legal0, p=p0_legal))
        else:
            # Legacy: resample per (action, k) -- not coupled across candidates.
            r = np.random.default_rng(cfg.search_seed + self._decision_counter)
            for j, i in enumerate(active_idx):
                p = probs[j]
                legal = np.where(p > 0)[0]
                opp_actions[i] = (
                    int(r.choice(legal, p=p[legal] / p[legal].sum()))
                    if legal.size
                    else 0
                )
        return opp_actions

    def _rollout_step(self, br: _Branches) -> None:
        active_idx = np.where(br.active)[0]
        if active_idx.size == 0:
            return
        prev_active = br.active.copy()
        for i in active_idx:
            br.lanes[i].mark_settled()
        eval_actions = self._sample_rollout_actions(
            br, self.env.eval_side, is_eval=True
        )
        opp_actions = self._sample_rollout_actions(br, self.env.opp_side, is_eval=False)
        for i in np.where(prev_active)[0]:
            br.prev_eval_state[i] = br.lanes[i].universal_state(self.env.eval_side)
        self._send_branch_choices(br, eval_actions, opp_actions)
        self._pump_branches(br)
        self._record_rollout_rewards(br, prev_active)
        br.depth_done[np.where(prev_active)[0]] += 1
        br.eval_steps[np.where(prev_active)[0]] += 1

    def _sample_rollout_actions(
        self, br: _Branches, side: str, is_eval: bool
    ) -> np.ndarray:
        """Sample actions for active branches from the frozen rollout policy.

        Uses deterministic keyed policy-RNG streams (skill §7) when a seed bank
        exists, so the same uniform variate is consumed at the same logical
        (k, step) where practical (paired sampling across candidate actions).
        """
        N = len(br.lanes)
        actions = np.zeros(N, dtype=np.int64)
        active_idx = np.where(br.active)[0]
        if active_idx.size == 0:
            return actions
        torch_obs, _, active_idx = self._branch_obs_batch(br, side)
        if torch_obs is None:
            return actions
        policy = self.eval_policy if is_eval else self.opponent_policy
        if is_eval:
            rl2 = (
                torch.from_numpy(br.eval_rl2s[active_idx]).to(self.device).unsqueeze(1)
            )
            tidx = (
                torch.from_numpy(br.eval_steps[active_idx])
                .to(self.device)
                .unsqueeze(1)
                .unsqueeze(1)
            )
            hidden = _index_hidden(br.eval_hidden, active_idx, self.device)
            emb, new_hidden = _state_embedding(policy, torch_obs, rl2, tidx, hidden)
            _scatter_hidden(br.eval_hidden, active_idx, new_hidden)
        else:
            rl2 = torch.from_numpy(br.opp_rl2s[active_idx]).to(self.device).unsqueeze(1)
            tidx = (
                torch.from_numpy(br.opp_steps[active_idx])
                .to(self.device)
                .unsqueeze(1)
                .unsqueeze(1)
            )
            hidden = _index_hidden(br.opp_hidden, active_idx, self.device)
            emb, new_hidden = _state_embedding(policy, torch_obs, rl2, tidx, hidden)
            _scatter_hidden(br.opp_hidden, active_idx, new_hidden)
        illegal = torch_obs["illegal_actions"].to(self.device)
        probs = _primary_probs(policy, emb, illegal).cpu().numpy()
        cfg = self.config
        for j, i in enumerate(active_idx):
            p = probs[j]
            legal = np.where(p > 0)[0]
            if legal.size == 0:
                actions[i] = 0
                continue
            if br.seed_bank is not None:
                k = int(br.rollout_index[i])
                step = int(br.depth_done[i])
                r = make_rng(
                    policy_rng_key(
                        cfg.search_seed,
                        self._battle_id,
                        side,
                        self._decision_counter,
                        k,
                        step,
                    )
                )
            else:
                r = np.random.default_rng(cfg.search_seed + i + br.depth_done[i])
            actions[i] = int(r.choice(legal, p=p[legal] / p[legal].sum()))
        return actions

    def _pump_branches(self, br: _Branches) -> None:
        """Pump until every active branch parks at its next eval-side decision or ends.

        Settles the root/rollout turn by auto-answering the same follow-ups the
        live env's ``_pump_settle`` + ``_advance_lanes`` resolve:

        * **opponent-only follow-ups** -- e.g. the opponent fainted and must
          switch while the eval side waits. This includes the both-sides-advanced
          case where the eval side is ``wait``: Showdown's ``makeRequest`` advances
          both serials together, but a ``wait`` side is never "answered", so the
          old ``not other_advanced`` guard left ``answered[eval]`` stuck below the
          eval serial and the opponent's follow-up was never answered -> the host
          went idle and ``pump_until`` timed out (skill §35).
        * **single-side ``|error|`` re-prompts** -- e.g. a revealed trap that
          invalidates a switch, re-answered with a uniform-legal rollout action
          (mirrors ``_pump_settle``'s reprompt + error-retry cases).

        A fresh eval-side move/force-switch decision is *never* auto-answered:
        that is where the rollout parks (the leaf for depth 0, the next rollout
        step for depth>0). The branch version also re-answers an eval
        ``|error|``-with-no-new-request stall because, unlike the live env, there
        is no outer ``step`` loop to re-apply the committed eval action.
        """
        proc = self.env.proc
        eval_side = self.env.eval_side
        opp_side = self.env.opp_side
        answerable = ("move", "forceswitch", "teampreview")
        answered = {
            i: {s: br.lanes[i].request_serial[s] for s in (eval_side, opp_side)}
            for i in range(len(br.lanes))
        }
        # Tracks the request serial we already re-answered for an |error| that
        # arrived *without* a fresh request (serial unchanged), so we repair it
        # at most once per serial (mirrors _pump_settle's err_handled).
        err_handled = {
            i: {s: -1 for s in (eval_side, opp_side)} for i in range(len(br.lanes))
        }

        def ready() -> bool:
            done = True
            entries = []
            for i in np.where(br.active)[0]:
                lane = br.lanes[i]
                if lane.ended:
                    continue
                eval_needs = lane.needs_agent_decision(eval_side)
                opp_needs = lane.needs_agent_decision(opp_side)
                # Park at the eval side's next decision (the leaf / next rollout
                # step), or once the lane is fully settled (neither side owes a
                # decision). Requires decision_ready so the cycle is synchronized.
                if lane.decision_ready() and (eval_needs or not opp_needs):
                    continue
                # Otherwise auto-resolve follow-ups so the turn can advance.
                for s in (opp_side, eval_side):
                    adv = lane.request_serial[s] > answered[i][s]
                    choose = False
                    if (
                        adv
                        and lane.reprompt_pending[s]
                        and lane._side_ready(s)
                        and lane.request_kind(s) in answerable
                    ):
                        # |error| re-prompt whose fresh request has arrived.
                        # Re-answer the opponent; a fresh eval re-prompt is the
                        # eval side's next decision -> park (do not re-answer).
                        choose = s == opp_side
                    elif (
                        not adv
                        and lane.error[s]
                        and err_handled[i][s] != lane.request_serial[s]
                    ):
                        # |error| with no new request yet: the host is blocked
                        # waiting for a valid choice -> re-answer to unblock.
                        choose = True
                        err_handled[i][s] = lane.request_serial[s]
                    elif (
                        adv
                        and not lane.reprompt_pending[s]
                        and lane._side_ready(s)
                        and lane.request_kind(s) in answerable
                        and s == opp_side
                        and not eval_needs
                    ):
                        # Opponent-only follow-up while the eval side waits
                        # (e.g. opp fainted -> forceswitch). Answer it regardless
                        # of whether the eval side's ``wait`` serial advanced --
                        # the old ``not other_advanced`` guard stalled here.
                        choose = True
                    if choose:
                        if s == opp_side:
                            a = self._sample_single(
                                br, i, opp_side, self.opponent_policy, is_eval=False
                            )
                            ch = self._action_to_choice(
                                a, lane, opp_side, self.opponent_action_space
                            )
                        else:
                            a = self._sample_single(
                                br, i, eval_side, self.eval_policy, is_eval=True
                            )
                            ch = self._action_to_choice(
                                a, lane, eval_side, self.eval_action_space
                            )
                        entries.append((lane.lane_id, s, ch))
                        answered[i][s] = lane.request_serial[s]
                        lane.reprompt_pending[s] = False
                # Re-check park: an eval re-prompt (case above) may have made the
                # lane decision_ready with eval owing a decision.
                if lane.decision_ready() and (eval_needs or not opp_needs):
                    continue
                done = False
            if entries:
                proc.choose_batch(entries)
            return done

        proc.pump_until(ready, timeout=60.0, idle_timeout=20.0)
        for i in np.where(br.active)[0]:
            if br.lanes[i].ended:
                br.terminal[i] = True
                br.active[i] = False

    def _sample_single(
        self, br: _Branches, i: int, side: str, policy, is_eval: bool
    ) -> int:
        """Sample one action for one branch (single-side follow-ups).

        Uniform over the legal mask (cheap); deterministically keyed by
        (k, step) when a seed bank exists.
        """
        lane = br.lanes[i]
        state = lane.universal_state(side)
        if is_eval:
            legal = lane.legal_action_indices(side, self.eval_action_space, state)
        else:
            legal = lane.legal_action_indices(side, self.opponent_action_space, state)
        cfg = self.config
        if br.seed_bank is not None:
            k = int(br.rollout_index[i])
            step = int(br.depth_done[i])
            r = make_rng(
                policy_rng_key(
                    cfg.search_seed,
                    self._battle_id,
                    side,
                    self._decision_counter,
                    k,
                    step,
                )
            )
        else:
            r = np.random.default_rng(cfg.search_seed + i + br.depth_done[i])
        return int(r.choice(legal)) if legal else 0

    def _record_rollout_rewards(self, br: _Branches, prev_active: np.ndarray) -> None:
        """Accumulate discounted, multiplier-scaled rewards for branches that
        participated in this settlement (skill §5).

        Records the reward for EVERY branch that was active at the start of this
        settlement -- including ones that just reached a terminal state -- so
        the terminal settlement's +200 victory term is captured exactly once
        (BUG A fix). Rewards are scaled by ``reward_multiplier`` to match the
        critic's training units (BUG B fix). The discount exponent is the number
        of settlements completed BEFORE this one (``depth_done``), and the caller
        increments ``depth_done`` afterward, so the leaf bootstrap is discounted
        by gamma**(D+1) (BUG C fix).
        """
        if not self.config.search_include_intermediate_rewards:
            # Even with intermediate rewards disabled, the terminal settlement
            # reward (victory) must be recorded so terminal branches are valued
            # correctly. Record only for branches that became terminal this round.
            for i in np.where(prev_active & br.terminal)[0]:
                prev = br.prev_eval_state[i]
                if prev is None:
                    continue
                new = br.lanes[i].universal_state(self.env.eval_side)
                r = float(self.eval_reward_function(prev, new)) * self.reward_multiplier
                br.cum_reward[i] += (br.gamma ** int(br.depth_done[i])) * r
                br.prev_eval_state[i] = new
                br.eval_rl2s[i] = 0.0
                br.eval_rl2s[i, 0] = r / self.reward_multiplier
            return
        for i in np.where(prev_active)[0]:
            prev = br.prev_eval_state[i]
            if prev is None:
                continue
            new = br.lanes[i].universal_state(self.env.eval_side)
            r_env = float(self.eval_reward_function(prev, new))
            r = r_env * self.reward_multiplier
            br.cum_reward[i] += (br.gamma ** int(br.depth_done[i])) * r
            br.prev_eval_state[i] = new
            # RL²: [reward, last_action_one_hot] -- preserve the existing
            # convention (reward in slot 0; action part left as-is/zeroed).
            br.eval_rl2s[i, 0] = r_env

    def _leaf_values(self, br: _Branches) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Leaf values: discounted bootstrap for nonterminal branches; cum_reward
        (no bootstrap) for terminal branches (skill §5/§10)."""
        cfg = self.config
        N = len(br.lanes)
        vals = np.zeros(N, dtype=np.float64)
        diag: Dict[str, Any] = {}
        term_idx = np.where(br.terminal)[0]
        for i in term_idx:
            vals[i] = br.cum_reward[i]  # terminal: victory counted once, bootstrap 0
        active_idx = np.where(br.active)[0]
        if active_idx.size == 0:
            diag.update(
                intermediate_reward_mean=(
                    float(br.cum_reward[term_idx].mean()) if term_idx.size else 0.0
                ),
                bootstrap_mean=0.0,
                n_settled_mean=float(br.depth_done.mean()) if N else 0.0,
                critic_disagreement=0.0,
            )
            return vals, diag

        torch_obs, _, active_idx = self._branch_obs_batch(br, self.env.eval_side)
        rl2 = torch.from_numpy(br.eval_rl2s[active_idx]).to(self.device).unsqueeze(1)
        tidx = (
            torch.from_numpy(br.eval_steps[active_idx])
            .to(self.device)
            .unsqueeze(1)
            .unsqueeze(1)
        )
        hidden = _index_hidden(br.eval_hidden, active_idx, self.device)
        emb, _ = _state_embedding(self.eval_policy, torch_obs, rl2, tidx, hidden)
        illegal = torch_obs["illegal_actions"].to(self.device)  # (n_active, 1, A)
        horizon = cfg.critic_horizon_index

        if cfg.search_leaf_value_mode == "policy_expectation":
            v_pi, q_all, probs, q_per_head = _exact_leaf_v_pi(
                self.eval_policy, emb, illegal, self.action_dim, horizon
            )
            v = v_pi.cpu().numpy()
            # critic disagreement: mean over active branches of per-action std over heads
            legal_2d = ~illegal[:, 0, :].cpu().numpy()
            disagrees = []
            for j in range(v.size):
                if legal_2d[j].any():
                    disagrees.append(
                        float(q_per_head[j][legal_2d[j]].std(dim=-1).mean().item())
                    )
            diag["critic_disagreement"] = (
                float(np.mean(disagrees)) if disagrees else 0.0
            )
            for j, i in enumerate(active_idx):
                vals[i] = br.cum_reward[i] + (
                    br.gamma ** int(br.depth_done[i])
                ) * float(v[j])
        else:  # sampled_action (legacy)
            probs_np = _primary_probs(self.eval_policy, emb, illegal).cpu().numpy()
            n_active = active_idx.size
            r = np.random.default_rng(cfg.search_seed + self._decision_counter)
            sampled = np.zeros(n_active, dtype=np.int64)
            for j in range(n_active):
                p = probs_np[j]
                legal = np.where(p > 0)[0]
                sampled[j] = (
                    int(r.choice(legal, p=p[legal] / p[legal].sum()))
                    if legal.size
                    else 0
                )
            q = (
                _critic_leaf_values(
                    self.eval_policy,
                    emb,
                    torch.from_numpy(sampled).long().to(self.device),
                    self.action_dim,
                    horizon,
                )
                .cpu()
                .numpy()
            )
            diag["critic_disagreement"] = 0.0
            for j, i in enumerate(active_idx):
                vals[i] = br.cum_reward[i] + (
                    br.gamma ** int(br.depth_done[i])
                ) * float(q[j])

        diag["intermediate_reward_mean"] = float(br.cum_reward[active_idx].mean())
        diag["bootstrap_mean"] = float(
            np.mean([vals[i] - br.cum_reward[i] for i in active_idx])
        )
        diag["n_settled_mean"] = float(br.depth_done[active_idx].mean())
        return vals, diag

    def _cleanup_branches(self, br: _Branches) -> None:
        proc = self.env.proc
        for bid in br.lane_ids:
            try:
                proc.reset(bid)
            except Exception:
                pass
            if bid in self._active_fork_lanes:
                self._active_fork_lanes.remove(bid)
        try:
            proc.release_snapshot(br.snap_id)
        except Exception:
            pass
        br.eval_hidden = None
        br.opp_hidden = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Hidden-state indexing helpers
# ---------------------------------------------------------------------------


def _index_hidden(hidden, idx: np.ndarray, device: torch.device):
    """Return a hidden state whose batch is the lanes at ``idx`` (view, not copy)."""
    from amago.nets.transformer import Cache, TformerHiddenState

    idx_t = torch.as_tensor(idx, dtype=torch.long, device=device)
    k = hidden.key_cache.data[:, idx_t]
    v = hidden.val_cache.data[:, idx_t]
    kc = Cache.__new__(Cache)
    kc.data = k
    kc.max_seq_len = hidden.key_cache.max_seq_len
    kc.device = device
    vc = Cache.__new__(Cache)
    vc.data = v
    vc.max_seq_len = hidden.val_cache.max_seq_len
    vc.device = device
    sl = hidden.seq_lens[idx_t]
    return TformerHiddenState(key_cache=kc, val_cache=vc, seq_lens=sl)


def _scatter_hidden(hidden, idx: np.ndarray, new_hidden) -> None:
    """Write the per-branch new hidden state back into the batched hidden."""
    idx_t = torch.as_tensor(idx, dtype=torch.long, device=hidden.key_cache.data.device)
    hidden.key_cache.data[:, idx_t] = new_hidden.key_cache.data
    hidden.val_cache.data[:, idx_t] = new_hidden.val_cache.data
    hidden.seq_lens[idx_t] = new_hidden.seq_lens
