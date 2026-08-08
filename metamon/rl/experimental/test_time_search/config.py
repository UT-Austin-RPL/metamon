"""Configuration for eval-only test-time search over a frozen Metamon policy.

Search is **opt-in**: ``search_mode="none"`` (the default) reproduces the frozen
baseline exactly. Only ``oracle-root-mc`` activates search.

The defaults are the **research-safe** values from skill §15. The legacy
prototype configuration (per-root z-scoring, 5%-of-max pruning, inherited trunk
RNG, sampled-action leaf bootstrap, every-5th-decision, base-fallback on error)
is still reachable -- pass the individual legacy flags, or use
``--legacy_prototype`` on the CLI which sets them all at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SearchConfig:
    """All knobs for the oracle root Monte Carlo search wrapper.

    Depth is measured in **settled evaluated-player decisions** (our action +
    opponent simultaneous action + RNG + faint cascade + forced switches /
    re-prompts -> next eval-side request or terminal), matching the env's
    ``step`` semantics.
    """

    # ------------------------------------------------------------------
    # top-level
    # ------------------------------------------------------------------
    search_mode: str = "none"  # "none" | "oracle-root-mc"

    # ------------------------------------------------------------------
    # rollout budget / depth / cadence
    # ------------------------------------------------------------------
    search_rollouts_per_action: int = 16  # K per retained root action
    search_depth: int = 0  # settled eval decisions to roll out (0 = bootstrap now)
    search_every_n_decisions: int = 1  # search every N-th settled eval decision

    # ------------------------------------------------------------------
    # root candidate selection (skill §9)
    # ------------------------------------------------------------------
    # "all_legal" (research default) | "relative_threshold" (legacy prune) |
    # "cumulative_mass" (retain until cumulative actor mass >= threshold).
    search_root_candidate_mode: str = "all_legal"
    # Used by "relative_threshold": keep actions whose base prob is at least
    # this fraction of the max legal prob. 0.0 = keep all legal.
    search_root_prob_threshold: float = 0.0
    # Used by "cumulative_mass": retain actions until cumulative actor mass
    # >= this, with a minimum number retained.
    search_cumulative_mass_threshold: float = 0.99
    search_min_root_actions: int = 2
    # Hard cap on retained root actions (None = no cap; applied AFTER the
    # candidate mode as a safety net). The base argmax is always retained.
    search_max_root_actions: Optional[int] = None

    # ------------------------------------------------------------------
    # branch chance / RNG (skill §7)
    # ------------------------------------------------------------------
    # "resample_crn" (research default): each rollout index k gets an
    # independent branch-only Showdown PRNG seed, shared across candidate
    # actions (common random numbers); the trunk is never reseeded.
    # "inherited_trunk_rng": branches inherit the trunk's exact future PRNG
    # stream (future-chance oracle DIAGNOSTIC only; never the primary result).
    search_chance_mode: str = "resample_crn"
    # Couple the opponent's root action across candidate actions for a fixed
    # rollout index k (one opp action per k, reused across candidates).
    search_root_opponent_coupling: bool = True

    # ------------------------------------------------------------------
    # leaf-value estimator (skill §10)
    # ------------------------------------------------------------------
    # "policy_expectation" (research default): V_pi(h) = sum_a pi(a|h) Q(h,a)
    #   over ALL legal actions, exact, fixed-shape batched critic forward.
    # "sampled_action": sample one action per branch and score just it
    #   (legacy; avoids all-action compile storms but adds MC variance).
    # "root_critic_only": no simulator rollout; Q_root(a) = frozen critic
    #   Q(h_root, a) for all legal a, then the improvement operator. Ablation
    #   that isolates whether gains come from the critic vs. real transitions.
    search_leaf_value_mode: str = "policy_expectation"

    # ------------------------------------------------------------------
    # value scale / normalization (skill §11)
    # ------------------------------------------------------------------
    # Per-root z-scoring of advantages (legacy; off in primary research mode
    # because it maps tiny noisy Q gaps to ~[-1,+1] and makes beta root-dep).
    search_value_normalization: bool = False
    # "raw" | "environment_units" | "global_standardized" | "legacy_zscore".
    # Describes the advantage units beta is expressed in (logged on every root).
    #   raw               -> raw critic-return advantages (scale=None)
    #   environment_units -> divide by the checkpoint reward_multiplier (10.0)
    #   global_standardized -> divide by search_global_advantage_scale (frozen)
    #   legacy_zscore     -> per-root z-score (sets search_value_normalization)
    search_value_scale_mode: str = "raw"
    # A single frozen scale (in raw Q units) for "global_standardized". Must not
    # vary by root/matchup/run after selection.
    search_global_advantage_scale: Optional[float] = None

    # ------------------------------------------------------------------
    # policy-improvement operator (skill §12)
    # ------------------------------------------------------------------
    # "single_anchor_kl" (research default; "kl_anchor" is a legacy alias)
    #   pi_search(a) ~ pi_base(a) * exp(A(a)/beta)
    # "confidence_gated_kl" (Phase C, skill §37): single_anchor_kl with a
    #   per-root z-score gate. Suppresses the update (returns pi_base) when
    #   the best action's advantage over its nearest competitor is not
    #   statistically separated (paired z-score < search_z_gate). Optionally
    #   scales beta with confidence (search_adaptive_beta).
    # "magnetic_kl"  : single-anchor + uniform magnetic term
    #   pi_search(a) ~ rho(a)^(alpha/(alpha+beta)) pi_base(a)^(beta/(alpha+beta))
    #                  * exp(Q(a)/(alpha+beta)),  rho uniform over legal
    # "softmax_q" | "argmax_q" | "base_only" : ablations / plumbing controls.
    search_ablation: str = "single_anchor_kl"
    search_beta: float = 1.0  # policy-anchor strength
    search_magnet_alpha: float = 0.0  # magnetic-anchor strength (magnetic_kl)
    search_policy_prior_floor: float = 0.0  # floor on pi_base before anchoring
    search_root_selection: str = "sample"  # "sample" | "argmax"

    # --- Phase C: confidence-gated update (skill §37) ---
    # z-score threshold for the confidence gate. When > 0 (and the operator is
    # confidence_gated_kl, or any operator with z_gate > 0), the update is
    # suppressed if the best action's min paired z-score < this threshold.
    # 0.0 = no gating (confidence_gated_kl == single_anchor_kl).
    search_z_gate: float = 0.0
    # When True, scale beta as beta_eff = beta * z_gate / max(min_z, z_gate)
    # so the update strengthens with confidence (skill §37 Phase C).
    search_adaptive_beta: bool = False

    # --- Phase B: adaptive-K evaluation (skill §37) ---
    # When True, the search driver uses a multi-round fork: an initial pilot of
    # ``search_k_pilot`` rollouts per action, then batches of ``search_k_batch``
    # additional rollouts, stopping early when the best action's paired z-score
    # exceeds ``search_k_z_stop`` (or ``search_k_max`` is reached). This makes
    # high-K-quality search affordable for live paired eval (Phase 2 screen):
    # easy roots stop at K_pilot, hard roots get up to K_max. The per-``k`` branch
    # seed is K-independent (``rng.RootSeedBank`` keys on ``(root, k)``, not on
    # ``K``), so each round's rollouts use the same chance streams as a
    # standalone K=K_max run at those k indices. Recommended for D=0 only (D>0
    # deeper-rollout policy RNG keys use local k, which breaks CRN across rounds).
    search_adaptive_k: bool = False
    search_k_pilot: int = 4  # initial rollouts per action
    search_k_max: int = 64  # maximum rollouts per action
    search_k_batch: int = 4  # additional rollouts per action per round
    search_k_z_stop: float = 2.0  # z-score early-stopping threshold

    # ------------------------------------------------------------------
    # rollout policy
    # ------------------------------------------------------------------
    search_rollout_temperature: float = 1.0  # rollout actor temperature
    search_critic_horizon: Optional[int] = None  # gamma index; None=primary (-1)
    search_include_intermediate_rewards: bool = True  # MC return vs leaf-only
    # kimi-search M3: path to a trained WinHead checkpoint. When
    # search_leaf_value_mode="win_head", leaf values come from the win
    # head's policy-expectation P(win) instead of the shaped critic.
    search_win_head_path: Optional[str] = None

    # ------------------------------------------------------------------
    # execution / logging / errors (skill §15, §19)
    # ------------------------------------------------------------------
    search_lane_batch_size: int = 64  # max simultaneous branches
    search_seed: int = 0  # base RNG seed for the deterministic seed bank
    search_log_roots: Optional[str] = None  # JSONL per-search record path
    # "raise" (research default): a search error propagates and fails the run.
    # "base_fallback": log and fall back to the base action (legacy; dangerous
    # in research because it can make a broken config look stable).
    search_error_policy: str = "raise"
    # Log per-branch rollout returns / seeds / opponent actions to the JSONL
    # record (verbose; needed for estimator auditing).
    search_log_branch_details: bool = False

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.search_mode not in ("none", "oracle-root-mc"):
            raise ValueError(f"unknown search_mode: {self.search_mode}")
        if self.search_root_selection not in ("sample", "argmax"):
            raise ValueError(
                f"unknown search_root_selection: {self.search_root_selection}"
            )
        # "kl_anchor" is accepted as a legacy alias for "single_anchor_kl".
        ablation = (
            "single_anchor_kl"
            if self.search_ablation == "kl_anchor"
            else self.search_ablation
        )
        if ablation not in (
            "single_anchor_kl",
            "confidence_gated_kl",
            "magnetic_kl",
            "argmax_q",
            "softmax_q",
            "base_only",
        ):
            raise ValueError(f"unknown search_ablation: {self.search_ablation}")
        if self.search_chance_mode not in ("resample_crn", "inherited_trunk_rng"):
            raise ValueError(f"unknown search_chance_mode: {self.search_chance_mode}")
        if self.search_leaf_value_mode not in (
            "policy_expectation",
            "sampled_action",
            "root_critic_only",
            "win_head",
        ):
            raise ValueError(
                f"unknown search_leaf_value_mode: {self.search_leaf_value_mode}"
            )
        if self.search_leaf_value_mode == "win_head" and not self.search_win_head_path:
            raise ValueError(
                "search_leaf_value_mode='win_head' requires search_win_head_path"
            )
        if self.search_root_candidate_mode not in (
            "all_legal",
            "relative_threshold",
            "cumulative_mass",
        ):
            raise ValueError(
                f"unknown search_root_candidate_mode: {self.search_root_candidate_mode}"
            )
        if self.search_value_scale_mode not in (
            "raw",
            "environment_units",
            "global_standardized",
            "legacy_zscore",
        ):
            raise ValueError(
                f"unknown search_value_scale_mode: {self.search_value_scale_mode}"
            )
        if self.search_error_policy not in ("raise", "base_fallback"):
            raise ValueError(f"unknown search_error_policy: {self.search_error_policy}")
        if self.search_beta <= 0:
            raise ValueError("search_beta must be positive")
        if self.search_magnet_alpha < 0:
            raise ValueError("search_magnet_alpha must be non-negative")
        if self.search_rollouts_per_action < 1:
            raise ValueError("search_rollouts_per_action must be >= 1")
        if self.search_depth < 0:
            raise ValueError("search_depth must be >= 0")
        if self.search_lane_batch_size < 1:
            raise ValueError("search_lane_batch_size must be >= 1")
        if self.search_every_n_decisions < 1:
            raise ValueError("search_every_n_decisions must be >= 1")
        if not 0.0 <= self.search_root_prob_threshold <= 1.0:
            raise ValueError("search_root_prob_threshold must be in [0, 1]")
        if not 0.0 < self.search_cumulative_mass_threshold <= 1.0:
            raise ValueError("search_cumulative_mass_threshold must be in (0, 1]")
        if self.search_min_root_actions < 1:
            raise ValueError("search_min_root_actions must be >= 1")
        if self.search_value_scale_mode == "global_standardized" and not (
            self.search_global_advantage_scale
            and self.search_global_advantage_scale > 0
        ):
            raise ValueError(
                "search_value_scale_mode='global_standardized' requires a "
                "positive search_global_advantage_scale"
            )
        if self.search_z_gate < 0.0:
            raise ValueError("search_z_gate must be non-negative")
        if self.search_adaptive_k:
            if self.search_k_pilot < 1:
                raise ValueError("search_k_pilot must be >= 1")
            if self.search_k_max < self.search_k_pilot:
                raise ValueError("search_k_max must be >= search_k_pilot")
            if self.search_k_batch < 1:
                raise ValueError("search_k_batch must be >= 1")
            if self.search_k_z_stop < 0.0:
                raise ValueError("search_k_z_stop must be non-negative")

    # ------------------------------------------------------------------
    # derived helpers
    # ------------------------------------------------------------------
    @property
    def critic_horizon_index(self) -> int:
        if self.search_critic_horizon is not None:
            return int(self.search_critic_horizon)
        return -1  # primary gamma (0.999 for the frozen checkpoint)

    @property
    def search_enabled(self) -> bool:
        return self.search_mode == "oracle-root-mc"

    @property
    def improvement_operator(self) -> str:
        """Canonical operator name (resolves the ``kl_anchor`` legacy alias)."""
        return (
            "single_anchor_kl"
            if self.search_ablation == "kl_anchor"
            else self.search_ablation
        )

    def apply_legacy_prototype_defaults(self) -> "SearchConfig":
        """Restore the pre-correction prototype defaults in place (skill §28).

        Returns ``self`` for chaining. Used by the CLI ``--legacy_prototype``
        flag so the historical 100-game result stays reproducible under a
        labeled mode.
        """
        self.search_value_normalization = True
        self.search_value_scale_mode = "legacy_zscore"
        self.search_root_candidate_mode = "relative_threshold"
        self.search_root_prob_threshold = 0.05
        self.search_ablation = "kl_anchor"
        self.search_chance_mode = "inherited_trunk_rng"
        self.search_root_opponent_coupling = False
        self.search_leaf_value_mode = "sampled_action"
        self.search_every_n_decisions = 5
        self.search_error_policy = "base_fallback"
        return self
