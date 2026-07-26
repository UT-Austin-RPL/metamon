"""Policy-improvement operators for test-time search Q-estimates.

Given the frozen actor distribution ``pi_base`` over legal root actions and a
Monte Carlo search-Q estimate ``Q_search`` per action, construct an improved
root policy.

The primary operator is the **single-anchor KL** update (skill §12)::

    A(a) = Q(a) - sum_a' pi_base(a') * Q(a')
    pi_search(a) ~ pi_base(a) * exp(A(a) / beta)

This is the zero-magnet special case of the **magnetic KL** operator
(Ataraxos-style)::

    maximize_pi  E_pi[Q] - alpha * KL(pi || rho) - beta * KL(pi || pi_base)

whose closed-form legal-action solution is::

    pi_search(a) ~ rho(a) ** (alpha / (alpha + beta))
                 * pi_base(a) ** (beta / (alpha + beta))
                 * exp(Q(a) / (alpha + beta))

where ``rho`` is a magnetic reference distribution (uniform over legal actions
by default), ``beta`` is the policy-anchor strength, and ``alpha`` is the
magnet strength. ``alpha = 0`` recovers ``single_anchor_kl`` exactly.

Value scaling (skill §11): the primary research mode uses **no per-root
z-scoring** and a single frozen global advantage scale so ``beta`` has a stable
interpretation across roots. Per-root z-scoring is retained only as a legacy
ablation (``normalize_advantages=True``) because, with few retained actions, it
maps almost any noisy Q gap to approximately [-1, +1] and makes ``beta``
root-dependent.

Operators / ablations (selectable via ``ablation``):

  * ``single_anchor_kl`` (default; ``kl_anchor`` is a legacy alias)
  * ``magnetic_kl``      : single-anchor + uniform magnetic term
  * ``softmax_q``        : sample ~ softmax(Q) with no actor prior
  * ``argmax_q``         : greedy Q, no anchor
  * ``base_only``        : ignore search, return pi_base (plumbing control)

Properties (tested in ``test_improvement.py``):

  * illegal actions stay zero probability (the base mask is respected)
  * equal Q -> pi_search == pi_base (anchor invariance)
  * beta -> large: pi_search -> pi_base
  * beta -> 0+: probability concentrates on the highest-Q action **within the
    support of pi_base** (not an unanchored softmax unless prior_floor > 0 or
    a uniform magnet with alpha > 0 is used)
  * adding a constant to all Q leaves pi_search unchanged (constant-shift invariance)
  * alpha = 0: ``magnetic_kl`` == ``single_anchor_kl``
  * all outputs sum to 1 over legal actions

All math is in log-space for numerical stability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Canonical operator names. ``kl_anchor`` is accepted as a legacy alias for
# ``single_anchor_kl`` so existing configs/CLI commands keep working.
_LEGACY_ALIAS = {"kl_anchor": "single_anchor_kl"}
_OPERATORS = {"single_anchor_kl", "magnetic_kl", "softmax_q", "argmax_q", "base_only"}


@dataclass
class SearchImprovementResult:
    """Output of the improvement step for one searched root."""

    searched_probs: np.ndarray  # improved policy over action_dim (legal only)
    advantages: np.ndarray  # A_search over action_dim (illegal = 0)
    base_probs: np.ndarray  # pi_base over action_dim
    search_q: np.ndarray  # mean Q per action (illegal = nan)
    search_q_std: np.ndarray
    search_q_sem: np.ndarray
    search_q_min: np.ndarray
    search_q_max: np.ndarray
    rollout_counts: np.ndarray  # per action
    terminal_frac: np.ndarray  # per action
    selected_action: int
    base_argmax: int
    kl_to_base: float  # KL(pi_search || pi_base)
    base_entropy: float
    searched_entropy: float
    changed_argmax: bool
    n_legal: int
    legal_actions: np.ndarray
    # --- new research-mode fields (skill §11/§12) ---
    operator: str = "single_anchor_kl"
    alpha: float = 0.0
    beta: float = 1.0
    value_scale_mode: str = "raw"  # "raw" | "global_standardized" | "legacy_zscore"
    global_advantage_scale: Optional[float] = None
    magnet_rho: Optional[np.ndarray] = None  # rho over action_dim (illegal=0)


def _safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.clip(x, 1e-12, None))


def kl(pi: np.ndarray, q: np.ndarray) -> float:
    """KL(pi || q) for discrete distributions (nats)."""
    mask = pi > 0
    return float(np.sum(pi[mask] * (_safe_log(pi[mask]) - _safe_log(q[mask]))))


def entropy(p: np.ndarray) -> float:
    mask = p > 0
    return float(-np.sum(p[mask] * _safe_log(p[mask])))


def _normalize_legal(p: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    out = np.where(legal_mask, p, 0.0)
    s = out.sum()
    if not np.isfinite(s) or s <= 0:
        out = np.where(legal_mask, 1.0, 0.0)
        s = out.sum()
    return out / s


def improve_policy(
    base_probs: np.ndarray,
    search_q_mean: np.ndarray,
    search_q_std: np.ndarray,
    rollout_counts: np.ndarray,
    terminal_frac: np.ndarray,
    legal_mask: np.ndarray,
    beta: float,
    prior_floor: float = 0.0,
    normalize_advantages: bool = True,
    ablation: str = "single_anchor_kl",
    root_selection: str = "sample",
    rng: Optional[np.random.Generator] = None,
    alpha: float = 0.0,
    global_advantage_scale: Optional[float] = None,
) -> SearchImprovementResult:
    """Construct the improved root policy and select the live action.

    Args:
        base_probs: frozen actor probabilities over the full action_dim
            (illegal actions are 0).
        search_q_mean: mean search Q per action (illegal/norollout = nan).
        search_q_std: std of search Q per action.
        rollout_counts: number of rollouts that contributed per action.
        terminal_frac: fraction of rollouts that reached a terminal state.
        legal_mask: bool array over action_dim (True = legal).
        beta: policy-anchor strength. For ``magnetic_kl`` this is the
            ``KL(pi || pi_base)`` weight; ``alpha`` is the magnet weight.
        prior_floor: floor applied to pi_base before anchoring.
        normalize_advantages: legacy per-root z-scoring of advantages before
            the exponential update (skill §11: off in primary research mode).
        ablation: operator name. ``kl_anchor`` is a legacy alias for
            ``single_anchor_kl``.
        root_selection: "sample" | "argmax".
        rng: sampler for "sample" selection and tie-breaking.
        alpha: magnetic-anchor strength (only used by ``magnetic_kl``);
            ``alpha=0`` makes ``magnetic_kl`` identical to ``single_anchor_kl``.
        global_advantage_scale: a single frozen scale (in raw Q units) by which
            advantages are divided when ``normalize_advantages`` is False. Makes
            ``beta`` interpretable across roots. None = raw advantages.

    Returns:
        :class:`SearchImprovementResult`.
    """
    if rng is None:
        rng = np.random.default_rng()
    operator = _LEGACY_ALIAS.get(ablation, ablation)
    if operator not in _OPERATORS:
        raise ValueError(
            f"unknown improvement operator: {ablation!r}. "
            f"valid: {sorted(_OPERATORS)} (legacy alias 'kl_anchor' -> 'single_anchor_kl')"
        )
    if root_selection not in ("sample", "argmax"):
        raise ValueError(f"unknown root_selection: {root_selection!r}")
    if beta <= 0:
        raise ValueError("beta must be positive")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if operator == "magnetic_kl" and (alpha + beta) <= 0:
        raise ValueError("alpha + beta must be positive for magnetic_kl")
    if not 0.0 <= prior_floor < 1.0:
        raise ValueError("prior_floor must be in [0, 1)")

    action_dim = legal_mask.shape[0]
    legal = np.where(legal_mask)[0]
    n_legal = int(legal.size)
    if n_legal == 0:
        raise ValueError("improve_policy: no legal actions")

    q = np.where(legal_mask, np.nan_to_num(search_q_mean, nan=0.0), 0.0)
    counts = np.where(legal_mask, rollout_counts, 0).astype(np.float64)
    sem = np.where(
        counts > 1,
        search_q_std / np.sqrt(np.maximum(counts, 1)),
        0.0,
    )

    # Advantage centered by the base-policy-weighted mean Q (only legal).
    # Centering is a constant shift across actions, so it does not change the
    # normalized distribution; it improves numerical stability and is the
    # quantity we log.
    base_legal = base_probs[legal_mask]
    q_legal = q[legal_mask]
    baseline = float(np.sum(base_legal * q_legal)) if n_legal else 0.0
    adv = np.zeros(action_dim, dtype=np.float64)
    adv[legal_mask] = q_legal - baseline

    # --- value scaling (skill §11) ---
    if normalize_advantages and n_legal > 1:
        a_legal = adv[legal_mask]
        sd = float(a_legal.std())
        if sd > 1e-8:
            adv[legal_mask] = a_legal / sd
        value_scale_mode = "legacy_zscore"
        scale_used: Optional[float] = None
    elif global_advantage_scale is not None and float(global_advantage_scale) > 0.0:
        g = float(global_advantage_scale)
        adv[legal_mask] = adv[legal_mask] / g
        value_scale_mode = "global_standardized"
        scale_used = g
    else:
        value_scale_mode = "raw"
        scale_used = None

    # --- construct the searched policy ---
    magnet_rho = np.zeros(action_dim, dtype=np.float64)
    if operator == "base_only":
        searched = _normalize_legal(base_probs, legal_mask)
    elif operator == "argmax_q":
        searched = np.zeros(action_dim, dtype=np.float64)
        q_legal = q[legal_mask]
        best = np.isclose(q_legal, q_legal.max())
        pick = int(rng.choice(legal[best]))
        searched[pick] = 1.0
    elif operator == "softmax_q":
        logits = np.full(action_dim, -np.inf, dtype=np.float64)
        logits[legal_mask] = q[legal_mask] / max(beta, 1e-6)
        shifted = logits - np.max(logits[legal_mask])
        exp = np.exp(shifted)
        searched = np.where(legal_mask, exp, 0.0)
    elif operator == "single_anchor_kl":
        prior = np.where(legal_mask, np.clip(base_probs, prior_floor, 1.0), 0.0)
        prior = prior / prior.sum()
        log_prior = _safe_log(prior)
        logits = np.full(action_dim, -np.inf, dtype=np.float64)
        logits[legal_mask] = log_prior[legal_mask] + adv[legal_mask] / max(beta, 1e-6)
        shifted = logits - np.max(logits[legal_mask])
        exp = np.exp(shifted)
        searched = np.where(legal_mask, exp, 0.0)
    else:  # magnetic_kl
        denom = alpha + beta
        # Uniform magnetic reference over legal actions.
        magnet_rho[legal_mask] = 1.0 / n_legal
        log_rho = _safe_log(magnet_rho)
        prior = np.where(legal_mask, np.clip(base_probs, prior_floor, 1.0), 0.0)
        prior = prior / prior.sum()
        log_prior = _safe_log(prior)
        logits = np.full(action_dim, -np.inf, dtype=np.float64)
        # adv is centered Q; constant-shift invariance holds for the exp term.
        logits[legal_mask] = (
            (alpha / denom) * log_rho[legal_mask]
            + (beta / denom) * log_prior[legal_mask]
            + adv[legal_mask] / denom
        )
        shifted = logits - np.max(logits[legal_mask])
        exp = np.exp(shifted)
        searched = np.where(legal_mask, exp, 0.0)

    s = searched.sum()
    if not np.isfinite(s) or s <= 0:
        # numerical fallback: revert to base policy on legal actions
        searched = np.where(legal_mask, base_probs, 0.0)
        s = searched.sum()
    searched = searched / s

    # --- select the live action ---
    base_argmax = int(legal[np.argmax(base_probs[legal_mask])])
    if operator == "argmax_q":
        selected = int(np.argmax(searched))
    elif root_selection == "argmax":
        selected = int(legal[np.argmax(searched[legal_mask])])
    else:
        selected = int(rng.choice(action_dim, p=searched))

    return SearchImprovementResult(
        searched_probs=searched,
        advantages=adv,
        base_probs=base_probs,
        search_q=np.where(legal_mask, search_q_mean, np.nan),
        search_q_std=np.where(legal_mask, search_q_std, np.nan),
        search_q_sem=sem,
        search_q_min=np.where(
            legal_mask, np.nan_to_num(search_q_mean, nan=0.0), np.nan
        ),
        search_q_max=np.where(
            legal_mask, np.nan_to_num(search_q_mean, nan=0.0), np.nan
        ),
        rollout_counts=rollout_counts.astype(np.int64),
        terminal_frac=terminal_frac,
        selected_action=selected,
        base_argmax=base_argmax,
        kl_to_base=kl(searched, np.where(legal_mask, base_probs, 1e-12)),
        base_entropy=entropy(np.where(legal_mask, base_probs, 0.0)),
        searched_entropy=entropy(searched),
        changed_argmax=(selected != base_argmax),
        n_legal=n_legal,
        legal_actions=legal,
        operator=operator,
        alpha=float(alpha),
        beta=float(beta),
        value_scale_mode=value_scale_mode,
        global_advantage_scale=scale_used,
        magnet_rho=magnet_rho,
    )
