"""Per-step "is the anchor out-of-distribution?" diagnostics.

Computes, at the test-time (last) gamma, a set of scalar features contrasting the
anchor member against the consensus of the *other* members. These are logged into
each step's ``diagnostics['disagreement']`` block (independent of which decider is
active) so the JSONL can later be mined for signals that predict anchor failure /
OOD-ness, and used to train a learned router.

Two families of features:

* **Anchor self-uncertainty** -- the spread across the anchor's *own* critic
  ensemble (``MemberStepFeatures.q_std``). High intra-agent critic disagreement is
  a classic epistemic-uncertainty / OOD proxy.
* **Anchor-vs-rest disagreement** -- how much the anchor's policy / value departs
  from the consensus of the other members (favorite-action prob gap, JS divergence,
  value gap, argmax agreement, how "alone" the anchor's pick is).

All outputs are plain Python ``float``/``int`` so they serialize cleanly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from metamon.rl.experimental.ensemblev2.decision import EnsembleDecisionContext


def _entropy_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def _js_divergence_bits(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits, in ``[0, 1]``. Inputs sum to 1."""
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def compute_disagreement_features(
    context: "EnsembleDecisionContext",
) -> dict[str, float]:
    """Anchor-vs-others disagreement + anchor self-uncertainty at the target gamma.

    Requires >= 2 members (an anchor and at least one other). Q-based features are
    emitted only when Q-values were gathered; otherwise just the policy features.
    """
    legal = context.legal_actions
    others = [m for i, m in enumerate(context.members) if i != context.anchor_index]
    if not legal or not others:
        return {}

    anchor = context.anchor
    legal_idx = np.asarray(legal, dtype=np.int64)

    anchor_p = np.asarray(anchor.rollout_probs(), dtype=np.float64)
    others_p = np.stack(
        [np.asarray(m.rollout_probs(), dtype=np.float64) for m in others]
    )  # (M, 13)
    others_mean_p = others_p.mean(axis=0)

    # Restrict the distributions to legal actions (already legal-normalized, so
    # these still sum to ~1) for divergence / entropy.
    a_leg = anchor_p[legal_idx]
    o_leg = others_mean_p[legal_idx]
    a_leg = a_leg / max(a_leg.sum(), 1e-12)
    o_leg = o_leg / max(o_leg.sum(), 1e-12)

    anchor_top = int(legal_idx[int(np.argmax(anchor_p[legal_idx]))])
    consensus_top = int(legal_idx[int(np.argmax(others_mean_p[legal_idx]))])

    feats: dict[str, float] = {
        "num_legal": int(len(legal)),
        "num_others": int(len(others)),
        "anchor_top_action": anchor_top,
        "consensus_top_action": consensus_top,
        "top_action_agree": int(anchor_top == consensus_top),
        # --- favorite-action prob vs the average of the rest ---
        "anchor_top_prob": float(anchor_p[anchor_top]),
        "others_mean_prob_on_anchor_top": float(others_mean_p[anchor_top]),
        "prob_gap_anchor_top": float(anchor_p[anchor_top] - others_mean_p[anchor_top]),
        # --- distributional disagreement ---
        "js_anchor_vs_others": _js_divergence_bits(a_leg, o_leg),
        "anchor_entropy_bits": _entropy_bits(a_leg),
        "others_entropy_bits": _entropy_bits(o_leg),
        # fraction of other members whose own top pick differs from the anchor's
        "anchor_alone_frac": float(
            np.mean(
                [
                    int(int(legal_idx[int(np.argmax(p[legal_idx]))]) != anchor_top)
                    for p in others_p
                ]
            )
        ),
    }

    # --- Q-based features (target gamma); only if gathered ---
    anchor_q = anchor.rollout_q()
    anchor_q_std = anchor.rollout_q_std()
    have_q = anchor_q is not None and not bool(np.all(np.isnan(anchor_q)))

    if have_q and anchor_q_std is not None:
        std_leg = anchor_q_std[legal_idx]
        feats["anchor_q_std_top"] = float(anchor_q_std[anchor_top])
        feats["anchor_q_std_mean_legal"] = float(np.nanmean(std_leg))
        feats["anchor_q_std_max_legal"] = float(np.nanmax(std_leg))
        # the other members' average intra-agent critic spread, for context
        others_std = [
            m.rollout_q_std() for m in others if m.rollout_q_std() is not None
        ]
        if others_std:
            feats["others_q_std_mean_legal"] = float(
                np.nanmean([np.nanmean(s[legal_idx]) for s in others_std])
            )

    if have_q:
        others_q = np.stack(
            [
                np.asarray(m.rollout_q(), dtype=np.float64)
                for m in others
                if m.rollout_q() is not None
            ]
        )  # (M', 13)
        others_mean_q = np.nanmean(others_q, axis=0)
        a_q = np.asarray(anchor_q, dtype=np.float64)
        # value gap on the anchor's preferred action (raw reward units; shared
        # reward fn across this trio makes this directly comparable)
        feats["q_gap_anchor_top"] = float(a_q[anchor_top] - others_mean_q[anchor_top])
        anchor_q_top = int(legal_idx[int(np.nanargmax(a_q[legal_idx]))])
        others_q_top = int(legal_idx[int(np.nanargmax(others_mean_q[legal_idx]))])
        feats["q_argmax_agree"] = int(anchor_q_top == others_q_top)
        feats["anchor_q_argmax_action"] = anchor_q_top

    return feats
