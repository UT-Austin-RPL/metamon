"""Concrete :class:`EnsembleDecision` strategies.

Two layers live here:

* Pure controls (``anchor``, ``mean_prob``) -- minimal strategies with no safety
  logic. ``anchor`` is an exact passthrough of the anchor policy's argmax and is
  the reference point for "does this ensemble beat the anchor?".
* :class:`HeuristicSafetyDecision` -- the intended ROOT for all tuned strategies.
  It ports the robustness "hacks" from the v1 ensemble (legal-action guards,
  forced-switch awareness, and the cycle/stall breaker) while staying agnostic to
  *how* the desired action is scored. Subclasses only implement
  :meth:`HeuristicSafetyDecision.score_actions`; the base wraps that with safety.

Add new strategies by subclassing :class:`HeuristicSafetyDecision` and registering
with ``@register_ensemble_decision``; select one via the ensemble config's
``decision`` field (or the ``METAMON_ENSEMBLEV2_DECISION`` env override).
"""

from __future__ import annotations

import math
from abc import abstractmethod
from typing import Any, Optional

import numpy as np

from metamon.rl.experimental.ensemblev2.action_remap import CANONICAL_ACTION_DIM
from metamon.rl.experimental.ensemblev2.decision import (
    EnsembleDecision,
    EnsembleDecisionContext,
    register_ensemble_decision,
)


# --------------------------------------------------------------------------- #
# Pure controls (no safety logic)                                             #
# --------------------------------------------------------------------------- #


@register_ensemble_decision("anchor")
class AnchorPassthroughDecision(EnsembleDecision):
    """Defer entirely to the anchor model.

    Picks the legal action with the highest anchor probability at the test-time
    (last) gamma. Exactly equivalent to running the anchor policy alone with
    deterministic argmax selection -- the ensemble baseline / control.
    """

    def __call__(self, context: EnsembleDecisionContext) -> int:
        legal = context.legal_actions
        if not legal:
            return 0
        if len(legal) == 1:
            return int(legal[0])
        anchor_probs = context.anchor.rollout_probs()
        return int(max(legal, key=lambda action: float(anchor_probs[action])))


@register_ensemble_decision("anchor_q")
class AnchorQArgmaxDecision(EnsembleDecision):
    """Defer to the anchor's *critic*: pick the legal action with the highest Q.

    Unlike ``anchor`` (which takes the argmax of the anchor actor head's action
    probabilities), this ranks legal actions by the anchor policy's Q-values at
    the test-time (last) gamma. It's a clean control for an interesting question:
    does the value function pick better actions than the policy head when the two
    disagree? Falls back to the actor argmax if no legal action has a finite Q
    (e.g. Q gathering disabled, or all legal actions inexpressible by the critic).
    """

    def __call__(self, context: EnsembleDecisionContext) -> int:
        legal = context.legal_actions
        if not legal:
            return 0
        if len(legal) == 1:
            return int(legal[0])
        q = context.anchor.rollout_q()
        finite = [a for a in legal if a < len(q) and np.isfinite(q[a])]
        if not finite:
            anchor_probs = context.anchor.rollout_probs()
            return int(max(legal, key=lambda action: float(anchor_probs[action])))
        return int(max(finite, key=lambda action: float(q[action])))


@register_ensemble_decision("mean_prob")
class MeanProbDecision(EnsembleDecision):
    """Average every member's legal action distribution and take the argmax."""

    def __call__(self, context: EnsembleDecisionContext) -> int:
        legal = context.legal_actions
        if not legal:
            return 0
        if len(legal) == 1:
            return int(legal[0])
        legal_arr, weights = _average_legal_distribution(context)
        return int(legal_arr[int(np.argmax(weights))])


def _average_legal_distribution(
    context: EnsembleDecisionContext,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean of per-member *legal-conditional* actor distributions.

    For each member we:
      1. take test-time rollout probs on the current legal action set only,
      2. zero non-finite / negative mass,
      3. renormalize over legal (uniform fallback if the member put no mass there).

    Then we average those legal-length vectors. The result is a distribution that
    puts **zero** mass on illegal actions by construction (it is only defined on
    ``legal``). Callers must sample/argmax over the returned ``legal_arr`` with
    the returned ``weights`` — never over the full 13-wide action space.
    """
    legal = context.legal_actions
    if not legal:
        raise ValueError(
            "_average_legal_distribution requires at least one legal action"
        )
    legal_arr = np.asarray(legal, dtype=np.int64)
    n_legal = len(legal_arr)
    acc = np.zeros(n_legal, dtype=np.float64)
    for member in context.members:
        probs = np.asarray(member.rollout_probs(), dtype=np.float64)
        weights = probs[legal_arr].astype(np.float64, copy=True)
        weights[~np.isfinite(weights)] = 0.0
        np.maximum(weights, 0.0, out=weights)
        total = float(weights.sum())
        if total <= 0.0:
            weights[:] = 1.0 / n_legal
        else:
            weights /= total
        acc += weights
    acc /= max(len(context.members), 1)
    # Belt-and-suspenders: re-normalize in case of tiny numerical drift.
    total = float(acc.sum())
    if total <= 0.0 or not np.isfinite(total):
        acc[:] = 1.0 / n_legal
    else:
        acc /= total
    return legal_arr, acc


def _sample_from_legal(
    legal_arr: np.ndarray,
    weights: np.ndarray,
    *,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Draw one action from a distribution defined only on ``legal_arr``."""
    if len(legal_arr) == 0:
        return 0
    if len(legal_arr) == 1:
        return int(legal_arr[0])
    w = np.asarray(weights, dtype=np.float64).copy()
    w[~np.isfinite(w)] = 0.0
    np.maximum(w, 0.0, out=w)
    total = float(w.sum())
    if total <= 0.0:
        w[:] = 1.0 / len(legal_arr)
    else:
        w /= total
    draw = np.random.default_rng() if rng is None else rng
    return int(draw.choice(legal_arr, p=w))


def _member_legal_weights(probs: np.ndarray, legal_arr: np.ndarray) -> np.ndarray:
    """Legal-conditional distribution for one member (sums to 1 over legal)."""
    weights = np.asarray(probs, dtype=np.float64)[legal_arr].astype(
        np.float64, copy=True
    )
    weights[~np.isfinite(weights)] = 0.0
    np.maximum(weights, 0.0, out=weights)
    total = float(weights.sum())
    if total <= 0.0:
        weights[:] = 1.0 / len(legal_arr)
    else:
        weights /= total
    return weights


def _member_preferred_action(probs: np.ndarray, legal_arr: np.ndarray) -> int:
    """Legal argmax of a member's actor distribution."""
    weights = _member_legal_weights(probs, legal_arr)
    return int(legal_arr[int(np.argmax(weights))])


@register_ensemble_decision("sample_mean_prob")
class SampleMeanProbDecision(EnsembleDecision):
    """Average members' legal action distributions, then sample once from the mean.

    Illegal / masked actions are excluded before averaging (each member is
    renormalized over the current legal set), so they can never be sampled.
    """

    def __call__(self, context: EnsembleDecisionContext) -> int:
        legal = context.legal_actions
        if not legal:
            return 0
        if len(legal) == 1:
            return int(legal[0])
        legal_arr, weights = _average_legal_distribution(context)
        chosen = _sample_from_legal(legal_arr, weights)
        context.diagnostics["sample_mean"] = {
            "legal_probs": {
                int(a): float(p) for a, p in zip(legal_arr.tolist(), weights.tolist())
            },
            "chosen": chosen,
        }
        return chosen


def _gated_anchor_or_majority(
    context: EnsembleDecisionContext,
    *,
    samples_per_member: int,
    min_dissenters: int,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Sample from the anchor unless enough members disagree with its preference.

    1. Anchor preference = legal argmax of the anchor actor.
    2. Count how many ensemble members have a different legal argmax
       (``dissenters``). The anchor itself never counts as a dissenter.
    3. If ``dissenters >= min_dissenters`` (default 2 = "more than one"), fall
       back to sampled majority vote over all members.
    4. Otherwise sample once from the anchor's legal-conditional distribution.

    Illegal actions are excluded from preferences, sampling, and voting.
    """
    if min_dissenters < 1:
        raise ValueError(f"min_dissenters must be >= 1, got {min_dissenters}")
    legal = context.legal_actions
    legal_arr = np.asarray(legal, dtype=np.int64)
    anchor = context.anchor
    anchor_probs = np.asarray(anchor.rollout_probs(), dtype=np.float64)
    anchor_pref = _member_preferred_action(anchor_probs, legal_arr)

    dissenters: list[int] = []
    for member in context.members:
        if member.member_index == context.anchor_index:
            continue
        pref = _member_preferred_action(
            np.asarray(member.rollout_probs(), dtype=np.float64), legal_arr
        )
        if pref != anchor_pref:
            dissenters.append(int(member.member_index))

    use_majority = len(dissenters) >= min_dissenters
    context.diagnostics["anchor_gate"] = {
        "anchor_pref": int(anchor_pref),
        "dissenters": dissenters,
        "min_dissenters": int(min_dissenters),
        "use_majority": bool(use_majority),
    }

    if use_majority:
        scores = _majority_vote_scores(
            context, samples_per_member=samples_per_member, rng=rng
        )
        chosen = int(max(legal, key=lambda action: float(scores[action])))
        context.diagnostics["anchor_gate"]["chosen"] = chosen
        context.diagnostics["anchor_gate"]["path"] = "majority"
        return chosen

    anchor_weights = _member_legal_weights(anchor_probs, legal_arr)
    chosen = _sample_from_legal(legal_arr, anchor_weights, rng=rng)
    context.diagnostics["anchor_gate"]["chosen"] = chosen
    context.diagnostics["anchor_gate"]["path"] = "anchor_sample"
    return chosen


@register_ensemble_decision("anchor_gated_majority")
class AnchorGatedMajorityDecision(EnsembleDecision):
    """Sample from the anchor unless ``min_dissenters`` members disagree, else
    sampled majority.

    Args:
        samples_per_member: Draws per member when falling back to majority
            (default 16).
        min_dissenters: Override to majority when at least this many *non-anchor*
            members have a different legal argmax than the anchor (default 2 =
            "more than one").
    """

    def __init__(
        self,
        samples_per_member: int = 16,
        min_dissenters: int = 2,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.samples_per_member = int(samples_per_member)
        self.min_dissenters = int(min_dissenters)

    def __call__(self, context: EnsembleDecisionContext) -> int:
        legal = context.legal_actions
        if not legal:
            return 0
        if len(legal) == 1:
            return int(legal[0])
        return _gated_anchor_or_majority(
            context,
            samples_per_member=self.samples_per_member,
            min_dissenters=self.min_dissenters,
        )


def _majority_vote_scores(
    context: EnsembleDecisionContext,
    *,
    samples_per_member: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sampled majority vote counts with mean-prob fractional tie-break.

    Each member draws ``samples_per_member`` independent samples from its
    legal-normalized test-time actor distribution; every sample is one vote.
    Scores are ``vote_count + mean_prob`` so ties break toward the soft consensus
    without changing the hard-vote ranking when counts differ.
    """
    if samples_per_member < 1:
        raise ValueError(f"samples_per_member must be >= 1, got {samples_per_member}")
    legal = context.legal_actions
    legal_arr = np.asarray(legal, dtype=np.int64)
    vote_counts = np.zeros(CANONICAL_ACTION_DIM, dtype=np.float64)
    member_probs = []
    draw = np.random.default_rng() if rng is None else rng
    for member in context.members:
        probs = np.asarray(member.rollout_probs(), dtype=np.float64)
        member_probs.append(probs)
        weights = probs[legal_arr].astype(np.float64, copy=True)
        total = float(weights.sum())
        if total <= 0.0 or not np.isfinite(total):
            # Degenerate: fall back to uniform over legal actions.
            weights[:] = 1.0 / len(legal_arr)
        else:
            weights /= total
        votes = draw.choice(legal_arr, size=samples_per_member, p=weights)
        for action in votes:
            vote_counts[int(action)] += 1.0
    mean_probs = np.mean(member_probs, axis=0)
    context.diagnostics["majority"] = {
        "samples_per_member": int(samples_per_member),
        "vote_counts": {
            int(a): int(vote_counts[a]) for a in legal if vote_counts[a] > 0
        },
    }
    return vote_counts + mean_probs


@register_ensemble_decision("majority_vote")
class MajorityVoteDecision(EnsembleDecision):
    """Sampled majority: each member votes ``samples_per_member`` times from its
    actor distribution; ties break soft (mean-prob).

    Args:
        samples_per_member: Independent categorical draws per member (default 16).
    """

    def __init__(self, samples_per_member: int = 16, **kwargs: Any):
        super().__init__(**kwargs)
        self.samples_per_member = int(samples_per_member)

    def __call__(self, context: EnsembleDecisionContext) -> int:
        legal = context.legal_actions
        if not legal:
            return 0
        if len(legal) == 1:
            return int(legal[0])
        scores = _majority_vote_scores(
            context, samples_per_member=self.samples_per_member
        )
        return int(max(legal, key=lambda action: float(scores[action])))


# --------------------------------------------------------------------------- #
# Heuristic-safety root (ports the v1 robustness hacks)                        #
# --------------------------------------------------------------------------- #


class HeuristicSafetyDecision(EnsembleDecision):
    """Root strategy that wraps a (subclass-defined) score with v1 safety hacks.

    Subclasses implement :meth:`score_actions`, returning a per-action preference
    score over canonical universal indices (higher = better; illegal slots are
    ignored). The base class then:

      1. Short-circuits trivial cases (no legal action -> 0; single legal action).
      2. Masks illegal actions.
      3. Applies the **cycle/stall breaker** ported from the v1 ensemble: if the
         battle has been stuck repeating the same (state, action) -- a periodic
         ``ABAB``/``AAA`` loop or a 4x dead-repeat -- with negligible reward
         progress, the offending action is penalized so a different legal action
         wins. Disabled during forced switches (active fainted), matching v1.
      4. Returns the argmax legal action.

    All thresholds are constructor kwargs so the behavior is tunable per config
    (``decision_kwargs``). This is the intended base for all further tweaks; the
    only thing a subclass must decide is how to *score* actions.

    Args:
        cycle_penalty: Severity (as a fraction of the legal-score spread) applied
            to an action caught in a periodic ``ABAB`` cycle.
        repeat_penalty: Severity for an action caught in a 4x dead-repeat (the
            strongest stall signal).
        period1_bonus: Extra severity added to ``cycle_penalty`` for a tight
            period-1 (``AAA``) loop.
        reward_cap: Max |reward| in a window still considered "no progress".
        mean_reward_cap: Max mean |reward| in a window still considered "no
            progress".
        min_history: Minimum completed transitions before stall logic engages.
    """

    def __init__(
        self,
        cycle_penalty: float = 0.6,
        repeat_penalty: float = 1.0,
        period1_bonus: float = 0.2,
        reward_cap: float = 0.08,
        mean_reward_cap: float = 0.025,
        min_history: int = 4,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.cycle_penalty = cycle_penalty
        self.repeat_penalty = repeat_penalty
        self.period1_bonus = period1_bonus
        self.reward_cap = reward_cap
        self.mean_reward_cap = mean_reward_cap
        self.min_history = min_history

    # -- subclass hook ----------------------------------------------------- #

    @abstractmethod
    def score_actions(self, context: EnsembleDecisionContext) -> np.ndarray:
        """Return a length-13 canonical score vector (higher = better).

        Illegal / inexpressible slots are ignored by the base class, so their
        values don't matter (0 or NaN are both fine).
        """
        raise NotImplementedError

    # -- decision point ---------------------------------------------------- #

    def __call__(self, context: EnsembleDecisionContext) -> int:
        legal = context.legal_actions
        if not legal:
            return 0
        if len(legal) == 1:
            return int(legal[0])

        scores = np.array(self.score_actions(context), dtype=np.float64).copy()
        mask = np.ones(CANONICAL_ACTION_DIM, dtype=bool)
        mask[legal] = False
        scores[mask] = -np.inf

        intended = int(max(legal, key=lambda action: scores[action]))

        penalties = self._stall_penalties(context)
        if penalties:
            legal_scores = scores[legal]
            spread = float(np.max(legal_scores) - np.min(legal_scores))
            base = spread if spread > 0 else 1.0
            for action, severity in penalties.items():
                if action in legal:
                    scores[action] -= severity * base

        # Break score ties against penalized (stalling) actions so the strongest
        # stall signal reliably flips even when a penalty only equalizes scores.
        chosen = int(
            max(
                legal,
                key=lambda action: (scores[action], -penalties.get(action, 0.0)),
            )
        )

        if penalties:
            # Tier is implied by severity (repeat_penalty is the strongest signal).
            context.diagnostics["safety"] = {
                "intended_action": intended,
                "overridden": bool(chosen != intended),
                "penalties": {int(a): float(s) for a, s in penalties.items()},
            }

        return chosen

    # -- ported v1 safety helpers ------------------------------------------ #

    @staticmethod
    def _forced_switch(legal: list[int]) -> bool:
        """True when every legal action is a switch (active fainted)."""
        return all(4 <= action <= 8 for action in legal)

    def _completed_transitions(
        self, context: EnsembleDecisionContext
    ) -> list[tuple[Any, int, float]]:
        """Reconstruct ``(state_hash, action, resulting_reward)`` for past turns.

        The reward earned by the action chosen at turn ``i`` shows up as the
        ``prev_reward`` of turn ``i+1`` (and ``context.prev_reward`` for the most
        recent turn), mirroring v1's pending-transition finalization.
        """
        history = context.history
        transitions: list[tuple[Any, int, float]] = []
        for i, record in enumerate(history):
            state = record.get("state_hash")
            action = record.get("chosen_action")
            if i + 1 < len(history):
                reward = history[i + 1].get("prev_reward", 0.0)
            else:
                reward = context.prev_reward
            if not math.isfinite(reward):
                reward = 0.0
            transitions.append((state, int(action), float(reward)))
        return transitions

    def _low_progress(self, window: list[tuple[Any, int, float]]) -> bool:
        magnitudes = [abs(reward) for _, _, reward in window]
        if not magnitudes:
            return False
        return (
            max(magnitudes) <= self.reward_cap
            and (sum(magnitudes) / len(magnitudes)) <= self.mean_reward_cap
        )

    def _stall_penalties(self, context: EnsembleDecisionContext) -> dict[int, float]:
        """Port of v1 ``_stall_penalties``: penalize actions stuck in a stall.

        Only fires on clearly persistent loops (``AAA`` / ``ABAB``) or a 4x
        dead-repeat in the *current* state with negligible reward progress.
        """
        legal = context.legal_actions
        if self._forced_switch(legal):
            return {}
        transitions = self._completed_transitions(context)
        if len(transitions) < self.min_history:
            return {}

        current_state = context.state_hash
        penalties: dict[int, float] = {}

        # Periodic cycles: last 3 blocks of `period` are identical (state+action),
        # the current state matches the cycle, and the window made no progress.
        for period in (1, 2):
            if len(transitions) < 3 * period:
                continue
            recent = transitions[-3 * period :]
            blocks = [recent[k * period : (k + 1) * period] for k in range(3)]
            state_blocks = [[t[0] for t in block] for block in blocks]
            action_blocks = [[t[1] for t in block] for block in blocks]
            if not (
                state_blocks[0] == state_blocks[1] == state_blocks[2]
                and action_blocks[0] == action_blocks[1] == action_blocks[2]
            ):
                continue
            if current_state != state_blocks[-1][0]:
                continue
            if not self._low_progress(recent):
                continue
            cycle_action = action_blocks[-1][0]
            penalty = self.cycle_penalty + (self.period1_bonus if period == 1 else 0.0)
            penalties[cycle_action] = max(penalties.get(cycle_action, 0.0), penalty)

        # Dead-repeat: same state + same action four turns running, no progress.
        recent = transitions[-4:]
        if (
            all(t[0] == current_state for t in recent)
            and len({t[1] for t in recent}) == 1
            and self._low_progress(recent)
        ):
            repeat_action = recent[-1][1]
            penalties[repeat_action] = max(
                penalties.get(repeat_action, 0.0), self.repeat_penalty
            )

        return penalties


@register_ensemble_decision("safe_anchor")
class SafeAnchorDecision(HeuristicSafetyDecision):
    """Anchor passthrough wrapped in the heuristic safety layer.

    Identical to ``anchor`` except a detected stall/cycle nudges the choice off
    the looping action. The natural first "tuned" strategy and a clean template
    for richer scorers.
    """

    def score_actions(self, context: EnsembleDecisionContext) -> np.ndarray:
        return np.asarray(context.anchor.rollout_probs(), dtype=np.float64)


@register_ensemble_decision("safe_anchor_q")
class SafeAnchorQDecision(HeuristicSafetyDecision):
    """Anchor critic-Q argmax wrapped in the heuristic safety layer.

    The ``anchor_q`` sibling of :class:`SafeAnchorDecision`: scores legal actions
    by the anchor policy's test-time Q-values (critic-ensemble mean) rather than
    the actor head, then applies the cycle/stall safety hacks. Legal actions the
    critic cannot express are scored below the worst finite legal Q; if no legal
    action has a finite Q at all, falls back to the actor distribution.
    """

    def score_actions(self, context: EnsembleDecisionContext) -> np.ndarray:
        q = np.asarray(context.anchor.rollout_q(), dtype=np.float64)
        legal_q = [
            q[a] for a in context.legal_actions if a < len(q) and np.isfinite(q[a])
        ]
        if not legal_q:
            return np.asarray(context.anchor.rollout_probs(), dtype=np.float64)
        # Keep scores finite (the safety layer needs a finite legal-score spread).
        return np.where(np.isfinite(q), q, min(legal_q) - 1.0)


@register_ensemble_decision("safe_anchor_q_gamma")
class SafeAnchorQGammaDecision(HeuristicSafetyDecision):
    """``safe_anchor_q`` with a configurable (optionally turn-dependent) gamma.

    Ranks legal actions by the anchor critic's Q at a chosen gamma *index* instead
    of always the last (longest-horizon) gamma. Optionally switches to a shorter
    horizon late in the battle, triggered by either the turn index
    (``turn_idx >= late_turn``) or -- preferred -- the opponent being down to a
    handful of Pokemon (``opponents_remaining <= late_opp_remaining``). The
    opponent-count signal comes straight off the ``UniversalState`` and is a much
    cleaner "endgame" indicator than raw turn count.

    The anchor's gammas are ``[0.1, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999]`` →
    indices ``0..6``. Defaults (``gamma_index=-1``, no late switch) reproduce
    ``safe_anchor_q`` exactly.

    Args:
        gamma_index: gamma index used for ranking (negative allowed; -1 = last).
        late_gamma_index: if set, the (shorter) gamma index to use once an
            endgame trigger fires.
        late_turn: turn index at/after which ``late_gamma_index`` takes over.
        late_opp_remaining: opponent Pokemon count at/below which
            ``late_gamma_index`` takes over (e.g. ``1`` = only the opponent's last
            mon). Uses ``context.opponents_remaining``; ignored if that is None.
    """

    def __init__(
        self,
        gamma_index: int = -1,
        late_gamma_index: Optional[int] = None,
        late_turn: int = 1_000_000,
        late_opp_remaining: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.gamma_index = int(gamma_index)
        self.late_gamma_index = (
            None if late_gamma_index is None else int(late_gamma_index)
        )
        self.late_turn = int(late_turn)
        self.late_opp_remaining = (
            None if late_opp_remaining is None else int(late_opp_remaining)
        )

    def _is_late(self, context: EnsembleDecisionContext) -> bool:
        if self.late_gamma_index is None:
            return False
        if context.turn_idx >= self.late_turn:
            return True
        if (
            self.late_opp_remaining is not None
            and context.opponents_remaining is not None
            and context.opponents_remaining <= self.late_opp_remaining
        ):
            return True
        return False

    def score_actions(self, context: EnsembleDecisionContext) -> np.ndarray:
        anchor = context.anchor
        q_all = np.asarray(anchor.q_values, dtype=np.float64)  # (num_gammas, 13)
        num_gammas = q_all.shape[0]
        gi = self.gamma_index
        late = self._is_late(context)
        if late:
            gi = self.late_gamma_index
        if not -num_gammas <= gi < num_gammas:
            raise ValueError(
                f"gamma_index {gi} out of range for {num_gammas} gammas "
                f"(valid: 0..{num_gammas - 1} or -1..-{num_gammas}). "
                "Note: there is NO wrap-around."
            )
        gi %= num_gammas
        q = q_all[gi]
        legal_q = [
            q[a] for a in context.legal_actions if a < len(q) and np.isfinite(q[a])
        ]
        if not legal_q:
            return np.asarray(anchor.rollout_probs(), dtype=np.float64)
        context.diagnostics["gamma"] = {
            "gamma_index": int(gi),
            "gamma": float(anchor.gammas[gi]),
            "late": bool(late),
            "opp_remaining": context.opponents_remaining,
        }
        return np.where(np.isfinite(q), q, min(legal_q) - 1.0)


@register_ensemble_decision("safe_consensus")
class SafeConsensusDecision(HeuristicSafetyDecision):
    """Ensemble consensus (mean legal action distribution) + safety layer, argmax.

    Scores each canonical action by the average of every member's legal-conditional
    rollout distribution, so actions multiple members agree on win. The
    safety-wrapped sibling of the pure ``mean_prob`` control: same consensus
    pick, but a detected stall/cycle nudges it off the looping action.

    Prefer ``safe_sample_consensus`` when you want to *sample* from that mean
    instead of taking its argmax.
    """

    def score_actions(self, context: EnsembleDecisionContext) -> np.ndarray:
        legal_arr, weights = _average_legal_distribution(context)
        scores = np.zeros(CANONICAL_ACTION_DIM, dtype=np.float64)
        scores[legal_arr] = weights
        return scores


@register_ensemble_decision("safe_sample_consensus")
class SafeSampleConsensusDecision(HeuristicSafetyDecision):
    """Average members' legal distributions, apply stall downweights, then sample.

    Same mean construction as ``sample_mean_prob`` / ``safe_consensus``, but the
    final action is drawn from the (possibly stall-penalized) mean distribution
    rather than taking an argmax. Illegal actions are never given mass: averaging
    and sampling both happen strictly over ``context.legal_actions``.
    """

    def score_actions(self, context: EnsembleDecisionContext) -> np.ndarray:
        # Kept for the abstract API / diagnostics; __call__ samples instead of
        # argmaxing these scores.
        legal_arr, weights = _average_legal_distribution(context)
        scores = np.zeros(CANONICAL_ACTION_DIM, dtype=np.float64)
        scores[legal_arr] = weights
        return scores

    def __call__(self, context: EnsembleDecisionContext) -> int:
        legal = context.legal_actions
        if not legal:
            return 0
        if len(legal) == 1:
            return int(legal[0])

        legal_arr, weights = _average_legal_distribution(context)
        intended = int(legal_arr[int(np.argmax(weights))])

        penalties = self._stall_penalties(context)
        if penalties:
            adjusted = weights.copy()
            for i, action in enumerate(legal_arr.tolist()):
                severity = penalties.get(int(action), 0.0)
                if severity > 0.0:
                    adjusted[i] *= max(0.0, 1.0 - float(severity))
            total = float(adjusted.sum())
            if total <= 0.0 or not np.isfinite(total):
                # All mass wiped by penalties — fall back to uniform over legal.
                adjusted[:] = 1.0 / len(legal_arr)
            else:
                adjusted /= total
            weights = adjusted

        chosen = _sample_from_legal(legal_arr, weights)
        context.diagnostics["sample_mean"] = {
            "legal_probs": {
                int(a): float(p) for a, p in zip(legal_arr.tolist(), weights.tolist())
            },
            "chosen": int(chosen),
            "intended_argmax": intended,
        }
        if penalties:
            context.diagnostics["safety"] = {
                "intended_action": intended,
                "overridden": bool(chosen != intended),
                "penalties": {int(a): float(s) for a, s in penalties.items()},
            }
        return chosen


@register_ensemble_decision("safe_majority")
class SafeMajorityDecision(HeuristicSafetyDecision):
    """Sampled majority vote + safety layer (sibling of ``majority_vote``).

    Each member draws ``samples_per_member`` votes from its legal actor
    distribution; vote counts (plus a mean-prob fractional tie-break) are the
    score vector that the heuristic safety layer may nudge off a detected
    stall/cycle.

    Args:
        samples_per_member: Independent categorical draws per member (default 16).
    """

    def __init__(self, samples_per_member: int = 16, **kwargs: Any):
        super().__init__(**kwargs)
        self.samples_per_member = int(samples_per_member)

    def score_actions(self, context: EnsembleDecisionContext) -> np.ndarray:
        return _majority_vote_scores(
            context, samples_per_member=self.samples_per_member
        )


@register_ensemble_decision("safe_anchor_gated_majority")
class SafeAnchorGatedMajorityDecision(HeuristicSafetyDecision):
    """Sample from the anchor unless enough teammates disagree, else majority.

    Default path matches solo-anchor stochastic play (sample from the anchor's
    legal distribution). If at least ``min_dissenters`` non-anchor members have a
    different legal argmax than the anchor, fall back to sampled majority. Stall
    / cycle penalties downweight the chosen path's distribution or vote scores
    before the final pick.

    Args:
        samples_per_member: Draws per member on the majority path (default 16).
        min_dissenters: Majority override threshold (default 2 = more than one).
    """

    def __init__(
        self,
        samples_per_member: int = 16,
        min_dissenters: int = 2,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.samples_per_member = int(samples_per_member)
        self.min_dissenters = int(min_dissenters)

    def score_actions(self, context: EnsembleDecisionContext) -> np.ndarray:
        # Used only if something calls the argmax safety path; primary logic is
        # in __call__.
        return _majority_vote_scores(
            context, samples_per_member=self.samples_per_member
        )

    def __call__(self, context: EnsembleDecisionContext) -> int:
        legal = context.legal_actions
        if not legal:
            return 0
        if len(legal) == 1:
            return int(legal[0])

        legal_arr = np.asarray(legal, dtype=np.int64)
        anchor = context.anchor
        anchor_probs = np.asarray(anchor.rollout_probs(), dtype=np.float64)
        anchor_pref = _member_preferred_action(anchor_probs, legal_arr)

        dissenters: list[int] = []
        for member in context.members:
            if member.member_index == context.anchor_index:
                continue
            pref = _member_preferred_action(
                np.asarray(member.rollout_probs(), dtype=np.float64), legal_arr
            )
            if pref != anchor_pref:
                dissenters.append(int(member.member_index))

        use_majority = len(dissenters) >= self.min_dissenters
        penalties = self._stall_penalties(context)
        context.diagnostics["anchor_gate"] = {
            "anchor_pref": int(anchor_pref),
            "dissenters": dissenters,
            "min_dissenters": int(self.min_dissenters),
            "use_majority": bool(use_majority),
        }

        if use_majority:
            scores = _majority_vote_scores(
                context, samples_per_member=self.samples_per_member
            )
            # Mirror HeuristicSafetyDecision: mask illegal, apply stall penalties,
            # argmax (majority path stays discrete).
            mask = np.ones(CANONICAL_ACTION_DIM, dtype=bool)
            mask[legal] = False
            scores = scores.copy()
            scores[mask] = -np.inf
            intended = int(max(legal, key=lambda action: scores[action]))
            if penalties:
                legal_scores = scores[legal]
                finite = legal_scores[np.isfinite(legal_scores)]
                spread = float(np.max(finite) - np.min(finite)) if len(finite) else 1.0
                base = spread if spread > 0 else 1.0
                for action, severity in penalties.items():
                    if action in legal:
                        scores[action] -= severity * base
            chosen = int(
                max(
                    legal,
                    key=lambda action: (
                        scores[action],
                        -penalties.get(action, 0.0),
                    ),
                )
            )
            path = "majority"
        else:
            weights = _member_legal_weights(anchor_probs, legal_arr)
            intended = int(legal_arr[int(np.argmax(weights))])
            if penalties:
                adjusted = weights.copy()
                for i, action in enumerate(legal_arr.tolist()):
                    severity = penalties.get(int(action), 0.0)
                    if severity > 0.0:
                        adjusted[i] *= max(0.0, 1.0 - float(severity))
                total = float(adjusted.sum())
                if total <= 0.0 or not np.isfinite(total):
                    adjusted[:] = 1.0 / len(legal_arr)
                else:
                    adjusted /= total
                weights = adjusted
            chosen = _sample_from_legal(legal_arr, weights)
            path = "anchor_sample"

        context.diagnostics["anchor_gate"]["chosen"] = int(chosen)
        context.diagnostics["anchor_gate"]["path"] = path
        if penalties:
            context.diagnostics["safety"] = {
                "intended_action": int(intended),
                "overridden": bool(chosen != intended),
                "penalties": {int(a): float(s) for a, s in penalties.items()},
            }
        return chosen


@register_ensemble_decision("safe_teammates")
class SafeTeammatesDecision(HeuristicSafetyDecision):
    """Consensus of the *non-anchor* members only (+ safety layer).

    Unlike ``safe_consensus`` (which averages in the confident anchor and so almost
    always reproduces its argmax), this excludes the anchor entirely: it scores by
    the mean test-time distribution of the teammates. This deliberately *creates*
    disagreement -- the executed action differs from the anchor's pick whenever the
    teammates collectively prefer something else -- giving a real-disagreement /
    counterfactual dataset for studying when overriding the anchor helps or hurts.

    Falls back to the anchor's distribution if there are no teammates.
    """

    def score_actions(self, context: EnsembleDecisionContext) -> np.ndarray:
        teammate_probs = [
            np.asarray(member.rollout_probs(), dtype=np.float64)
            for member in context.members
            if member.member_index != context.anchor_index
        ]
        if not teammate_probs:
            return np.asarray(context.anchor.rollout_probs(), dtype=np.float64)
        return np.mean(teammate_probs, axis=0)


@register_ensemble_decision("safe_anchor_q_fs")
class SafeAnchorQUnforcedSwitchDecision(SafeAnchorQDecision):
    """``safe_anchor_q`` that gates the anchor's *unforced* switches on agreement.

    Base behavior is ``safe_anchor_q`` (anchor critic-Q argmax + safety layer).
    The one change: a *voluntary* switch (the anchor's Q-best action is a switch on
    a turn where a move is also legal) is only allowed when at least ``min_agree``
    non-anchor teammates agree that switch is the best possible switch:

      * Each teammate's best switch is its own argmax over the legal switches, by
        critic Q when available else actor prob (``agreement_basis``).
      * If a teammate corroborates the anchor's switch, take it.
      * Otherwise the switch is blocked and we stay in, taking the anchor's
        Q-best *move* instead.

    Forced switches (every legal action a switch) and turns where the anchor
    already prefers a move are untouched -- pure ``safe_anchor_q``. The outcome is
    recorded in ``diagnostics['unforced_switch']``.
    """

    def __init__(self, agreement_basis: str = "q", min_agree: int = 1, **kwargs: Any):
        super().__init__(**kwargs)
        if agreement_basis not in ("q", "prob"):
            raise ValueError("agreement_basis must be 'q' or 'prob'")
        self.agreement_basis = agreement_basis
        self.min_agree = int(min_agree)

    @staticmethod
    def _is_switch(action: int) -> bool:
        return 4 <= action <= 8

    def _member_top_switch(self, member: Any, switches: list[int]) -> int:
        if self.agreement_basis == "q":
            q = np.asarray(member.rollout_q(), dtype=np.float64)
            cand = [s for s in switches if s < len(q) and np.isfinite(q[s])]
            if cand:
                return int(max(cand, key=lambda s: float(q[s])))
        probs = np.asarray(member.rollout_probs(), dtype=np.float64)
        return int(max(switches, key=lambda s: float(probs[s])))

    def score_actions(self, context: EnsembleDecisionContext) -> np.ndarray:
        scores = super().score_actions(context)  # safe_anchor_q anchor-Q vector
        legal = context.legal_actions
        switches = [a for a in legal if self._is_switch(a)]
        moves = [a for a in legal if not self._is_switch(a)]
        # Nothing to gate: forced switch (no move available) or no switch at all.
        if not switches or not moves:
            return scores

        anchor_choice = int(max(legal, key=lambda a: float(scores[a])))
        if not self._is_switch(anchor_choice):
            return scores  # anchor wants to move; unforced switch never considered

        agree = sum(
            self._member_top_switch(member, switches) == anchor_choice
            for member in context.members
            if member.member_index != context.anchor_index
        )
        blocked = agree < self.min_agree
        context.diagnostics["unforced_switch"] = {
            "anchor_switch": anchor_choice,
            "agree_count": int(agree),
            "blocked": bool(blocked),
        }
        if not blocked:
            return scores

        # Block switching: demote every switch below the worst legal move so the
        # anchor's best *move* wins, while keeping the move ordering intact.
        scores = scores.copy()
        floor = min(float(scores[m]) for m in moves) - 1.0
        for s in switches:
            scores[s] = floor
        return scores
