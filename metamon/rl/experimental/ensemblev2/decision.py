from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Type

import numpy as np

from metamon.rl.experimental.ensemblev2.action_remap import CANONICAL_ACTION_DIM


@dataclass
class MemberStepFeatures:
    """Everything one member contributes about one decision, in canonical indices.

    ``probs`` and ``q_values`` are indexed ``[gamma, universal_action_idx]`` where
    the universal action layout is the 13-wide ``DefaultActionSpace`` (0-3 moves,
    4-8 switches, 9-12 tera-moves). Probabilities for illegal / inexpressible
    actions are 0.0; Q-values for actions the member cannot express are NaN.
    """

    member_index: int
    model_name: str
    checkpoint: Optional[int]
    action_space_name: str
    is_anchor: bool
    gammas: list[float]
    probs: np.ndarray  # (num_gammas, 13)
    q_values: np.ndarray  # (num_gammas, 13) — mean over the critic ensemble
    # std across this member's own critic ensemble, same layout as q_values.
    # NaN where inexpressible; may be all-NaN if Q gathering is disabled.
    q_std: Optional[np.ndarray] = None  # (num_gammas, 13)

    def rollout_probs(self) -> np.ndarray:
        """Action distribution at the test-time (last) gamma."""
        return self.probs[-1]

    def rollout_q(self) -> np.ndarray:
        """Q-values (critic-ensemble mean) at the test-time (last) gamma."""
        return self.q_values[-1]

    def rollout_q_std(self) -> Optional[np.ndarray]:
        """Critic-ensemble std at the test-time (last) gamma (None if absent)."""
        return None if self.q_std is None else self.q_std[-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "member_index": self.member_index,
            "model_name": self.model_name,
            "checkpoint": self.checkpoint,
            "action_space_name": self.action_space_name,
            "is_anchor": self.is_anchor,
            "gammas": list(self.gammas),
            "probs": np.nan_to_num(self.probs, nan=0.0).tolist(),
            "q_values": [
                [None if math.isnan(v) else float(v) for v in row]
                for row in self.q_values.tolist()
            ],
            "q_std": (
                None
                if self.q_std is None
                else [
                    [None if math.isnan(v) else float(v) for v in row]
                    for row in self.q_std.tolist()
                ]
            ),
        }


@dataclass
class EnsembleDecisionContext:
    """All information available when choosing one action for one battle lane.

    The decider sees the current per-member features plus the full per-battle
    history of prior step records (same information, one dict per earlier turn).
    """

    turn_idx: int
    legal_actions: list[int]  # canonical universal action indices
    members: list[MemberStepFeatures]
    anchor_index: int
    prev_reward: float
    history: list[dict[str, Any]] = field(default_factory=list)
    # Compact, coarse signature of the current battle state (anchor obs hash).
    # Lets stateless deciders detect repeated states across turns (cycle/stall
    # breaking) by comparing against ``state_hash`` entries in ``history``.
    state_hash: Optional[str] = None
    # Opponent Pokemon remaining (0-6), read straight off the UniversalState via
    # the ``ensemble/opponents_remaining`` obs key. None if unavailable.
    opponents_remaining: Optional[float] = None
    # Scratch space a decider may populate to record *why* it chose an action
    # (e.g. safety overrides). Included in ``step_record`` when non-empty so the
    # logs can be mined for how often each mechanism fired.
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def anchor(self) -> MemberStepFeatures:
        return self.members[self.anchor_index]

    def step_record(self, chosen_action: int) -> dict[str, Any]:
        """JSON-serializable snapshot of this decision (for logging + history)."""
        record = {
            "turn_idx": self.turn_idx,
            "legal_actions": list(self.legal_actions),
            "prev_reward": self.prev_reward,
            "anchor_index": self.anchor_index,
            "state_hash": self.state_hash,
            "opponents_remaining": self.opponents_remaining,
            "chosen_action": int(chosen_action),
            "members": [member.to_dict() for member in self.members],
        }
        if self.diagnostics:
            record["diagnostics"] = self.diagnostics
        return record


# --------------------------------------------------------------------------- #
# Pluggable decision strategies (registry mirrors action spaces / reward funcs) #
# --------------------------------------------------------------------------- #

ALL_ENSEMBLE_DECISIONS: dict[str, Type["EnsembleDecision"]] = {}


def register_ensemble_decision(name: Optional[str] = None):
    """Decorator to register an :class:`EnsembleDecision` strategy class.

    Usage::

        @register_ensemble_decision("anchor")
        class AnchorPassthroughDecision(EnsembleDecision):
            ...
    """

    def _register(cls: Type["EnsembleDecision"]):
        decision_name = name if name is not None else cls.__name__
        if decision_name in ALL_ENSEMBLE_DECISIONS:
            raise ValueError(
                f"Ensemble decision '{decision_name}' is already registered!"
            )
        ALL_ENSEMBLE_DECISIONS[decision_name] = cls
        return cls

    return _register


def get_ensemble_decision_names() -> list[str]:
    """All registered ensemble decision strategy names."""
    return sorted(ALL_ENSEMBLE_DECISIONS.keys())


def get_ensemble_decision(name: str, **kwargs: Any) -> "EnsembleDecision":
    """Instantiate a registered ensemble decision strategy by name."""
    if name not in ALL_ENSEMBLE_DECISIONS:
        raise ValueError(
            f"Unknown ensemble decision '{name}' "
            f"(available: {get_ensemble_decision_names()})"
        )
    return ALL_ENSEMBLE_DECISIONS[name](**kwargs)


class EnsembleDecision(ABC):
    """Base class for ensemble action-selection strategies.

    A strategy maps an :class:`EnsembleDecisionContext` (every member's per-gamma
    probs + Q-values on canonical action indices, the legal actions, and the
    per-battle history) to a single chosen canonical ``UniversalAction`` index.
    ``__call__`` is the decision point.

    Subclass, implement :meth:`__call__`, and register with
    :func:`register_ensemble_decision`; then select the strategy via the ensemble
    config's ``decision`` field (and optional ``decision_kwargs``). Strategies
    should be stateless across calls -- any per-battle memory lives in
    ``context.history`` -- because a single instance serves all batched lanes.
    """

    def __init__(self, **kwargs: Any):
        # Subclasses may accept configuration kwargs; the base ignores extras so
        # configs can pass through ``decision_kwargs`` uniformly.
        self.config_kwargs = kwargs

    @abstractmethod
    def __call__(self, context: EnsembleDecisionContext) -> int:
        """Return the chosen canonical universal action index."""
        raise NotImplementedError
