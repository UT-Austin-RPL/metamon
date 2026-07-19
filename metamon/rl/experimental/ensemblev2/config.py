from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_PRESETS_PATH = Path(__file__).with_name("presets.json")


@dataclass(frozen=True)
class EnsembleV2MemberSpec:
    """Configuration for one inference-only EnsembleV2 member.

    Args:
        model_name: Registry name of a ``PretrainedModel`` (e.g. ``"TaurosV0"``).
        checkpoint: Checkpoint epoch to load. ``None`` uses the model's
            ``default_checkpoint``.
        anchor: Whether this member is the ensemble's anchor (the policy the
            baseline decider defers to). Exactly one member should set this; if
            none do, the first member is treated as the anchor.
    """

    model_name: str
    checkpoint: Optional[int] = None
    anchor: bool = False


@dataclass(frozen=True)
class EnsembleV2Config:
    """A named combination of EnsembleV2 members plus a decision strategy.

    Args:
        members: The ensemble members.
        name: Preset/config name (used for the agent name and feature-log path).
        decision: Registered :class:`EnsembleDecision` strategy name (see
            ``deciders.py``). Defaults to ``"anchor"`` (anchor passthrough).
        decision_kwargs: Optional kwargs forwarded to the decision strategy's
            constructor.
    """

    members: tuple[EnsembleV2MemberSpec, ...]
    name: str = "ensemblev2"
    decision: str = "anchor"
    decision_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.members:
            raise ValueError("EnsembleV2Config requires at least one member")
        anchors = [i for i, m in enumerate(self.members) if m.anchor]
        if len(anchors) > 1:
            raise ValueError(
                f"EnsembleV2Config '{self.name}' has multiple anchors at {anchors}; "
                "exactly one member may set anchor=True"
            )

    @property
    def anchor_index(self) -> int:
        for idx, member in enumerate(self.members):
            if member.anchor:
                return idx
        return 0


def member_prefix(index: int) -> str:
    """Canonical namespace prefix for a member's observation keys.

    Used by both :class:`MultiObservationSpace` (when building the combined obs
    the env produces) and :class:`EnsembleV2Policy` (when splitting that obs back
    out per member). Index-based so duplicate models/checkpoints stay distinct.
    """
    return f"m{index}"


def _spec_from_dict(raw: dict) -> EnsembleV2MemberSpec:
    return EnsembleV2MemberSpec(
        model_name=raw["model_name"],
        checkpoint=raw.get("checkpoint"),
        anchor=bool(raw.get("anchor", False)),
    )


def _members_from_same_model(body: dict) -> tuple[EnsembleV2MemberSpec, ...]:
    """Expand a same-model multi-checkpoint shorthand into member specs.

    Expected shape under ``same_model``::

        {
          "model_name": "SmallG1OnlineV3",
          "checkpoints": [250, 300, 400, 500, 525],
          "anchor_checkpoint": 300
        }

    ``anchor_checkpoint`` may be omitted (defaults to the first checkpoint).
    """
    same = body["same_model"]
    model_name = same["model_name"]
    checkpoints = list(same["checkpoints"])
    if not checkpoints:
        raise ValueError("same_model.checkpoints must be a non-empty list")
    anchor_ckpt = same.get("anchor_checkpoint", checkpoints[0])
    if anchor_ckpt not in checkpoints:
        raise ValueError(
            f"same_model.anchor_checkpoint {anchor_ckpt} is not in checkpoints "
            f"{checkpoints}"
        )
    return tuple(
        EnsembleV2MemberSpec(
            model_name=model_name,
            checkpoint=int(ckpt),
            anchor=(ckpt == anchor_ckpt),
        )
        for ckpt in checkpoints
    )


def _members_from_preset_body(body: dict) -> tuple[EnsembleV2MemberSpec, ...]:
    if "same_model" in body and "members" in body:
        raise ValueError("Preset cannot set both 'same_model' and 'members'; pick one")
    if "same_model" in body:
        return _members_from_same_model(body)
    if "members" not in body:
        raise ValueError("Preset requires either 'members' or 'same_model'")
    return tuple(_spec_from_dict(spec) for spec in body["members"])


def load_ensemblev2_presets() -> dict[str, EnsembleV2Config]:
    """Load all named EnsembleV2 configs from ``presets.json``."""
    raw_presets = json.loads(_PRESETS_PATH.read_text())
    presets: dict[str, EnsembleV2Config] = {}
    for name, body in raw_presets.items():
        members = _members_from_preset_body(body)
        presets[name] = EnsembleV2Config(
            members=members,
            name=name,
            decision=body.get("decision", "anchor"),
            decision_kwargs=dict(body.get("decision_kwargs", {})),
        )
    return presets


def get_ensemblev2_preset(name: str) -> EnsembleV2Config:
    presets = load_ensemblev2_presets()
    if name not in presets:
        raise ValueError(
            f"Unknown EnsembleV2 preset '{name}' (available: {sorted(presets)})"
        )
    return presets[name]
