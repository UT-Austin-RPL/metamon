from __future__ import annotations

import json
from pathlib import Path
from typing import Type

import yaml

from metamon.rl.experimental.ensemble import EnsembleMemberSpec

_AGENTS_PATH = Path(__file__).with_name("agents.yaml")
_PRESETS_PATH = Path(__file__).with_name("ensemble_presets.json")

_FAMILY_BASES: dict[str, Type] = {}


def _load_presets() -> dict[str, list[EnsembleMemberSpec]]:
    raw_presets = json.loads(_PRESETS_PATH.read_text())
    return {
        name: [
            EnsembleMemberSpec(
                **{
                    **spec,
                    "proposal_roles": tuple(spec.get("proposal_roles", [])),
                }
            )
            for spec in member_specs
        ]
        for name, member_specs in raw_presets.items()
    }


def _make_nickname_class(
    nickname: str,
    base_cls: Type,
    member_specs: list[EnsembleMemberSpec],
) -> Type:
    class NicknamedEnsemble(base_cls):
        MEMBER_SPECS = member_specs

        @classmethod
        def _member_specs_from_env(cls) -> list[EnsembleMemberSpec]:
            return cls.MEMBER_SPECS

    NicknamedEnsemble.__name__ = nickname
    NicknamedEnsemble.__qualname__ = nickname
    NicknamedEnsemble.__module__ = __name__
    return NicknamedEnsemble


def register_nickname_agents() -> None:
    from metamon.rl.pretrained import (
        KakunaEnsemble,
        TaurosEnsemble,
        pretrained_model,
    )

    global _FAMILY_BASES
    _FAMILY_BASES = {
        "kakuna": KakunaEnsemble,
        "tauros": TaurosEnsemble,
    }

    presets = _load_presets()
    config = yaml.safe_load(_AGENTS_PATH.read_text())
    for nickname, spec in config["agents"].items():
        family = spec["family"]
        preset_name = spec["preset"]
        if family not in _FAMILY_BASES:
            raise ValueError(f"Unknown ensemble family '{family}' for agent {nickname}")
        if preset_name not in presets:
            raise ValueError(
                f"Unknown preset '{preset_name}' for agent {nickname} "
                f"(available: {sorted(presets)})"
            )
        base_cls = _FAMILY_BASES[family]
        agent_cls = _make_nickname_class(nickname, base_cls, presets[preset_name])
        pretrained_model(nickname)(agent_cls)


register_nickname_agents()
