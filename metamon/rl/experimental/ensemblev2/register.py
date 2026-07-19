from __future__ import annotations

from typing import Type

from metamon.rl.experimental.ensemblev2.config import (
    EnsembleV2Config,
    load_ensemblev2_presets,
)

# Public nickname -> preset name (in presets.json).
_NICKNAMES: dict[str, str] = {
    "EnsembleV2Gen1": "gen1_kakuna_tauros_smallg1",
    "EnsembleV2SmallG1V3CkptBag": "smallg1v3_ckpt_bag",
    "EnsembleV2TaurosV1CkptBag": "taurosv1_ckpt_bag",
    "EnsembleV2TaurosV1SmallG1V3CkptBag": "taurosv1_smallg1v3_ckpt_bag",
}


def _make_nickname_class(
    nickname: str,
    base_cls: Type,
    config: EnsembleV2Config,
) -> Type:
    class NicknamedEnsembleV2(base_cls):
        CONFIG = config

    NicknamedEnsembleV2.__name__ = nickname
    NicknamedEnsembleV2.__qualname__ = nickname
    NicknamedEnsembleV2.__module__ = __name__
    return NicknamedEnsembleV2


def register_nickname_agents() -> None:
    from metamon.rl.pretrained import EnsembleV2Model, pretrained_model

    presets = load_ensemblev2_presets()
    for nickname, preset_name in _NICKNAMES.items():
        if preset_name not in presets:
            raise ValueError(
                f"Unknown EnsembleV2 preset '{preset_name}' for agent {nickname} "
                f"(available: {sorted(presets)})"
            )
        agent_cls = _make_nickname_class(
            nickname, EnsembleV2Model, presets[preset_name]
        )
        pretrained_model(nickname)(agent_cls)


register_nickname_agents()
