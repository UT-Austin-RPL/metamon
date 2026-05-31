"""Opponent pool config for vectorized Showdown eval and training.

Same YAML shape as ladder self-play (``agents`` + ``defaults``) and h2h
``policies``. Reuses :mod:`metamon.rl.evaluate.common` for loading, merging,
and sampling — one shared opponent per env ``reset()``.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from metamon.env import get_metamon_teams
from metamon.env.wrappers import TeamSet
from metamon.rl.evaluate.common import (
    PolicySpec,
    load_config,
    merge_defaults,
    sample_policy_from_merged,
)


def parse_opponent_pool_dict(raw: Dict[str, Any]) -> List[Tuple[str, dict]]:
    """Parse ``agents`` / ``policies`` + ``defaults`` into (name, merged_config) rows."""
    defaults = raw.get("defaults", {})
    agents = raw.get("agents")
    if agents is None:
        agents = raw.get("policies")
    if not agents:
        raise ValueError(
            "Opponent pool config must define an 'agents' or 'policies' section"
        )

    rows: List[Tuple[str, dict]] = []
    for base_name, agent_config in agents.items():
        merged = merge_defaults(defaults, agent_config or {})
        if "model_name" not in merged:
            merged["model_name"] = base_name
        rows.append((base_name, merged))
    return rows


class OpponentPoolConfig:
    """Parsed opponent pool — sample one shared opponent per env ``reset()``."""

    def __init__(
        self,
        agents: List[Tuple[str, dict]],
        battle_format: str,
        rng: Optional[random.Random] = None,
    ):
        if not agents:
            raise ValueError("OpponentPoolConfig requires at least one agent entry")
        self.agents = agents
        self.battle_format = battle_format
        self.rng = rng or random.Random()
        self._team_cache: Dict[str, TeamSet] = {}

    @classmethod
    def from_dict(
        cls,
        raw: Dict[str, Any],
        battle_format: str,
        rng: Optional[random.Random] = None,
    ) -> "OpponentPoolConfig":
        return cls(
            agents=parse_opponent_pool_dict(raw),
            battle_format=battle_format,
            rng=rng,
        )

    def sample_opponent(self) -> PolicySpec:
        """Pick an agent, then sample checkpoint / temperature / team set."""
        name, merged = self.rng.choice(self.agents)
        return sample_policy_from_merged(name, merged)

    def team_set_for(self, team_set_name: str) -> TeamSet:
        if team_set_name not in self._team_cache:
            self._team_cache[team_set_name] = get_metamon_teams(
                self.battle_format, team_set_name
            )
        return self._team_cache[team_set_name]


def make_simple_opponent_pool_dict(
    opponent_agent: str,
    team_set: str = "competitive",
    checkpoint: Optional[int] = None,
    temperature: float = 1.0,
    battle_backend: str = "metamon",
) -> Dict[str, Any]:
    """Minimal one-agent pool (``--eval_type metamon --opponent_agent`` + ``--team_set``)."""
    return {
        "defaults": {
            "team_set": team_set,
            "battle_backend": battle_backend,
            "checkpoints": [checkpoint],
            "temperatures": [temperature],
        },
        "agents": {
            opponent_agent: {"model_name": opponent_agent},
        },
    }


def load_simple_opponent_pool(
    opponent_agent: str,
    battle_format: str,
    team_set: str = "competitive",
    checkpoint: Optional[int] = None,
    temperature: float = 1.0,
    battle_backend: str = "metamon",
    rng: Optional[random.Random] = None,
) -> OpponentPoolConfig:
    raw = make_simple_opponent_pool_dict(
        opponent_agent=opponent_agent,
        team_set=team_set,
        checkpoint=checkpoint,
        temperature=temperature,
        battle_backend=battle_backend,
    )
    return OpponentPoolConfig.from_dict(raw, battle_format=battle_format, rng=rng)


def load_opponent_pool(
    config_path: str,
    battle_format: str,
    template_vars: Optional[Dict[str, str]] = None,
) -> OpponentPoolConfig:
    raw = load_config(config_path, template_vars=template_vars)
    return OpponentPoolConfig.from_dict(raw, battle_format=battle_format)
