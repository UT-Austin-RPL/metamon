"""Opponent pool config for vectorized Showdown eval and training.

Same YAML shape as ladder self-play (``agents`` + ``defaults``) and h2h
``policies``. Reuses :mod:`metamon.rl.evaluate.common` for loading, merging,
and sampling — one shared opponent per env ``reset()``.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

from metamon.env import get_metamon_team_set_or_mix, get_metamon_team_set_from_schedule
from metamon.env.wrappers import TeamSet
from metamon.rl.evaluate.common import (
    PolicySpec,
    expand_agent_pool_entries,
    load_config,
    merge_defaults,
    sample_policy_from_merged,
)


def parse_opponent_pool_dict(raw: Dict[str, Any]) -> List[Tuple[str, dict]]:
    """Parse ``agents`` / ``policies`` + ``defaults`` into (name, merged_config) rows.

    Each agent is expanded by its merged ``num_agents`` field (default 1). A value
    of ``N`` adds ``N`` equally-weighted rows, matching ladder self-play expansion.
    """
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
        rows.extend(expand_agent_pool_entries(base_name, merged))
    return rows


class OpponentPoolConfig:
    """Parsed opponent pool — sample one shared opponent per env ``reset()``.

    Optional ``weights`` (aligned with ``agents``) reweight the row sampling for
    PSRO-Lite prioritized collection. ``None`` ⇒ uniform (the default, so all
    existing call sites are unchanged).
    """

    def __init__(
        self,
        agents: List[Tuple[str, dict]],
        battle_format: str,
        rng: Optional[random.Random] = None,
        weights: Optional[List[float]] = None,
        team_schedule: Optional["TeamMixSchedule"] = None,
        epoch_ref: Optional["EpochRef"] = None,
    ):
        if not agents:
            raise ValueError("OpponentPoolConfig requires at least one agent entry")
        self.agents = agents
        self.battle_format = battle_format
        self.rng = rng or random.Random()
        self._team_cache: Dict[str, TeamSet] = {}
        self._weights: Optional[List[float]] = None
        self.set_weights(weights)
        # Optional schedule-aware team sets: when an agent's team_set is the
        # marker string "@schedule", team_set_for() returns a WeightedMixedTeamSet
        # that lazily follows this schedule + epoch_ref.
        self._team_schedule = team_schedule
        self._epoch_ref = epoch_ref

    @classmethod
    def from_dict(
        cls,
        raw: Dict[str, Any],
        battle_format: str,
        rng: Optional[random.Random] = None,
        team_schedule=None,
        epoch_ref=None,
    ) -> "OpponentPoolConfig":
        return cls(
            agents=parse_opponent_pool_dict(raw),
            battle_format=battle_format,
            rng=rng,
            team_schedule=team_schedule,
            epoch_ref=epoch_ref,
        )

    def set_weights(self, weights: Optional[List[float]]) -> None:
        """Set per-row sampling weights (aligned with ``self.agents``).

        ``None`` resets to uniform. Validates length and normalizes. Non-finite
        or all-zero inputs fall back to uniform so the env never hard-fails on
        weighting.
        """
        if weights is None:
            self._weights = None
            return
        if len(weights) != len(self.agents):
            raise ValueError(
                f"weights length {len(weights)} != agents length "
                f"{len(self.agents)}"
            )
        w = [float(x) for x in weights]
        if any(not math.isfinite(x) or x < 0.0 for x in w) or sum(w) <= 0.0:
            self._weights = None
            return
        total = sum(w)
        self._weights = [x / total for x in w]

    @property
    def weights(self) -> Optional[List[float]]:
        return None if self._weights is None else list(self._weights)

    def sample_opponent(self) -> PolicySpec:
        """Pick an agent, then sample checkpoint / temperature / team set."""
        if self._weights is None:
            name, merged = self.rng.choice(self.agents)
        else:
            name, merged = self.rng.choices(
                self.agents, weights=self._weights, k=1
            )[0]
        return sample_policy_from_merged(name, merged)

    def sample_opponent_for_agent(self, name: str) -> PolicySpec:
        """Sample checkpoint / temperature / team set for a *specific* agent row.

        Used by the PSRO-Lite quota phase to guarantee a chosen agent row gets
        played. Raises ``KeyError`` if ``name`` is not a row in ``self.agents``.
        """
        for nm, merged in self.agents:
            if nm == name:
                return sample_policy_from_merged(nm, merged)
        raise KeyError(f"Agent {name!r} not in pool")

    def team_set_for(self, team_set_name: str) -> TeamSet:
        if team_set_name == "@schedule":
            if self._team_schedule is None or self._epoch_ref is None:
                raise ValueError(
                    "opponent pool team_set '@schedule' requires a TeamMixSchedule "
                    "and EpochRef (pass --train_team_schedule to online_rl)"
                )
            if "@schedule" not in self._team_cache:
                self._team_cache["@schedule"] = get_metamon_team_set_from_schedule(
                    self.battle_format, self._team_schedule, self._epoch_ref
                )
            return self._team_cache["@schedule"]
        if team_set_name not in self._team_cache:
            self._team_cache[team_set_name] = get_metamon_team_set_or_mix(
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
    team_schedule=None,
    epoch_ref=None,
) -> OpponentPoolConfig:
    raw = load_config(config_path, template_vars=template_vars)
    return OpponentPoolConfig.from_dict(
        raw, battle_format=battle_format, team_schedule=team_schedule, epoch_ref=epoch_ref
    )
