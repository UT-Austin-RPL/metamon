from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

Team = tuple[int, ...]


@dataclass(frozen=True)
class PokemonSet:
    """One canonical set used to represent a Pokemon species in team search/simulation."""

    species: str
    moves: tuple[str, ...]
    showdown_set: str
    usage: float = 0.0
    ability: str | None = None


@dataclass(frozen=True)
class BattleExample:
    """One supervised training example for the team-vs-team win model."""

    team_A: Team
    team_B: Team
    y: int

    def swapped(self) -> "BattleExample":
        return BattleExample(team_A=self.team_B, team_B=self.team_A, y=1 - int(self.y))


def canonical_team(team: Iterable[int], team_size: int | None = None) -> Team:
    """Canonical team representation: sorted tuple of unique Pokemon IDs."""

    values = tuple(int(x) for x in team)
    if len(values) != len(set(values)):
        raise ValueError(f"Team contains duplicate Pokemon IDs: {values}")
    if team_size is not None and len(values) != team_size:
        raise ValueError(f"Expected team size {team_size}, got {len(values)}")
    return tuple(sorted(values))


def parse_int_team(raw: Sequence[int] | str, team_size: int | None = None) -> Team:
    """Parse comma/space-separated team IDs or pass-through existing integer sequences."""

    if isinstance(raw, str):
        parts = [p.strip() for p in raw.replace(",", " ").split() if p.strip()]
        values = [int(p) for p in parts]
    else:
        values = [int(x) for x in raw]
    return canonical_team(values, team_size=team_size)


def battle_example_to_json_dict(example: BattleExample) -> dict:
    return {
        "team_A": list(example.team_A),
        "team_B": list(example.team_B),
        "y": int(example.y),
    }


def battle_example_from_json_dict(data: dict) -> BattleExample:
    team_a = canonical_team(data["team_A"])
    team_b = canonical_team(data["team_B"])
    y = int(data["y"])
    if y not in (0, 1):
        raise ValueError(f"Label y must be 0/1, got {y}")
    return BattleExample(team_A=team_a, team_B=team_b, y=y)


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)
