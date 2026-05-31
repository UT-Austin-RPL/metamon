"""Resolve metamon team sets to Showdown ``>player`` specs.

``TeamSet.yield_team()`` already returns a Showdown *packed* team string, which is
exactly what ``BattleStream``'s ``>player p1 {"team": ...}`` command expects, so
the adapter is thin. Random-battle formats carry no team (the sim generates one),
so we omit the ``team`` field for them.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from poke_env.teambuilder import Teambuilder


def is_random_format(battle_format: str) -> bool:
    fmt = (battle_format or "").lower()
    return "random" in fmt or "factory" in fmt


def player_spec(
    name: str,
    team_set: Optional[Teambuilder],
    battle_format: str,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Build a ``>player`` spec dict and return the source team file (if any)."""
    spec: Dict[str, Any] = {"name": name}
    team_file: Optional[str] = None
    if team_set is not None and not is_random_format(battle_format):
        spec["team"] = team_set.yield_team()
        team_file = getattr(team_set, "most_recent_team_file", None)
    return spec, team_file
