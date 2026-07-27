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


def coupled_player_specs(
    p1_name: str,
    p2_name: str,
    eval_team_set: Optional[Teambuilder],
    opp_team_set: Optional[Teambuilder],
    battle_format: str,
    eval_side: str,
) -> Tuple[Dict[str, Any], Optional[str], Dict[str, Any], Optional[str]]:
    """Build ``>player`` specs for both sides with coupled team-pool selection.

    Both sides draw from the **same team pool** per battle whenever possible, so
    PSRO-Lite collection never pairs a learner on one team composition against an
    opponent on a different composition (the mismatch that inflated per-opponent
    win-rate estimates relative to same-pool ladder play).

    Coupling rules (``WeightedMixedTeamSet`` is a schedule-driven mix; a plain
    ``TeamSet`` is pinned to one team set):

      * **Both sides are a mix** (the ``@schedule`` case for both player and
        opponent): one component index is sampled from the *eval* side's
        weights (so the learner's curriculum still drives the mix) and the
        opponent is drawn from its component whose ``team_file_dir`` matches —
        same pool for both.
      * **Eval is a mix, opponent is pinned** (a pool agent with a concrete
        ``team_set``, e.g. ``smogon_pass2``): the eval is drawn from the mix
        component matching the opponent's directory, so both share the
        opponent's pool. The pinned side dictates the pool because it cannot
        move.
      * **Opponent is a mix, eval is pinned**: symmetric — the opponent is
        drawn from the component matching the eval's directory.
      * **Neither is a mix** (both plain, or a side is ``None``), or a random
        battle format: both draw independently — the previous behavior (when
        both plain and from the same set, they already share the pool).

      When a match cannot be found (e.g. a pinned opponent whose set is not one
      of the mix's components), that side falls back to an independent draw;
      the residual mismatch is unavoidable without re-pinning the opponent.

    Args:
        p1_name/p2_name: Showdown player names for the two physical sides.
        eval_team_set: the evaluated (learner) side's ``TeamSet``.
        opp_team_set: the in-the-loop opponent's ``TeamSet``.
        battle_format: e.g. ``"gen1ou"`` (random formats carry no teams).
        eval_side: ``"p1"`` or ``"p2"`` — which physical side the eval side
            plays. The other side is the opponent.

    Returns:
        ``(p1_spec, p1_file, p2_spec, p2_file)`` where ``*_file`` is the source
        team file path (or ``None`` for random formats / missing team sets).
    """
    # Lazy import avoids any import-order dependency at module load time.
    from metamon.env.wrappers import WeightedMixedTeamSet

    # Random formats: the sim generates teams; neither side draws.
    if is_random_format(battle_format):
        return {"name": p1_name}, None, {"name": p2_name}, None

    eval_is_mix = isinstance(eval_team_set, WeightedMixedTeamSet)
    opp_is_mix = isinstance(opp_team_set, WeightedMixedTeamSet)

    if eval_is_mix and opp_is_mix:
        # Curriculum drives the pick; both sides draw from the same component.
        idx = eval_team_set.sample_component_index()
        eval_team = eval_team_set.yield_team_from_component(idx)
        eval_file = eval_team_set.most_recent_team_file
        shared_dir = getattr(eval_team_set.team_sets[idx], "team_file_dir", None)
        opp_idx = opp_team_set.index_for_team_file_dir(shared_dir)
        if opp_idx is not None:
            opp_team = opp_team_set.yield_team_from_component(opp_idx)
        else:
            # Components diverged (misconfigured schedules) → independent draw.
            opp_team = opp_team_set.yield_team()
        opp_file = opp_team_set.most_recent_team_file

    elif eval_is_mix and opp_team_set is not None and not opp_is_mix:
        # Opponent is pinned to one set; the eval matches its pool.
        opp_dir = getattr(opp_team_set, "team_file_dir", None)
        eval_idx = eval_team_set.index_for_team_file_dir(opp_dir)
        if eval_idx is not None:
            eval_team = eval_team_set.yield_team_from_component(eval_idx)
        else:
            # Opponent's set isn't one of the mix's components → can't couple.
            eval_team = eval_team_set.yield_team()
        eval_file = eval_team_set.most_recent_team_file
        opp_team = opp_team_set.yield_team()
        opp_file = opp_team_set.most_recent_team_file

    elif opp_is_mix and eval_team_set is not None and not eval_is_mix:
        # Eval is pinned; the opponent matches its pool.
        eval_dir = getattr(eval_team_set, "team_file_dir", None)
        opp_idx = opp_team_set.index_for_team_file_dir(eval_dir)
        if opp_idx is not None:
            opp_team = opp_team_set.yield_team_from_component(opp_idx)
        else:
            opp_team = opp_team_set.yield_team()
        opp_file = opp_team_set.most_recent_team_file
        eval_team = eval_team_set.yield_team()
        eval_file = eval_team_set.most_recent_team_file

    else:
        # Both plain (or a side is None): independent draws.
        eval_team = eval_team_set.yield_team() if eval_team_set is not None else None
        eval_file = (
            getattr(eval_team_set, "most_recent_team_file", None)
            if eval_team_set is not None
            else None
        )
        opp_team = opp_team_set.yield_team() if opp_team_set is not None else None
        opp_file = (
            getattr(opp_team_set, "most_recent_team_file", None)
            if opp_team_set is not None
            else None
        )

    # Assign the drawn teams to physical p1/p2 sides.
    if eval_side == "p1":
        p1_team, p1_file = eval_team, eval_file
        p2_team, p2_file = opp_team, opp_file
    else:
        p1_team, p1_file = opp_team, opp_file
        p2_team, p2_file = eval_team, eval_file

    p1_spec: Dict[str, Any] = {"name": p1_name}
    if p1_team is not None:
        p1_spec["team"] = p1_team
    p2_spec: Dict[str, Any] = {"name": p2_name}
    if p2_team is not None:
        p2_spec["team"] = p2_team
    return p1_spec, p1_file, p2_spec, p2_file
