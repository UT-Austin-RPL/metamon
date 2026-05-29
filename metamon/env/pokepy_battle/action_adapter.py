"""Map metamon UniversalAction indices to pokepy step encodings."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from pokepy.core.constants import (
    OFF_META,
    OFF_SIDE0,
    OFF_SIDE1,
    M_ACTIVE0,
    M_ACTIVE1,
    PHASE_FORCED_SWITCH,
)
from pokepy.core.state import MultiFormatState
from pokepy.data.loader import GameData, IDMappings
from pokepy.engine.action_mask import get_action_mask

from metamon.interface import (
    ActionSpace,
    UniversalAction,
    UniversalState,
    consistent_move_order,
    consistent_pokemon_order,
)
from metamon.env.pokepy_battle.state_adapter import _move_name, _species_name


def _active_slot(battle: np.ndarray, side: int) -> int:
    return int(battle[OFF_META + (M_ACTIVE0 if side == 0 else M_ACTIVE1)])


def _find_move_slot(
    state: MultiFormatState,
    side: int,
    move_name: str,
    mappings: IDMappings,
) -> int:
    active = _active_slot(state.battle_state, side)
    moves_arr = state.team_moves if side == 0 else state.opp_moves
    for slot in range(4):
        mid = int(moves_arr[active, slot])
        if mid >= 0 and _move_name(mid, mappings) == move_name:
            return slot
    return 0


def _find_team_slot_for_pokemon(
    state: MultiFormatState,
    side: int,
    pokemon_name: str,
    move_names: tuple[str, ...],
    mappings: IDMappings,
) -> int:
    species_arr = state.team_species if side == 0 else state.opp_species
    moves_arr = state.team_moves if side == 0 else state.opp_moves
    active = _active_slot(state.battle_state, side)
    candidates = []
    for slot in range(6):
        if slot == active:
            continue
        sid = int(species_arr[slot])
        if sid < 0:
            continue
        if _species_name(sid, mappings) != pokemon_name:
            continue
        slot_moves = tuple(
            _move_name(int(moves_arr[slot, j]), mappings)
            for j in range(4)
            if int(moves_arr[slot, j]) >= 0
        )
        candidates.append((slot, slot_moves))
    if not candidates:
        raise ValueError(f"Could not find team slot for switch target {pokemon_name!r}")
    if len(candidates) == 1:
        return candidates[0][0]
    for slot, slot_moves in candidates:
        if slot_moves == move_names:
            return slot
    return candidates[0][0]


def universal_action_to_pokepy(
    action: UniversalAction,
    universal_state: UniversalState,
    pokepy_state: MultiFormatState,
    mappings: IDMappings,
    *,
    player_side: int = 0,
) -> Tuple[int, bool]:
    """Resolve a metamon action index to (pokepy_action, wants_tera).

    Index semantics match ``UniversalAction.action_idx_to_BattleOrder`` in
    ``interface.py`` (maybe_valid_actions / definitely_valid_actions).
    """
    idx = int(action.action_idx)
    wants_tera = False
    if idx >= 9:
        wants_tera = True
        idx -= 9

    switch_options = consistent_pokemon_order(universal_state.available_switches)
    move_options = consistent_move_order(universal_state.player_active_pokemon.moves)

    # Move slot (indices 0-3) — only when not forced to switch.
    if idx <= 3 and not universal_state.forced_switch:
        if idx >= len(move_options):
            raise ValueError(
                f"move index {idx} out of range "
                f"(only {len(move_options)} moves available)"
            )
        move_slot = _find_move_slot(
            pokepy_state,
            player_side,
            move_options[idx].name,
            mappings,
        )
        return move_slot, wants_tera

    # Switch slot (indices 4-8).
    if 4 <= idx <= 8:
        switch_idx = idx - 4
        if switch_idx >= len(switch_options):
            raise ValueError(
                f"switch index {switch_idx} out of range "
                f"(only {len(switch_options)} switches available)"
            )
        target = switch_options[switch_idx]
        move_names = tuple(m.name for m in target.moves if m.name != "nomove")
        team_slot = _find_team_slot_for_pokemon(
            pokepy_state,
            player_side,
            target.name,
            move_names,
            mappings,
        )
        return 4 + team_slot, False

    raise ValueError(
        f"action index {idx} invalid "
        f"(forced_switch={universal_state.forced_switch})"
    )


def _pokepy_legal_for_agent_action(
    pokepy_action: int,
    wants_tera: bool,
    *,
    universal_state: UniversalState,
    pokepy_state: MultiFormatState,
    pokepy_mask: np.ndarray,
) -> bool:
    if not (0 <= pokepy_action < len(pokepy_mask)) or not bool(
        pokepy_mask[pokepy_action]
    ):
        return False
    if pokepy_action < 4 and wants_tera and not universal_state.can_tera:
        return False
    return True


def build_illegal_actions_mask(
    action_space: ActionSpace,
    universal_state: UniversalState,
    pokepy_state: MultiFormatState,
    game_data: GameData,
    mappings: IDMappings,
    *,
    player_side: int = 0,
) -> np.ndarray:
    """Build metamon illegal_actions mask (True = illegal).

    Candidate actions follow ``UniversalAction.maybe_valid_actions``; each
    candidate is kept only if it converts and is legal in pokepy (the pokepy
    analogue of ``definitely_valid_actions`` via Showdown ``BattleOrder``).
    """
    n = action_space.gym_space.n
    illegal = np.ones(n, dtype=bool)
    pokepy_mask = get_action_mask(pokepy_state, player_side, game_data)

    for action in UniversalAction.maybe_valid_actions(universal_state):
        idx = int(action.action_idx)
        if idx < 0 or idx >= n:
            continue
        try:
            pokepy_action, wants_tera = universal_action_to_pokepy(
                action,
                universal_state,
                pokepy_state,
                mappings,
                player_side=player_side,
            )
        except (ValueError, IndexError):
            continue

        if _pokepy_legal_for_agent_action(
            pokepy_action,
            wants_tera,
            universal_state=universal_state,
            pokepy_state=pokepy_state,
            pokepy_mask=pokepy_mask,
        ):
            illegal[idx] = False
    return illegal


def legal_action_indices(
    action_space: ActionSpace,
    universal_state: UniversalState,
    pokepy_state: MultiFormatState,
    game_data: GameData,
    mappings: IDMappings,
    *,
    player_side: int = 0,
) -> list[int]:
    illegal = build_illegal_actions_mask(
        action_space,
        universal_state,
        pokepy_state,
        game_data,
        mappings,
        player_side=player_side,
    )
    return [i for i in range(action_space.gym_space.n) if not illegal[i]]
