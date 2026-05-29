"""pokepy MultiFormatState -> metamon UniversalState."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from pokepy.core.constants import (
    OFF_SIDE0,
    OFF_SIDE1,
    OFF_FIELD,
    OFF_META,
    POKEMON_SIZE,
    M_ACTIVE0,
    M_ACTIVE1,
    PHASE_FORCED_SWITCH,
    F_WEATHER,
    F_TERRAIN,
    F_LAST_MOVE_0,
    F_LAST_MOVE_1,
    F_SCREENS_0,
    F_SCREENS_1,
    F_HAZARDS_0,
    F_HAZARDS_1,
    WEATHER_NAMES,
    TERRAIN_NAMES,
    STATUS_NAMES,
    TYPE_NAMES,
)
from pokepy.core.bitpack import extract_boost, get_status, get_spikes_layers
from pokepy.core.state import MultiFormatState
from pokepy.data.loader import GameData, IDMappings
from pokepy.obs.universal import clean_name, clean_no_numbers

from metamon.interface import UniversalMove, UniversalPokemon, UniversalState


def _species_name(idx: int, mappings: IDMappings) -> str:
    if idx < 0:
        return "<blank>"
    name = mappings.species_names.get(int(idx), "")
    return clean_name(name) if name else "<blank>"


def _move_name(idx: int, mappings: IDMappings) -> str:
    if idx < 0:
        return "nomove"
    name = mappings.move_names.get(int(idx), "")
    return clean_name(name) if name else "nomove"


def _item_name(idx: int, mappings: IDMappings) -> str:
    if idx < 0:
        return "noitem"
    name = mappings.item_names.get(int(idx), "")
    return clean_name(name) if name else "noitem"


def _ability_name(idx: int, mappings: IDMappings) -> str:
    if idx < 0:
        return "noability"
    name = mappings.ability_names.get(int(idx), "")
    return clean_name(name) if name else "noability"


def _type_name(idx: int) -> str:
    if idx < 0 or idx >= len(TYPE_NAMES):
        return "notype"
    return clean_name(TYPE_NAMES[idx])


def _types_string(type1: int, type2: int) -> str:
    t1 = _type_name(type1)
    t2 = _type_name(type2) if type2 >= 0 else "notype"
    return " ".join(sorted([t1, t2]))


def _weather_name(idx: int) -> str:
    if 0 <= idx < len(WEATHER_NAMES) and WEATHER_NAMES[idx]:
        return WEATHER_NAMES[idx]
    return "noweather"


def _terrain_name(idx: int) -> str:
    if 0 <= idx < len(TERRAIN_NAMES) and TERRAIN_NAMES[idx]:
        return TERRAIN_NAMES[idx]
    return "nofield"


def _status_name(idx: int) -> str:
    if 0 <= idx < len(STATUS_NAMES) and STATUS_NAMES[idx]:
        return STATUS_NAMES[idx]
    return "nostatus"


def _build_move(
    move_id: int,
    current_pp: int,
    max_pp: int,
    game_data: GameData,
    mappings: IDMappings,
) -> UniversalMove:
    if move_id < 0:
        return UniversalMove.blank_move()
    return UniversalMove(
        name=_move_name(move_id, mappings),
        move_type=_type_name(int(game_data.move_type[move_id])),
        category=str(int(game_data.move_category[move_id])),
        base_power=int(game_data.move_base_power[move_id]),
        accuracy=float(int(game_data.move_accuracy[move_id])) / 100.0,
        priority=int(game_data.move_priority[move_id]),
        current_pp=int(current_pp),
        max_pp=int(max_pp) if int(max_pp) > 0 else int(game_data.move_pp[move_id]),
    )


def _live_stats_from_battle_offset(battle: np.ndarray, poff: int) -> dict:
    """Read computed in-battle stats from a pokepy battle_state pokemon slot."""
    return {
        "hp": int(battle[poff + 2]),
        "atk": int(battle[poff + 7]),
        "def": int(battle[poff + 8]),
        "spa": int(battle[poff + 9]),
        "spd": int(battle[poff + 10]),
        "spe": int(battle[poff + 11]),
    }


def _build_static_pokemon(
    slot: int,
    species_arr: np.ndarray,
    moves_arr: np.ndarray,
    items_arr: np.ndarray,
    abilities_arr: np.ndarray,
    tera_arr: np.ndarray,
    pp_arr: np.ndarray,
    game_data: GameData,
    mappings: IDMappings,
    *,
    hp_pct: float = 1.0,
    status: str = "nostatus",
    boosts: Optional[List[int]] = None,
    live_stats: Optional[dict] = None,
) -> UniversalPokemon:
    species_id = int(species_arr[slot])
    if species_id < 0:
        return UniversalPokemon.from_dict(
            {
                "name": "<blank>",
                "base_species": "<blank>",
                "hp_pct": 0.0,
                "types": "notype notype",
                "item": "noitem",
                "ability": "noability",
                "lvl": 100,
                "status": "nostatus",
                "effect": "noeffect",
                "moves": [],
                "tera_type": "notype",
            }
        )
    bs = game_data.species_base_stats[species_id]
    types = game_data.species_types[species_id]
    type1 = int(types[0])
    type2 = int(types[1])
    mon_moves: List[UniversalMove] = []
    for j in range(4):
        mid = int(moves_arr[slot, j])
        mon_moves.append(
            _build_move(
                mid, int(pp_arr[slot, j]), int(pp_arr[slot, j]), game_data, mappings
            )
        )
    if boosts is None:
        boosts = [0] * 7
    stat_fields = UniversalPokemon._stat_fields(live_stats or {})
    return UniversalPokemon(
        name=_species_name(species_id, mappings),
        base_species=_species_name(species_id, mappings),
        hp_pct=float(hp_pct),
        types=_types_string(type1, type2),
        item=_item_name(int(items_arr[slot]), mappings),
        ability=_ability_name(int(abilities_arr[slot]), mappings),
        lvl=100,
        status=status,
        effect="noeffect",
        moves=mon_moves,
        atk_boost=boosts[0],
        def_boost=boosts[1],
        spa_boost=boosts[2],
        spd_boost=boosts[3],
        spe_boost=boosts[4],
        accuracy_boost=boosts[5],
        evasion_boost=boosts[6],
        base_hp=int(bs[0]),
        base_atk=int(bs[1]),
        base_def=int(bs[2]),
        base_spa=int(bs[3]),
        base_spd=int(bs[4]),
        base_spe=int(bs[5]),
        tera_type=(
            _type_name(int(tera_arr[slot])) if int(tera_arr[slot]) >= 0 else "notype"
        ),
        **stat_fields,
    )


def _build_active_pokemon_from_battle(
    state: MultiFormatState,
    side: int,
    game_data: GameData,
    mappings: IDMappings,
    *,
    fog_of_war: bool = False,
) -> UniversalPokemon:
    """Build the active Pokemon's UniversalState view.

    When ``fog_of_war`` is True the Pokemon is being observed by the *opposing*
    player, so hidden information is masked to match the Showdown/poke-env
    training distribution (see ``UniversalPokemon.from_Pokemon``): the item and
    ability read as ``unknownitem`` / ``unknownability`` until revealed, only
    moves the engine has flagged as revealed are exposed, and the exact computed
    stats are dropped (``MISSING_STAT``). Visible-on-field attributes — species,
    types, HP%, level, status, stat boosts, and (once terastallized) tera type —
    are always shown. The player's own active Pokemon is built with
    ``fog_of_war=False`` so it retains full information.
    """
    battle = state.battle_state
    side_base = OFF_SIDE0 if side == 0 else OFF_SIDE1
    active_idx = int(battle[OFF_META + (M_ACTIVE0 if side == 0 else M_ACTIVE1)])
    if active_idx < 0:
        return UniversalPokemon.from_dict(
            {
                "name": "<blank>",
                "base_species": "<blank>",
                "hp_pct": 0.0,
                "types": "notype notype",
                "item": "noitem",
                "ability": "noability",
                "lvl": 100,
                "status": "nostatus",
                "effect": "noeffect",
                "moves": [],
                "tera_type": "notype",
            }
        )

    poff = side_base + active_idx * POKEMON_SIZE
    species_id = int(battle[poff + 0])
    cur_hp = int(battle[poff + 1])
    max_hp = int(battle[poff + 2])
    level = int(battle[poff + 3])
    types_packed = int(battle[poff + 4])
    type1 = types_packed & 0xFF
    type2 = (types_packed >> 8) & 0xFF
    if type2 == 0xFF or type2 == 255:
        type2 = -1
    ability_id = int(battle[poff + 5])
    item_id = int(battle[poff + 6])
    status_field = int(battle[poff + 12])
    boosts13 = int(battle[poff + 13])
    boosts14 = int(battle[poff + 14])
    flags = int(battle[poff + 15])

    atk_boost = extract_boost(boosts13, 0)
    def_boost = extract_boost(boosts13, 4)
    spa_boost = extract_boost(boosts13, 8)
    spd_boost = extract_boost(boosts13, 12)
    spe_boost = extract_boost(boosts14, 0)
    acc_boost = extract_boost(boosts14, 4)
    eva_boost = extract_boost(boosts14, 8)
    tera_type_idx = (boosts14 >> 12) & 0xF
    has_tera = (flags & 0x4) != 0

    status_id = get_status(status_field)

    moves_arr = state.team_moves if side == 0 else state.opp_moves
    pp_arr = state.team_pp if side == 0 else state.opp_pp
    # team_*_revealed track side 0's reveals; opp_*_revealed track side 1's.
    moves_revealed_arr = (
        state.team_moves_revealed if side == 0 else state.opp_moves_revealed
    )
    mon_moves: List[UniversalMove] = []
    for j in range(4):
        mid = int(moves_arr[active_idx, j])
        if fog_of_war and not bool(moves_revealed_arr[active_idx, j]):
            # Opponent move not yet revealed: hide it entirely (matches
            # poke-env, where unrevealed moves are simply absent).
            continue
        mon_moves.append(
            _build_move(
                mid,
                int(pp_arr[active_idx, j]),
                int(pp_arr[active_idx, j]),
                game_data,
                mappings,
            )
        )

    bs = game_data.species_base_stats[species_id]
    if fog_of_war:
        # Showdown reveals an opponent's item/ability only once it has actually
        # activated (an `-item`/`-ability` protocol message). The engine sets
        # the per-slot reveal arrays at those activation sites; until then the
        # field reads as unknown. Exact computed stats are never public, so
        # they stay MISSING. team_* track side-0's mons, opp_* track side-1's.
        item_revealed_arr = (
            state.team_items_revealed if side == 0 else state.opp_items_revealed
        )
        ability_revealed_arr = (
            state.team_abilities_revealed if side == 0 else state.opp_abilities_revealed
        )
        item_name = (
            _item_name(item_id, mappings)
            if bool(item_revealed_arr[active_idx])
            else "unknownitem"
        )
        ability_name = (
            _ability_name(ability_id, mappings)
            if bool(ability_revealed_arr[active_idx])
            else "unknownability"
        )
        stat_fields = UniversalPokemon._stat_fields(UniversalPokemon.MISSING_STATS)
    else:
        item_name = _item_name(item_id, mappings)
        ability_name = _ability_name(ability_id, mappings)
        stat_fields = UniversalPokemon._stat_fields(
            _live_stats_from_battle_offset(battle, poff)
        )
    return UniversalPokemon(
        name=_species_name(species_id, mappings),
        base_species=_species_name(species_id, mappings),
        hp_pct=(cur_hp / max_hp) if max_hp > 0 else 0.0,
        types=_types_string(type1, type2),
        item=item_name,
        ability=ability_name,
        lvl=level,
        status=_status_name(status_id),
        effect="noeffect",
        moves=mon_moves,
        atk_boost=atk_boost,
        def_boost=def_boost,
        spa_boost=spa_boost,
        spd_boost=spd_boost,
        spe_boost=spe_boost,
        accuracy_boost=acc_boost,
        evasion_boost=eva_boost,
        base_hp=int(bs[0]),
        base_atk=int(bs[1]),
        base_def=int(bs[2]),
        base_spa=int(bs[3]),
        base_spd=int(bs[4]),
        base_spe=int(bs[5]),
        tera_type=_type_name(tera_type_idx) if has_tera else "notype",
        **stat_fields,
    )


def _conditions_string(side_screens: int, side_hazards: int) -> str:
    if side_screens & 0x7:
        return "reflect"
    if (side_screens >> 3) & 0x7:
        return "lightscreen"
    if (side_screens >> 6) & 0x7:
        return "auroraveil"
    if (side_screens >> 9) & 0x7:
        return "tailwind"
    if get_spikes_layers(side_hazards):
        return "spikes"
    if (side_hazards >> 2) & 0x1:
        return "stealthrock"
    return "noconditions"


def pokepy_state_to_universal(
    state: MultiFormatState,
    game_data: GameData,
    mappings: IDMappings,
    *,
    format_str: str = "gen9ou",
    player_side: int = 0,
) -> UniversalState:
    """Convert pokepy state to metamon UniversalState."""
    battle = state.battle_state
    opp_side = 1 - player_side

    player = _build_active_pokemon_from_battle(state, player_side, game_data, mappings)
    # Opponent's active Pokemon is partially observed: mask hidden information
    # (item/ability/unrevealed moves/exact stats) to preserve fog of war.
    opponent = _build_active_pokemon_from_battle(
        state, opp_side, game_data, mappings, fog_of_war=True
    )

    side_base = OFF_SIDE0 if player_side == 0 else OFF_SIDE1
    active_idx = int(battle[OFF_META + (M_ACTIVE0 if player_side == 0 else M_ACTIVE1)])

    species_arr = state.team_species if player_side == 0 else state.opp_species
    moves_arr = state.team_moves if player_side == 0 else state.opp_moves
    items_arr = state.team_items if player_side == 0 else state.opp_items
    abilities_arr = state.team_abilities if player_side == 0 else state.opp_abilities
    tera_arr = state.team_tera if player_side == 0 else state.opp_tera
    pp_arr = state.team_pp if player_side == 0 else state.opp_pp

    available_switches: List[UniversalPokemon] = []
    for slot in range(6):
        if slot == active_idx:
            continue
        if int(species_arr[slot]) < 0:
            continue
        poff = side_base + slot * POKEMON_SIZE
        cur_hp = int(battle[poff + 1])
        max_hp = int(battle[poff + 2])
        flags = int(battle[poff + 15])
        fainted = (flags & 0x1) != 0
        if fainted or cur_hp == 0:
            continue
        available_switches.append(
            _build_static_pokemon(
                slot,
                species_arr,
                moves_arr,
                items_arr,
                abilities_arr,
                tera_arr,
                pp_arr,
                game_data,
                mappings,
                hp_pct=(cur_hp / max_hp) if max_hp > 0 else 0.0,
                live_stats=_live_stats_from_battle_offset(battle, poff),
            )
        )

    opponents_remaining = 0
    opp_side_base = OFF_SIDE1 if player_side == 0 else OFF_SIDE0
    for slot in range(6):
        poff = opp_side_base + slot * POKEMON_SIZE
        if int(battle[poff + 0]) < 0:
            continue
        flags = int(battle[poff + 15])
        if (flags & 0x1) == 0:
            opponents_remaining += 1

    weather = _weather_name(int(battle[OFF_FIELD + F_WEATHER]))
    battle_field = _terrain_name(int(battle[OFF_FIELD + F_TERRAIN]))

    p_screens = int(
        battle[OFF_FIELD + (F_SCREENS_0 if player_side == 0 else F_SCREENS_1)]
    )
    p_hazards = int(
        battle[OFF_FIELD + (F_HAZARDS_0 if player_side == 0 else F_HAZARDS_1)]
    )
    o_screens = int(
        battle[OFF_FIELD + (F_SCREENS_1 if player_side == 0 else F_SCREENS_0)]
    )
    o_hazards = int(
        battle[OFF_FIELD + (F_HAZARDS_1 if player_side == 0 else F_HAZARDS_0)]
    )
    player_conditions = _conditions_string(p_screens, p_hazards)
    opponent_conditions = _conditions_string(o_screens, o_hazards)

    p_last_id = int(
        battle[OFF_FIELD + (F_LAST_MOVE_0 if player_side == 0 else F_LAST_MOVE_1)]
    )
    o_last_id = int(
        battle[OFF_FIELD + (F_LAST_MOVE_1 if player_side == 0 else F_LAST_MOVE_0)]
    )
    player_prev_move = (
        _build_move(p_last_id, 0, 0, game_data, mappings)
        if p_last_id >= 0
        else UniversalMove.blank_move()
    )
    opponent_prev_move = (
        _build_move(o_last_id, 0, 0, game_data, mappings)
        if o_last_id >= 0
        else UniversalMove.blank_move()
    )

    # Symmetric forced-switch requests: ``forced_switch_side`` is 0, 1, or 2
    # (both). Match pokepy ``state_to_universal`` so side-1 eval agents get
    # switch-only ``maybe_valid_actions`` and correct illegal-action masks.
    _fs_side = int(getattr(state, "forced_switch_side", -1))
    forced_switch = bool(getattr(state, "forced_switch_slot", -1) >= 0) or (
        int(state.phase) == PHASE_FORCED_SWITCH and _fs_side in (player_side, 2)
    )

    from pokepy.core.gen_profile import profile_for_format

    profile = profile_for_format(format_str)
    can_tera = profile.has_tera
    if can_tera:
        for slot in range(6):
            poff = side_base + slot * POKEMON_SIZE
            flags = int(battle[poff + 15])
            if (flags & 0x8) != 0:
                can_tera = False
                break

    opp_species_arr = state.opp_species if player_side == 0 else state.team_species
    teampreview: List[str] = []
    if profile.has_teampreview:
        for slot in range(6):
            sid = int(opp_species_arr[slot])
            if sid >= 0:
                teampreview.append(_species_name(sid, mappings))

    return UniversalState(
        format=format_str,
        player_active_pokemon=player,
        opponent_active_pokemon=opponent,
        available_switches=available_switches,
        player_prev_move=player_prev_move,
        opponent_prev_move=opponent_prev_move,
        opponents_remaining=opponents_remaining,
        player_conditions=player_conditions,
        opponent_conditions=opponent_conditions,
        weather=weather,
        battle_field=battle_field,
        forced_switch=forced_switch,
        battle_won=bool(state.winner == player_side),
        battle_lost=bool(state.winner == (1 - player_side)),
        can_tera=can_tera,
        opponent_teampreview=teampreview,
    )
