"""Convert metamon TeamSet / Showdown teams to pokepy team dicts."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from poke_env.teambuilder import Teambuilder
from pokepy.data.loader import IDMappings, load_id_mappings

from metamon.backend.replay_parser.str_parsing import clean_name, pokemon_name
from metamon.env.wrappers import TeamSet


def _lookup(mapping: Dict[str, int], raw_name: str, kind: str) -> int:
    candidates = [
        pokemon_name(raw_name),
        clean_name(raw_name),
        raw_name.lower().replace(" ", "").replace("-", ""),
        raw_name.lower().replace(" ", ""),
    ]
    if "-" in raw_name:
        base = raw_name.split("-", 1)[0]
        candidates.extend(
            [
                pokemon_name(base),
                clean_name(base),
                base.lower().replace(" ", "").replace("-", ""),
            ]
        )
    for key in candidates:
        if key in mapping:
            return int(mapping[key])
    raise KeyError(f"Unmapped pokepy {kind} id for {raw_name!r} (tried {candidates})")


def _cleanup_move_id(move_id: str) -> str:
    move_id = clean_name(move_id)
    if move_id.startswith("hiddenpower"):
        return "hiddenpower"
    if move_id == "vicegrip":
        return "visegrip"
    if move_id.startswith("return"):
        return "return"
    return move_id


def _parse_evs(evs: Optional[str]) -> Optional[List[int]]:
    if not evs:
        return None
    out = [0] * 6
    for part in evs.split("/"):
        stat, _, val = part.partition(":")
        stat = stat.strip().lower()
        val = int(val.strip())
        stat_map = {
            "hp": 0,
            "atk": 1,
            "def": 2,
            "spa": 3,
            "spd": 4,
            "spe": 5,
        }
        if stat in stat_map:
            out[stat_map[stat]] = val
    return out


def _parse_ivs(ivs: Optional[str]) -> Optional[List[int]]:
    if not ivs:
        return None
    out = [31] * 6
    for part in ivs.split("/"):
        stat, _, val = part.partition(":")
        stat = stat.strip().lower()
        val = int(val.strip())
        stat_map = {
            "hp": 0,
            "atk": 1,
            "def": 2,
            "spa": 3,
            "spd": 4,
            "spe": 5,
        }
        if stat in stat_map:
            out[stat_map[stat]] = val
    return out


def _parse_packed_stat_spread(spread: str, default: int) -> List[int]:
    """Parse Showdown packed EV/IV spread (comma-separated, blanks = default)."""
    if not spread:
        return [default] * 6
    parts = spread.split(",")
    while len(parts) < 6:
        parts.append("")
    out: List[int] = []
    for part in parts[:6]:
        part = part.strip()
        out.append(int(part) if part else default)
    return out


def _parse_packed_tera(end_field: str) -> Optional[str]:
    if not end_field:
        return None
    chunks = [chunk for chunk in end_field.split(",") if chunk]
    return chunks[-1] if chunks else None


def _parse_packed_showdown_team(packed_team: str) -> List[dict]:
    """Parse a Showdown packed team string (mons joined by ``]``)."""
    mons: List[dict] = []
    for mon_str in packed_team.split("]"):
        mon_str = mon_str.strip()
        if not mon_str:
            continue
        parts = mon_str.split("|")
        while len(parts) < 12:
            parts.append("")

        nickname = parts[0]
        species = parts[1] or nickname
        if not species:
            continue

        moves = [m for m in parts[4].split(",") if m]
        tera_raw = _parse_packed_tera(parts[11])

        mons.append(
            dict(
                species=species,
                moves=moves,
                item=parts[2] or None,
                ability=parts[3] or None,
                nature=parts[5] or None,
                evs=_parse_packed_stat_spread(parts[6], default=0),
                ivs=_parse_packed_stat_spread(parts[8], default=31),
                level=int(parts[10]) if parts[10] else 100,
                tera_type=tera_raw,
            )
        )
    return mons


def showdown_team_to_pokepy_dict(
    packed_team: str,
    mappings: Optional[IDMappings] = None,
) -> Dict[str, Any]:
    """Parse a Showdown team string into pokepy's init_battle_state format."""
    mappings = mappings or load_id_mappings()

    if "]" in packed_team:
        parsed_mons = _parse_packed_showdown_team(packed_team)
    else:
        parsed_mons = []
        for mon in Teambuilder.parse_showdown_team(packed_team):
            species_name = mon.species or mon.nickname
            if not species_name:
                continue
            parsed_mons.append(
                dict(
                    species=species_name,
                    moves=list(mon.moves),
                    item=mon.item,
                    ability=mon.ability,
                    nature=mon.nature,
                    evs=list(mon.evs) if mon.evs is not None else None,
                    ivs=list(mon.ivs) if mon.ivs is not None else None,
                    level=int(mon.level or 100),
                    tera_type=getattr(mon, "tera_type", None)
                    or getattr(mon, "teratype", None),
                )
            )

    species: List[int] = []
    moves: List[List[int]] = []
    items: List[int] = []
    abilities: List[int] = []
    tera_types: List[int] = []
    levels: List[int] = []
    evs_list: List[List[int]] = []
    ivs_list: List[List[int]] = []
    natures: List[Optional[str]] = []

    for mon in parsed_mons:
        species.append(_lookup(mappings.species_to_idx, mon["species"], "species"))
        move_ids = []
        for move in mon["moves"]:
            move_ids.append(
                _lookup(mappings.move_to_idx, _cleanup_move_id(move), "move")
            )
        while len(move_ids) < 4:
            move_ids.append(-1)
        moves.append(move_ids[:4])

        item_name = mon["item"] if mon["item"] else ""
        items.append(
            _lookup(mappings.item_to_idx, item_name, "item") if item_name else 0
        )
        ability_name = mon["ability"] if mon["ability"] else ""
        abilities.append(
            _lookup(mappings.ability_to_idx, ability_name, "ability")
            if ability_name
            else 0
        )

        tera_raw = mon.get("tera_type")
        if tera_raw:
            tera_types.append(_lookup(mappings.type_to_idx, str(tera_raw), "type"))
        else:
            tera_types.append(-1)

        levels.append(int(mon.get("level") or 100))
        evs_list.append(mon.get("evs"))
        ivs_list.append(mon.get("ivs"))
        natures.append(mon.get("nature"))

    while len(species) < 6:
        species.append(-1)
        moves.append([-1, -1, -1, -1])
        items.append(0)
        abilities.append(0)
        tera_types.append(-1)
        levels.append(100)
        evs_list.append(None)
        ivs_list.append(None)
        natures.append(None)

    return dict(
        species=species[:6],
        moves=moves[:6],
        items=items[:6],
        abilities=abilities[:6],
        tera_types=tera_types[:6],
        levels=levels[:6],
        evs=evs_list[:6],
        ivs=ivs_list[:6],
        natures=natures[:6],
    )


def team_set_to_pokepy_dict(
    team_set: TeamSet,
    mappings: Optional[IDMappings] = None,
) -> Dict[str, Any]:
    """Sample a team from a metamon TeamSet and convert to pokepy format."""
    packed = team_set.yield_team()
    return showdown_team_to_pokepy_dict(packed, mappings=mappings)
