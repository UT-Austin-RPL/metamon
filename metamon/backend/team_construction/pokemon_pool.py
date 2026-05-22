from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Sequence

from metamon.backend import format_to_gen
from metamon.backend.replay_parser.str_parsing import pokemon_name
from metamon.backend.showdown_dex.dex import Dex
from metamon.backend.team_prediction.usage_stats import (
    DEFAULT_USAGE_RANK,
    get_usage_stats,
)
from metamon.backend.team_construction.teams.parse import parse_species_name

from .core import PokemonSet, Team, canonical_team


def _month_to_date(usage_month: str) -> dt.date:
    try:
        year, month = usage_month.split("-")
        return dt.date(year=int(year), month=int(month), day=1)
    except Exception as exc:
        raise ValueError(f"usage_month must be YYYY-MM, got '{usage_month}'") from exc


def _norm(name: str) -> str:
    return pokemon_name(name)


def _sorted_top_items(raw: dict[str, float], k: int) -> list[str]:
    banned = {"", "other", "nothing"}
    cleaned = [(str(name).strip(), float(value)) for name, value in raw.items()]
    cleaned = [
        (name, value)
        for name, value in cleaned
        if _norm(name) not in banned and value > 0
    ]
    cleaned.sort(key=lambda item: (item[1], item[0]), reverse=True)
    return [name for name, _ in cleaned[:k]]


def _top_ability(
    stats_entry: dict[str, Any], gen: int, dex_entry: dict[str, Any]
) -> str | None:
    if gen <= 2:
        return "No Ability"

    abilities = stats_entry.get("abilities", {}) or {}
    top = _sorted_top_items(abilities, k=1)
    if top:
        return top[0]

    dex_abilities = dex_entry.get("abilities", {}) or {}
    if dex_abilities:
        ordered = sorted(dex_abilities.items(), key=lambda kv: kv[0])
        return str(ordered[0][1])
    return None


def _required_item(
    stats_entry: dict[str, Any], dex_entry: dict[str, Any]
) -> str | None:
    required_item = dex_entry.get("requiredItem")
    if isinstance(required_item, str) and required_item.strip():
        return required_item.strip()

    required_items = dex_entry.get("requiredItems")
    if isinstance(required_items, list):
        options = [str(item).strip() for item in required_items if str(item).strip()]
        if not options:
            return None

        option_by_norm = {_norm(item): item for item in options}
        top_items = _sorted_top_items(stats_entry.get("items", {}) or {}, k=16)
        for item in top_items:
            chosen = option_by_norm.get(_norm(item))
            if chosen is not None:
                return chosen
        return options[0]

    return None


def _ensure_required_move(moves: list[str], dex_entry: dict[str, Any]) -> list[str]:
    required_move = dex_entry.get("requiredMove")
    if not isinstance(required_move, str):
        return moves
    required_move = required_move.strip()
    if not required_move:
        return moves
    if any(_norm(move) == _norm(required_move) for move in moves):
        return moves
    if len(moves) >= 4:
        return [required_move, *moves[:3]]
    return [required_move, *moves]


def build_standardized_showdown_set(
    *,
    species: str,
    moves: list[str],
    gen: int,
    ability: str | None,
    item: str | None = None,
    nature: str = "Serious",
    strict_max_evs: bool = False,
) -> str:
    """Create a standardized per-species Showdown export block.

    In early generations we keep the set minimal because EV/nature syntax differs.
    """

    if len(moves) < 4:
        raise ValueError(f"Need at least 4 moves for {species}, got {moves}")

    header = f"{species} @ {item}" if gen >= 2 and item else species
    lines: list[str] = [header]

    resolved_ability = str(ability).strip() if ability else ""
    if gen <= 2 and not resolved_ability:
        resolved_ability = "No Ability"
    if resolved_ability:
        lines.append(f"Ability: {resolved_ability}")

    if gen >= 3:
        if strict_max_evs:
            lines.append(
                "EVs: 252 HP / 252 Atk / 252 Def / 252 SpA / 252 SpD / 252 Spe"
            )
        else:
            lines.append("EVs: 84 HP / 84 Atk / 84 Def / 84 SpA / 84 SpD / 84 Spe")
        lines.append(f"{nature} Nature")

    for move in moves[:4]:
        lines.append(f"- {move}")
    return "\n".join(lines)


def _load_replication_moves(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}

    if isinstance(payload, dict):
        for species, moves in payload.items():
            if isinstance(moves, dict):
                raw_moves = moves.get("moves", [])
            else:
                raw_moves = moves
            if not isinstance(raw_moves, list):
                continue
            cleaned = [str(m).strip() for m in raw_moves if str(m).strip()]
            if cleaned:
                out[_norm(species)] = cleaned
        return out

    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                continue
            species = str(row.get("species", "")).strip()
            moves = row.get("moves", [])
            if not species or not isinstance(moves, list):
                continue
            cleaned = [str(m).strip() for m in moves if str(m).strip()]
            if cleaned:
                out[_norm(species)] = cleaned
        return out

    raise ValueError(f"Replication moves file must be a dict or list of dicts: {path}")


def _load_manual_sets(path: Path, gen: int) -> list[PokemonSet]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Manual sets file must be a JSON list.")

    dex = Dex.from_gen(gen)
    out: list[PokemonSet] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        species = str(row.get("species", "")).strip()
        moves = row.get("moves", [])
        if not species or not isinstance(moves, list):
            continue
        clean_moves = [str(m).strip() for m in moves if str(m).strip()]
        try:
            dex_entry = dex.get_pokedex_entry(species)
        except KeyError:
            dex_entry = {}
        clean_moves = _ensure_required_move(clean_moves, dex_entry=dex_entry)
        if len(clean_moves) < 4:
            raise ValueError(
                f"Manual set for {species} must provide at least 4 moves, got {clean_moves}"
            )
        showdown_set = str(row.get("showdown_set", "")).strip()
        ability = row.get("ability")
        resolved_ability = (
            str(ability).strip()
            if ability
            else _top_ability({}, gen=gen, dex_entry=dex_entry)
        )
        item = row.get("item") or _required_item({}, dex_entry=dex_entry)
        if not showdown_set:
            showdown_set = build_standardized_showdown_set(
                species=species,
                moves=clean_moves,
                gen=gen,
                ability=resolved_ability,
                item=str(item) if item else None,
            )
        out.append(
            PokemonSet(
                species=species,
                moves=tuple(clean_moves[:4]),
                showdown_set=showdown_set,
                usage=float(row.get("usage", 0.0) or 0.0),
                ability=resolved_ability,
            )
        )

    if not out:
        raise ValueError(f"No valid manual sets found in {path}")
    return out


def get_eligible_pokemon(
    format_id: str,
    usage_month: str,
    usage_threshold: float,
    *,
    rank: int = DEFAULT_USAGE_RANK,
    replication_movesets_json: Path | None = None,
    manual_sets_json: Path | None = None,
    strict_max_evs: bool = False,
) -> list[PokemonSet]:
    """Build the eligible Pokemon pool with one standardized set per species."""

    gen = format_to_gen(format_id)

    if manual_sets_json is not None:
        return _load_manual_sets(manual_sets_json, gen=gen)

    month_date = _month_to_date(usage_month)
    stats = get_usage_stats(
        format_id,
        start_date=month_date,
        end_date=month_date,
        rank=rank,
    )
    raw_movesets = stats.movesets
    if not raw_movesets:
        raise ValueError(
            f"No usage stats available for {format_id} at {usage_month} (rank {rank})."
        )

    replication_moves: dict[str, list[str]] = {}
    if replication_movesets_json is not None:
        replication_moves = _load_replication_moves(replication_movesets_json)

    dex = Dex.from_gen(gen)
    total = sum(float(entry.get("count", 0)) for entry in raw_movesets.values())
    if total <= 0:
        raise ValueError(
            f"Usage stats for {format_id} at {usage_month} have zero total count."
        )

    out: list[PokemonSet] = []
    for species_id, entry in raw_movesets.items():
        usage = float(entry.get("count", 0.0)) / total
        if usage < usage_threshold:
            continue

        try:
            dex_entry = dex.get_pokedex_entry(species_id)
            species_name = str(dex_entry.get("name", species_id))
        except KeyError:
            dex_entry = {}
            species_name = species_id

        override_moves = replication_moves.get(
            _norm(species_name)
        ) or replication_moves.get(_norm(species_id))
        if override_moves is not None:
            moves = override_moves[:4]
        else:
            moves = _sorted_top_items(entry.get("moves", {}) or {}, k=4)

        moves = _ensure_required_move(moves, dex_entry=dex_entry)
        if len(moves) < 4:
            continue

        ability = _top_ability(entry, gen=gen, dex_entry=dex_entry)
        item = _required_item(entry, dex_entry=dex_entry)
        showdown = build_standardized_showdown_set(
            species=species_name,
            moves=moves,
            gen=gen,
            ability=ability,
            item=item,
            strict_max_evs=strict_max_evs,
        )
        out.append(
            PokemonSet(
                species=species_name,
                moves=tuple(moves),
                showdown_set=showdown,
                usage=usage,
                ability=ability,
            )
        )

    out.sort(key=lambda p: (p.usage, p.species), reverse=True)
    if not out:
        raise ValueError(
            "Filtering removed all Pokemon. Try lowering --usage-threshold or supplying manual sets."
        )
    return out


def save_pool_artifact(
    out_path: Path,
    *,
    format_id: str,
    usage_month: str,
    usage_threshold: float,
    pokemon_sets: list[PokemonSet],
    metadata: dict[str, Any] | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_id": format_id,
        "usage_month": usage_month,
        "usage_threshold": float(usage_threshold),
        "pool_size": len(pokemon_sets),
        "pokemon_sets": [
            {
                "id": idx,
                "species": item.species,
                "usage": float(item.usage),
                "ability": item.ability,
                "moves": list(item.moves),
                "showdown_set": item.showdown_set,
            }
            for idx, item in enumerate(pokemon_sets)
        ],
        "metadata": metadata or {},
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_pool_artifact(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "pokemon_sets" not in data or not isinstance(data["pokemon_sets"], list):
        raise ValueError(f"Invalid pool artifact: {path}")
    return data


def pool_pokemon_sets(data: dict[str, Any]) -> list[PokemonSet]:
    out: list[PokemonSet] = []
    for row in data["pokemon_sets"]:
        out.append(
            PokemonSet(
                species=str(row["species"]),
                moves=tuple(str(m) for m in row.get("moves", [])),
                showdown_set=str(row["showdown_set"]),
                usage=float(row.get("usage", 0.0) or 0.0),
                ability=(str(row["ability"]) if row.get("ability") else None),
            )
        )
    return out


def build_species_to_id_map(pokemon_sets: list[PokemonSet]) -> dict[str, int]:
    species_to_id: dict[str, int] = {}
    for idx, pset in enumerate(pokemon_sets):
        species_to_id[_norm(pset.species)] = idx
    return species_to_id


def build_species_clause_keys(
    format_id: str,
    pokemon_sets: Sequence[PokemonSet],
) -> dict[int, object]:
    gen = format_to_gen(format_id)
    dex = Dex.from_gen(gen)

    out: dict[int, object] = {}
    for idx, pset in enumerate(pokemon_sets):
        default_key: object = f"name:{_norm(pset.species)}"
        try:
            dex_entry = dex.get_pokedex_entry(pset.species)
        except KeyError:
            out[idx] = default_key
            continue

        num = dex_entry.get("num")
        if isinstance(num, int):
            out[idx] = f"num:{num}"
            continue

        base_species = dex_entry.get("baseSpecies")
        if isinstance(base_species, str) and base_species.strip():
            out[idx] = f"base:{_norm(base_species)}"
            continue

        out[idx] = default_key
    return out


def extract_species_from_team_string(team_text: str) -> list[str]:
    blocks = [block for block in team_text.split("\n\n") if block.strip()]
    out: list[str] = []
    for block in blocks:
        species = parse_species_name(block)
        if species:
            out.append(species)
    return out


def team_string_to_ids(
    team_text_or_names: str,
    pokemon_sets: list[PokemonSet],
    *,
    team_size: int | None = None,
) -> Team:
    species_to_id = build_species_to_id_map(pokemon_sets)
    text = team_text_or_names.strip()

    if "\n" in text:
        species_names = extract_species_from_team_string(text)
    elif "," in text:
        species_names = [chunk.strip() for chunk in text.split(",") if chunk.strip()]
    else:
        species_names = [chunk.strip() for chunk in text.split() if chunk.strip()]

    if not species_names:
        raise ValueError("Could not parse species from team input.")

    team_ids: list[int] = []
    missing: list[str] = []
    for name in species_names:
        idx = species_to_id.get(_norm(name))
        if idx is None:
            missing.append(name)
            continue
        team_ids.append(idx)

    if missing:
        raise ValueError(
            "Unrecognized species in team input: "
            + ", ".join(missing)
            + ". Rebuild the pool or use matching species names."
        )

    return canonical_team(team_ids, team_size=team_size)


def team_ids_to_showdown(team: Team, pokemon_sets: list[PokemonSet]) -> str:
    blocks = [pokemon_sets[idx].showdown_set.strip() for idx in team]
    return "\n\n".join(blocks)
