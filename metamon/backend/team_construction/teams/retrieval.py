import argparse
import csv
import itertools
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

TEAM_FILE_RE = re.compile(r"^(team_\d+)\.(gen[0-9][a-z0-9]*)_team$", re.IGNORECASE)


def normalize_name(name: str) -> str:
    return name.strip().lower()


def normalize_move(move: str) -> str:
    return move.strip().lower()


def load_index(csv_path: Path) -> Tuple[Dict[str, Set[str]], Dict[str, str]]:
    """Load pokemon -> team_ids mapping from CSV.

    Returns:
        teams_by_name: normalized pokemon name -> team id set
        canonical_name: normalized pokemon name -> original cased pokemon name
    """
    teams_by_name: Dict[str, Set[str]] = {}
    canonical_name: Dict[str, str] = {}

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"pokemon_name", "team_ids"}
        if not required_cols.issubset(reader.fieldnames or set()):
            raise ValueError(
                f"CSV {csv_path} must have columns: {sorted(required_cols)}"
            )

        for row in reader:
            raw_name = (row.get("pokemon_name") or "").strip()
            key = normalize_name(raw_name)
            if not key:
                continue

            raw_team_ids = (row.get("team_ids") or "").strip()
            team_ids = {t.strip() for t in raw_team_ids.split(",") if t.strip()}
            teams_by_name[key] = team_ids
            canonical_name[key] = raw_name

    if not teams_by_name:
        raise ValueError(f"CSV {csv_path} contained no usable pokemon rows.")
    return teams_by_name, canonical_name


def load_moveset_index(
    csv_path: Path,
) -> Dict[str, Dict[str, List[Set[str]]]]:
    """Load moveset index as team_id -> pokemon_name -> list[move-set]."""
    moves_by_team: Dict[str, Dict[str, List[Set[str]]]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"pokemon_name", "moveset", "team_ids"}
        if not required_cols.issubset(reader.fieldnames or set()):
            raise ValueError(
                f"CSV {csv_path} must have columns: {sorted(required_cols)}"
            )

        for row in reader:
            raw_name = (row.get("pokemon_name") or "").strip()
            name_key = normalize_name(raw_name)
            if not name_key:
                continue

            raw_moveset = (row.get("moveset") or "").strip()
            move_set = {
                normalize_move(m) for m in raw_moveset.split("/") if normalize_move(m)
            }

            raw_team_ids = (row.get("team_ids") or "").strip()
            team_ids = [t.strip() for t in raw_team_ids.split(",") if t.strip()]
            for team_id in team_ids:
                by_pokemon = moves_by_team.setdefault(team_id, {})
                by_pokemon.setdefault(name_key, []).append(move_set)

    return moves_by_team


def build_team_file_lookup(team_dir: Path) -> Dict[str, Path]:
    """Map team IDs (e.g. '0001') to concrete team files under team_dir."""
    lookup: Dict[str, Path] = {}
    for path in sorted(team_dir.rglob("*.gen*_team")):
        match = TEAM_FILE_RE.match(path.name)
        if not match:
            continue
        team_id = match.group(1).split("_", 1)[1]
        lookup[team_id] = path
    return lookup


def dedupe_names(names: Sequence[str]) -> List[str]:
    normalized = [normalize_name(n) for n in names if normalize_name(n)]
    seen: Set[str] = set()
    unique_names: List[str] = []
    for n in normalized:
        if n not in seen:
            unique_names.append(n)
            seen.add(n)
    return unique_names


def parse_moveset_specs(specs: Sequence[str]) -> Dict[str, Set[str]]:
    """Parse repeated --moveset-spec values.

    Format: "PokemonName:Move1|Move2|Move3"
    """
    out: Dict[str, Set[str]] = {}
    for spec in specs:
        if ":" not in spec:
            raise ValueError(
                f"Invalid --moveset-spec '{spec}'. Expected format: Pokemon:Move1|Move2"
            )
        raw_name, raw_moves = spec.split(":", 1)
        name_key = normalize_name(raw_name)
        if not name_key:
            raise ValueError(f"Invalid --moveset-spec '{spec}': empty Pokemon name.")
        moves = {normalize_move(m) for m in raw_moves.split("|") if normalize_move(m)}
        if not moves:
            raise ValueError(
                f"Invalid --moveset-spec '{spec}': provide at least one move."
            )
        out[name_key] = moves
    return out


def moves_matched_ratio(
    team_id: str,
    moveset_requests: Dict[str, Set[str]],
    moves_by_team: Dict[str, Dict[str, List[Set[str]]]],
) -> float:
    if not moveset_requests:
        return 0.0

    team_moves = moves_by_team.get(team_id, {})
    matched = 0
    total = 0
    for pokemon_name, wanted_moves in moveset_requests.items():
        total += len(wanted_moves)
        moveset_options = team_moves.get(pokemon_name, [])
        if not moveset_options:
            continue
        best_overlap = 0
        for option in moveset_options:
            best_overlap = max(best_overlap, len(wanted_moves & option))
        matched += best_overlap
    if total == 0:
        return 0.0
    return matched / total


def best_match_by_names_then_moves(
    names: Sequence[str],
    teams_by_name: Dict[str, Set[str]],
    moveset_requests: Dict[str, Set[str]],
    moves_by_team: Dict[str, Dict[str, List[Set[str]]]],
) -> Tuple[Optional[str], List[str], int, float]:
    """Return best team using (name match count, moves matched ratio) ranking."""
    unique_names = dedupe_names(names)
    if not unique_names:
        return None, [], 0, 0.0
    available = [n for n in unique_names if n in teams_by_name and teams_by_name[n]]
    if not available:
        return None, [], 0, 0.0

    for k in range(len(available), 0, -1):
        best_choice: Optional[Tuple[str, Tuple[str, ...], float]] = None
        for combo in itertools.combinations(available, k):
            intersection = set(teams_by_name[combo[0]])
            for n in combo[1:]:
                intersection &= teams_by_name[n]
                if not intersection:
                    break
            if not intersection:
                continue
            for team_id in sorted(intersection):
                ratio = moves_matched_ratio(team_id, moveset_requests, moves_by_team)
                if best_choice is None:
                    best_choice = (team_id, combo, ratio)
                    continue
                _, _, best_ratio = best_choice
                if ratio > best_ratio:
                    best_choice = (team_id, combo, ratio)
        if best_choice is not None:
            team_id, combo, ratio = best_choice
            return team_id, list(combo), k, ratio

    return None, [], 0, 0.0


def retrieve_team(
    pokemon_names: Sequence[str],
    csv_path: Path,
    team_dir: Path,
    moveset_csv_path: Optional[Path] = None,
    moveset_specs: Optional[Sequence[str]] = None,
) -> Tuple[Path, str, List[str], int, float]:
    teams_by_name, canonical_name = load_index(csv_path)
    moveset_requests = parse_moveset_specs(moveset_specs or [])
    moves_by_team: Dict[str, Dict[str, List[Set[str]]]] = {}
    if moveset_requests:
        if moveset_csv_path is None:
            raise ValueError(
                "Moveset specs were provided but --moveset-csv is missing."
            )
        moves_by_team = load_moveset_index(moveset_csv_path)

    team_file_lookup = build_team_file_lookup(team_dir)
    if not team_file_lookup:
        raise FileNotFoundError(f"No team files found under: {team_dir}")

    team_id, matched_norm_names, match_size, moves_ratio = (
        best_match_by_names_then_moves(
            pokemon_names, teams_by_name, moveset_requests, moves_by_team
        )
    )
    if team_id is None:
        raise ValueError("Could not match any provided pokemon names to indexed teams.")

    team_file = team_file_lookup.get(team_id)
    if team_file is None:
        raise FileNotFoundError(
            f"Matched team id {team_id}, but no corresponding file was found in {team_dir}"
        )

    team_text = team_file.read_text(encoding="utf-8", errors="replace")
    matched_names = [canonical_name.get(n, n) for n in matched_norm_names]
    return team_file, team_text, matched_names, match_size, moves_ratio


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve a preconstructed team matching up to six pokemon names. "
            "If no 6-way match exists, automatically falls back to 5, 4, 3, etc."
        )
    )
    parser.add_argument(
        "pokemon_names",
        nargs="+",
        help="Pokemon names to match (usually 6 names).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("/tmp/team_construction/gen1ou_pokemon_team_index.csv"),
        help=(
            "Path to pokemon index CSV. Default: /tmp/team_construction/gen1ou_pokemon_team_index.csv"
        ),
    )
    parser.add_argument(
        "--team-dir",
        type=Path,
        required=True,
        help="Directory containing team files (*.gen*_team).",
    )
    parser.add_argument(
        "--moveset-csv",
        type=Path,
        default=Path("/tmp/team_construction/gen1ou_pokemon_moveset_index.csv"),
        help=(
            "Path to pokemon moveset index CSV. "
            "Used only when --moveset-spec is provided."
        ),
    )
    parser.add_argument(
        "--moveset-spec",
        action="append",
        default=[],
        help=(
            "Optional move constraints. Repeat as needed. "
            "Format: Pokemon:Move1|Move2|Move3"
        ),
    )
    parser.add_argument(
        "--show-team",
        action="store_true",
        help="Print full team content after selection.",
    )

    args = parser.parse_args()

    try:
        team_file, team_text, matched_names, match_size, moves_ratio = retrieve_team(
            pokemon_names=args.pokemon_names,
            csv_path=args.csv,
            team_dir=args.team_dir,
            moveset_csv_path=args.moveset_csv,
            moveset_specs=args.moveset_spec,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Selected team file: {team_file}")
    print(f"Matched {match_size} pokemon: {', '.join(matched_names)}")
    if args.moveset_spec:
        print(f"Moves matched: {moves_ratio * 100:.1f}%")
    if args.show_team:
        print("\n--- TEAM ---")
        print(team_text.rstrip())


if __name__ == "__main__":
    main()
