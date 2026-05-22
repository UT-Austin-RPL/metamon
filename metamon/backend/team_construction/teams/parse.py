import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

FORMAT_RE = re.compile(r"\.(gen[0-9][a-z0-9]*)_team$", re.IGNORECASE)


def infer_format(
    team_file: Path, fallback_format: Optional[str] = None
) -> Optional[str]:
    match = FORMAT_RE.search(team_file.name)
    if match:
        return match.group(1).lower()
    if fallback_format:
        return fallback_format.lower()
    return None


def team_id_from_filename(team_file: Path) -> str:
    name = team_file.name
    match = FORMAT_RE.search(name)
    if match:
        base = name[: match.start()]
    else:
        base = team_file.stem
    team_num_match = re.fullmatch(r"team_(\d+)", base)
    if team_num_match:
        return team_num_match.group(1)
    return base


def sort_team_ids(team_ids: Iterable[str]) -> List[str]:
    def key_fn(value: str):
        num_match = re.search(r"\d+", value)
        if num_match:
            return (0, int(num_match.group()), value)
        return (1, value)

    return sorted(team_ids, key=key_fn)


def parse_species_name(block: str) -> Optional[str]:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if not lines:
        return None

    header = lines[0]
    if header.startswith(("-", "Ability:", "EVs:", "IVs:", "Level:", "Tera Type:")):
        return None

    if " @ " in header:
        header = header.split(" @ ", 1)[0].strip()

    form_match = re.search(r"\(([^()]*)\)\s*$", header)
    if form_match:
        candidate = form_match.group(1).strip()
        if candidate:
            return candidate

    if header.endswith(")"):
        return None

    return header


def parse_moveset(block: str) -> Tuple[str, ...]:
    moves: List[str] = []
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith("-"):
            continue
        move = line[1:].strip()
        if move:
            moves.append(move)
    return tuple(moves)


def parse_team_file(team_file: Path) -> List[Tuple[str, Tuple[str, ...]]]:
    content = team_file.read_text(encoding="utf-8", errors="replace")
    blocks = [block for block in content.split("\n\n") if block.strip()]
    parsed: List[Tuple[str, Tuple[str, ...]]] = []
    seen: Set[str] = set()
    for block in blocks:
        name = parse_species_name(block)
        if not name or name in seen:
            continue
        seen.add(name)
        parsed.append((name, parse_moveset(block)))
    return parsed


def parse_teams(
    team_dir: Path, fallback_format: Optional[str]
) -> Tuple[Dict[str, Set[str]], Dict[str, Dict[Tuple[str, ...], Set[str]]]]:
    pokemon_to_teams: Dict[str, Set[str]] = defaultdict(set)
    pokemon_to_movesets: Dict[str, Dict[Tuple[str, ...], Set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    team_files = sorted(team_dir.rglob("*.gen*_team"))

    if not team_files:
        raise FileNotFoundError(f"No team files found under: {team_dir}")

    failed_files: List[str] = []

    for team_file in team_files:
        format_name = infer_format(team_file, fallback_format=fallback_format)
        if not format_name:
            failed_files.append(f"{team_file} (cannot infer format)")
            continue

        try:
            parsed_entries = parse_team_file(team_file)
        except Exception as exc:
            failed_files.append(f"{team_file} ({exc})")
            continue

        if not parsed_entries:
            failed_files.append(f"{team_file} (no Pokemon parsed)")
            continue

        team_id = team_id_from_filename(team_file)
        for pokemon_name, moveset in parsed_entries:
            pokemon_to_teams[pokemon_name].add(team_id)
            pokemon_to_movesets[pokemon_name][moveset].add(team_id)

    if failed_files:
        print("Skipped team files:", file=sys.stderr)
        for line in failed_files:
            print(f"  - {line}", file=sys.stderr)

    if not pokemon_to_teams:
        raise RuntimeError("No Pokemon names were parsed from the provided team files.")

    return pokemon_to_teams, pokemon_to_movesets


def write_team_index_csv(
    pokemon_to_teams: Dict[str, Set[str]], output_csv: Path
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pokemon_name", "team_count", "team_ids"])
        for pokemon_name in sorted(pokemon_to_teams):
            team_ids = sort_team_ids(pokemon_to_teams[pokemon_name])
            writer.writerow([pokemon_name, len(team_ids), ",".join(team_ids)])


def write_moveset_index_csv(
    pokemon_to_movesets: Dict[str, Dict[Tuple[str, ...], Set[str]]], output_csv: Path
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pokemon_name", "moveset", "team_count", "team_ids"])
        for pokemon_name in sorted(pokemon_to_movesets):
            rows: List[Tuple[int, str, str, str]] = []
            for moveset, team_ids in pokemon_to_movesets[pokemon_name].items():
                ordered_team_ids = sort_team_ids(team_ids)
                moveset_str = " / ".join(moveset)
                rows.append(
                    (
                        -len(ordered_team_ids),
                        moveset_str,
                        str(len(ordered_team_ids)),
                        ",".join(ordered_team_ids),
                    )
                )
            for _, moveset_str, team_count, team_id_str in sorted(rows):
                writer.writerow([pokemon_name, moveset_str, team_count, team_id_str])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create CSV indexes from team files: "
            "pokemon->team IDs and pokemon->movesets."
        )
    )
    parser.add_argument(
        "team_dir", type=Path, help="Directory containing showdown team files."
    )
    parser.add_argument(
        "-o",
        "--team-output",
        type=Path,
        default=Path("/tmp/team_construction/gen1ou_pokemon_team_index.csv"),
        help=(
            "Output path for pokemon->team index CSV. "
            "Default: /tmp/team_construction/gen1ou_pokemon_team_index.csv"
        ),
    )
    parser.add_argument(
        "--moveset-output",
        type=Path,
        default=Path("/tmp/team_construction/gen1ou_pokemon_moveset_index.csv"),
        help=(
            "Output path for pokemon->moveset index CSV. "
            "Default: /tmp/team_construction/gen1ou_pokemon_moveset_index.csv"
        ),
    )
    parser.add_argument(
        "--skip-team-index",
        action="store_true",
        help="Do not write the pokemon->team index CSV.",
    )
    parser.add_argument(
        "--skip-moveset-index",
        action="store_true",
        help="Do not write the pokemon->moveset index CSV.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default=None,
        help="Optional fallback format name (e.g., gen1ou) if not inferable from filename.",
    )

    args = parser.parse_args()
    if args.skip_team_index and args.skip_moveset_index:
        raise ValueError("Both outputs were skipped. Enable at least one output CSV.")

    pokemon_to_teams, pokemon_to_movesets = parse_teams(
        args.team_dir, fallback_format=args.format
    )

    if not args.skip_team_index:
        write_team_index_csv(pokemon_to_teams, args.team_output)
        print(
            f"Wrote {len(pokemon_to_teams)} Pokemon rows to {args.team_output} from {args.team_dir}"
        )

    if not args.skip_moveset_index:
        write_moveset_index_csv(pokemon_to_movesets, args.moveset_output)
        print(
            "Wrote moveset index rows to " f"{args.moveset_output} from {args.team_dir}"
        )


if __name__ == "__main__":
    main()
