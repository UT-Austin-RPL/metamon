"""Write index.csv for team set directories under METAMON_CACHE_DIR/teams."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import metamon
from metamon.backend.team_prediction.team_index import (
    PUBLIC_TEAM_SETS,
    iter_format_dirs,
    refresh_team_index,
    should_index_set_dir,
)


def write_indexes(
    teams_root: Path,
    set_names: list[str] | None = None,
    formats: list[str] | None = None,
    public_only: bool = True,
) -> list[tuple[str, str, int, Path]]:
    written: list[tuple[str, str, int, Path]] = []
    if set_names is None and public_only:
        set_names = sorted(PUBLIC_TEAM_SETS)
    set_dirs = sorted(
        p
        for p in teams_root.iterdir()
        if p.is_dir()
        and should_index_set_dir(p.name, public_only=public_only)
        and (set_names is None or p.name in set_names)
    )

    for set_dir in set_dirs:
        for format_dir, battle_format in iter_format_dirs(set_dir):
            if formats and battle_format not in formats:
                continue
            index_path, count = refresh_team_index(format_dir, battle_format)
            written.append((set_dir.name, battle_format, count, index_path))
            print(f"Wrote {index_path} ({count:,} teams)")
    return written


def main() -> None:
    default_root = (
        Path(metamon.METAMON_CACHE_DIR) / "teams" if metamon.METAMON_CACHE_DIR else None
    )
    parser = argparse.ArgumentParser(
        description="Generate index.csv for team sets under METAMON_CACHE_DIR/teams."
    )
    parser.add_argument(
        "--teams-root",
        type=Path,
        default=default_root,
        help="Root directory containing team set folders (default: $METAMON_CACHE_DIR/teams)",
    )
    parser.add_argument(
        "--set",
        action="append",
        dest="sets",
        metavar="SET",
        help="Only index this set (repeatable)",
    )
    parser.add_argument(
        "--format",
        action="append",
        dest="formats",
        metavar="FORMAT",
        help="Only index this format (repeatable)",
    )
    parser.add_argument(
        "--all-sets",
        action="store_true",
        help="Index every team dir (not just public HF sets)",
    )
    args = parser.parse_args()

    if args.teams_root is None:
        raise SystemExit(
            "METAMON_CACHE_DIR is not set and --teams-root was not provided"
        )

    if not args.teams_root.is_dir():
        raise SystemExit(f"Teams root not found: {args.teams_root}")

    entries = write_indexes(
        args.teams_root,
        set_names=args.sets,
        formats=args.formats,
        public_only=not args.all_sets,
    )
    if not entries:
        print("No team directories indexed.")
    else:
        total = sum(count for _, _, count, _ in entries)
        print(f"Indexed {len(entries)} format dirs ({total:,} teams total)")


if __name__ == "__main__":
    main()
