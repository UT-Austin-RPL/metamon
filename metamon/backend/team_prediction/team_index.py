"""index.csv helpers for Metamon team set directories."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Optional

INDEX_FILENAME = "index.csv"
_FORMAT_DIR_RE = re.compile(r"^gen\d+[a-z0-9]+$")


def team_file_suffix(battle_format: str) -> str:
    return f".{battle_format.lower()}_team"


def index_path_for(format_dir: os.PathLike[str] | str) -> Path:
    return Path(format_dir) / INDEX_FILENAME


def resolve_format_dir(team_root: os.PathLike[str] | str, battle_format: str) -> Path:
    """Return the directory containing team files for a format."""
    root = Path(team_root)
    nested = root / battle_format.lower()
    if nested.is_dir():
        return nested
    return root


def scan_team_filenames(
    format_dir: os.PathLike[str] | str, battle_format: str
) -> list[str]:
    """List team filenames relative to format_dir (sorted)."""
    format_dir = Path(format_dir)
    suffix = team_file_suffix(battle_format)
    rel_paths: list[str] = []
    for dirpath, _, filenames in os.walk(format_dir):
        for name in filenames:
            if not name.endswith(suffix):
                continue
            full = Path(dirpath) / name
            rel_paths.append(full.relative_to(format_dir).as_posix())
    rel_paths.sort()
    return rel_paths


def write_team_index(
    format_dir: os.PathLike[str] | str, filenames: Iterable[str]
) -> Path:
    format_dir = Path(format_dir)
    index_path = index_path_for(format_dir)
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("filename\n")
        for name in filenames:
            f.write(f"{name}\n")
    return index_path


def refresh_team_index(
    format_dir: os.PathLike[str] | str, battle_format: str
) -> tuple[Path, int]:
    filenames = scan_team_filenames(format_dir, battle_format)
    index_path = write_team_index(format_dir, filenames)
    return index_path, len(filenames)


def load_team_files(
    team_root: os.PathLike[str] | str,
    battle_format: str,
) -> tuple[list[str], bool]:
    """
    Load absolute paths to team files from index.csv if present, else scan disk.

    Returns (paths, loaded_from_index).
    """
    format_dir = resolve_format_dir(team_root, battle_format)
    index_path = index_path_for(format_dir)
    if not index_path.is_file():
        names = scan_team_filenames(format_dir, battle_format)
        return [str(format_dir / name) for name in names], False

    suffix = team_file_suffix(battle_format)
    team_files: list[str] = []
    missing = 0
    with open(index_path, encoding="utf-8") as f:
        f.readline()  # header
        for line in f:
            rel = line.strip()
            if not rel:
                continue
            full = format_dir / rel
            if full.is_file():
                team_files.append(str(full))
            else:
                missing += 1

    if not team_files:
        names = scan_team_filenames(format_dir, battle_format)
        return [str(format_dir / name) for name in names], False

    if missing:
        print(
            f"Warning: {missing} entries in {index_path} missing on disk; "
            f"loaded {len(team_files):,} teams"
        )

    return team_files, True


def infer_battle_format(format_dir: Path) -> Optional[str]:
    for entry in sorted(format_dir.iterdir()):
        if not entry.is_file():
            continue
        match = re.match(r".+\.(gen\d+[a-z0-9]+)_team$", entry.name)
        if match:
            return match.group(1)
    return None


def is_format_team_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if _FORMAT_DIR_RE.match(path.name):
        return infer_battle_format(path) is not None
    return infer_battle_format(path) is not None


def iter_format_dirs(set_dir: Path) -> Iterable[tuple[Path, str]]:
    """Yield (format_dir, battle_format) under a team set directory."""
    if is_format_team_dir(set_dir):
        fmt = infer_battle_format(set_dir) or set_dir.name
        yield set_dir, fmt
        return

    for child in sorted(set_dir.iterdir()):
        if not child.is_dir():
            continue
        if not is_format_team_dir(child):
            continue
        fmt = infer_battle_format(child) or child.name
        yield child, fmt


PUBLIC_TEAM_SETS = frozenset(
    {
        "competitive",
        "paper_variety",
        "paper_replays",
        "modern_replays",
        "modern_replays_v2",
        "gl_05_26",
        "hl_05_26",
    }
)


def should_index_set_dir(name: str, public_only: bool = True) -> bool:
    if name.startswith("."):
        return False
    if name.endswith("-unfiltered"):
        return False
    if name in {"analysis", "select"}:
        return False
    if public_only and name not in PUBLIC_TEAM_SETS:
        return False
    return True
