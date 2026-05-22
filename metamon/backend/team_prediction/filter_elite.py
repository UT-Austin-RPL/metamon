import json
import os
import shutil
import datetime
from pathlib import Path

import tqdm

from metamon.backend.team_prediction.dataset import (
    FilteredTeamsFromReplaysDataset,
    parse_replay_team_filename,
)
from metamon.backend.team_prediction.team_index import write_team_index


def _find_smogtours_files(team_path: Path, format_name: str) -> set[str]:
    suffix = f".{format_name}_team"
    selected: set[str] = set()
    for root, _, files in os.walk(team_path):
        for filename in files:
            if not filename.endswith(suffix):
                continue
            if "smogtours" not in filename.lower():
                continue
            full_path = Path(root) / filename
            rel_path = full_path.relative_to(team_path).as_posix()
            selected.add(rel_path)
    return selected


def _write_index_csv(output_dir: Path, rel_paths: list[str]) -> None:
    write_team_index(output_dir, rel_paths)


def _filter_files_by_min_date(
    files: set[str], format_name: str, min_date: str | None
) -> set[str]:
    if min_date is None:
        return files
    cutoff = datetime.datetime.strptime(min_date, "%m-%d-%Y")
    filtered = set()
    for rel_path in files:
        meta = parse_replay_team_filename(Path(rel_path).name, format_name)
        if meta is None:
            continue
        if meta.date >= cutoff:
            filtered.add(rel_path)
    return filtered


def select_elite_filenames(
    replay_teamfile_dir: str,
    format_name: str,
    min_rating: int = 1400,
    min_date: str | None = None,
) -> list[str]:
    """Return high-ELO ∪ smogtours filenames (no copy)."""
    dataset = FilteredTeamsFromReplaysDataset(
        replay_teamfile_dir=replay_teamfile_dir,
        format=format_name,
        min_rating=min_rating,
        min_date=min_date,
    )
    source_team_path = Path(dataset.team_path)
    high_elo_files = _filter_files_by_min_date(
        set(dataset.filenames), format_name, min_date
    )
    smogtours_files = _find_smogtours_files(source_team_path, format_name)
    smogtours_files = _filter_files_by_min_date(smogtours_files, format_name, min_date)
    return sorted(high_elo_files | smogtours_files)


def write_team_index_csv(output_dir: Path, rel_paths: list[str]) -> None:
    write_team_index(output_dir, rel_paths)


def filter_elite_sets(
    replay_teamfile_dir: str,
    base_output_dir: str,
    format_name: str = "gen1ou",
    min_rating: int = 1400,
    min_date: str | None = None,
    max_teams: int | None = None,
    overwrite: bool = False,
) -> dict:
    selected_files = select_elite_filenames(
        replay_teamfile_dir=replay_teamfile_dir,
        format_name=format_name,
        min_rating=min_rating,
        min_date=min_date,
    )
    if max_teams is not None:
        selected_files = selected_files[:max_teams]

    source_team_path = Path(replay_teamfile_dir) / format_name
    high_elo_files = set()
    for f in selected_files:
        meta = parse_replay_team_filename(Path(f).name, format_name)
        if meta and meta.rating_int >= min_rating:
            high_elo_files.add(f)
    smogtours_files = {f for f in selected_files if "smogtours" in Path(f).name.lower()}
    intersection_files = high_elo_files & smogtours_files

    output_dir = Path(base_output_dir) / format_name
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for rel_path in tqdm.tqdm(
        selected_files,
        desc=f"Copying {format_name} elite sets",
        total=len(selected_files),
    ):
        src = source_team_path / rel_path
        if not src.exists():
            continue
        dst = output_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    _write_index_csv(output_dir, selected_files)

    metadata = {
        "format": format_name,
        "source_team_path": str(source_team_path),
        "min_rating": min_rating,
        "min_date": min_date,
        "total_selected": len(selected_files),
        "copied_files": copied,
        "high_elo_count": len(high_elo_files),
        "smogtours_count": len(smogtours_files),
        "intersection_count": len(intersection_files),
        "union_count": len(selected_files),
        "overwrite": overwrite,
    }
    with open(output_dir / "elite_filter_meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata
