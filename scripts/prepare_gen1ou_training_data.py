#!/usr/bin/env python3
"""Prepare local Gen1 OU replay data for metamon training.

This script does not copy replay payloads. It creates a flat symlink view for
locally generated trajectories so MetamonDataset can load them as one custom
dataset, and it writes filename caches for official self-play tar archives.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


FORMAT = "gen1ou"
SELF_PLAY_SUBSETS = ("pac-base", "pac-exploratory")


def default_cache_dir() -> Path:
    return Path(os.environ.get("METAMON_CACHE_DIR", "~/metamon_cache")).expanduser()


def collect_local_files(source_roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in source_roots:
        if not root.exists():
            continue
        files.extend(sorted(root.rglob(f"{FORMAT}/*.json.lz4")))
    return sorted(files)


def ensure_unique_basenames(files: list[Path]) -> None:
    counts = Counter(path.name for path in files)
    duplicates = [name for name, count in counts.items() if count > 1]
    if duplicates:
        preview = ", ".join(duplicates[:10])
        raise RuntimeError(
            f"Found {len(duplicates)} duplicate trajectory basenames. "
            f"Cannot create a flat symlink view safely. Examples: {preview}"
        )


def write_flat_index(dataset_root: Path, filenames: list[str]) -> Path:
    index_path = dataset_root / "index.csv"
    with index_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename"])
        for name in filenames:
            writer.writerow([f"{FORMAT}/{name}"])
    return index_path


def create_local_view(files: list[Path], output_root: Path) -> dict:
    dataset_root = output_root / "local-generated"
    format_dir = dataset_root / FORMAT
    format_dir.mkdir(parents=True, exist_ok=True)

    linked = 0
    for source in files:
        target = format_dir / source.name
        if target.exists() or target.is_symlink():
            if not target.is_symlink() or target.resolve() != source.resolve():
                raise RuntimeError(f"Refusing to overwrite existing path: {target}")
            continue
        target.symlink_to(source)
        linked += 1

    index_path = write_flat_index(dataset_root, [path.name for path in files])

    by_dir = defaultdict(int)
    for path in files:
        by_dir[str(path.parent)] += 1

    return {
        "dataset_root": str(dataset_root),
        "format_dir": str(format_dir),
        "index_csv": str(index_path),
        "files": len(files),
        "new_symlinks": linked,
        "source_dirs": dict(sorted(by_dir.items())),
    }


def count_human_replays(cache_dir: Path) -> dict:
    root = cache_dir / "parsed-replays"
    format_dir = root / FORMAT
    count = sum(1 for _ in format_dir.glob("*.json.lz4")) if format_dir.exists() else 0
    return {
        "dataset_root": str(root),
        "format_dir": str(format_dir),
        "files": count,
    }


def write_self_play_index(cache_dir: Path, subset: str) -> dict:
    subset_root = cache_dir / "self-play" / subset
    tar_path = subset_root / f"{FORMAT}.tar"
    sqlite_path = subset_root / subset / f"{FORMAT}.tar.index.sqlite"
    text_index_path = subset_root / f"{FORMAT}.tar.index.txt"

    info = {
        "subset": subset,
        "dataset_root": str(subset_root),
        "tar": str(tar_path),
        "sqlite_index": str(sqlite_path),
        "text_index": str(text_index_path),
        "files": 0,
        "written": False,
        "available": tar_path.exists(),
    }

    if not tar_path.exists():
        return info
    if not sqlite_path.exists():
        raise RuntimeError(f"Missing tar SQLite index: {sqlite_path}")

    con = sqlite3.connect(sqlite_path)
    try:
        rows = con.execute(
            "select path, name from files where name like '%.json.lz4' order by path, name"
        )
        with text_index_path.open("w") as f:
            for path, name in rows:
                member = f"{str(path).strip('/')}/{name}"
                f.write(member + "\n")
                info["files"] += 1
        info["written"] = True
    finally:
        con.close()

    return info


def write_manifest(output_root: Path, manifest: dict) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Gen1 OU replay data for metamon training."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_cache_dir() / "training" / "gen1ou",
        help="Where to create the prepared training view.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=default_cache_dir(),
        help="Metamon cache directory containing parsed-replays and self-play.",
    )
    parser.add_argument(
        "--source-roots",
        type=Path,
        nargs="+",
        default=[Path("~/metamon/trajectories"), Path("~/metamon/other")],
        help="Roots to search for local */gen1ou/*.json.lz4 trajectories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    source_roots = [path.expanduser().resolve() for path in args.source_roots]

    local_files = collect_local_files(source_roots)
    ensure_unique_basenames(local_files)

    local = create_local_view(local_files, output_root)
    human = count_human_replays(cache_dir)
    self_play = [write_self_play_index(cache_dir, subset) for subset in SELF_PLAY_SUBSETS]

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "format": FORMAT,
        "cache_dir": str(cache_dir),
        "output_root": str(output_root),
        "local_generated": local,
        "official_human_replays": human,
        "official_self_play": self_play,
    }
    manifest_path = write_manifest(output_root, manifest)

    print(f"Prepared Gen1 OU data manifest: {manifest_path}")
    print(
        "Local generated: "
        f"{local['files']} files at {local['dataset_root']} "
        f"({local['new_symlinks']} new symlinks)"
    )
    print(
        "Official human replays: "
        f"{human['files']} files at {human['dataset_root']}"
    )
    for item in self_play:
        status = "indexed" if item["written"] else "missing"
        print(f"Official self-play {item['subset']}: {item['files']} files ({status})")


if __name__ == "__main__":
    main()
