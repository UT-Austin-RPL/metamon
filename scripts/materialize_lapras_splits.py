#!/usr/bin/env python3
"""Materialize manifest splits as MetamonDataset-compatible symlink trees."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path("/home/eddie/metamon/trajectories/lapras/manifest.csv")
DEFAULT_OUTPUT_DIR = Path("/home/eddie/metamon/trajectories/lapras/splits/v1")


def read_manifest(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("pyarrow is required to read parquet manifests") from exc
        return pq.read_table(path).to_pylist()

    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_index(split_root: Path, rel_paths: list[str]) -> None:
    with (split_root / "index.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename"])
        for rel_path in sorted(rel_paths):
            writer.writerow([rel_path])


def materialize(
    rows: list[dict[str, Any]],
    output_dir: Path,
    viewpoint: str,
    overwrite: bool,
) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    index_rows: dict[Path, list[str]] = {}

    for row in rows:
        if row["viewpoint"] != viewpoint:
            continue
        split = row["split"]
        if split not in counts:
            continue
        fmt = row["battle_format"]
        source = Path(row["trajectory_path"]).resolve()
        split_root = output_dir / split / viewpoint
        target_dir = split_root / fmt
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name

        if target.exists() or target.is_symlink():
            if target.is_symlink() and target.resolve() == source:
                pass
            elif overwrite:
                target.unlink()
                target.symlink_to(source)
            else:
                raise FileExistsError(
                    f"{target} already exists and does not point to {source}"
                )
        else:
            target.symlink_to(source)

        index_rows.setdefault(split_root, []).append(f"{fmt}/{target.name}")
        counts[split] += 1

    for split_root, rel_paths in index_rows.items():
        write_index(split_root, rel_paths)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--viewpoint", default="lapras")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = read_manifest(args.manifest.expanduser().resolve())
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = materialize(
        rows=rows,
        output_dir=output_dir,
        viewpoint=args.viewpoint,
        overwrite=args.overwrite,
    )
    print(f"Materialized {args.viewpoint} splits under {output_dir}")
    for split, count in sorted(counts.items()):
        print(f"  {split}: {count:,}")


if __name__ == "__main__":
    main()
