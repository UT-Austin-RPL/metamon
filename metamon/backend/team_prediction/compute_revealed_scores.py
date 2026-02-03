#!/usr/bin/env python3
"""
Compute revealed scores for all teams in the dataset.

The revealed score is the fraction of "relevant" attributes that are known,
where "relevant" means attributes that could be masked during training
(respecting generation constraints).

Output:
  - index_scored.csv: filename, gen, revealed_score
  - index_scored_meta.json: per-generation statistics (count, mean, median, quartiles)
"""

import argparse
import csv
import json
import os
import pathlib
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from tqdm import tqdm

import metamon.data.download
from metamon.backend.team_prediction.team import TeamSet, PokemonSet


# Global for include_stats (set by worker initializer)
_include_stats = False


def _init_worker(include_stats: bool):
    """Initialize worker process with global state."""
    global _include_stats
    _include_stats = include_stats


def _process_single_file(args: Tuple[str, str]) -> Optional[Tuple[str, int, float]]:
    """
    Worker function to process a single team file.

    Args:
        args: (full_path, rel_path) tuple

    Returns:
        (rel_path, gen, score) on success, None on error
    """
    full_path, rel_path = args

    try:
        # Infer format from filename
        # Files are like: path/to/file.gen9ou_team
        path = pathlib.Path(full_path)
        format_str = path.suffix.replace("_team", "").lstrip(".")
        if not format_str:
            format_str = path.parent.name

        team = TeamSet.from_showdown_file(full_path, format_str)
        score = team.revealed_score(_include_stats)
        gen = team.gen

        return (rel_path, gen, score)
    except Exception:
        return None


def compute_gen_statistics(scores: List[float]) -> Dict[str, float]:
    """Compute statistics for a list of scores."""
    if not scores:
        return {}

    arr = np.array(scores)
    return {
        "count": len(scores),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "q75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
    }


def process_directory(
    data_dir: str,
    output_filename: str = "index_scored.csv",
    include_stats: bool = False,
    verbose: bool = True,
    num_workers: Optional[int] = None,
) -> Tuple[int, int, List[Tuple[str, int, float]], Dict[str, Any]]:
    """
    Process all team files in a directory and compute revealed scores.

    Returns: (num_processed, num_errors, results, metadata)
        results: List of (filename, gen, score) tuples
        metadata: Per-generation statistics
    """
    d_path = pathlib.Path(data_dir)

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    # Read file list from index.csv (must already exist)
    index_path = d_path / "index.csv"
    if not index_path.exists():
        raise FileNotFoundError(
            f"index.csv not found at {index_path}. "
            "Create it first or run with --scan to generate it."
        )

    if verbose:
        print(f"Reading file list from {index_path}...")

    work_items = []  # (full_path, rel_path)
    with open(index_path, "r") as f:
        lines = f.read().splitlines()[1:]  # skip header
        for rel_path in lines:
            if rel_path:
                full_path = str(d_path / rel_path)
                work_items.append((full_path, rel_path))

    if verbose:
        print(f"Loaded {len(work_items)} team files from index.csv")
        print(f"Processing with {num_workers} workers...")

    results = []  # (filename, gen, score)
    scores_by_gen: Dict[int, List[float]] = defaultdict(list)
    num_errors = 0

    # Process in parallel
    with Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(include_stats,),
    ) as pool:
        # Use imap_unordered for better performance with progress bar
        iterator = pool.imap_unordered(_process_single_file, work_items, chunksize=100)

        if verbose:
            iterator = tqdm(iterator, total=len(work_items), desc="Computing scores")

        for result in iterator:
            if result is None:
                num_errors += 1
            else:
                rel_path, gen, score = result
                results.append((rel_path, gen, score))
                scores_by_gen[gen].append(score)

    # Compute per-generation statistics
    metadata = {
        "total_count": len(results),
        "total_errors": num_errors,
        "include_stats": include_stats,
        "per_generation": {},
    }

    for gen in sorted(scores_by_gen.keys()):
        gen_stats = compute_gen_statistics(scores_by_gen[gen])
        gen_stats["max_attrs_per_pokemon"] = PokemonSet.max_relevant_attrs(
            gen, include_stats
        )
        metadata["per_generation"][f"gen{gen}"] = gen_stats

    # Sort by gen, then by score descending
    results.sort(key=lambda x: (x[1], -x[2]))

    # Write CSV output
    output_path = d_path / output_filename
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "gen", "revealed_score"])
        for filename, gen, score in results:
            writer.writerow([filename, gen, f"{score:.4f}"])

    # Write metadata JSON
    meta_filename = output_filename.replace(".csv", "_meta.json")
    meta_path = d_path / meta_filename
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"\nWrote {len(results)} entries to {output_path}")
        print(f"Wrote metadata to {meta_path}")
        print(f"Errors: {num_errors}")

        # Print per-generation statistics
        print(f"\nPer-generation statistics:")
        print(
            f"{'Gen':<6} {'Count':>8} {'Mean':>8} {'Median':>8} {'Q25':>8} {'Q75':>8} {'Attrs/Mon':>10}"
        )
        print("-" * 60)
        for gen in sorted(scores_by_gen.keys()):
            stats = metadata["per_generation"][f"gen{gen}"]
            print(
                f"Gen {gen:<2} {stats['count']:>8} {stats['mean']:>7.1%} {stats['median']:>7.1%} "
                f"{stats['q25']:>7.1%} {stats['q75']:>7.1%} {stats['max_attrs_per_pokemon']:>10}"
            )

        # Overall histogram
        all_scores = [s for _, _, s in results]
        if all_scores:
            print(f"\nOverall score distribution:")
            print(
                f"  Min: {min(all_scores):.2%}, Max: {max(all_scores):.2%}, Mean: {np.mean(all_scores):.2%}"
            )

            buckets = [0] * 10
            for s in all_scores:
                bucket = min(int(s * 10), 9)
                buckets[bucket] += 1
            print(f"\n  Histogram (10% buckets):")
            for i, count in enumerate(buckets):
                pct = count / len(all_scores) * 100
                bar = "█" * int(pct / 2)
                print(f"    {i*10:2d}-{(i+1)*10:2d}%: {count:6d} ({pct:5.1f}%) {bar}")

        # Find and print most/least revealed teams per generation
        print(f"\n{'='*60}")
        print("Most and least revealed teams per generation:")
        print(f"{'='*60}")

        # Group by generation
        results_by_gen: Dict[int, List[Tuple[str, int, float]]] = defaultdict(list)
        for rel_path, gen, score in tqdm(results, desc="Grouping by generation"):
            results_by_gen[gen].append((rel_path, gen, score))

        for gen in sorted(results_by_gen.keys()):
            gen_results = results_by_gen[gen]
            if not gen_results:
                continue

            # Find min and max
            most_revealed = max(gen_results, key=lambda x: x[2])
            least_revealed = min(gen_results, key=lambda x: x[2])

            print(f"\n--- Gen {gen} ---")

            # Most revealed
            print(f"\nMOST REVEALED (score: {most_revealed[2]:.1%}):")
            print(f"  File: {most_revealed[0]}")
            try:
                most_path = d_path / most_revealed[0]
                # Infer format from filename
                format_str = most_path.suffix.replace("_team", "").lstrip(".")
                if not format_str:
                    format_str = most_path.parent.name
                team = TeamSet.from_showdown_file(str(most_path), format_str)
                print(team.to_str())
            except Exception as e:
                print(f"  (Could not load team: {e})")

            # Least revealed
            print(f"\nLEAST REVEALED (score: {least_revealed[2]:.1%}):")
            print(f"  File: {least_revealed[0]}")
            try:
                least_path = d_path / least_revealed[0]
                format_str = least_path.suffix.replace("_team", "").lstrip(".")
                if not format_str:
                    format_str = least_path.parent.name
                team = TeamSet.from_showdown_file(str(least_path), format_str)
                print(team.to_str())
            except Exception as e:
                print(f"  (Could not load team: {e})")

    return len(results), num_errors, results, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Compute revealed scores for team files"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing team files. Defaults to downloaded revealed_teams.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="index_scored.csv",
        help="Output filename (will be created in each data directory)",
    )
    parser.add_argument(
        "--include-stats",
        action="store_true",
        help="Include nature/EVs/IVs in the score calculation",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count - 1)",
    )

    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = metamon.data.download.download_revealed_teams()

    # Process each format subdirectory
    data_path = pathlib.Path(args.data_dir)

    # Check if this is a top-level dir with format subdirs, or a single format dir
    subdirs = [d for d in data_path.iterdir() if d.is_dir()]
    has_team_files = any(
        str(f).endswith("team") for f in data_path.rglob("*") if f.is_file()
    )

    if subdirs and not has_team_files:
        # Process each subdirectory (format) separately
        for subdir in sorted(subdirs):
            if not args.quiet:
                print(f"\n{'='*60}")
                print(f"Processing {subdir.name}")
                print(f"{'='*60}")
            _, _, _, _ = process_directory(
                str(subdir),
                output_filename=args.output,
                include_stats=args.include_stats,
                verbose=not args.quiet,
                num_workers=args.workers,
            )
    else:
        # Process single directory
        _, _, _, _ = process_directory(
            args.data_dir,
            output_filename=args.output,
            include_stats=args.include_stats,
            verbose=not args.quiet,
            num_workers=args.workers,
        )


if __name__ == "__main__":
    main()
