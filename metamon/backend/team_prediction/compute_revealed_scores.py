"""
Compute revealed scores for all teams in the dataset.

Output:
  - index_scored.csv: filename, gen, revealed_score (sorted by gen, then score desc)
  - index_scored_meta.json: per-generation statistics
"""

import argparse
import csv
import json
import pathlib
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

import metamon.data.download
from metamon.backend.team_prediction.team import TeamSet, PokemonSet

_include_stats = False


def _init_worker(include_stats: bool):
    global _include_stats
    _include_stats = include_stats


def _process_single_file(args):
    full_path, rel_path = args
    try:
        path = pathlib.Path(full_path)
        format_str = path.suffix.replace("_team", "").lstrip(".")
        if not format_str:
            format_str = path.parent.name

        team = TeamSet.from_showdown_file(full_path, format_str)
        return (rel_path, team.gen, team.revealed_score(_include_stats))
    except Exception:
        return None


def compute_gen_statistics(scores):
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
    num_workers: int = None,
):
    """Process all team files and compute revealed scores."""
    d_path = pathlib.Path(data_dir)
    num_workers = num_workers or max(1, cpu_count() - 1)

    index_path = d_path / "index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"index.csv not found at {index_path}")

    if verbose:
        print(f"Reading file list from {index_path}...")

    work_items = []
    with open(index_path, "r") as f:
        for rel_path in f.read().splitlines()[1:]:
            if rel_path:
                work_items.append((str(d_path / rel_path), rel_path))

    if verbose:
        print(f"Processing {len(work_items)} files with {num_workers} workers...")

    results = []
    scores_by_gen = defaultdict(list)
    num_errors = 0

    with Pool(num_workers, initializer=_init_worker, initargs=(include_stats,)) as pool:
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

    # Build metadata
    metadata = {
        "total_count": len(results),
        "total_errors": num_errors,
        "include_stats": include_stats,
        "per_generation": {},
    }
    for gen in sorted(scores_by_gen.keys()):
        stats = compute_gen_statistics(scores_by_gen[gen])
        stats["max_attrs_per_pokemon"] = PokemonSet.max_relevant_attrs(
            gen, include_stats
        )
        metadata["per_generation"][f"gen{gen}"] = stats

    # Sort by gen, then score descending
    results.sort(key=lambda x: (x[1], -x[2]))

    # Write outputs
    output_path = d_path / output_filename
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "gen", "revealed_score"])
        for filename, gen, score in results:
            writer.writerow([filename, gen, f"{score:.4f}"])

    meta_path = d_path / output_filename.replace(".csv", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"\nWrote {len(results)} entries to {output_path}")
        print(f"Errors: {num_errors}")
        _print_statistics(scores_by_gen, metadata, results, d_path)

    return len(results), num_errors, results, metadata


def _print_statistics(scores_by_gen, metadata, results, d_path):
    """Print per-generation stats and example teams."""
    print(f"\n{'Gen':<6} {'Count':>8} {'Mean':>8} {'Median':>8} {'Q25':>8} {'Q75':>8}")
    print("-" * 50)
    for gen in sorted(scores_by_gen.keys()):
        s = metadata["per_generation"][f"gen{gen}"]
        print(
            f"Gen {gen:<2} {s['count']:>8} {s['mean']:>7.1%} {s['median']:>7.1%} {s['q25']:>7.1%} {s['q75']:>7.1%}"
        )

    all_scores = [s for _, _, s in results]
    if all_scores:
        print(
            f"\nOverall: min={min(all_scores):.1%}, max={max(all_scores):.1%}, mean={np.mean(all_scores):.1%}"
        )

        buckets = [0] * 10
        for s in all_scores:
            buckets[min(int(s * 10), 9)] += 1
        print(f"\nHistogram:")
        for i, count in enumerate(buckets):
            pct = count / len(all_scores) * 100
            print(
                f"  {i*10:2d}-{(i+1)*10:2d}%: {count:6d} ({pct:4.1f}%) {'█' * int(pct / 2)}"
            )

    # Show most/least revealed per gen
    print(f"\n{'='*50}\nMost/least revealed per generation:\n{'='*50}")
    results_by_gen = defaultdict(list)
    for r in results:
        results_by_gen[r[1]].append(r)

    for gen in sorted(results_by_gen.keys()):
        gen_results = results_by_gen[gen]
        if not gen_results:
            continue

        most = max(gen_results, key=lambda x: x[2])
        least = min(gen_results, key=lambda x: x[2])

        print(f"\n--- Gen {gen} ---")
        for label, item in [("MOST", most), ("LEAST", least)]:
            print(f"\n{label} ({item[2]:.1%}): {item[0]}")
            try:
                path = d_path / item[0]
                fmt = path.suffix.replace("_team", "").lstrip(".") or path.parent.name
                print(TeamSet.from_showdown_file(str(path), fmt).to_str())
            except Exception as e:
                print(f"  (Could not load: {e})")


def main():
    parser = argparse.ArgumentParser(
        description="Compute revealed scores for team files"
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default="index_scored.csv")
    parser.add_argument(
        "--include-stats", action="store_true", help="Include nature/EVs/IVs"
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = metamon.data.download.download_revealed_teams()

    data_path = pathlib.Path(args.data_dir)
    subdirs = [d for d in data_path.iterdir() if d.is_dir()]
    has_team_files = any(
        str(f).endswith("team") for f in data_path.rglob("*") if f.is_file()
    )

    if subdirs and not has_team_files:
        for subdir in sorted(subdirs):
            if not args.quiet:
                print(f"\n{'='*50}\nProcessing {subdir.name}\n{'='*50}")
            process_directory(
                str(subdir),
                output_filename=args.output,
                include_stats=args.include_stats,
                verbose=not args.quiet,
                num_workers=args.workers,
            )
    else:
        process_directory(
            args.data_dir,
            output_filename=args.output,
            include_stats=args.include_stats,
            verbose=not args.quiet,
            num_workers=args.workers,
        )


if __name__ == "__main__":
    main()
