"""
Build replay-derived team sets from cached revealed_teams.

Set definitions live in YAML (see team_sets_gl_hl_05_26.yaml, team_sets_may26_gen9ou.yaml).
Each set has set_type: gl (general ladder) or hl (high ladder ∪ smogtours).

Pipeline per format:
  1. Select filenames (FilteredTeamsFromReplaysDataset or filter_elite.select_elite_filenames)
  2. Fill with NaiveUsagePredictor (+ optional post-fill revealed_score filter for hl)
  3. Write index.csv (+ predictions_meta.json)

Example (year-window gl/hl):
  python -m metamon.backend.team_prediction.build_replay_team_sets --set all

Example (May 2026 gen9ou supplement):
  python -m metamon.backend.team_prediction.build_replay_team_sets \\
    --config metamon/backend/team_prediction/team_sets_may26_gen9ou.yaml \\
    --set all --formats gen9ou --validate
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Optional

import tqdm
import yaml

from metamon.config import METAMON_CACHE_DIR
from metamon.backend.team_prediction.dataset import (
    FilteredTeamsFromReplaysDataset,
    default_revealed_teams_dir,
    parse_replay_team_filename,
)
from metamon.backend.team_prediction.filter_elite import (
    select_elite_filenames,
    write_team_index_csv,
)
from metamon.backend.team_prediction.predictor import NaiveUsagePredictor
from metamon.backend.team_prediction.team import TeamSet

DEFAULT_CONFIG = Path(__file__).resolve().parent / "team_sets_gl_hl_05_26.yaml"
FORMATS = ["gen1ou", "gen2ou", "gen3ou", "gen4ou", "gen9ou"]
_FILL_PREDICTOR: NaiveUsagePredictor | None = None


def _default_workers() -> int:
    return max(1, min(64, cpu_count() - 4))


def _init_fill_worker() -> None:
    global _FILL_PREDICTOR
    _FILL_PREDICTOR = NaiveUsagePredictor()


def load_config(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def config_set_names(cfg: dict[str, Any]) -> list[str]:
    return [key for key in cfg if key != "window"]


def resolve_set_type(set_name: str, set_cfg: dict[str, Any]) -> str:
    set_type = set_cfg.get("set_type")
    if set_type is not None:
        return set_type
    if set_name.startswith("gl"):
        return "gl"
    if set_name.startswith("hl"):
        return "hl"
    raise ValueError(
        f"Set {set_name!r} needs set_type: gl|hl in config (or a gl_/hl_ prefix)"
    )


def resolve_min_rating(selection: dict, format_name: str) -> Optional[int]:
    per_fmt = selection.get("per_format") or {}
    if format_name in per_fmt and "min_rating" in per_fmt[format_name]:
        return per_fmt[format_name]["min_rating"]
    default = selection.get("default") or {}
    return default.get("min_rating")


def resolve_min_revealed_score(hl_cfg: dict, format_name: str) -> Optional[float]:
    scores = hl_cfg.get("min_revealed_score") or {}
    if format_name in scores:
        return scores[format_name]
    return scores.get("default")


def select_gl_filenames(
    revealed_dir: str,
    format_name: str,
    window: dict,
    selection: dict,
) -> list[str]:
    dataset = FilteredTeamsFromReplaysDataset(
        replay_teamfile_dir=revealed_dir,
        format=format_name,
        min_date=window.get("min_date"),
        max_date=window.get("max_date"),
        min_rating=resolve_min_rating(selection, format_name),
        sort_by_date=bool(selection.get("sort_by_date", False)),
        max_teams=selection.get("max_teams"),
    )
    return list(dataset.filenames)


def select_hl_filenames(
    revealed_dir: str,
    format_name: str,
    window: dict,
    selection: dict,
) -> list[str]:
    min_rating = resolve_min_rating(selection, format_name)
    if min_rating is None:
        raise ValueError(f"hl_05_26 requires min_rating for {format_name}")
    return select_elite_filenames(
        replay_teamfile_dir=revealed_dir,
        format_name=format_name,
        min_rating=min_rating,
        min_date=window.get("min_date"),
    )


def _fill_one(args: tuple) -> Optional[dict]:
    global _FILL_PREDICTOR
    if _FILL_PREDICTOR is None:
        _FILL_PREDICTOR = NaiveUsagePredictor()
    idx, rel_path, src_root, format_name, min_revealed_score, output_dir = args
    src_path = os.path.join(src_root, rel_path)
    basename = os.path.basename(rel_path)
    meta = parse_replay_team_filename(basename, format_name)
    if meta is None:
        return None
    try:
        team = TeamSet.from_showdown_file(src_path, format=format_name)
        predicted = _FILL_PREDICTOR.predict(
            team,
            date=meta.date.date(),
            rating=meta.rating_raw,
            gameid=meta.battle_id,
        )
        score = predicted.revealed_score(include_stats=False)
        if min_revealed_score is not None and score < min_revealed_score:
            return None
        out_name = f"team_{idx:06d}.{format_name}_team"
        predicted.write_to_file(os.path.join(output_dir, out_name))
        return {
            "output_file": out_name,
            "revealed_score": float(score),
            "source_file": rel_path,
        }
    except Exception:
        return None


def fill_and_write(
    revealed_dir: str,
    format_name: str,
    rel_paths: list[str],
    output_dir: Path,
    min_revealed_score: Optional[float],
    workers: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    src_root = os.path.join(revealed_dir, format_name)
    out_dir_str = str(output_dir)
    work = [
        (i, rel_path, src_root, format_name, min_revealed_score, out_dir_str)
        for i, rel_path in enumerate(rel_paths)
    ]
    chunksize = max(1, len(work) // (max(1, workers) * 8))
    kept = []
    with Pool(max(1, workers), initializer=_init_fill_worker) as pool:
        for result in tqdm.tqdm(
            pool.imap_unordered(_fill_one, work, chunksize=chunksize),
            total=len(work),
            desc=f"Fill {format_name}",
        ):
            if result is not None:
                kept.append(result)

    kept.sort(key=lambda row: row["output_file"])
    out_names = [row["output_file"] for row in kept]
    meta_rows = [
        {
            "output_file": row["output_file"],
            "revealed_score": row["revealed_score"],
            "source_file": row["source_file"],
        }
        for row in kept
    ]

    write_team_index_csv(output_dir, out_names)
    with open(output_dir / "predictions_meta.json", "w") as f:
        json.dump(meta_rows, f, indent=2)

    return {
        "format": format_name,
        "input_count": len(rel_paths),
        "output_count": len(out_names),
        "min_revealed_score": min_revealed_score,
        "output_dir": str(output_dir),
    }


def run_validate(
    format_name: str,
    input_root: Path,
    output_root: Path,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "metamon.backend.team_prediction.validate",
        format_name,
        "--input-path",
        str(input_root),
        "--output-path",
        str(output_root),
    ]
    subprocess.run(cmd, check=True)


def build_set(
    set_name: str,
    cfg: dict,
    revealed_dir: str,
    cache_dir: str,
    formats: list[str],
    workers: int,
    validate: bool,
) -> None:
    window = cfg["window"]
    set_cfg = cfg[set_name]
    selection = set_cfg["selection"]
    out_unfiltered = Path(cache_dir) / set_cfg["output"]["unfiltered"]
    out_verified = Path(cache_dir) / set_cfg["output"]["verified"]

    set_type = resolve_set_type(set_name, set_cfg)
    print(f"\n=== {set_name} ({set_type}) ===")
    summary = []
    for format_name in formats:
        if set_type == "gl":
            rel_paths = select_gl_filenames(
                revealed_dir, format_name, window, selection
            )
            min_score = None
        elif set_type == "hl":
            rel_paths = select_hl_filenames(
                revealed_dir, format_name, window, selection
            )
            min_score = resolve_min_revealed_score(set_cfg, format_name)
        else:
            raise ValueError(f"Unknown set_type {set_type!r} for {set_name}")

        print(f"{format_name}: selected {len(rel_paths):,} teams")
        if not rel_paths:
            continue

        meta = fill_and_write(
            revealed_dir=revealed_dir,
            format_name=format_name,
            rel_paths=rel_paths,
            output_dir=out_unfiltered / format_name,
            min_revealed_score=min_score,
            workers=workers,
        )
        summary.append(meta)
        print(json.dumps(meta, indent=2))

        if validate:
            print(f"Validating {format_name} -> {out_verified / format_name}")
            run_validate(format_name, out_unfiltered, out_verified)

    summary_path = out_unfiltered / "build_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Build replay-derived team sets")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--set",
        nargs="+",
        default=["all"],
        metavar="SET",
        help="Set name(s) from config, or 'all' (default: all sets in config)",
    )
    parser.add_argument("--formats", nargs="+", default=FORMATS)
    parser.add_argument("--revealed-teams-dir", default=None)
    parser.add_argument("--cache-dir", default=METAMON_CACHE_DIR)
    parser.add_argument("--workers", type=int, default=_default_workers())
    parser.add_argument(
        "--min-date",
        type=str,
        default=None,
        help='Override window min_date (MM-DD-YYYY), e.g. "05-01-2026"',
    )
    parser.add_argument(
        "--max-date",
        type=str,
        default=None,
        help="Override window max_date (MM-DD-YYYY)",
    )
    parser.add_argument(
        "--max-teams",
        type=int,
        default=None,
        help="Override gl set max_teams cap from config",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validate.py after fill (Showdown legality check)",
    )
    args = parser.parse_args()

    if args.cache_dir is None:
        raise ValueError("METAMON_CACHE_DIR must be set (or pass --cache-dir)")

    revealed_dir = args.revealed_teams_dir or default_revealed_teams_dir()
    cfg = load_config(args.config)
    available = config_set_names(cfg)
    if "all" in args.set:
        sets = available
    else:
        unknown = set(args.set) - set(available)
        if unknown:
            raise ValueError(
                f"Unknown set(s) {sorted(unknown)}; available: {available}"
            )
        sets = args.set

    window = dict(cfg["window"])
    if args.min_date is not None:
        window["min_date"] = args.min_date
    if args.max_date is not None:
        window["max_date"] = args.max_date
    cfg = {**cfg, "window": window}

    print(f"Config: {args.config}")
    print(f"Sets: {sets}")
    print(f"Window: {window.get('min_date')} .. {window.get('max_date') or 'now'}")
    print(f"Revealed teams: {revealed_dir}")
    print(f"Cache output: {args.cache_dir}/teams/")

    for set_name in sets:
        set_cfg = dict(cfg[set_name])
        if args.max_teams is not None and resolve_set_type(set_name, set_cfg) == "gl":
            set_cfg = {
                **set_cfg,
                "selection": {**set_cfg["selection"], "max_teams": args.max_teams},
            }
        build_set(
            set_name=set_name,
            cfg={**cfg, set_name: set_cfg},
            revealed_dir=revealed_dir,
            cache_dir=args.cache_dir,
            formats=args.formats,
            workers=args.workers,
            validate=args.validate,
        )


if __name__ == "__main__":
    main()
