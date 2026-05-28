#!/usr/bin/env python3
"""Build a manifest and split report for Lapras specialist trajectories.

The saved trajectory files only contain ``states`` and ``actions``, so model
identity and team-set metadata are inferred from the collection usernames in
the filename. Override the defaults with ``--username-condition`` when adding
new collection jobs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lz4.frame


DEFAULT_SOURCE_ROOT = Path("/home/eddie/metamon/trajectories/lapras")
DEFAULT_OUTPUT_DIR = DEFAULT_SOURCE_ROOT
DEFAULT_TEAM_FILE = Path(
    "/home/eddie/metamon_cache/teams/lapras/gen1ou/"
    "team_066e7c8ed917b9d7_chansey_jynx_lapras_v2.gen1ou_team"
)

DEFAULT_USERNAME_CONDITIONS = {
    "articuno01": ("Articuno", "smogon_pass2"),
    "articuno02": ("Articuno", "smogon_pass2"),
    "tauros": ("TaurosV0", "smogon_pass2"),
    "taurose": ("TaurosEnsemble", "smogon_pass2"),
}

MANIFEST_FIELDS = [
    "battle_id",
    "trajectory_path",
    "viewpoint",
    "player_username",
    "opponent_username",
    "player_agent",
    "opponent_agent",
    "opponent_key",
    "player_team_hash",
    "opponent_team_hash",
    "battle_format",
    "result",
    "num_turns",
    "sampling_temperature",
    "source_dir",
    "snapshot_id",
    "split",
    "nonfinal_missing_action_labels",
    "negative_action_labels",
]


def _sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _strip_known_suffixes(name: str) -> str:
    if name.endswith(".json.lz4"):
        return name[: -len(".json.lz4")]
    if name.endswith(".json"):
        return name[: -len(".json")]
    return name


def parse_filename(path: Path) -> dict[str, str]:
    name = _strip_known_suffixes(path.name)
    prefix, _timestamp, result = name.rsplit("_", 2)
    left, opponent_username = prefix.rsplit("_vs_", 1)
    battle_part, _rating, player_username = left.split("_", 2)
    battle_bits = battle_part.split("-")
    battle_id = battle_bits[-1]
    battle_format = battle_bits[-2] if len(battle_bits) >= 2 else "unknown"
    return {
        "battle_id": battle_id,
        "battle_format": battle_format,
        "player_username": player_username,
        "opponent_username": opponent_username,
        "result": result.lower(),
    }


def parse_username_conditions(values: list[str]) -> dict[str, tuple[str, str]]:
    conditions = dict(DEFAULT_USERNAME_CONDITIONS)
    for value in values:
        try:
            username, rest = value.split("=", 1)
            agent, team_set = rest.split("/", 1)
        except ValueError as exc:
            raise ValueError(
                "--username-condition must have the form username=Agent/team_set"
            ) from exc
        conditions[username.lower()] = (agent, team_set)
    return conditions


def infer_viewpoint(path: Path, source_root: Path) -> str:
    rel = path.relative_to(source_root)
    return rel.parts[0]


def infer_agent(
    username: str,
    viewpoint: str,
    lapras_agent: str,
    username_conditions: dict[str, tuple[str, str]],
) -> str:
    if viewpoint == "lapras" or username.lower().startswith("lapras"):
        return lapras_agent
    return username_conditions.get(username.lower(), (username, "unknown"))[0]


def infer_opponent_key(
    opponent_username: str,
    opponent_agent: str,
    username_conditions: dict[str, tuple[str, str]],
) -> str:
    _agent, team_set = username_conditions.get(
        opponent_username.lower(), (opponent_agent, "unknown")
    )
    return f"{opponent_agent}/{team_set}"


def canonical_team_from_state(state: dict[str, Any]) -> str:
    pokemon = [state["player_active_pokemon"], *state.get("available_switches", [])]
    entries = []
    for mon in pokemon:
        moves = sorted(move.get("name", "").lower() for move in mon.get("moves", []))
        entries.append(f"{mon.get('name', '').lower()}|{','.join(moves)}")
    return "\n".join(sorted(entries))


def hash_team_file(path: Path) -> str | None:
    if not path.exists():
        return None
    text = "\n".join(
        line.strip().lower() for line in path.read_text().splitlines() if line.strip()
    )
    return _sha16(text)


def load_trajectory_summary(path: Path) -> dict[str, Any]:
    with lz4.frame.open(path, "rb") as f:
        data = json.loads(f.read().decode("utf-8"))
    actions = data["actions"]
    nonfinal_actions = actions[:-1]
    first_state = data["states"][0]
    return {
        "num_turns": max(len(actions) - 1, 0),
        "player_team_hash": _sha16(canonical_team_from_state(first_state)),
        "nonfinal_missing_action_labels": sum(
            1 for action in nonfinal_actions if action < 0
        ),
        "negative_action_labels": sum(1 for action in actions if action < 0),
    }


def split_battles(
    rows: list[dict[str, Any]],
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> dict[str, str]:
    battle_meta: dict[str, tuple[str, str]] = {}
    fallback: dict[str, tuple[str, str]] = {}
    for row in rows:
        meta = (row["opponent_key"], row["result"])
        fallback.setdefault(row["battle_id"], meta)
        if row["viewpoint"] == "lapras":
            battle_meta[row["battle_id"]] = meta
    for battle_id, meta in fallback.items():
        battle_meta.setdefault(battle_id, meta)

    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for battle_id, meta in battle_meta.items():
        strata[meta].append(battle_id)

    assignments: dict[str, str] = {}
    rng = random.Random(seed)
    for key in sorted(strata):
        battle_ids = sorted(strata[key])
        rng.shuffle(battle_ids)
        n_total = len(battle_ids)
        n_train = int(n_total * train_fraction)
        n_val = int(n_total * val_fraction)
        for battle_id in battle_ids[:n_train]:
            assignments[battle_id] = "train"
        for battle_id in battle_ids[n_train : n_train + n_val]:
            assignments[battle_id] = "val"
        for battle_id in battle_ids[n_train + n_val :]:
            assignments[battle_id] = "test"
    return assignments


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_parquet(rows: list[dict[str, Any]], path: Path) -> str | None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return "pyarrow is not installed; skipped parquet output"
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)
    return None


def percentile(values: list[int], pct: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = min(round((len(values) - 1) * pct), len(values) - 1)
    return float(values[idx])


def build_report(
    rows: list[dict[str, Any]], parquet_warning: str | None
) -> dict[str, Any]:
    by_split = Counter(row["split"] for row in rows)
    by_viewpoint = Counter(row["viewpoint"] for row in rows)
    by_split_battles = defaultdict(set)
    by_opponent_result = Counter()
    team_hashes = defaultdict(set)
    lengths = []
    leakage = defaultdict(set)

    for row in rows:
        by_split_battles[row["split"]].add(row["battle_id"])
        by_opponent_result[
            (row["viewpoint"], row["opponent_key"], row["result"])
        ] += 1
        team_hashes[row["viewpoint"]].add(row["player_team_hash"])
        lengths.append(int(row["num_turns"]))
        leakage[row["battle_id"]].add(row["split"])

    battle_leakage = sorted(
        battle_id for battle_id, splits in leakage.items() if len(splits) > 1
    )
    missing_labels = sum(int(row["nonfinal_missing_action_labels"]) for row in rows)
    negative_labels = sum(int(row["negative_action_labels"]) for row in rows)

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(rows),
        "n_battles_by_split": {
            split: len(battle_ids)
            for split, battle_ids in sorted(by_split_battles.items())
        },
        "n_trajectories_by_split": dict(sorted(by_split.items())),
        "n_trajectories_by_viewpoint": dict(sorted(by_viewpoint.items())),
        "win_rate_by_viewpoint_and_opponent": {
            f"{viewpoint}|{opponent_key}": {
                "wins": by_opponent_result[(viewpoint, opponent_key, "win")],
                "losses": by_opponent_result[(viewpoint, opponent_key, "loss")],
                "win_rate": (
                    by_opponent_result[(viewpoint, opponent_key, "win")]
                    / max(
                        by_opponent_result[(viewpoint, opponent_key, "win")]
                        + by_opponent_result[(viewpoint, opponent_key, "loss")],
                        1,
                    )
                ),
            }
            for viewpoint, opponent_key in sorted(
                {(row["viewpoint"], row["opponent_key"]) for row in rows}
            )
        },
        "battle_length": {
            "mean": statistics.fmean(lengths) if lengths else None,
            "p50": percentile(lengths, 0.50),
            "p95": percentile(lengths, 0.95),
        },
        "missing_action_labels": {
            "nonfinal": missing_labels,
            "all_negative_including_final": negative_labels,
        },
        "team_hash_cardinality_by_viewpoint": {
            viewpoint: len(hashes)
            for viewpoint, hashes in sorted(team_hashes.items())
        },
        "battle_id_leakage_across_splits": battle_leakage[:50],
        "battle_id_leakage_count": len(battle_leakage),
        "parquet_warning": parquet_warning,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--snapshot_id", default="v1")
    parser.add_argument("--lapras_agent", default="Articuno")
    parser.add_argument("--sampling_temperature", type=float, default=1.5)
    parser.add_argument("--train_fraction", type=float, default=0.70)
    parser.add_argument("--val_fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress_interval", type=int, default=10000)
    parser.add_argument("--lapras_team_file", type=Path, default=DEFAULT_TEAM_FILE)
    parser.add_argument(
        "--username-condition",
        action="append",
        default=[],
        help="Override username metadata as username=Agent/team_set.",
    )
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    username_conditions = parse_username_conditions(args.username_condition)
    lapras_team_hash = hash_team_file(args.lapras_team_file.expanduser())

    paths = sorted(source_root.glob("*/*/*.json.lz4"))
    if args.limit is not None:
        paths = paths[: args.limit]

    rows = []
    for idx, path in enumerate(paths, 1):
        meta = parse_filename(path)
        viewpoint = infer_viewpoint(path, source_root)
        player_agent = infer_agent(
            meta["player_username"], viewpoint, args.lapras_agent, username_conditions
        )
        opponent_agent = infer_agent(
            meta["opponent_username"],
            "opponent" if viewpoint == "lapras" else "lapras",
            args.lapras_agent,
            username_conditions,
        )
        opponent_key = infer_opponent_key(
            meta["opponent_username"], opponent_agent, username_conditions
        )
        summary = load_trajectory_summary(path)
        if viewpoint == "lapras" and lapras_team_hash is not None:
            summary["player_team_hash"] = lapras_team_hash

        rows.append(
            {
                "battle_id": meta["battle_id"],
                "trajectory_path": str(path.resolve()),
                "viewpoint": viewpoint,
                "player_username": meta["player_username"],
                "opponent_username": meta["opponent_username"],
                "player_agent": player_agent,
                "opponent_agent": opponent_agent,
                "opponent_key": opponent_key,
                "player_team_hash": summary["player_team_hash"],
                "opponent_team_hash": "",
                "battle_format": meta["battle_format"],
                "result": meta["result"],
                "num_turns": summary["num_turns"],
                "sampling_temperature": args.sampling_temperature,
                "source_dir": str(path.parent.parent.resolve()),
                "snapshot_id": args.snapshot_id,
                "split": "",
                "nonfinal_missing_action_labels": summary[
                    "nonfinal_missing_action_labels"
                ],
                "negative_action_labels": summary["negative_action_labels"],
            }
        )
        if args.progress_interval and idx % args.progress_interval == 0:
            print(f"Processed {idx:,} / {len(paths):,} trajectories")

    assignments = split_battles(
        rows,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    for row in rows:
        row["split"] = assignments[row["battle_id"]]

    csv_path = output_dir / "manifest.csv"
    parquet_path = output_dir / "manifest.parquet"
    report_path = output_dir / "dataset_report.json"
    write_csv(rows, csv_path)
    parquet_warning = write_parquet(rows, parquet_path)
    report = build_report(rows, parquet_warning)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"Wrote {len(rows):,} rows to {csv_path}")
    if parquet_warning:
        print(parquet_warning)
    else:
        print(f"Wrote parquet manifest to {parquet_path}")
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
