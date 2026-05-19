#!/usr/bin/env python3
"""
Filter Gen 1 self-play trajectories for basic quality issues.
"""

from __future__ import annotations

import argparse
import json
import lz4.frame
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_trajectory(filepath: str) -> Dict:
    with lz4.frame.open(filepath, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def save_trajectory(filepath: str, data: Dict) -> None:
    temp_path = filepath + ".tmp"
    with lz4.frame.open(temp_path, "wb") as f:
        f.write(json.dumps(data).encode("utf-8"))
    os.replace(temp_path, filepath)


def count_invalid_actions(trajectory: Dict) -> Tuple[int, int]:
    actions = trajectory.get("actions", [])
    invalid = sum(1 for a in actions if a == -1)
    return invalid, len(actions)


def get_turn_count(trajectory: Dict) -> int:
    return max(len(trajectory.get("states", [])) - 1, 0)


def get_outcome(filepath: str) -> str:
    name = Path(filepath).name
    if "WIN" in name:
        return "WIN"
    if "LOSS" in name:
        return "LOSS"
    return "UNKNOWN"


def filter_trajectory(
    filepath: str,
    max_invalid_rate: float = 0.05,
    min_turns: int = 10,
    max_turns: int = 1000,
) -> Tuple[bool, str]:
    try:
        traj = load_trajectory(filepath)
    except Exception as exc:
        return False, f"load_error: {exc}"

    turns = get_turn_count(traj)
    if turns < min_turns:
        return False, f"too_short: {turns}"
    if turns > max_turns:
        return False, f"too_long: {turns}"

    invalid, total = count_invalid_actions(traj)
    if total == 0:
        return False, "no_actions"
    if invalid / total > max_invalid_rate:
        return False, f"invalid_actions: {invalid / total:.1%}"

    return True, "passed"


def balance_outcomes(filepaths: List[str]) -> List[str]:
    wins = [fp for fp in filepaths if get_outcome(fp) == "WIN"]
    losses = [fp for fp in filepaths if get_outcome(fp) == "LOSS"]
    unknown = [fp for fp in filepaths if get_outcome(fp) == "UNKNOWN"]
    target = min(len(wins), len(losses))
    if len(wins) > target:
        wins = random.sample(wins, target)
    if len(losses) > target:
        losses = random.sample(losses, target)
    print(f"Wins: {len(wins)} | Losses: {len(losses)} | Unknown: {len(unknown)}")
    return wins + losses + unknown


def filter_dataset(
    input_dir: str,
    output_dir: str,
    max_invalid_rate: float = 0.05,
    min_turns: int = 10,
    max_turns: int = 1000,
    balance: bool = False,
) -> None:
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")

    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(input_dir)

    traj_files = list(input_path.glob("*.json.lz4"))
    print(f"Found {len(traj_files)} files")

    passed: List[str] = []
    failed_reasons: Counter[str] = Counter()
    for idx, filepath in enumerate(traj_files, start=1):
        if idx % 1000 == 0:
            print(f"Processed {idx}/{len(traj_files)}")
        ok, reason = filter_trajectory(
            str(filepath),
            max_invalid_rate=max_invalid_rate,
            min_turns=min_turns,
            max_turns=max_turns,
        )
        if ok:
            passed.append(str(filepath))
        else:
            failed_reasons[reason] += 1

    if balance and passed:
        passed = balance_outcomes(passed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for idx, filepath in enumerate(passed, start=1):
        if idx % 1000 == 0:
            print(f"Copied {idx}/{len(passed)}")
        traj = load_trajectory(filepath)
        save_trajectory(str(output_path / Path(filepath).name), traj)

    stats = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "input_count": len(traj_files),
        "output_count": len(passed),
        "failed_reasons": dict(failed_reasons),
        "filters": {
            "max_invalid_rate": max_invalid_rate,
            "min_turns": min_turns,
            "max_turns": max_turns,
            "balance_outcomes": balance,
        },
    }
    with open(output_path / "filter_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter self-play trajectories")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_invalid_rate", type=float, default=0.05)
    parser.add_argument("--min_turns", type=int, default=10)
    parser.add_argument("--max_turns", type=int, default=1000)
    parser.add_argument("--balance_outcomes", action="store_true")
    args = parser.parse_args()

    filter_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_invalid_rate=args.max_invalid_rate,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        balance=args.balance_outcomes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
