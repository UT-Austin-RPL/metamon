#!/usr/bin/env python3
"""
Generate Gen 1 self-play trajectories from local ladder battles.

This is a convenience wrapper around the existing ladder evaluation helpers.
It runs one or more local players in parallel and saves `.json.lz4` trajectories
under the requested output directory.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from metamon.rl.gen1_binary_models import *  # noqa: F401,F403
from metamon.rl.pretrained import get_pretrained_model_names


def _split_battles(num_battles: int, parallel_instances: int) -> list[int]:
    base = num_battles // parallel_instances
    remainder = num_battles % parallel_instances
    return [base + (1 if i < remainder else 0) for i in range(parallel_instances)]


def _build_snippet(
    repo_root: str,
    model_name: str,
    username: str,
    battle_format: str,
    team_set_name: str,
    battles: int,
    battle_backend: str,
    output_dir: str,
) -> str:
    return f"""
import sys
sys.path.insert(0, {repo_root!r})
from metamon.rl.gen1_binary_models import *  # noqa: F401,F403
from metamon.rl.evaluate import pretrained_vs_local_ladder
from metamon.rl.pretrained import get_pretrained_model
from metamon.env import get_metamon_teams

model = get_pretrained_model({model_name!r})
team_set = get_metamon_teams({battle_format!r}, {team_set_name!r})
pretrained_vs_local_ladder(
    pretrained_model=model,
    username={username!r},
    battle_format={battle_format!r},
    team_set=team_set,
    total_battles={battles},
    battle_backend={battle_backend!r},
    save_trajectories_to={output_dir!r},
    log_to_wandb=False,
)
"""


def generate_selfplay(
    model_name: str,
    opponent_model_name: str,
    battle_format: str,
    team_set_name: str,
    num_battles: int,
    output_dir: str,
    battle_backend: str = "metamon",
    parallel_instances: int = 2,
) -> int:
    trajectories_dir = os.path.join(output_dir, battle_format)
    os.makedirs(trajectories_dir, exist_ok=True)

    print(f"Model: {model_name}")
    print(f"Opponent: {opponent_model_name}")
    print(f"Format: {battle_format}")
    print(f"Battles: {num_battles}")
    print(f"Output: {trajectories_dir}")

    battle_counts = _split_battles(num_battles, parallel_instances)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    processes = []

    for idx, battles in enumerate(battle_counts):
        if battles <= 0:
            continue

        username_a = f"SelfPlay_{model_name[:8]}_{timestamp}_A{idx}"
        username_b = f"SelfPlay_{opponent_model_name[:8]}_{timestamp}_B{idx}"

        common_snippet = _build_snippet(
            repo_root=str(Path(__file__).parent.parent),
            model_name=model_name,
            username=username_a,
            battle_format=battle_format,
            team_set_name=team_set_name,
            battles=battles,
            battle_backend=battle_backend,
            output_dir=output_dir,
        )

        opponent_snippet = _build_snippet(
            repo_root=str(Path(__file__).parent.parent),
            model_name=opponent_model_name,
            username=username_b,
            battle_format=battle_format,
            team_set_name=team_set_name,
            battles=battles,
            battle_backend=battle_backend,
            output_dir=output_dir,
        )

        proc_a = subprocess.Popen(
            [sys.executable, "-c", common_snippet],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(3)
        proc_b = subprocess.Popen(
            [sys.executable, "-c", opponent_snippet],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append((proc_a, proc_b, username_a, username_b))
        time.sleep(5)

    print("Collecting battles...")
    start = time.time()
    while True:
        active = sum(
            1
            for proc_a, proc_b, _, _ in processes
            if proc_a.poll() is None or proc_b.poll() is None
        )
        if active == 0:
            break
        time.sleep(30)
        count = len(list(Path(trajectories_dir).glob("*.json.lz4")))
        elapsed = (time.time() - start) / 3600
        print(f"[{elapsed:.1f}h] {count}/{num_battles} battles | {active}/{len(processes)} instances active")

    for proc_a, proc_b, user_a, user_b in processes:
        proc_a.wait()
        proc_b.wait()
        if proc_a.returncode != 0:
            print(f"WARNING: {user_a} exited with {proc_a.returncode}")
            print((proc_a.stderr.read() or "")[:500])
        if proc_b.returncode != 0:
            print(f"WARNING: {user_b} exited with {proc_b.returncode}")
            print((proc_b.stderr.read() or "")[:500])

    final_count = len(list(Path(trajectories_dir).glob("*.json.lz4")))
    print(f"Saved {final_count} trajectories to {trajectories_dir}")
    return final_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Gen 1 self-play data")
    parser.add_argument("--model", help="Registered model name")
    parser.add_argument(
        "--opponent_model",
        default=None,
        help="Opponent model name. Defaults to the same model for pure self-play.",
    )
    parser.add_argument("--num_battles", type=int, default=10_000)
    parser.add_argument(
        "--output_dir",
        default=os.path.expanduser("~/metamon/trajectories"),
        help="Root output directory",
    )
    parser.add_argument("--battle_format", default="gen1ou")
    parser.add_argument("--team_set", default="competitive")
    parser.add_argument("--battle_backend", default="metamon")
    parser.add_argument("--parallel_instances", type=int, default=2)
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="Print registered model names and exit",
    )
    args = parser.parse_args()

    if args.list_models:
        for model_name in get_pretrained_model_names():
            print(model_name)
        return 0

    if not args.model:
        parser.error("--model is required unless --list_models is set")

    opponent = args.opponent_model or args.model
    generate_selfplay(
        model_name=args.model,
        opponent_model_name=opponent,
        battle_format=args.battle_format,
        team_set_name=args.team_set,
        num_battles=args.num_battles,
        output_dir=args.output_dir,
        battle_backend=args.battle_backend,
        parallel_instances=args.parallel_instances,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
