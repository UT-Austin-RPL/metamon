#!/usr/bin/env python3
"""
Quick evaluation helper for Gen 1 model checkpoints.

This is a lightweight alternative to the full tournament scripts when you just
want a fast sanity check against a couple of heuristic baselines.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from metamon.rl.gen1_binary_models import *  # noqa: F401,F403
from metamon.rl.evaluate import pretrained_vs_baselines
from metamon.rl.pretrained import get_pretrained_model, get_pretrained_model_names
from metamon.env import get_metamon_teams


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick evaluation against baselines")
    parser.add_argument("--model", help="Registered pretrained model name")
    parser.add_argument("--format", default="gen1ou", help="Battle format")
    parser.add_argument("--team_set", default="competitive", help="Team set")
    parser.add_argument("--battles", type=int, default=50, help="Total battles")
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

    model = get_pretrained_model(args.model)
    team_set = get_metamon_teams(args.format, args.team_set)
    results = pretrained_vs_baselines(
        pretrained_model=model,
        battle_format=args.format,
        team_set=team_set,
        total_battles=args.battles,
        parallel_actors_per_baseline=1,
        baselines=["PokeEnvHeuristic", "MaxBPBaseline"],
    )
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
