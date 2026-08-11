#!/usr/bin/env python
"""Read-only pre-flight smoke for PSRO-Lite.

Points ``compute_prioritized_weights`` at the current live buffer (read-only)
and prints the weights it *would* produce at the configured start epoch. This
validates the filename regex against real filenames and shows the starting
weight distribution before committing to a relaunch — no sidecar is written.

Usage::

    METAMON_CACHE_DIR=... python -m scripts.psro_preflight \\
        --buffer_dir /path/to/buffer \\
        --battle_format gen1ou \\
        --train_pool metamon/rl/configs/opponent_pools/hl_gen1ou.yaml \\
        [--psro_window 50000] [--psro_min_games 20] [--psro_temp 1.0] \\
        [--psro_floor 0.05] [--psro_ema 0.7]
"""

from __future__ import annotations

import argparse
import os
import sys

from metamon.rl.evaluate.opponent_pool import load_opponent_pool
from metamon.rl.psro_lite import compute_prioritized_weights, weight_entropy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buffer_dir", type=str, required=True)
    parser.add_argument("--battle_format", type=str, required=True)
    parser.add_argument("--train_pool", type=str, required=True)
    parser.add_argument("--psro_window", type=int, default=50000)
    parser.add_argument("--psro_min_games", type=int, default=20)
    parser.add_argument("--psro_temp", type=float, default=1.0)
    parser.add_argument("--psro_floor", type=float, default=0.05)
    parser.add_argument("--psro_ema", type=float, default=0.7)
    parser.add_argument("--laplace_alpha", type=float, default=1.0)
    args = parser.parse_args()

    pool = load_opponent_pool(args.train_pool, battle_format=args.battle_format)
    agent_names = [row[0] for row in pool.agents]
    print(f"Pool agents ({len(agent_names)}): {agent_names}")
    print(
        f"Buffer: {os.path.join(os.path.abspath(args.buffer_dir), args.battle_format)}"
    )
    print()

    weights, diag = compute_prioritized_weights(
        buffer_dir=args.buffer_dir,
        battle_format=args.battle_format,
        agent_names=agent_names,
        window=args.psro_window,
        min_games=args.psro_min_games,
        temp=args.psro_temp,
        floor=args.psro_floor,
        ema=0.0,  # pre-flight: show the raw first-update weights
        prev_weights=None,
        laplace_alpha=args.laplace_alpha,
    )

    print(f"{'agent':<28} {'n':>8} {'win_rate':>10} {'weight':>10}")
    print("-" * 60)
    for name in agent_names:
        d = diag.get(name, {})
        n = d.get("n", 0)
        wr = d.get("win_rate")
        wr_s = f"{wr:.3f}" if wr is not None else "  N/A"
        print(f"{name:<28} {n:>8} {wr_s:>10} {d.get('weight', 0.0):>10.4f}")
    print("-" * 60)
    unmatched = diag.get("_unmatched_files", 0)
    print(f"unmatched files in window: {unmatched}")
    print(
        f"weight_entropy: {weight_entropy(weights):.4f}  (uniform={len(agent_names)} agents → {__import__('math').log(len(agent_names)):.4f})"
    )
    print()
    print("These are the weights the FIRST prioritized update would produce.")
    print(
        "No sidecar was written. Relaunch the collector with --psro_weighting to apply."
    )


if __name__ == "__main__":
    main()
