"""Profile VectorizedShowdownEnv step() throughput at high lane counts.

Usage:
    cd metamon/env/vectorized && npm ci   # once
    python -m metamon.env.vectorized._profile_step --lanes 128 --steps 100

Set METAMON_VEC_PROFILE=1 (on by default here) for a per-section timing breakdown
printed when the env closes.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from metamon.interface import (
    DefaultActionSpace,
    DefaultObservationSpace,
    DefaultShapedReward,
)
from metamon.env.vectorized import RandomBatchedOpponent, VectorizedShowdownEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lanes", type=int, default=128)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--format", default="gen9randombattle")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="Number of Node processes (lanes split across workers)",
    )
    parser.add_argument(
        "--no-section-profile",
        action="store_true",
        help="Disable METAMON_VEC_PROFILE section breakdown",
    )
    args = parser.parse_args()

    if not args.no_section_profile:
        os.environ["METAMON_VEC_PROFILE"] = "1"

    n = int(args.lanes)
    obs_space = DefaultObservationSpace()
    act_space = DefaultActionSpace()
    reward = DefaultShapedReward()
    action_dim = act_space.gym_space.n

    opponent = RandomBatchedOpponent(num_lanes=n, action_dim=action_dim)
    env = VectorizedShowdownEnv(
        player_team_set=None,
        opponent_team_set=None,
        opponent=opponent,
        opponent_obs_space=obs_space,
        opponent_action_space=act_space,
        eval_obs_space=obs_space,
        eval_action_space=act_space,
        eval_reward_function=reward,
        opponent_reward_function=reward,
        batched_envs=n,
        battle_format=args.format,
        turn_limit=200,
        seed=args.seed,
        n_workers=args.n_workers,
    )

    try:
        t_reset0 = time.perf_counter()
        obs, info = env.reset()
        reset_s = time.perf_counter() - t_reset0
        print(f"reset: {reset_s:.3f}s ({n} lanes, {args.format})")

        rng = np.random.default_rng(args.seed)
        total_done = 0
        t_steps0 = time.perf_counter()
        for step in range(args.steps):
            actions = []
            for la in info["legal_actions"]:
                actions.append(int(rng.choice(la)) if la else 0)
            obs, rewards, terminated, truncated, info = env.step(np.array(actions))
            total_done += int((terminated | truncated).sum())

        steps_s = time.perf_counter() - t_steps0
        print(
            f"step: {args.steps} steps in {steps_s:.3f}s "
            f"({args.steps / steps_s:.2f} steps/s, "
            f"{args.steps * n / steps_s:.1f} lane-steps/s); "
            f"battles finished={total_done}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
