"""throwaway: end-to-end VectorizedShowdownEnv smoke with a random opponent."""

import numpy as np

from metamon.interface import (
    DefaultObservationSpace,
    DefaultActionSpace,
    DefaultShapedReward,
)
from metamon.env.vectorized import VectorizedShowdownEnv, RandomBatchedOpponent


def main():
    n = 4
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
        battle_format="gen9randombattle",
        turn_limit=200,
        seed=0,
    )
    print(
        "observation_space keys:", list(env.observation_space.spaces.keys())[:5], "..."
    )
    print("action_space:", env.action_space)

    try:
        obs, info = env.reset()
        print("reset obs illegal_actions shape:", obs["illegal_actions"].shape)
        print("legal_actions per lane:", [len(la) for la in info["legal_actions"]])

        rng = np.random.default_rng(0)
        total_done = 0
        reward_sum = 0.0
        for step in range(300):
            actions = []
            for la in info["legal_actions"]:
                actions.append(int(rng.choice(la)) if la else 0)
            obs, rewards, terminated, truncated, info = env.step(np.array(actions))
            reward_sum += float(rewards.sum())
            done = terminated | truncated
            total_done += int(done.sum())
            if done.any():
                won = info.get("won")
                idxs = np.where(done)[0].tolist()
                print(f"step {step}: done lanes={idxs} won={[won[i] for i in idxs]}")
            assert obs["illegal_actions"].shape == (n, action_dim)

        print(
            f"completed 300 steps; battles finished={total_done}; reward_sum={reward_sum:.3f}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
