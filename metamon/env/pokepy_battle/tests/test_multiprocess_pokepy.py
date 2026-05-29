"""Correctness checks for multiprocess VectorizedPokepyEnv."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

POKEPY_ENGINE = Path(__file__).resolve().parents[1] / "pokepy-engine"
if POKEPY_ENGINE.exists():
    os.environ.setdefault(
        "POKEPY_DATA_PATH", str(POKEPY_ENGINE / "pokepy" / "data" / "extracted")
    )


def _make_env(batched_envs: int, num_workers: int, eval_player_side: int = 0):
    from pokepy.data.loader import get_data_path

    if not (get_data_path() / "type_chart.npy").exists():
        pytest.skip("pokepy extracted data not available")

    try:
        from metamon.env import get_metamon_teams
        from metamon.rl.metamon_to_amago import make_pokepy_env
        from metamon.rl.pretrained import get_pretrained_model
    except ImportError as e:
        pytest.skip(str(e))

    model = get_pretrained_model("Kakuna")
    if not model.model_name:
        pytest.skip("could not load Kakuna pretrained model")
    team_set = get_metamon_teams("gen9ou", "competitive")
    opponent_agent = model.initialize_agent(log=False)
    kwargs = dict(
        battle_format="gen9ou",
        observation_space=model.observation_space,
        action_space=model.action_space,
        reward_function=model.reward_function,
        team_set=team_set,
        opponent_model=model,
        batched_envs=batched_envs,
        turn_limit=80,
        eval_player_side=eval_player_side,
        num_workers=num_workers,
    )
    if num_workers <= 1:
        kwargs["opponent_policy"] = opponent_agent.policy
        kwargs["opponent_obs_space"] = model.observation_space
        kwargs["opponent_action_space"] = model.action_space
    return make_pokepy_env(**kwargs)


def _run_until_wins(env, max_steps: int = 500):
    obs, info = env.inner_reset()
    env.add_illegal_action_mask_to_obs(obs, info)
    wins = []
    for _ in range(max_steps):
        illegal = obs["illegal_actions"]
        actions = (~illegal).argmax(axis=-1)
        obs, _, terminated, truncated, info = env.inner_step(actions)
        env.add_illegal_action_mask_to_obs(obs, info)
        if "won" in info:
            for lane, w in enumerate(info["won"]):
                if w is not None:
                    wins.append(bool(w))
    return wins


@pytest.mark.slow
@pytest.mark.parametrize("eval_player_side", [0, 1])
def test_multiprocess_mirror_parity(eval_player_side: int):
    batched_envs = 8
    single = _make_env(batched_envs, num_workers=1, eval_player_side=eval_player_side)
    multi = _make_env(batched_envs, num_workers=4, eval_player_side=eval_player_side)

    single_wins = _run_until_wins(single, max_steps=200)
    multi_wins = _run_until_wins(multi, max_steps=200)
    multi.env.close()

    assert len(single_wins) > 0
    assert len(multi_wins) > 0
    single_rate = np.mean(single_wins)
    multi_rate = np.mean(multi_wins)
    print(
        f"eval_side={eval_player_side} single_win_rate={single_rate:.3f} "
        f"multi_win_rate={multi_rate:.3f} "
        f"battles(single={len(single_wins)}, multi={len(multi_wins)})"
    )
    # Parity check: multiprocess should match single-process stats within noise.
    # (Greedy first-legal actions are not ~50/50; full mirror eval uses the agent.)
    assert abs(single_rate - multi_rate) < 0.25
    assert abs(len(single_wins) - len(multi_wins)) <= max(
        4, int(0.3 * len(single_wins))
    )
