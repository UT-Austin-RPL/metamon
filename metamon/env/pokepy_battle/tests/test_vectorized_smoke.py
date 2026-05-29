"""Smoke test for vectorized pokepy self-play (requires pretrained checkpoints)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

POKEPY_ENGINE = Path(__file__).resolve().parents[1] / "pokepy-engine"
if POKEPY_ENGINE.exists():
    os.environ.setdefault(
        "POKEPY_DATA_PATH", str(POKEPY_ENGINE / "pokepy" / "data" / "extracted")
    )


@pytest.mark.slow
def test_vectorized_self_play_smoke():
    from pokepy.data.loader import get_data_path

    if not (get_data_path() / "type_chart.npy").exists():
        pytest.skip("pokepy extracted data not available")

    try:
        from metamon.rl.pretrained import get_pretrained_model
        from metamon.env import get_metamon_teams
        from metamon.rl.metamon_to_amago import make_metamon_env
    except ImportError as e:
        pytest.skip(str(e))

    model_names = []
    try:
        from metamon.rl.pretrained import get_pretrained_model_names

        model_names = get_pretrained_model_names()
    except Exception:
        pass
    if not model_names:
        pytest.skip("no pretrained models registered")

    model_name = model_names[0]
    model = get_pretrained_model(model_name)
    if not model.model_name:
        pytest.skip("could not load pretrained model")

    team_set = get_metamon_teams("gen9ou", "competitive")
    opponent_agent = model.initialize_agent(log=False)
    env = make_metamon_env(
        battle_format="gen9ou",
        observation_space=model.observation_space,
        action_space=model.action_space,
        reward_function=model.reward_function,
        team_set=team_set,
        opponent_model=model,
        opponent_policy=opponent_agent.policy,
        opponent_obs_space=model.observation_space,
        opponent_action_space=model.action_space,
        batched_envs=4,
        turn_limit=50,
    )
    obs, info = env.inner_reset()
    env.add_illegal_action_mask_to_obs(obs, info)
    n_steps = 100
    t0 = time.time()
    for _ in range(n_steps):
        illegal = obs["illegal_actions"]
        actions = (~illegal).argmax(axis=-1)
        obs, _, terminated, truncated, info = env.inner_step(actions)
        env.add_illegal_action_mask_to_obs(obs, info)
    fps = n_steps * env.batched_envs / max(time.time() - t0, 1e-6)
    print(f"Smoke: {n_steps} steps x {env.batched_envs} lanes, FPS={fps:.1f}")
    assert fps > 0
