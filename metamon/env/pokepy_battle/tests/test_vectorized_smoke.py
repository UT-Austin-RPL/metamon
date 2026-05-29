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


def _make_smoke_env(batched_envs: int):
    """Build a self-play env or skip if data / checkpoints are unavailable."""
    from pokepy.data.loader import get_data_path

    if not (get_data_path() / "type_chart.npy").exists():
        pytest.skip("pokepy extracted data not available")

    try:
        from metamon.rl.pretrained import get_pretrained_model
        from metamon.env import get_metamon_teams
        from metamon.rl.metamon_to_amago import make_pokepy_env
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

    model = get_pretrained_model(model_names[0])
    if not model.model_name:
        pytest.skip("could not load pretrained model")

    team_set = get_metamon_teams("gen9ou", "competitive")
    opponent_agent = model.initialize_agent(log=False)
    return make_pokepy_env(
        battle_format="gen9ou",
        observation_space=model.observation_space,
        action_space=model.action_space,
        reward_function=model.reward_function,
        team_set=team_set,
        opponent_model=model,
        opponent_policy=opponent_agent.policy,
        opponent_obs_space=model.observation_space,
        opponent_action_space=model.action_space,
        batched_envs=batched_envs,
        turn_limit=50,
    )


def _run_smoke(env, n_steps: int = 100):
    obs, info = env.inner_reset()
    env.add_illegal_action_mask_to_obs(obs, info)
    t0 = time.time()
    for _ in range(n_steps):
        illegal = obs["illegal_actions"]
        actions = (~illegal).argmax(axis=-1)
        obs, _, terminated, truncated, info = env.inner_step(actions)
        env.add_illegal_action_mask_to_obs(obs, info)
    return n_steps * env.batched_envs / max(time.time() - t0, 1e-6)


@pytest.mark.slow
def test_vectorized_self_play_smoke():
    env = _make_smoke_env(batched_envs=4)
    fps = _run_smoke(env)
    print(f"Smoke (batched): {env.batched_envs} lanes, FPS={fps:.1f}")
    assert fps > 0


@pytest.mark.slow
def test_single_battle_self_play_smoke():
    # batched_envs=1 routes through the standalone, readable PokepyEnv.
    env = _make_smoke_env(batched_envs=1)
    fps = _run_smoke(env)
    print(f"Smoke (single): FPS={fps:.1f}")
    assert fps > 0
