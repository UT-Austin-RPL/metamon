"""Standalone throughput harness for VectorizedPokepyEnv (opt-in / slow)."""

from __future__ import annotations

import cProfile
import io
import os
import pstats
import time
from pathlib import Path

import numpy as np
import pytest

POKEPY_ENGINE = Path(__file__).resolve().parents[1] / "pokepy-engine"
if POKEPY_ENGINE.exists():
    os.environ.setdefault(
        "POKEPY_DATA_PATH", str(POKEPY_ENGINE / "pokepy" / "data" / "extracted")
    )


def _make_bench_env(batched_envs: int, *, num_workers: int = 1):
    from pokepy.data.loader import get_data_path

    if not (get_data_path() / "type_chart.npy").exists():
        pytest.skip("pokepy extracted data not available")

    try:
        from metamon.env import get_metamon_teams
        from metamon.rl.metamon_to_amago import make_pokepy_env
        from metamon.rl.pretrained import (
            get_pretrained_model,
            get_pretrained_model_names,
        )
    except ImportError as e:
        pytest.skip(str(e))

    model_names = get_pretrained_model_names()
    if not model_names:
        pytest.skip("no pretrained models registered")

    model = get_pretrained_model(model_names[0])
    if not model.model_name:
        pytest.skip("could not load pretrained model")

    team_set = get_metamon_teams("gen9ou", "competitive")
    opponent_agent = model.initialize_agent(log=False)
    kwargs = dict(
        battle_format="gen9ou",
        observation_space=model.observation_space,
        action_space=model.action_space,
        reward_function=model.reward_function,
        team_set=team_set,
        opponent_model=model,
        opponent_checkpoint=None,
        batched_envs=batched_envs,
        turn_limit=50,
        num_workers=num_workers,
    )
    if num_workers <= 1:
        kwargs.update(
            opponent_policy=opponent_agent.policy,
            opponent_obs_space=model.observation_space,
            opponent_action_space=model.action_space,
        )
    return make_pokepy_env(**kwargs)


def _run_steps(env, n_steps: int = 50):
    obs, info = env.inner_reset()
    env.add_illegal_action_mask_to_obs(obs, info)
    for _ in range(n_steps):
        illegal = obs["illegal_actions"]
        actions = (~illegal).argmax(axis=-1)
        obs, _, terminated, truncated, info = env.inner_step(actions)
        env.add_illegal_action_mask_to_obs(obs, info)
    return n_steps


def _profile_one_run(env, n_steps: int = 30) -> str:
    prof = cProfile.Profile()
    prof.enable()
    _run_steps(env, n_steps=n_steps)
    prof.disable()
    stream = io.StringIO()
    stats = pstats.Stats(prof, stream=stream)
    stats.sort_stats("cumtime")
    stats.print_stats(25)
    return stream.getvalue()


@pytest.mark.slow
@pytest.mark.parametrize("batched_envs", [1, 4, 16])
def test_bench_vector_env_fps_and_profile(batched_envs: int):
    env = _make_bench_env(batched_envs)
    raw_env = env.env
    if hasattr(raw_env, "enable_profiling"):
        raw_env.enable_profiling()

    t0 = time.perf_counter()
    steps = _run_steps(env, n_steps=40)
    elapsed = time.perf_counter() - t0
    fps = steps * env.batched_envs / max(elapsed, 1e-6)

    profile_text = ""
    summary = {}
    if hasattr(raw_env, "profile_summary"):
        summary = raw_env.profile_summary()
        profile_text = str(summary)

    print(f"bench batched_envs={batched_envs} fps={fps:.1f} " f"profile={profile_text}")
    assert fps > 0
    if summary:
        assert summary.get("per_lane_loop_ms", 0) >= 0

    if batched_envs == 4:
        print(_profile_one_run(env, n_steps=20))


@pytest.mark.slow
def test_bench_multiprocess_scaling():
    batched_envs = 8
    single = _make_bench_env(batched_envs, num_workers=1)
    multi = _make_bench_env(batched_envs, num_workers=4)

    def timed(env):
        t0 = time.perf_counter()
        _run_steps(env, n_steps=30)
        return 30 * env.batched_envs / max(time.perf_counter() - t0, 1e-6)

    fps_single = timed(single)
    fps_multi = timed(multi)
    print(f"multiprocess scaling: single={fps_single:.1f} multi={fps_multi:.1f}")
    assert fps_multi > 0
    multi.env.close()
