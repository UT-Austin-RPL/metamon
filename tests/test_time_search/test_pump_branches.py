"""Regression tests for ``SearchEvalRunner._pump_branches`` settle logic
(skill §17/§35) -- GPU/checkpoint-gated.

Auto-skip without CUDA/the frozen checkpoint (see ``conftest.gpu_required``);
run on GPU with ``METAMON_CACHE_DIR`` set:

    METAMON_CACHE_DIR=/home/eddie/metamon_cache uv run python -m pytest \\
        tests/test_time_search/test_pump_branches.py -q

Background
----------
``_pump_branches`` settles each search branch after the root (and deeper
rollout) actions are submitted, auto-answering follow-ups until the eval side
parks at its next decision. A pre-existing settle-cascade bug -- the
both-sides-advanced case where the eval side is ``wait`` and the opponent has a
``forceswitch`` (e.g. the opponent fainted during the root exchange) -- left the
opponent forever unanswered because a ``wait`` side is never "answered" so
``answered[eval]`` never caught up to its serial and the old
``not other_advanced`` guard blocked the opponent's answer. The host then went
idle and ``pump_until`` raised ``pump_until idle for 20.0s``.

This was masked by ``base_fallback`` in the legacy prototype (12/128 roots) and
surfaced once ``error_policy=raise`` was the default. The fix mirrors the live
env's ``_pump_settle`` + ``_advance_lanes`` semantics: answer the opponent-only
follow-up regardless of whether the eval ``wait`` serial advanced, and re-answer
single-side ``|error|`` re-prompts.

These tests exercise the real production settle path (frozen checkpoint + real
``BattleAgainstMetamon`` env) through the stall-prone early decisions and assert
no ``pump_until idle`` timeout occurs.
"""

from __future__ import annotations

import numpy as np
import pytest

from metamon.rl.experimental.test_time_search.config import SearchConfig

from .conftest import gpu_required  # noqa: F401

pytestmark = gpu_required


def _step_search_decisions(bundle, runner, cfg, max_searched: int):
    """Play the env like ``eval_search.run_search_eval`` but stop after
    ``max_searched`` search roots have completed on lane 0. Returns the list of
    ``SearchRootRecord``\\ s produced for lane 0. Uses the correctness config
    (exhaustive actions, resampled CRN chance, exact leaf expectation,
    ``error_policy=raise``) so any settle timeout propagates as a
    ``ShowdownSimProcessError("pump_until idle ...")``.
    """
    from metamon.env.vectorized.obs_utils import unstack_obs_dicts

    env = bundle.env
    n = env.batched_envs
    obs, info = env.reset()
    lane0_records = []
    steps = 0
    max_steps = max_searched * 60 + 100  # safety cap
    while len(lane0_records) < max_searched and steps < max_steps:
        steps += 1
        obs_list = unstack_obs_dicts(obs)
        actions = np.zeros(n, dtype=np.int64)
        for i in range(n):
            lane = env.lanes[i]
            if lane.ended or not lane.needs_agent_decision(env.eval_side):
                actions[i] = 0
                continue
            legal = info["legal_actions"][i]
            runner._battle_id = "pump0"
            runner._decision_counter = steps
            # Search every eval decision (every_n=1) so we exercise settle on
            # both normal turns and the faint/forceswitch cascades that stall.
            action, rec = runner.search_root(i, obs_list[i], legal)
            actions[i] = action
            if i == 0:
                lane0_records.append(rec)
        obs, rewards, terminated, truncated, info = env.step(actions)
        for i in range(n):
            bundle.eval_driver.observe(i, float(rewards[i]), int(actions[i]))
        done = terminated | truncated
        if done.any():
            for i in np.where(done)[0]:
                bundle.eval_driver.reset_lanes(
                    np.array([i == j for j in range(n)], dtype=bool)
                )
                env.opponent.reset_lanes(
                    np.array([i == j for j in range(n)], dtype=bool)
                )
            # If lane 0 ended before we gathered enough records, stop early --
            # the test still asserts whatever settle calls happened were clean.
            if done[0]:
                break
    return lane0_records


def test_pump_branches_no_idle_timeout_through_faint_cascade(frozen_env_bundle):
    """The canonical regression: play the early decisions of a seeded battle
    (seed 42, the same setup that stalled at decision 3 in the §18G smoke eval)
    with the correctness config and ``error_policy=raise``. Before the fix, a
    both-sides-advanced ``wait``/``forceswitch`` settle raised
    ``pump_until idle for 20.0s`` within the first few decisions. After the fix
    every ``search_root`` completes cleanly."""
    bundle = frozen_env_bundle
    cfg = SearchConfig(
        search_mode="oracle-root-mc",
        search_rollouts_per_action=2,  # small K keeps the gated test fast
        search_depth=0,
        search_every_n_decisions=1,
        search_root_candidate_mode="all_legal",
        search_chance_mode="resample_crn",
        search_leaf_value_mode="policy_expectation",
        search_value_normalization=False,
        search_ablation="single_anchor_kl",
        search_error_policy="raise",
    )
    runner = bundle.make_runner(cfg)
    try:
        records = _step_search_decisions(bundle, runner, cfg, max_searched=8)
        # We must have actually searched at least a few decisions (the stall
        # historically hit at decision ~3). If lane 0 ended immediately, skip.
        if len(records) < 3:
            pytest.skip(
                f"only {len(records)} search roots gathered before lane 0 ended; "
                "could not exercise the faint-cascade settle path"
            )
        for rec in records:
            assert rec.error == "", (
                f"search_root errored at b{rec.battle_id} d{rec.decision}: "
                f"{rec.error!r}"
            )
    finally:
        runner.close()


def test_pump_branches_answers_opp_forceswitch_while_eval_waits(frozen_env_bundle):
    """Stress the specific settle state over more decisions: exhaustive actions
    with K=4 rollouts creates many branches, raising the chance that at least
    one branch hits an opponent faint -> forceswitch while the eval side waits.
    Asserts zero errors across a larger window (the smoke-eval failure mode was
    ~9% of roots; K=4 over ~12 decisions exercises hundreds of branch settles)."""
    bundle = frozen_env_bundle
    cfg = SearchConfig(
        search_mode="oracle-root-mc",
        search_rollouts_per_action=4,
        search_depth=0,
        search_every_n_decisions=1,
        search_root_candidate_mode="all_legal",
        search_chance_mode="resample_crn",
        search_leaf_value_mode="policy_expectation",
        search_value_normalization=False,
        search_ablation="single_anchor_kl",
        search_error_policy="raise",
    )
    runner = bundle.make_runner(cfg)
    try:
        records = _step_search_decisions(bundle, runner, cfg, max_searched=12)
        if len(records) < 3:
            pytest.skip(
                f"only {len(records)} search roots gathered before lane 0 ended"
            )
        errors = [r for r in records if r.error]
        assert not errors, (
            f"{len(errors)}/{len(records)} search roots errored; first: "
            f"{errors[0].error!r} at b{errors[0].battle_id} d{errors[0].decision}"
        )
    finally:
        runner.close()
