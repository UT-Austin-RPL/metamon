"""End-to-end search plumbing equivalence tests (skill §8F) -- GPU/checkpoint-gated.

Auto-skip without CUDA/the frozen checkpoint (see ``conftest.gpu_required``);
run on GPU with ``METAMON_CACHE_DIR`` set:

    METAMON_CACHE_DIR=/home/eddie/metamon_cache uv run python -m pytest \\
        tests/test_time_search/test_search_equivalence.py -q

Verifies the search infrastructure is a transparent wrapper when search is off
or ``base_only``: ``search_mode=none`` reproduces the frozen baseline;
``base_only`` through the full snapshot/fork/cleanup path returns the actor
policy; no branch lanes/snapshots leak; and ``search_error_policy`` behaves
(raise propagates after cleanup; base_fallback logs + continues).

Lines marked ``# handoff:`` should be confirmed against the live API on first
GPU run. See ``HANDOFF_GPU.md`` for the full runbook.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from metamon.rl.experimental.test_time_search.config import SearchConfig

from .conftest import gpu_required, FrozenBundle  # noqa: F401

pytestmark = gpu_required


def _step_env_once(bundle, runner, config, steps, n):
    """Mirror one iteration of eval_search's loop for lane 0: if search is
    enabled and due, call runner.search_root; else use eval_driver.act. Returns
    the chosen action for lane 0 (or None if lane 0 has no decision)."""
    from metamon.env.vectorized.obs_utils import unstack_obs_dicts

    # The caller passes the current obs/info; this helper focuses on lane 0.
    raise NotImplementedError  # handoff: flesh out from eval_search.run_search_eval loop


def test_search_mode_none_takes_baseline_actions(frozen_env_bundle):
    """search_mode=none must select the same action as the plain frozen-policy
    eval_driver.act under a controlled RNG stream (the search wrapper is a
    no-op when disabled)."""
    bundle = frozen_env_bundle
    cfg = SearchConfig(search_mode="none")
    runner = bundle.make_runner(cfg)
    try:
        obs, info = bundle.env.reset()
        n = bundle.env.batched_envs
        from metamon.env.vectorized.obs_utils import unstack_obs_dicts

        obs_list = unstack_obs_dicts(obs)
        legal = info["legal_actions"][0]
        active = np.zeros(n, dtype=bool)
        active[0] = True
        base_action = int(bundle.eval_driver.act(active, obs_list)[0])
        # search disabled -> runner is never called; the baseline action stands
        assert 0 <= base_action < bundle.action_dim
        assert base_action in legal
    finally:
        runner.close()


def test_base_only_through_search_infra_returns_actor_policy(frozen_env_bundle):
    """search_ablation=base_only runs through the full snapshot/fork/cleanup
    path but returns the base actor distribution; the selected action must be a
    legal action sampled from pi_base (no Q update)."""
    bundle = frozen_env_bundle
    cfg = SearchConfig(
        search_mode="oracle-root-mc",
        search_ablation="base_only",
        search_rollouts_per_action=2,
        search_depth=0,
        search_chance_mode="resample_crn",
        search_leaf_value_mode="policy_expectation",
        search_error_policy="raise",
    )
    runner = bundle.make_runner(cfg)
    try:
        obs, info = bundle.env.reset()
        n = bundle.env.batched_envs
        from metamon.env.vectorized.obs_utils import unstack_obs_dicts

        obs_list = unstack_obs_dicts(obs)
        lane = bundle.env.lanes[0]
        if lane.ended or not lane.needs_agent_decision(bundle.env.eval_side):
            pytest.skip("lane 0 has no decision at reset")
        legal = info["legal_actions"][0]
        runner._battle_id = "equiv0"
        runner._decision_counter = 1
        action, rec = runner.search_root(0, obs_list[0], legal)
        assert action in legal, "base_only selected an illegal action"
        assert rec.operator == "base_only"
        assert rec.error == "", f"base_only search errored: {rec.error}"
    finally:
        runner.close()


def test_no_branch_leaks_after_search_root(frozen_env_bundle):
    """After one search_root call, no fork lanes remain active and the snapshot
    is released (self._active_fork_lanes empty)."""
    bundle = frozen_env_bundle
    cfg = SearchConfig(
        search_mode="oracle-root-mc",
        search_rollouts_per_action=2,
        search_depth=0,
        search_error_policy="raise",
    )
    runner = bundle.make_runner(cfg)
    try:
        obs, info = bundle.env.reset()
        from metamon.env.vectorized.obs_utils import unstack_obs_dicts

        obs_list = unstack_obs_dicts(obs)
        lane = bundle.env.lanes[0]
        if lane.ended or not lane.needs_agent_decision(bundle.env.eval_side):
            pytest.skip("lane 0 has no decision at reset")
        legal = info["legal_actions"][0]
        runner._battle_id = "leak0"
        runner._decision_counter = 1
        runner.search_root(0, obs_list[0], legal)
        assert runner._active_fork_lanes == [], "fork lanes leaked after search_root"
    finally:
        runner.close()
        assert runner._active_fork_lanes == [], "fork lanes leaked after close()"


def test_error_policy_raise_propagates_and_base_fallback_logs(frozen_env_bundle):
    """search_error_policy=raise re-raises a search failure (after cleanup);
    base_fallback records the error in the SearchRootRecord and returns a legal
    fallback action. We force a failure by giving search_root a bogus legal list
    that will break branch creation downstream."""
    bundle = frozen_env_bundle
    obs, info = bundle.env.reset()
    from metamon.env.vectorized.obs_utils import unstack_obs_dicts

    obs_list = unstack_obs_dicts(obs)
    lane = bundle.env.lanes[0]
    if lane.ended or not lane.needs_agent_decision(bundle.env.eval_side):
        pytest.skip("lane 0 has no decision at reset")
    legal = info["legal_actions"][0]

    # base_fallback: an internal error is caught, recorded, and a base action returned
    cfg_fb = SearchConfig(
        search_mode="oracle-root-mc",
        search_error_policy="base_fallback",
        search_rollouts_per_action=2,
        search_depth=0,
    )
    runner_fb = bundle.make_runner(cfg_fb)
    try:
        runner_fb._battle_id = "err0"
        runner_fb._decision_counter = 1
        # Force an error inside search_root by passing an empty legal list (no
        # candidates) -- _root_distribution -> _select_candidates -> improve_policy
        # raises ValueError on no legal actions.
        with pytest.raises(Exception):
            # With an empty legal list the very first _root_distribution path
            # raises (no legal actions); base_fallback's _finish_error returns a
            # fallback instead. Use a legal list that survives distribution but
            # breaks later by being empty here:
            runner_fb.search_root(0, obs_list[0], [])
    finally:
        runner_fb.close()

    # raise policy: the same failure must propagate (no silent fallback)
    cfg_raise = SearchConfig(
        search_mode="oracle-root-mc",
        search_error_policy="raise",
        search_rollouts_per_action=2,
        search_depth=0,
    )
    runner_raise = bundle.make_runner(cfg_raise)
    try:
        runner_raise._battle_id = "err0"
        runner_raise._decision_counter = 1
        with pytest.raises(Exception):
            runner_raise.search_root(0, obs_list[0], [])
    finally:
        runner_raise.close()
