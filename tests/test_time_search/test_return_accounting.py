"""Return-accounting tests (skill §5 / §14).

Verifies the search return uses the same convention as the critic training
target. The returns audit found three critical bugs, all fixed in
``search_driver.py``:

  * BUG A: the terminal victory reward was dropped (``_record_rollout_rewards``
    skipped branches already marked terminal -> immediate wins got Q=0).
  * BUG B: ``reward_multiplier`` (10.0) was not applied to search rewards (env
    units mixed with 10x critic units).
  * BUG C: the discount exponent was off-by-one (at depth 0 the bootstrap used
    gamma**0 instead of gamma**1).

These tests use a stub runner + mock lanes + monkeypatched policy calls (no
checkpoint / GPU) to check the bookkeeping formulas directly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from metamon.rl.experimental.test_time_search.config import SearchConfig
from metamon.rl.experimental.test_time_search import search_driver as sd
from metamon.rl.experimental.test_time_search.search_driver import (
    SearchEvalRunner,
    _Branches,
)


class _MockLane:
    def __init__(self, i):
        self.i = i

    def universal_state(self, side):
        return self.i


def _make_runner(monkeypatch, gamma_idx=-1, include_intermediate=True):
    cfg = SearchConfig(
        search_mode="oracle-root-mc",
        search_include_intermediate_rewards=include_intermediate,
        search_leaf_value_mode="policy_expectation",
    )
    runner = SearchEvalRunner.__new__(SearchEvalRunner)
    runner.config = cfg
    runner.eval_reward_function = None  # set per-test via the queue
    runner.reward_multiplier = 10.0
    runner.action_dim = 9
    runner.device = torch.device("cpu")
    runner.eval_policy = object()  # unused; _state_embedding is monkeypatched
    runner._decision_counter = 0
    runner._battle_id = "test"

    class _EnvStub:
        eval_side = "p1"
        opp_side = "p2"

    runner.env = _EnvStub()

    # Monkeypatch the policy-embedding + exact-V helpers so _leaf_values is
    # deterministic and independent of the frozen checkpoint.
    monkeypatch.setattr(
        sd,
        "_state_embedding",
        lambda policy, obs, rl2, tidx, hidden: (torch.zeros(1), None),
    )
    monkeypatch.setattr(sd, "_index_hidden", lambda hidden, idx, device: None)
    monkeypatch.setattr(sd, "_scatter_hidden", lambda hidden, idx, new_hidden: None)

    def _branch_obs_batch(self, br, side):
        active_idx = np.where(br.active)[0]
        n = int(active_idx.size)
        torch_obs = {"illegal_actions": torch.zeros(n, 1, self.action_dim, dtype=bool)}
        return torch_obs, [], active_idx

    monkeypatch.setattr(SearchEvalRunner, "_branch_obs_batch", _branch_obs_batch)
    return runner


def _make_branches(N, gamma):
    return _Branches(
        lane_ids=list(range(N)),
        lanes=[_MockLane(i) for i in range(N)],
        root_action=np.arange(N),
        rollout_index=np.arange(N),
        active=np.ones(N, dtype=bool),
        depth_done=np.zeros(N, dtype=np.int64),
        cum_reward=np.zeros(N, dtype=np.float64),
        terminal=np.zeros(N, dtype=bool),
        eval_hidden=None,
        eval_rl2s=np.zeros((N, 10), dtype=np.float64),
        eval_steps=np.zeros(N, dtype=np.int64),
        opp_hidden=None,
        opp_rl2s=np.zeros((N, 10), dtype=np.float64),
        opp_steps=np.zeros(N, dtype=np.int64),
        snap_id=0,
        trunk_lane=0,
        prev_eval_state=[0] * N,
        gamma=gamma,
        seed_bank=None,
    )


def _set_reward_queue(runner, queue):
    """reward_fn pops env-scale rewards in call order (index order)."""
    q = list(queue)

    def fn(prev, new):
        return q.pop(0)

    runner.eval_reward_function = fn
    return q


def test_record_rewards_applies_multiplier_and_root_discount(monkeypatch):
    """BUG B: rewards are scaled by reward_multiplier (10x). Root reward gets
    gamma**0 (skill §5)."""
    runner = _make_runner(monkeypatch)
    br = _make_branches(N=3, gamma=0.9)
    _set_reward_queue(runner, [1.0, 2.0, 5.0])  # env rewards for the root settlement
    runner._record_rollout_rewards(br, prev_active=br.active.copy())
    # cum += gamma**0 * 10 * r  -> [10, 20, 50]
    assert np.allclose(br.cum_reward, [10.0, 20.0, 50.0])
    assert br.eval_rl2s[0, 0] == 1.0  # env (unscaled) reward stored in rl2


def test_terminal_victory_reward_recorded_once(monkeypatch):
    """BUG A: the terminal settlement's +200 victory reward is recorded into
    cum_reward for the branch that just terminated (it was in prev_active)."""
    runner = _make_runner(monkeypatch)
    br = _make_branches(N=2, gamma=1.0)
    # root settlement: both active, r=1
    _set_reward_queue(runner, [1.0, 1.0])
    runner._record_rollout_rewards(br, br.active.copy())
    br.depth_done[:] += 1
    # deeper settlement: branch 0 wins (victory reward 200), branch 1 r=2.
    # Both were active at the start of this settlement (prev_active includes 0).
    br.terminal[0] = True
    br.active[0] = False
    _set_reward_queue(runner, [200.0, 2.0])
    runner._record_rollout_rewards(br, prev_active=np.array([True, True]))
    assert (
        br.cum_reward[0] == 10.0 + 1.0 * 200.0 * 10.0
    )  # root(10) + victory(2000) = 2010
    assert br.cum_reward[1] == 10.0 + 1.0 * 2.0 * 10.0  # root(10) + deeper(20) = 30


def test_leaf_bootstrap_discount_exponent_is_settlements_count(monkeypatch):
    """BUG C: the leaf bootstrap is discounted by gamma**(D+1) where D+1 is the
    number of settled decisions (root + deeper), NOT gamma**D. At depth 0 this
    is gamma**1, not gamma**0."""
    runner = _make_runner(monkeypatch)
    N = 2
    br = _make_branches(N=N, gamma=0.9)
    # one settlement (root): r=1 for both, none terminal
    _set_reward_queue(runner, [1.0, 1.0])
    runner._record_rollout_rewards(br, br.active.copy())
    br.depth_done[:] += 1  # =1 settlement
    # monkeypatch _exact_leaf_v_pi to return V_pi = 5.0 for every active branch
    v_pi = torch.full((N,), 5.0)
    q_per_head = torch.zeros(N, runner.action_dim, 4)
    monkeypatch.setattr(
        sd,
        "_exact_leaf_v_pi",
        lambda policy, emb, illegal, ad, h: (v_pi, None, None, q_per_head),
    )
    vals, diag = runner._leaf_values(br)
    # vals = cum (10) + gamma**1 * 5 = 10 + 0.9*5 = 14.5  (NOT gamma**0 * 5 = 15)
    assert np.allclose(vals, [10.0 + 0.9 * 5.0, 10.0 + 0.9 * 5.0])
    assert not np.allclose(
        vals, [15.0, 15.0]
    ), "bootstrap must be discounted by gamma**1"


def test_leaf_terminal_branch_uses_cum_reward_no_bootstrap(monkeypatch):
    """BUG A: terminal branches value = cum_reward (victory counted once),
    bootstrap exactly zero."""
    runner = _make_runner(monkeypatch)
    N = 2
    br = _make_branches(N=N, gamma=0.9)
    # root: r=1, branch 0 terminates with victory (r includes +200)
    br.terminal[0] = True
    br.active[0] = False
    _set_reward_queue(runner, [201.0, 1.0])  # branch 0: shaped+victory=201; branch 1: 1
    runner._record_rollout_rewards(br, prev_active=np.array([True, True]))
    br.depth_done[:] += 1
    # branch 1 bootstrap only
    v_pi = torch.tensor([5.0])
    q_per_head = torch.zeros(1, runner.action_dim, 4)
    monkeypatch.setattr(
        sd,
        "_exact_leaf_v_pi",
        lambda policy, emb, illegal, ad, h: (v_pi, None, None, q_per_head),
    )
    vals, diag = runner._leaf_values(br)
    # terminal branch 0: cum = 10*201 = 2010, no bootstrap
    assert vals[0] == pytest.approx(2010.0)
    # nonterminal branch 1: cum (10) + 0.9**1 * 5 = 14.5
    assert vals[1] == pytest.approx(10.0 + 0.9 * 5.0)


def test_include_intermediate_rewards_false_still_records_terminal(monkeypatch):
    """Even with intermediate rewards disabled, the terminal settlement reward
    (victory) must be recorded so terminal branches are valued correctly."""
    runner = _make_runner(monkeypatch, include_intermediate=False)
    N = 2
    br = _make_branches(N=N, gamma=1.0)
    # non-terminal settlement first (should be skipped with intermediate off)
    _set_reward_queue(runner, [5.0, 5.0])
    runner._record_rollout_rewards(br, br.active.copy())
    assert br.cum_reward.tolist() == [
        0.0,
        0.0,
    ], "non-terminal intermediate reward skipped"
    br.depth_done[:] += 1
    # terminal settlement: branch 0 wins
    br.terminal[0] = True
    br.active[0] = False
    _set_reward_queue(runner, [200.0, 3.0])
    runner._record_rollout_rewards(br, prev_active=np.array([True, True]))
    # branch 0 terminal -> recorded (10*200=2000); branch 1 non-terminal -> skipped
    assert br.cum_reward[0] == 2000.0
    assert br.cum_reward[1] == 0.0
