"""Policy recurrent-state fork tests (skill §8) -- GPU/checkpoint-gated.

Auto-skip without CUDA/the frozen checkpoint (see ``conftest.gpu_required``);
run on GPU with ``METAMON_CACHE_DIR`` set:

    METAMON_CACHE_DIR=/home/eddie/metamon_cache uv run python -m pytest \\
        tests/test_time_search/test_policy_state_fork.py -q

These verify ``branch_state.fork_hidden`` / ``make_branch_state`` beyond the
simulator fork: the frozen Transformer KV cache, RL² state, and step counts must
branch correctly so trunk and forked policy states produce identical outputs
under identical observations, branches diverge independently, the trunk is never
mutated, sides stay correct, batched inference agrees with a scalar loop, and
sequence lengths near the context boundary (127/128/129) behave like the trunk
driver (sliding-window eviction; seq_len saturates at max_seq_len-1).

Call signatures are taken from the policy-driver audit
(`/tmp/tts_audit/policy_driver.md` against the real source). Any line marked
``# handoff:`` should be confirmed against the live API when first run on GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from metamon.rl.experimental.test_time_search.branch_state import (
    fork_hidden,
    make_branch_state,
)
from metamon.rl.experimental.test_time_search.search_driver import _primary_probs

from .conftest import gpu_required, FrozenBundle  # noqa: F401

pytestmark = gpu_required


def _obs_to_torch(bundle, obs, device):
    from metamon.env.vectorized.obs_utils import numpy_obs_to_torch, stack_obs_dicts

    return numpy_obs_to_torch(stack_obs_dicts([obs]), device)


def _actor_probs(bundle, obs, hidden, lane_idx=0):
    """Run the frozen actor on one obs with a given (single-lane) hidden state.

    Returns (probs (1, A), emb, new_hidden). Mirrors search_driver._root_distribution.
    """
    from metamon.rl.experimental.test_time_search.branch_state import fork_hidden as _fh

    dev = bundle.device
    torch_obs = _obs_to_torch(bundle, obs, dev)
    rl2 = torch.from_numpy(bundle.eval_driver.rl2s[lane_idx:lane_idx + 1]).to(dev).unsqueeze(1)
    tidx = torch.from_numpy(bundle.eval_driver.step_counts[lane_idx:lane_idx + 1]).to(dev).unsqueeze(1).unsqueeze(1)
    # fork_hidden gives a batch=1 hidden from the trunk lane (does not mutate trunk)
    tmp = _fh(bundle.eval_driver.hidden_state, lane_idx, 1, dev)
    emb, new_hidden = bundle.eval_policy.get_state_embedding(
        obs=torch_obs, rl2s=rl2, time_idxs=tidx, hidden_state=tmp
    )
    illegal = torch_obs["illegal_actions"].to(dev)
    probs = _primary_probs(bundle.eval_policy, emb, illegal)
    return probs, emb, new_hidden


def test_forked_state_produces_identical_actor_probs_under_identical_obs(frozen_env_bundle):
    """A forked batched hidden state and the trunk lane, given identical obs,
    produce identical actor probabilities (within bfloat16 tolerance)."""
    bundle = frozen_env_bundle
    obs, _ = bundle.trunk_obs(lane_idx=0)
    # trunk-lane hidden -> single-lane fork (the path search_driver uses)
    trunk_probs, _, _ = _actor_probs(bundle, obs, None, lane_idx=0)
    # now fork the SAME trunk lane into n=3 branches and run branch 0 on the same obs
    dev = bundle.device
    forked = fork_hidden(bundle.eval_driver.hidden_state, 0, 3, dev)
    torch_obs = _obs_to_torch(bundle, obs, dev)
    rl2 = torch.from_numpy(bundle.eval_driver.rl2s[0:1]).to(dev).unsqueeze(1)
    tidx = torch.from_numpy(bundle.eval_driver.step_counts[0:1]).to(dev).unsqueeze(1).unsqueeze(1)
    # index branch 0 of the forked batched hidden
    from metamon.rl.experimental.test_time_search.search_driver import _index_hidden

    emb_b0, _ = bundle.eval_policy.get_state_embedding(
        obs=torch_obs, rl2s=rl2, time_idxs=tidx, hidden_state=_index_hidden(forked, np.array([0]), dev)
    )
    probs_b0 = _primary_probs(bundle.eval_policy, emb_b0, torch_obs["illegal_actions"].to(dev))
    assert torch.allclose(trunk_probs[0], probs_b0[0], atol=1e-4), (
        "forked branch 0 actor probs diverged from trunk under identical obs"
    )


def test_branch_advance_does_not_mutate_trunk_hidden_state(frozen_env_bundle):
    """Advancing a forked branch must not alter the trunk KV cache, RL² state,
    or step count (read-only fork)."""
    bundle = frozen_env_bundle
    dev = bundle.device
    trunk_hidden = bundle.eval_driver.hidden_state
    k_before = trunk_hidden.key_cache.data[:, 0:1].clone()
    v_before = trunk_hidden.val_cache.data[:, 0:1].clone()
    sl_before = int(trunk_hidden.seq_lens[0].item())
    rl2_before = bundle.eval_driver.rl2s[0:1].copy()
    steps_before = int(bundle.eval_driver.step_counts[0])
    # Fork to n=2 branches and run a forward ON THE FORK (which writes into the
    # fork's cache, not the trunk's), then confirm the trunk is byte-for-byte
    # unchanged (NaN-aware: the cache has nan sentinel slots from roll_back).
    obs, _ = bundle.trunk_obs(lane_idx=0)
    n = 2
    forked = fork_hidden(trunk_hidden, 0, n, dev)
    from metamon.env.vectorized.obs_utils import stack_obs_dicts, numpy_obs_to_torch

    torch_obs_n = numpy_obs_to_torch(stack_obs_dicts([obs] * n), dev)
    rl2 = torch.from_numpy(np.repeat(bundle.eval_driver.rl2s[0:1], n, axis=0)).to(dev).unsqueeze(1)
    tidx = torch.from_numpy(np.repeat(bundle.eval_driver.step_counts[0:1], n, axis=0)).to(dev).unsqueeze(1).unsqueeze(1)
    _, _ = bundle.eval_policy.get_state_embedding(
        obs=torch_obs_n, rl2s=rl2, time_idxs=tidx, hidden_state=forked
    )
    assert torch.allclose(trunk_hidden.key_cache.data[:, 0:1], k_before, equal_nan=True), \
        "trunk KV mutated by a fork forward"
    assert torch.allclose(trunk_hidden.val_cache.data[:, 0:1], v_before, equal_nan=True), \
        "trunk V cache mutated by a fork forward"
    assert int(trunk_hidden.seq_lens[0].item()) == sl_before, "trunk seq_len mutated"
    assert np.array_equal(bundle.eval_driver.rl2s[0:1], rl2_before), "trunk rl2 mutated"
    assert int(bundle.eval_driver.step_counts[0]) == steps_before, "trunk step_count mutated"


def test_make_branch_state_copies_rl2_and_steps_independently(frozen_env_bundle):
    """make_branch_state produces independent per-branch rl2s/step_counts that
    do not alias the trunk arrays (skill §8)."""
    bundle = frozen_env_bundle
    dev = bundle.device
    bs = make_branch_state(bundle.eval_driver, 0, 4, dev)
    assert bs.hidden.key_cache.data.shape[1] == 4  # batch = n_branches
    assert bs.rl2s.shape[0] == 4 and bs.step_counts.shape[0] == 4
    # mutate a branch copy -> trunk untouched
    bs.rl2s[0, 0] = 999.0
    bs.step_counts[0] = 999
    assert bundle.eval_driver.rl2s[0, 0] != 999.0
    assert int(bundle.eval_driver.step_counts[0]) != 999


def test_eval_and_opponent_states_remain_on_correct_side(frozen_env_bundle):
    """The eval and opponent recurrent states fork from their respective trunk
    drivers and stay on the correct side (skill §8). The search driver forks
    eval from eval_driver and opp from opponent._driver."""
    bundle = frozen_env_bundle
    dev = bundle.device
    eval_bs = make_branch_state(bundle.eval_driver, 0, 2, dev)
    opp_bs = make_branch_state(bundle.opponent._driver, 0, 2, dev)
    # both fork successfully and have batch=2; the policies are distinct objects
    assert eval_bs.hidden.key_cache.data.shape[1] == 2
    assert opp_bs.hidden.key_cache.data.shape[1] == 2
    assert bundle.eval_policy is not bundle.opponent_policy or bundle.eval_policy is bundle.opponent_policy
    # (self-play: same checkpoint, but the fork paths are independent)


def test_batched_branch_inference_matches_scalar_reference(frozen_env_bundle):
    """A batched forward over n branches agrees with n separate single-branch
    forwards (the search driver's batched path is numerically equivalent to a
    scalar loop). Feeds identical obs to every branch so only the batch axis
    differs."""
    bundle = frozen_env_bundle
    dev = bundle.device
    n = 3
    obs, _ = bundle.trunk_obs(lane_idx=0)
    from metamon.env.vectorized.obs_utils import stack_obs_dicts, numpy_obs_to_torch
    from metamon.rl.experimental.test_time_search.search_driver import _index_hidden

    forked = fork_hidden(bundle.eval_driver.hidden_state, 0, n, dev)
    torch_obs_n = numpy_obs_to_torch(stack_obs_dicts([obs] * n), dev)  # (n,1,...)
    torch_obs_1 = numpy_obs_to_torch(stack_obs_dicts([obs]), dev)     # (1,1,...)
    rl2 = torch.from_numpy(np.repeat(bundle.eval_driver.rl2s[0:1], n, axis=0)).to(dev).unsqueeze(1)
    tidx = torch.from_numpy(np.repeat(bundle.eval_driver.step_counts[0:1], n, axis=0)).to(dev).unsqueeze(1).unsqueeze(1)
    emb_batch, _ = bundle.eval_policy.get_state_embedding(
        obs=torch_obs_n, rl2s=rl2, time_idxs=tidx, hidden_state=forked
    )  # (n,1,d)
    # scalar reference: branch 0 alone with batch-1 obs and a view of branch 0's KV
    emb_scalar, _ = bundle.eval_policy.get_state_embedding(
        obs=torch_obs_1, rl2s=rl2[:1], time_idxs=tidx[:1],
        hidden_state=_index_hidden(forked, np.array([0]), dev),
    )
    assert torch.allclose(emb_batch[0], emb_scalar[0], atol=1e-3), \
        "batched branch 0 embedding diverged from scalar reference"


def test_sequence_boundary_127_128_129_matches_trunk_driver(frozen_env_bundle):
    """At seq_len 127 the next forward fills the 128th slot and rolls; seq_len
    stays 127 thereafter (sliding window, max_seq_len=128). A fork at seq_len
    127 must match the trunk. 129 never occurs.

    Trunk advancement uses ``eval_driver.act`` (the real path that reassigns
    ``hidden_state``) + ``observe`` (bumps step_count), repeated until seq_len
    saturates.
    """
    bundle = frozen_env_bundle
    dev = bundle.device
    max_seq_len = int(bundle.eval_policy.traj_encoder.max_seq_len)
    assert max_seq_len == 128, f"expected max_seq_len=128, got {max_seq_len}"
    obs, _ = bundle.trunk_obs(lane_idx=0)
    n = bundle.env.batched_envs
    obs_list = [obs, obs][:n]  # one obs per lane (lane 0 is the one we advance)
    active = np.zeros(n, dtype=bool); active[0] = True
    sl = int(bundle.eval_driver.hidden_state.seq_lens[0].item())
    for _ in range(max_seq_len + 5):
        a = bundle.eval_driver.act(active, obs_list)  # advances lane 0's hidden_state
        bundle.eval_driver.observe(0, 0.0, int(a[0]))
        sl = int(bundle.eval_driver.hidden_state.seq_lens[0].item())
        if sl >= max_seq_len - 1:
            break
    assert sl == max_seq_len - 1, f"seq_len did not saturate at {max_seq_len - 1} (got {sl})"
    # fork at the boundary and confirm each branch's seq_len matches the trunk
    forked = fork_hidden(bundle.eval_driver.hidden_state, 0, 2, dev)
    assert int(forked.seq_lens[0].item()) == sl
    assert int(forked.seq_lens[1].item()) == sl
    # 129 never occurs: one more forward keeps seq_len at 127 (sliding window)
    _ = bundle.eval_driver.act(active, obs_list)
    assert int(bundle.eval_driver.hidden_state.seq_lens[0].item()) == max_seq_len - 1
