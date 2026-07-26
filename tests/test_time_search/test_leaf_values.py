"""Exact leaf policy-expectation tests (skill §10).

Tests the fixed-shape all-action critic path and the exact
``V_pi(h) = sum_a pi(a|h) Q(h,a)`` leaf bootstrap with a MOCK policy on CPU
(no checkpoint / GPU needed). Verifies:

  * ``_all_action_q`` returns per-(branch, action) Q (mean over critics);
  * ``_exact_leaf_v_pi`` matches a hand-computed expectation;
  * the vectorized all-action call matches a brute-force one-action-at-a-time
    loop using the legacy ``_critic_leaf_values`` helper;
  * illegal actions are masked out of the expectation;
  * per-critic-head Q is returned for disagreement logging.
"""

from __future__ import annotations

import pytest
import torch

from metamon.rl.experimental.test_time_search.search_driver import (
    _all_action_q,
    _critic_leaf_values,
    _exact_leaf_v_pi,
    _primary_probs,
)

# ---------------------------------------------------------------------------
# mock policy
# ---------------------------------------------------------------------------


class _MockCategorical:
    def __init__(self, probs):
        self.probs = probs


class _MockBinDist:
    def __init__(self, a_oh, q_per_head):
        self.a_oh = a_oh
        self.q_per_head = q_per_head  # (B, A, C)


class _MockCritics:
    def __init__(self, q_per_head):
        self.q_per_head = q_per_head  # (B, A, C)

    def __call__(self, emb, a_oh):
        return _MockBinDist(a_oh, self.q_per_head)

    def bin_dist_to_raw_vals(self, bin_dist):
        a_oh = bin_dist.a_oh  # (K, B, 1, G, A)
        K, B, _L, G, A = a_oh.shape
        act = a_oh[:, :, 0, 0, :].argmax(-1)  # (K, B)
        C = self.q_per_head.shape[2]
        out = torch.zeros(K, B, 1, C, G, 1)
        for k in range(K):
            for b in range(B):
                out[k, b, 0, :, :, 0] = bin_dist.q_per_head[b, act[k, b], :].view(C, 1)
        return out


class _MockActor:
    def __init__(self, probs):
        self.probs = probs  # (B, A) prescribed, already renormalized over legal

    def __call__(self, emb, straight_from_obs=None):
        illegal = straight_from_obs["illegal_actions"]
        if illegal.ndim == 3:
            illegal = illegal[:, 0, :]
        p = self.probs.clone().masked_fill(illegal, 0.0)
        p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
        B, A = p.shape
        return _MockCategorical(p.unsqueeze(1).unsqueeze(1).expand(B, 1, 2, A))


class _MockPolicy:
    def __init__(self, probs, q_per_head):
        self.actor = _MockActor(probs)
        self.critics = _MockCritics(q_per_head)
        self.gammas = torch.tensor([0.9, 0.999])

    def popart(self, x, normalized=True):
        return x  # identity denormalization


def _make_policy():
    B, A, C = 2, 4, 2
    probs = torch.tensor(
        [
            [0.5, 0.3, 0.2, 0.0],  # action 3 illegal for branch 0
            [0.1, 0.4, 0.4, 0.1],
        ]
    )
    # q_per_head[b, a, c]
    q_per_head = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [9.0, 9.0]],  # critics agree (row 0)
            [
                [0.0, 0.0],
                [5.0, 7.0],
                [4.0, 4.0],
                [8.0, 8.0],
            ],  # critics disagree on a=1 (row 1)
        ]
    )
    policy = _MockPolicy(probs, q_per_head)
    illegal_3d = torch.tensor(
        [[[False, False, False, True]], [[False, False, False, False]]]
    )  # (B,1,A) True=illegal
    emb = torch.zeros(B, 1, 8)
    horizon = 1  # 0.999
    return policy, emb, illegal_3d, horizon, B, A, C


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_all_action_q_returns_per_action_mean_over_critics():
    policy, emb, illegal_3d, horizon, B, A, C = _make_policy()
    q_mean, q_per_head = _all_action_q(policy, emb, A, horizon)
    assert q_mean.shape == (B, A)
    assert q_per_head.shape == (B, A, C)
    expected = torch.tensor([[1.0, 2.0, 3.0, 9.0], [0.0, 6.0, 4.0, 8.0]])  # mean over C
    assert torch.allclose(q_mean, expected, atol=1e-6)


def test_exact_leaf_v_pi_matches_hand_computed():
    policy, emb, illegal_3d, horizon, B, A, C = _make_policy()
    v_pi, q_mean, probs, q_per_head = _exact_leaf_v_pi(
        policy, emb, illegal_3d, A, horizon
    )
    # branch 0: 0.5*1 + 0.3*2 + 0.2*3 = 1.7  (action 3 illegal -> excluded)
    # branch 1: 0.1*0 + 0.4*6 + 0.4*4 + 0.1*8 = 4.8
    assert torch.allclose(v_pi, torch.tensor([1.7, 4.8]), atol=1e-5)
    assert torch.allclose(probs[0], torch.tensor([0.5, 0.3, 0.2, 0.0]), atol=1e-6)
    assert torch.allclose(probs[1], torch.tensor([0.1, 0.4, 0.4, 0.1]), atol=1e-6)


def test_exact_leaf_v_pi_brute_force_equivalence():
    """The vectorized all-action expectation equals a one-action-at-a-time loop
    using the legacy ``_critic_leaf_values`` helper (skill §10 / Phase 0C)."""
    policy, emb, illegal_3d, horizon, B, A, C = _make_policy()
    v_vec, _, probs, _ = _exact_leaf_v_pi(policy, emb, illegal_3d, A, horizon)
    v_brute = torch.zeros(B)
    for a in range(A):
        q_a = _critic_leaf_values(
            policy, emb, torch.full((B,), a, dtype=torch.long), A, horizon
        )
        v_brute = (
            v_brute + probs[:, a] * q_a
        )  # illegal actions have p=0 -> no contribution
    assert torch.allclose(v_vec, v_brute, atol=1e-5)


def test_illegal_actions_excluded_from_expectation():
    policy, emb, illegal_3d, horizon, B, A, C = _make_policy()
    v_pi, q_mean, probs, _ = _exact_leaf_v_pi(policy, emb, illegal_3d, A, horizon)
    # branch 0 action 3 is illegal: prob 0 and excluded from V_pi
    assert probs[0, 3] == 0.0
    # a high-Q illegal action (Q=9) must not leak into V_pi
    assert float(v_pi[0]) == pytest.approx(1.7, abs=1e-5)


def test_primary_probs_respects_illegal_mask():
    policy, emb, illegal_3d, horizon, B, A, C = _make_policy()
    probs = _primary_probs(policy, emb, illegal_3d)  # (B, A)
    assert probs.shape == (B, A)
    assert probs[0, 3] == 0.0  # illegal -> 0
    assert torch.allclose(probs[0].sum(), torch.tensor(1.0), atol=1e-6)


def test_per_head_q_exposes_critic_disagreement():
    policy, emb, illegal_3d, horizon, B, A, C = _make_policy()
    _, q_per_head = _all_action_q(policy, emb, A, horizon)
    # branch 1, action 1: critics [5, 7] -> unbiased std sqrt(2) (torch default)
    disagree = q_per_head[1, 1].std(dim=-1)
    assert torch.allclose(disagree, torch.std(torch.tensor([5.0, 7.0])), atol=1e-5)
    # branch 0 critics agree on every action -> std 0
    assert torch.allclose(q_per_head[0].std(dim=-1), torch.zeros(A), atol=1e-6)
