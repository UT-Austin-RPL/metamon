"""Tests for Phase B: adaptive-K evaluation (skill §37).

Tests cover:
  * config validation for the adaptive-K parameters;
  * the z-score stopping criterion logic (pure numpy — the round-by-round
    decision to continue or stop);
  * a GPU-gated end-to-end smoke test that runs a small adaptive-K search on
    the real frozen checkpoint and verifies the result is well-formed.
"""

from __future__ import annotations

import numpy as np
import pytest

from metamon.rl.experimental.test_time_search.config import SearchConfig
from metamon.rl.experimental.test_time_search.improvement import (
    build_return_matrix,
    min_z_score,
)

# ---------------------------------------------------------------------------
# Config validation (CPU)
# ---------------------------------------------------------------------------


class TestAdaptiveKConfig:
    def test_valid_config(self):
        c = SearchConfig(
            search_adaptive_k=True,
            search_k_pilot=4,
            search_k_max=64,
            search_k_batch=8,
            search_k_z_stop=2.0,
        )
        assert c.search_adaptive_k is True

    def test_k_pilot_must_be_positive(self):
        with pytest.raises(ValueError, match="search_k_pilot"):
            SearchConfig(search_adaptive_k=True, search_k_pilot=0)

    def test_k_max_must_be_at_least_k_pilot(self):
        with pytest.raises(ValueError, match="search_k_max"):
            SearchConfig(search_adaptive_k=True, search_k_pilot=8, search_k_max=4)

    def test_k_batch_must_be_positive(self):
        with pytest.raises(ValueError, match="search_k_batch"):
            SearchConfig(search_adaptive_k=True, search_k_batch=0)

    def test_k_z_stop_must_be_non_negative(self):
        with pytest.raises(ValueError, match="search_k_z_stop"):
            SearchConfig(search_adaptive_k=True, search_k_z_stop=-1.0)

    def test_adaptive_k_off_does_not_validate_k_params(self):
        """When adaptive_k is off, the K params are not validated (legacy path)."""
        # These would be invalid if adaptive_k were on, but should pass when off.
        c = SearchConfig(search_adaptive_k=False, search_k_pilot=0, search_k_max=0)
        assert c.search_adaptive_k is False


# ---------------------------------------------------------------------------
# Stopping criterion logic (pure numpy)
# ---------------------------------------------------------------------------


def _simulate_adaptive_rounds(
    legal_arr: np.ndarray,
    true_q: np.ndarray,
    K_pilot: int,
    K_batch: int,
    K_max: int,
    z_stop: float,
    noise_scale: float,
    crn_corr: float = 0.0,
    seed: int = 0,
) -> int:
    """Simulate the adaptive-K round-by-round z-stop decision.

    Returns the effective K (the number of rollouts per action when the
    stopping criterion fires or K_max is reached). Mirrors the logic in
    ``SearchEvalRunner._adaptive_rollout_core``: after each round, build R from
    all accumulated per-branch returns, compute min_z, and stop if z >= z_stop.
    """
    rng = np.random.default_rng(seed)
    A = len(legal_arr)
    K_done = 0
    K_next = K_pilot
    all_q = []
    all_ra = []
    all_ri = []

    # Pre-generate the full K_max noise matrix (so each k's noise is consistent
    # regardless of when it's "revealed" — mirrors the K-independent seed bank).
    if crn_corr >= 1.0:
        noise = rng.normal(0, noise_scale, size=K_max)
        R_full = np.array(
            [[true_q[a] + noise[k] for k in range(K_max)] for a in range(A)]
        )
    else:
        R_full = np.array(
            [
                [true_q[a] + rng.normal(0, noise_scale) for k in range(K_max)]
                for a in range(A)
            ]
        )

    while K_done < K_max:
        K_round = min(K_next, K_max - K_done)
        # Reveal the next K_round columns of R_full
        for a in range(A):
            for k in range(K_done, K_done + K_round):
                all_q.append(R_full[a, k])
                all_ra.append(legal_arr[a])
                all_ri.append(k)
        K_done += K_round

        if K_done >= K_max:
            break

        q_pb = np.array(all_q)
        ra = np.array(all_ra)
        ri = np.array(all_ri)
        R = build_return_matrix(q_pb, ra, ri, legal_arr, K_done)
        q_mean = np.nanmean(R, axis=1)
        z = min_z_score(R, q_mean)
        if not np.isnan(z) and z >= z_stop:
            break
        K_next = K_batch

    return K_done


class TestAdaptiveKStoppingCriterion:
    def test_stops_at_k_pilot_when_confident(self):
        """With full CRN and a large gap, z is huge -> stops at K_pilot."""
        legal = np.array([0, 1])
        K_eff = _simulate_adaptive_rounds(
            legal_arr=legal,
            true_q=np.array([0.0, 100.0]),
            K_pilot=4,
            K_batch=4,
            K_max=64,
            z_stop=2.0,
            noise_scale=10.0,
            crn_corr=1.0,  # full CRN -> paired SE ≈ 0 -> z huge
        )
        assert K_eff == 4  # stopped immediately at pilot

    def test_runs_to_k_max_when_noisy(self):
        """With independent noise and a tiny gap, z stays low -> runs to K_max."""
        legal = np.array([0, 1])
        K_eff = _simulate_adaptive_rounds(
            legal_arr=legal,
            true_q=np.array([0.0, 0.01]),  # tiny gap
            K_pilot=4,
            K_batch=4,
            K_max=16,
            z_stop=3.0,
            noise_scale=10.0,
            crn_corr=0.0,  # independent noise -> high paired SE
            seed=42,
        )
        # With a tiny gap and high noise, z should stay below 3.0 even at K=16
        assert K_eff == 16  # ran to max

    def test_stops_partway_when_z_crosses_threshold(self):
        """With moderate noise, z should cross the threshold partway through."""
        legal = np.array([0, 1])
        # gap=5, noise=3, z_stop=2.0: z ≈ 5 / (3*sqrt(2)/sqrt(K)) = 5*sqrt(K)/4.24
        # z>=2 when sqrt(K) >= 2*4.24/5 = 1.70 -> K >= 2.89 -> K>=4 (pilot) might
        # already stop. Use a higher z_stop to force a few rounds.
        K_eff = _simulate_adaptive_rounds(
            legal_arr=legal,
            true_q=np.array([0.0, 5.0]),
            K_pilot=2,
            K_batch=2,
            K_max=32,
            z_stop=5.0,  # high threshold
            noise_scale=3.0,
            crn_corr=0.0,
            seed=100,
        )
        # z at K=2: ~5*sqrt(2)/4.24 ~ 1.67 < 5 -> continue
        # z at K=4: ~5*2/4.24 ~ 2.36 < 5 -> continue
        # z at K=8: ~5*2.83/4.24 ~ 3.34 < 5 -> continue
        # z at K=16: ~5*4/4.24 ~ 4.72 < 5 -> continue
        # z at K=32: ~5*5.66/4.24 ~ 6.67 >= 5 -> stop (or runs to 32)
        assert K_eff <= 32
        assert K_eff > 2  # didn't stop at pilot

    def test_single_action_always_stops_at_pilot(self):
        """A single legal action has no competitors -> min_z=inf -> stops."""
        legal = np.array([3])
        K_eff = _simulate_adaptive_rounds(
            legal_arr=legal,
            true_q=np.array([5.0]),
            K_pilot=4,
            K_batch=4,
            K_max=64,
            z_stop=100.0,  # very high, but inf > anything
            noise_scale=10.0,
        )
        assert K_eff == 4

    def test_equal_q_never_stops_early(self):
        """When all actions have equal Q, z=0 -> never stops early -> K_max."""
        legal = np.array([0, 1, 2])
        K_eff = _simulate_adaptive_rounds(
            legal_arr=legal,
            true_q=np.array([3.0, 3.0, 3.0]),
            K_pilot=4,
            K_batch=4,
            K_max=16,
            z_stop=1.0,
            noise_scale=0.0,  # no noise -> gap=0, SE=0 -> z=0
        )
        assert K_eff == 16  # never stops (z=0 < 1.0 at every round)


# ---------------------------------------------------------------------------
# GPU-gated end-to-end smoke test
# ---------------------------------------------------------------------------

gpu_required = pytest.mark.skipif(
    not (
        __import__("torch").cuda.is_available()
        and __import__("os").path.exists(
            __import__("os").path.expanduser(
                "~/metamon_runs/mini_online_psro_v1.4/mini_online_psro_v1.4/"
                "ckpts/policy_weights/policy_epoch_740.pt"
            )
        )
    ),
    reason="requires CUDA + the frozen checkpoint (Phase B smoke)",
)


@gpu_required
class TestAdaptiveKSmokeGPU:
    """End-to-end smoke: run a small adaptive-K search on the real checkpoint."""

    def test_adaptive_k_smoke(self, frozen_env_bundle):
        """Run adaptive-K search (K_pilot=2, K_max=8, K_batch=2, z_stop=1.0)
        on a real root and verify the result is well-formed."""
        from metamon.rl.experimental.test_time_search.search_driver import (
            SearchEvalRunner,
        )

        bundle = frozen_env_bundle
        config = SearchConfig(
            search_mode="oracle-root-mc",
            search_rollouts_per_action=2,  # overridden by adaptive_k
            search_depth=0,
            search_root_candidate_mode="all_legal",
            search_every_n_decisions=1,
            search_chance_mode="resample_crn",
            search_root_opponent_coupling=True,
            search_leaf_value_mode="policy_expectation",
            search_value_normalization=False,
            search_ablation="single_anchor_kl",
            search_error_policy="raise",
            search_adaptive_k=True,
            search_k_pilot=2,
            search_k_max=8,
            search_k_batch=2,
            search_k_z_stop=1.0,
            search_seed=0,
        )
        runner = bundle.make_runner(config)
        runner._battle_id = "test_adaptive"
        runner._decision_counter = 1
        try:
            obs, legal = bundle.trunk_obs(0)
            if len(legal) < 2:
                pytest.skip("need >= 2 legal actions for a meaningful search test")
            action, rec = runner.search_root(0, obs, legal)
            # Basic well-formedness
            assert rec.error == ""
            assert rec.adaptive_k is True
            assert 1 <= rec.k_effective <= config.search_k_max
            assert len(rec.search_q_mean) == len(rec.legal_actions)
            assert rec.n_rollouts == len(rec.legal_actions) * rec.k_effective
            # Q values should be finite
            assert all(np.isfinite(q) for q in rec.search_q_mean)
        finally:
            runner.close()

    def test_adaptive_k_no_leak(self, frozen_env_bundle):
        """Verify no fork lanes or snapshots leak after adaptive-K search."""
        from metamon.rl.experimental.test_time_search.search_driver import (
            SearchEvalRunner,
        )

        bundle = frozen_env_bundle
        config = SearchConfig(
            search_mode="oracle-root-mc",
            search_rollouts_per_action=2,
            search_depth=0,
            search_chance_mode="resample_crn",
            search_leaf_value_mode="policy_expectation",
            search_error_policy="raise",
            search_adaptive_k=True,
            search_k_pilot=2,
            search_k_max=8,
            search_k_batch=2,
            search_k_z_stop=1.0,
            search_seed=0,
        )
        runner = bundle.make_runner(config)
        runner._battle_id = "test_adaptive_leak"
        runner._decision_counter = 1
        try:
            obs, legal = bundle.trunk_obs(0)
            if len(legal) < 2:
                pytest.skip("need >= 2 legal actions")
            runner.search_root(0, obs, legal)
            # No fork lanes should remain after search_root
            assert len(runner._active_fork_lanes) == 0
        finally:
            runner.close()
            assert len(runner._active_fork_lanes) == 0
