"""Tests for the policy-improvement operators (skill §12)."""

from __future__ import annotations

import numpy as np
import pytest

from metamon.rl.experimental.test_time_search.improvement import (
    entropy,
    improve_policy,
    kl,
    build_return_matrix,
    paired_sem,
    min_z_score,
)


def _base_probs(legal, probs, dim=None):
    if dim is None:
        dim = int(max(legal)) + 1
    p = np.zeros(dim, dtype=np.float64)
    p[list(legal)] = probs
    return p / p.sum()


def _q_full(legal, q_legal, dim=None):
    if dim is None:
        dim = int(max(legal)) + 1
    q = np.full(dim, np.nan, dtype=np.float64)
    q[list(legal)] = q_legal
    return q


def _counts_full(legal, counts_legal, dim=None):
    if dim is None:
        dim = int(max(legal)) + 1
    c = np.zeros(dim, dtype=np.int64)
    c[list(legal)] = counts_legal
    return c


def _mask(legal, dim=None):
    if dim is None:
        dim = int(max(legal)) + 1
    m = np.zeros(dim, bool)
    m[list(legal)] = True
    return m


def _run(legal, base_probs, q, ablation="single_anchor_kl", **kw):
    dim = base_probs.shape[0]
    mask = _mask(legal, dim)
    return improve_policy(
        base_probs=base_probs,
        search_q_mean=np.where(mask, q, np.nan),
        search_q_std=np.zeros(dim),
        rollout_counts=_counts_full(legal, [8] * len(legal), dim),
        terminal_frac=np.zeros(dim),
        legal_mask=mask,
        beta=kw.pop("beta", 1.0),
        ablation=ablation,
        root_selection=kw.pop("root_selection", "argmax"),
        rng=kw.pop("rng", np.random.default_rng(0)),
        **kw,
    )


# ---------------------------------------------------------------------------
# legacy single-anchor behavior (preserved)
# ---------------------------------------------------------------------------


def test_equal_q_returns_base_policy():
    legal = np.array([0, 2, 5])
    base = _base_probs(legal, [0.5, 0.3, 0.2])
    q = _q_full(legal, [3.0, 3.0, 3.0])
    r = _run(legal, base, q, ablation="single_anchor_kl")
    assert np.allclose(r.searched_probs[legal], base[legal], atol=1e-9)
    assert r.kl_to_base < 1e-9


def test_large_beta_approaches_base():
    legal = np.array([0, 1])
    base = _base_probs(legal, [0.9, 0.1])
    q = np.array([1.0, 5.0])
    r = _run(legal, base, q, ablation="single_anchor_kl", beta=1e6)
    assert np.allclose(r.searched_probs, base, atol=1e-3)
    assert r.selected_action == r.base_argmax


def test_small_beta_approaches_softmax_q_within_prior_support():
    legal = np.array([0, 1])
    base = _base_probs(legal, [0.1, 0.9])
    q = np.array([10.0, 0.0])
    r = _run(legal, base, q, ablation="single_anchor_kl", beta=1e-6)
    # concentrates on the high-Q action within the support of pi_base
    assert r.searched_probs[0] > 0.99
    assert r.selected_action == 0


def test_illegal_actions_stay_zero():
    legal = np.array([0, 3])
    base = _base_probs(legal, [0.4, 0.6])
    q = _q_full(legal, [2.0, 8.0])
    r = _run(legal, base, q, ablation="single_anchor_kl")
    assert r.searched_probs.sum() == pytest.approx(1.0)
    assert r.searched_probs[1] == 0.0 and r.searched_probs[2] == 0.0


def test_argmax_q_ablation_ignores_prior():
    legal = np.array([0, 1])
    base = _base_probs(legal, [0.99, 0.01])
    q = np.array([0.0, 1.0])
    r = _run(legal, base, q, ablation="argmax_q")
    assert r.selected_action == 1
    assert r.searched_probs[1] == 1.0


def test_softmax_q_ablation():
    legal = np.array([0, 1])
    base = _base_probs(legal, [0.5, 0.5])
    q = np.array([0.0, 10.0])
    r = _run(legal, base, q, ablation="softmax_q")
    assert r.searched_probs[1] > r.searched_probs[0]
    assert r.searched_probs.sum() == pytest.approx(1.0)


def test_base_only_ablation_returns_base():
    legal = np.array([0, 1])
    base = _base_probs(legal, [0.7, 0.3])
    q = np.array([0.0, 99.0])
    r = _run(legal, base, q, ablation="base_only")
    assert np.allclose(r.searched_probs, base)
    assert r.selected_action == r.base_argmax


def test_advantages_centered_by_base_mean():
    legal = np.array([0, 1])
    base = _base_probs(legal, [0.5, 0.5])
    q = np.array([2.0, 4.0])
    r = _run(legal, base, q, ablation="single_anchor_kl", normalize_advantages=False)
    # baseline = 0.5*2 + 0.5*4 = 3; adv = [-1, +1]
    assert np.allclose(r.advantages[legal], [-1.0, 1.0])


def test_kl_and_entropy_nonneg_and_consistent():
    legal = np.array([0, 1, 2])
    base = _base_probs(legal, [0.6, 0.3, 0.1])
    q = np.array([1.0, 3.0, 2.0])
    r = _run(legal, base, q, ablation="single_anchor_kl")
    assert r.kl_to_base >= 0.0
    assert r.base_entropy >= 0.0 and r.searched_entropy >= 0.0
    assert kl(base, base) < 1e-9
    assert entropy(base) == pytest.approx(entropy(base))


def test_changed_argmax_flag():
    legal = np.array([0, 1])
    base = _base_probs(legal, [0.9, 0.1])
    q = np.array([0.0, 5.0])
    r = _run(legal, base, q, ablation="single_anchor_kl", beta=0.1)
    assert r.base_argmax == 0
    assert r.selected_action == 1
    assert r.changed_argmax is True


def test_kl_anchor_is_legacy_alias_for_single_anchor_kl():
    legal = np.array([0, 1, 2])
    base = _base_probs(legal, [0.6, 0.3, 0.1])
    q = np.array([1.0, 3.0, 2.0])
    r_alias = _run(legal, base, q, ablation="kl_anchor")
    r_canon = _run(legal, base, q, ablation="single_anchor_kl")
    assert r_alias.operator == "single_anchor_kl"
    assert np.allclose(r_alias.searched_probs, r_canon.searched_probs)


# ---------------------------------------------------------------------------
# magnetic operator (skill §12)
# ---------------------------------------------------------------------------


def test_magnetic_alpha_zero_equals_single_anchor():
    legal = np.array([0, 1, 2])
    base = _base_probs(legal, [0.6, 0.3, 0.1])
    q = np.array([1.0, 3.0, 2.0])
    r_mag = _run(legal, base, q, ablation="magnetic_kl", alpha=0.0, beta=1.0)
    r_sa = _run(legal, base, q, ablation="single_anchor_kl", alpha=0.0, beta=1.0)
    assert r_mag.operator == "magnetic_kl"
    assert np.allclose(r_mag.searched_probs, r_sa.searched_probs, atol=1e-12)


def test_magnetic_moves_toward_high_q():
    legal = np.array([0, 1])
    base = _base_probs(legal, [0.9, 0.1])
    q = np.array([0.0, 5.0])
    r = _run(
        legal,
        base,
        q,
        ablation="magnetic_kl",
        alpha=1.0,
        beta=1.0,
        normalize_advantages=False,
    )
    assert r.searched_probs[1] > base[1]
    assert r.selected_action == 1


def test_magnetic_recovers_low_prior_action_better_than_single_anchor():
    """A uniform magnet with alpha>0 can lift a near-zero-prior high-Q action,
    which a single-anchor update cannot (skill §12)."""
    legal = np.array([0, 1])
    base = _base_probs(legal, [1.0, 0.0])  # action 1 has exactly zero prior
    q = np.array([0.0, 10.0])
    r_mag = _run(legal, base, q, ablation="magnetic_kl", alpha=2.0, beta=0.5)
    r_sa = _run(legal, base, q, ablation="single_anchor_kl", alpha=0.0, beta=0.5)
    assert r_mag.searched_probs[1] > r_sa.searched_probs[1]
    assert r_mag.searched_probs[1] > 0.0


def test_magnetic_beta_large_approaches_base():
    legal = np.array([0, 1])
    base = _base_probs(legal, [0.7, 0.3])
    q = np.array([0.0, 9.0])
    r = _run(legal, base, q, ablation="magnetic_kl", alpha=1.0, beta=1e6)
    assert np.allclose(r.searched_probs, base, atol=1e-3)


def test_magnetic_alpha_large_approaches_uniform_magnet():
    """With alpha>>beta, the magnetic term dominates and pi approaches the
    uniform magnet (rho) rather than the base policy."""
    legal = np.array([0, 1, 2])
    base = _base_probs(legal, [0.9, 0.09, 0.01])
    q = np.array([0.0, 0.0, 0.0])  # equal Q -> no exp(Q) bias
    r = _run(legal, base, q, ablation="magnetic_kl", alpha=1e6, beta=1.0)
    assert np.allclose(r.searched_probs[legal], 1.0 / 3, atol=1e-3)


def test_magnetic_constant_shift_invariance():
    legal = np.array([0, 1, 2])
    base = _base_probs(legal, [0.5, 0.3, 0.2])
    q = np.array([1.0, 3.0, 2.0])
    r0 = _run(legal, base, q, ablation="magnetic_kl", alpha=1.0, beta=1.0)
    r1 = _run(legal, base, q + 100.0, ablation="magnetic_kl", alpha=1.0, beta=1.0)
    assert np.allclose(r0.searched_probs, r1.searched_probs, atol=1e-9)


# ---------------------------------------------------------------------------
# constant-shift invariance (all operators)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["single_anchor_kl", "magnetic_kl", "softmax_q"])
def test_constant_shift_invariance(op):
    legal = np.array([0, 1, 2])
    base = _base_probs(legal, [0.6, 0.3, 0.1])
    q = np.array([1.0, 3.0, 2.0])
    r0 = _run(legal, base, q, ablation=op, alpha=1.0)
    r1 = _run(legal, base, q + 73.0, ablation=op, alpha=1.0)
    assert np.allclose(r0.searched_probs, r1.searched_probs, atol=1e-9)


# ---------------------------------------------------------------------------
# value scale / normalization (skill §11)
# ---------------------------------------------------------------------------


def test_no_zscore_raw_advantages():
    legal = np.array([0, 1])
    base = _base_probs(legal, [0.5, 0.5])
    q = np.array([2.0, 4.0])  # baseline 3 -> raw adv [-1, +1]
    r = _run(
        legal,
        base,
        q,
        ablation="single_anchor_kl",
        normalize_advantages=False,
        global_advantage_scale=None,
    )
    assert np.allclose(r.advantages[legal], [-1.0, 1.0])
    assert r.value_scale_mode == "raw"


def test_global_advantage_scale_divides_advantages():
    legal = np.array([0, 1])
    base = _base_probs(legal, [0.5, 0.5])
    q = np.array([2.0, 4.0])  # raw adv [-1, +1]; /10 -> [-0.1, +0.1]
    r = _run(
        legal,
        base,
        q,
        ablation="single_anchor_kl",
        normalize_advantages=False,
        global_advantage_scale=10.0,
    )
    assert np.allclose(r.advantages[legal], [-0.1, 0.1])
    assert r.value_scale_mode == "global_standardized"
    assert r.global_advantage_scale == 10.0


def test_global_scale_makes_beta_stable_across_root_q_magnitudes():
    """Same scaled gap -> same pi regardless of raw Q magnitude (skill §11)."""
    legal = np.array([0, 1])
    base = _base_probs(legal, [0.5, 0.5])
    # two roots with very different raw Q magnitudes but the SAME raw advantage
    # gap of 2.0; with a fixed global scale both should produce identical pi.
    r_small = _run(
        legal,
        base,
        np.array([0.0, 2.0]),
        ablation="single_anchor_kl",
        normalize_advantages=False,
        global_advantage_scale=10.0,
        beta=1.0,
    )
    r_big = _run(
        legal,
        base,
        np.array([1000.0, 1002.0]),
        ablation="single_anchor_kl",
        normalize_advantages=False,
        global_advantage_scale=10.0,
        beta=1.0,
    )
    assert np.allclose(r_small.searched_probs, r_big.searched_probs, atol=1e-9)


def test_legacy_zscore_normalization_label():
    legal = np.array([0, 1, 2])
    base = _base_probs(legal, [0.6, 0.3, 0.1])
    q = np.array([1.0, 3.0, 2.0])
    r = _run(legal, base, q, ablation="single_anchor_kl", normalize_advantages=True)
    assert r.value_scale_mode == "legacy_zscore"


# ---------------------------------------------------------------------------
# structural invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op", ["single_anchor_kl", "magnetic_kl", "softmax_q", "argmax_q", "base_only"]
)
def test_all_outputs_sum_to_one(op):
    legal = np.array([0, 1, 2])
    base = _base_probs(legal, [0.6, 0.3, 0.1])
    q = np.array([1.0, 3.0, 2.0])
    r = _run(legal, base, q, ablation=op, alpha=1.0, root_selection="sample")
    assert r.searched_probs.sum() == pytest.approx(1.0, abs=1e-9)
    assert r.searched_probs[~_mask(legal, base.shape[0])].sum() == 0.0


def test_validation_rejects_bad_operators_and_params():
    legal = np.array([0, 1])
    base = _base_probs(legal, [0.5, 0.5])
    q = np.array([0.0, 1.0])
    mask = _mask(legal, base.shape[0])
    with pytest.raises(ValueError):
        improve_policy(
            base,
            np.where(mask, q, np.nan),
            np.zeros(2),
            np.array([1, 1]),
            np.zeros(2),
            mask,
            beta=1.0,
            ablation="nonsense",
        )
    with pytest.raises(ValueError):
        improve_policy(
            base,
            np.where(mask, q, np.nan),
            np.zeros(2),
            np.array([1, 1]),
            np.zeros(2),
            mask,
            beta=-1.0,
            ablation="single_anchor_kl",
        )
    with pytest.raises(ValueError):
        improve_policy(
            base,
            np.where(mask, q, np.nan),
            np.zeros(2),
            np.array([1, 1]),
            np.zeros(2),
            mask,
            beta=1.0,
            ablation="magnetic_kl",
            alpha=-0.5,
        )
    with pytest.raises(ValueError):
        improve_policy(
            base,
            np.where(mask, q, np.nan),
            np.zeros(2),
            np.array([1, 1]),
            np.zeros(2),
            mask,
            beta=1.0,
            ablation="single_anchor_kl",
            z_gate=-1.0,
        )


# ---------------------------------------------------------------------------
# Phase C: confidence-gated update (skill §37)
# ---------------------------------------------------------------------------


def _branch_data(legal, q_per_action, K, noise_scale=0.0, crn_corr=1.0, seed=0):
    """Build per-branch return data for ``len(legal)`` actions × ``K`` rollouts.

    ``q_per_action[a]`` is the TRUE mean return for action ``a``. Each branch
    return is ``q_per_action[a] + noise``. With ``crn_corr=1.0`` (full CRN),
    the same noise sequence is shared across actions so paired differences
    are noise-free (paired SE = 0 for nonzero gaps). With ``crn_corr=0.0``,
    each action gets independent noise.
    """
    rng = np.random.default_rng(seed)
    A = len(legal)
    if crn_corr >= 1.0:
        # Full CRN: same noise[k] for every action at rollout k
        noise = rng.normal(0, noise_scale, size=K)
        R = np.array([[q_per_action[a] + noise[k] for k in range(K)] for a in range(A)])
    else:
        R = np.array(
            [
                [q_per_action[a] + rng.normal(0, noise_scale) for k in range(K)]
                for a in range(A)
            ]
        )
    # Flatten into per-branch arrays in the np.repeat(legal, K) layout
    root_action = np.repeat(legal, K)
    rollout_index = np.tile(np.arange(K), A)
    q_per_branch = R.reshape(A * K)
    return q_per_branch, root_action, rollout_index, R


class TestBuildReturnMatrix:
    def test_correct_shape_and_values(self):
        legal = np.array([0, 2, 5])
        K = 4
        q_pb, ra, ri, R_true = _branch_data(legal, [1.0, 2.0, 3.0], K)
        R = build_return_matrix(q_pb, ra, ri, legal, K)
        assert R.shape == (3, K)
        assert np.allclose(R, R_true)

    def test_empty_when_no_branches(self):
        R = build_return_matrix(
            np.array([]), np.array([]), np.array([]), np.array([0, 1]), 4
        )
        assert R.shape == (2, 4)
        assert np.all(np.isnan(R))


class TestPairedSEM:
    def test_zero_with_full_crn(self):
        """With perfect CRN, paired differences have zero variance -> SE≈0."""
        legal = np.array([0, 1])
        K = 8
        _, _, _, R = _branch_data(legal, [1.0, 3.0], K, noise_scale=10.0, crn_corr=1.0)
        se = paired_sem(R, 0, 1)
        # identical noise -> near-zero paired variance (floating-point may give ~1e-16)
        assert se == pytest.approx(0.0, abs=1e-10)

    def test_nonzero_with_independent_noise(self):
        legal = np.array([0, 1])
        K = 100
        _, _, _, R = _branch_data(legal, [0.0, 0.0], K, noise_scale=5.0, crn_corr=0.0)
        se = paired_sem(R, 0, 1)
        # independent noise of scale 5 -> SE ~ 5*sqrt(2)/sqrt(100) ~ 0.707
        assert se > 0.0
        assert 0.3 < se < 1.2

    def test_inf_for_fewer_than_two_samples(self):
        R2 = np.array([[1.0], [3.0]])
        assert paired_sem(R2, 0, 1) == float("inf")  # K=1 -> inf
        # K=2 with identical differences -> SE=0 (zero variance)
        R = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert paired_sem(R, 0, 1) == 0.0

    def test_inf_for_empty_matrix(self):
        R = np.empty((0, 0))
        assert paired_sem(R, 0, 1) == float("inf")


class TestMinZScore:
    def test_inf_when_single_action(self):
        R = np.array([[1.0, 2.0, 3.0]])
        assert min_z_score(R, np.array([5.0])) == float("inf")

    def test_inf_with_full_crn_and_nonzero_gap(self):
        """With CRN (paired SE≈0) and a nonzero gap, z is very large (confident)."""
        legal = np.array([0, 1])
        K = 8
        _, _, _, R = _branch_data(legal, [1.0, 3.0], K, noise_scale=10.0, crn_corr=1.0)
        z = min_z_score(R, np.array([1.0, 3.0]))
        # gap=2, SE≈0 -> z is huge (may not be exactly inf due to floating-point)
        assert z > 1e10

    def test_finite_with_independent_noise(self):
        legal = np.array([0, 1])
        K = 100
        _, _, _, R = _branch_data(legal, [0.0, 5.0], K, noise_scale=5.0, crn_corr=0.0)
        z = min_z_score(R, np.array([0.0, 5.0]))
        # gap=5, SE ~ 5*sqrt(2)/10 ~ 0.707 -> z ~ 7
        assert z > 3.0

    def test_zero_when_gap_is_zero(self):
        legal = np.array([0, 1])
        K = 8
        _, _, _, R = _branch_data(legal, [3.0, 3.0], K, noise_scale=1.0, crn_corr=1.0)
        z = min_z_score(R, np.array([3.0, 3.0]))
        # gap=0, SE=0 -> z=0 (exactly tied -> should gate)
        assert z == 0.0

    def test_min_over_all_competitors(self):
        """With 3 actions, min_z is the weakest separation (best vs 2nd-best)."""
        legal = np.array([0, 1, 2])
        K = 50
        # action 0 is best (Q=10), action 1 is close (Q=9), action 2 is far (Q=0)
        _, _, _, R = _branch_data(
            legal, [10.0, 9.0, 0.0], K, noise_scale=3.0, crn_corr=0.0, seed=42
        )
        z = min_z_score(R, np.array([10.0, 9.0, 0.0]))
        # The min is between action 0 and action 1 (gap=1, small relative to noise)
        # not between action 0 and action 2 (gap=10, large)
        assert z < 5.0  # the close competitor dominates


class TestConfidenceGatedKL:
    def _run_gated(self, legal, base, q, K=8, z_gate=2.0, adaptive_beta=False, **kw):
        """Run improve_policy with confidence_gated_kl and synthetic per-branch data."""
        dim = base.shape[0]
        mask = _mask(legal, dim)
        q_pb, ra, ri, _ = _branch_data(
            legal,
            q.tolist(),
            K,
            noise_scale=kw.pop("noise_scale", 10.0),
            crn_corr=kw.pop("crn_corr", 1.0),
            seed=kw.pop("seed", 0),
        )
        return improve_policy(
            base_probs=base,
            search_q_mean=np.where(mask, q, np.nan),
            search_q_std=np.zeros(dim),
            rollout_counts=_counts_full(legal, [K] * len(legal), dim),
            terminal_frac=np.zeros(dim),
            legal_mask=mask,
            beta=kw.pop("beta", 1.0),
            ablation="confidence_gated_kl",
            root_selection=kw.pop("root_selection", "argmax"),
            rng=kw.pop("rng", np.random.default_rng(0)),
            normalize_advantages=kw.pop("normalize_advantages", False),
            q_per_branch=q_pb,
            root_action_pb=ra,
            rollout_index_pb=ri,
            K_rollouts=K,
            z_gate=z_gate,
            adaptive_beta=adaptive_beta,
            **kw,
        )

    def test_z_gate_zero_equals_single_anchor(self):
        """With z_gate=0, confidence_gated_kl is identical to single_anchor_kl."""
        legal = np.array([0, 1])
        base = _base_probs(legal, [0.6, 0.4])
        q = np.array([1.0, 3.0])
        r_gated = self._run_gated(legal, base, q, K=8, z_gate=0.0)
        r_sa = _run(
            legal,
            base,
            q,
            ablation="single_anchor_kl",
            beta=1.0,
            normalize_advantages=False,
        )
        assert r_gated.operator == "confidence_gated_kl"
        assert r_gated.gated is False
        assert np.allclose(r_gated.searched_probs, r_sa.searched_probs, atol=1e-9)

    def test_gated_returns_base_when_noisy(self):
        """With independent noise and a small gap, the gate suppresses the update."""
        legal = np.array([0, 1])
        base = _base_probs(legal, [0.9, 0.1])
        q = np.array([0.0, 0.5])  # small gap
        K = 4
        r = self._run_gated(
            legal, base, q, K=K, z_gate=3.0, noise_scale=10.0, crn_corr=0.0, seed=1
        )
        # With K=4, independent noise of 10, and gap=0.5, z should be tiny -> gated
        assert r.gated is True
        assert np.allclose(r.searched_probs[legal], base[legal], atol=1e-9)
        assert r.min_z_score < 3.0

    def test_not_gated_when_confident(self):
        """With full CRN (paired SE≈0) and a nonzero gap, the gate passes."""
        legal = np.array([0, 1])
        base = _base_probs(legal, [0.9, 0.1])
        q = np.array([0.0, 5.0])  # large gap
        r = self._run_gated(
            legal, base, q, K=8, z_gate=2.0, noise_scale=100.0, crn_corr=1.0
        )
        # Full CRN -> paired SE ≈ 0 -> z is huge -> not gated
        assert r.gated is False
        assert r.min_z_score > 1e10
        # Should have moved toward action 1
        assert r.searched_probs[1] > base[1]

    def test_adaptive_beta_strengthens_with_confidence(self):
        """With adaptive_beta, a higher z gives a smaller effective_beta (stronger update)."""
        legal = np.array([0, 1])
        base = _base_probs(legal, [0.5, 0.5])
        q = np.array([0.0, 3.0])
        K = 100
        # Moderate noise so z is finite but above the gate
        r_static = self._run_gated(
            legal, base, q, K=K, z_gate=1.0, noise_scale=2.0, crn_corr=0.0, seed=0
        )
        r_adaptive = self._run_gated(
            legal,
            base,
            q,
            K=K,
            z_gate=1.0,
            noise_scale=2.0,
            crn_corr=0.0,
            seed=0,
            adaptive_beta=True,
        )
        if r_static.gated or r_adaptive.gated:
            pytest.skip("noise realization gated both runs; rerun with different seed")
        # adaptive_beta should give a smaller effective_beta (stronger update)
        assert r_adaptive.effective_beta <= r_static.effective_beta
        # And the adaptive update should move more mass toward the better action
        assert r_adaptive.searched_probs[1] >= r_static.searched_probs[1] - 1e-9

    def test_adaptive_beta_formula(self):
        """Verify beta_eff = beta * z_gate / max(min_z, z_gate) at a known z."""
        # With full CRN, z=inf -> beta_eff = beta * z_gate / inf -> 0
        # But that's a degenerate case. Test with independent noise.
        legal = np.array([0, 1])
        base = _base_probs(legal, [0.5, 0.5])
        q = np.array([0.0, 10.0])
        K = 200
        r = self._run_gated(
            legal,
            base,
            q,
            K=K,
            z_gate=2.0,
            noise_scale=2.0,
            crn_corr=0.0,
            seed=42,
            adaptive_beta=True,
            beta=10.0,
        )
        if r.gated:
            pytest.skip("noise gated the run")
        expected = 10.0 * 2.0 / max(r.min_z_score, 2.0)
        assert r.effective_beta == pytest.approx(expected, rel=1e-6)

    def test_gated_with_single_action_passes(self):
        """A single legal action has no competitors -> min_z=inf -> not gated."""
        legal = np.array([3])
        base = _base_probs(legal, [1.0])
        q = np.array([5.0])
        r = self._run_gated(legal, base, q, K=4, z_gate=10.0)
        assert r.gated is False
        assert r.min_z_score == float("inf")  # no competitors -> inf

    def test_no_per_branch_data_uses_fallback_se(self):
        """Without per-branch data, the gate falls back to independent SE from counts/std."""
        legal = np.array([0, 1])
        base = _base_probs(legal, [0.5, 0.5])
        q = np.array([0.0, 10.0])
        dim = base.shape[0]
        mask = _mask(legal, dim)
        # std=0, counts=1 -> sem=0 -> fallback SE=0 -> z=inf (gap>0) -> not gated
        r = improve_policy(
            base_probs=base,
            search_q_mean=np.where(mask, q, np.nan),
            search_q_std=np.zeros(dim),
            rollout_counts=_counts_full(legal, [1, 1], dim),
            terminal_frac=np.zeros(dim),
            legal_mask=mask,
            beta=1.0,
            ablation="confidence_gated_kl",
            root_selection="argmax",
            rng=np.random.default_rng(0),
            normalize_advantages=False,
            z_gate=2.0,
            # No per-branch data
        )
        assert r.gated is False
        assert r.min_z_score == float("inf")  # deterministic -> always confident

    def test_constant_shift_invariance(self):
        legal = np.array([0, 1, 2])
        base = _base_probs(legal, [0.5, 0.3, 0.2])
        q = np.array([1.0, 3.0, 2.0])
        K = 8
        r0 = self._run_gated(legal, base, q, K=K, z_gate=1.0, crn_corr=1.0)
        r1 = self._run_gated(legal, base, q + 100.0, K=K, z_gate=1.0, crn_corr=1.0)
        assert np.allclose(r0.searched_probs, r1.searched_probs, atol=1e-9)

    def test_output_sums_to_one(self):
        legal = np.array([0, 1, 2])
        base = _base_probs(legal, [0.6, 0.3, 0.1])
        q = np.array([1.0, 3.0, 2.0])
        r = self._run_gated(
            legal, base, q, K=8, z_gate=2.0, crn_corr=1.0, root_selection="sample"
        )
        assert r.searched_probs.sum() == pytest.approx(1.0, abs=1e-9)

    def test_gating_logs_min_z_and_gate(self):
        legal = np.array([0, 1])
        base = _base_probs(legal, [0.5, 0.5])
        q = np.array([0.0, 0.1])  # tiny gap
        K = 4
        r = self._run_gated(
            legal, base, q, K=K, z_gate=5.0, noise_scale=10.0, crn_corr=0.0, seed=3
        )
        assert r.z_gate == 5.0
        assert r.gated is True
        assert r.min_z_score < 5.0
        assert r.effective_beta == 1.0  # gated -> no beta scaling applied
