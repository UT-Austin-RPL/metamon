"""Tests for the policy-improvement operators (skill §12)."""

from __future__ import annotations

import numpy as np
import pytest

from metamon.rl.experimental.test_time_search.improvement import (
    entropy,
    improve_policy,
    kl,
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
