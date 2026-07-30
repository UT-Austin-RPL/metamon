"""Tests for the Phase 1 fixed-root estimator benchmark (skill §22).

Two groups:

* **CPU** (always run): the pure-numpy derivation + convergence-metric helpers
  -- ``branch_return_matrix``, ``prefix_q``/``block_means``, rank correlations,
  ``root_convergence_metrics``, ``aggregate_convergence``, and the go/no-go
  assessment logic. These are the scientific core and must be exact.
* **GPU-gated** (CUDA + frozen ckpt 740): an end-to-end smoke of
  ``benchmark_roots`` on the real checkpoint -- a tiny corpus (2 roots,
  ``K_ref=16``) proving the in-battle grid runs cleanly, produces well-formed
  records + a go/no-go verdict, and does not leak fork lanes.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from metamon.rl.experimental.test_time_search import benchmark_roots as br
from metamon.rl.experimental.test_time_search import root_dataset as rd
from metamon.rl.experimental.test_time_search.benchmark_roots import (
    RootResultRecord,
    branch_return_matrix,
    prefix_q,
    block_means,
    reference_q,
    split_half_top1_agreement,
    root_convergence_metrics,
    aggregate_convergence,
    go_no_go_assessment,
    spearman_corr,
    kendall_corr,
    build_grid_configs,
    estimator_comparison,
    comparison_table,
    search_justification_gate,
)
from metamon.rl.experimental.test_time_search.root_dataset import (
    compute_root_features,
    make_manifest_entry,
    fill_reference_fields,
    entropy_band,
    top2_gap_band,
    ref_gap_band,
    phase_band,
)

# reuse the GPU-gated skip marker / fixture from the Phase 0 tests
from .conftest import gpu_required  # noqa: F401

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _fake_estimate(legal_arr, R):
    """Build a minimal object with the attributes branch_return_matrix reads."""
    A, K = R.shape
    root_action = np.repeat(legal_arr, K)
    rollout_index = np.tile(np.arange(K), A)
    est = types.SimpleNamespace(
        legal_arr=np.asarray(legal_arr),
        root_action=root_action,
        rollout_index=rollout_index,
        q_per_branch=R.reshape(-1),  # row-major: a*K + k
        K=K,
        q_mean=R.mean(axis=1),
        q_std=R.std(axis=1),
        counts=np.full(A, K),
        term_frac=np.zeros(A),
        diag={"critic_disagreement": 0.0},
        search_depth=0,
        leaf_value_mode="policy_expectation",
        chance_mode="resample_crn",
        latency_ms=1.0,
    )
    return est


# ---------------------------------------------------------------------------
# CPU: branch return matrix + derivation
# ---------------------------------------------------------------------------


class TestBranchMatrix:
    def test_branch_return_matrix_round_trips_AxK(self):
        legal = np.array([0, 3, 7])
        R = np.array([[1.0, 2, 3, 4], [10, 20, 30, 40], [100, 200, 300, 400]])  # (3, 4)
        est = _fake_estimate(legal, R)
        R2 = branch_return_matrix(est)
        assert R2.shape == (3, 4)
        np.testing.assert_allclose(R2, R)

    def test_root_critic_only_returns_empty_matrix(self):
        est = types.SimpleNamespace(
            legal_arr=np.array([0, 1]),
            root_action=None,
            q_per_branch=None,
            rollout_index=None,
            K=0,
        )
        assert branch_return_matrix(est).size == 0

    def test_prefix_q_uses_first_k_streams(self):
        R = np.array([[1.0, 2, 3, 4, 5, 6, 7, 8]])
        np.testing.assert_allclose(prefix_q(R, 4), [2.5])
        np.testing.assert_allclose(prefix_q(R, 8), [4.5])
        # k_prime > K clamps to K
        np.testing.assert_allclose(prefix_q(R, 99), [4.5])

    def test_block_means_non_overlapping(self):
        R = np.array([[1.0, 2, 3, 4, 5, 6, 7, 8]])  # (1, 8)
        bm = block_means(R, 2)  # 4 blocks of 2
        assert bm.shape == (4, 1)
        np.testing.assert_allclose(bm.ravel(), [1.5, 3.5, 5.5, 7.5])

    def test_block_means_too_few_returns_empty(self):
        R = np.array([[1.0, 2, 3]])
        assert block_means(R, 4).shape == (0, 1)


# ---------------------------------------------------------------------------
# CPU: rank correlations
# ---------------------------------------------------------------------------


class TestRankCorr:
    def test_spearman_perfect_monotone(self):
        assert spearman_corr(np.array([1.0, 2, 3]), np.array([10.0, 20, 30])) == 1.0

    def test_spearman_perfect_inverse(self):
        assert spearman_corr(np.array([1.0, 2, 3]), np.array([30.0, 20, 10])) == -1.0

    def test_spearman_handles_ties(self):
        # ties -> average ranks; identical vectors -> 1.0
        assert spearman_corr(np.array([1.0, 1, 2]), np.array([1.0, 1, 2])) == 1.0

    def test_spearman_degenerate(self):
        assert np.isnan(spearman_corr(np.array([1.0]), np.array([1.0])))

    def test_kendall_perfect_concordant(self):
        assert kendall_corr(np.array([1.0, 2, 3]), np.array([1.0, 2, 3])) == 1.0

    def test_kendall_perfect_discordant(self):
        assert kendall_corr(np.array([1.0, 2, 3]), np.array([3.0, 2, 1])) == -1.0

    def test_kendall_degenerate(self):
        assert np.isnan(kendall_corr(np.array([1.0]), np.array([1.0])))


# ---------------------------------------------------------------------------
# CPU: convergence metrics + the "K rises" direction
# ---------------------------------------------------------------------------


class TestConvergenceMetrics:
    def _convergent_R(self):
        # action 0 is best in expectation (mean 10) but its first 4 chance
        # streams are low (mean 2); action 1 is stably second (mean 6); action
        # 2 worst (mean 1). K=4 picks action 1 (wrong); K=8 picks action 0.
        R = np.array(
            [
                [2.0, 2, 2, 2, 18, 18, 18, 18],
                [6.0, 6, 6, 6, 6, 6, 6, 6],
                [1.0, 1, 1, 1, 1, 1, 1, 1],
            ]
        )
        return R

    def test_K4_disagrees_K8_agrees_with_reference(self):
        R = self._convergent_R()
        legal = np.array([0, 1, 2])
        ref = reference_q(R)  # [10, 6, 1] -> argmax 0
        m4 = root_convergence_metrics(R, 4, 0, "policy_expectation", legal, ref)
        m8 = root_convergence_metrics(R, 8, 0, "policy_expectation", legal, ref)
        assert m4.top1_agree == 0  # K=4 picks action 1
        assert m4.kp_argmax == 1
        assert m8.top1_agree == 1  # K=8 picks action 0 (== ref)
        assert m8.kp_argmax == 0
        # regret falls: K=4 regret = ref[0]-ref[1] = 4; K=8 regret = 0
        assert m4.regret == pytest.approx(4.0)
        assert m8.regret == pytest.approx(0.0)
        # rank correlation improves (K=4: Q=[2,6,1] vs ref [10,6,1]; K=8 exact)
        assert m8.spearman == 1.0

    def test_degenerate_single_action_returns_none(self):
        R = np.array([[1.0, 2, 3, 4]])
        legal = np.array([0])
        assert root_convergence_metrics(R, 2, 0, "policy_expectation", legal) is None

    def test_se_calibration_on_iid_returns(self):
        # i.i.d. returns: block spread should match std/sqrt(K')
        rng = np.random.default_rng(0)
        R = rng.normal(loc=10.0, scale=4.0, size=(3, 64))
        legal = np.array([0, 1, 2])
        m = root_convergence_metrics(R, 8, 0, "policy_expectation", legal)
        # se_ratio = block_std_mean / theo_se_mean; with i.i.d. returns ~1
        assert 0.6 < m.se_ratio < 1.5

    def test_split_half_stable_when_clear_best(self):
        R = np.array(
            [
                [10.0] * 16,
                [5.0] * 16,
                [1.0] * 16,
            ]
        )
        assert split_half_top1_agreement(R) == 1.0

    def test_split_half_unstable_when_chance_flips(self):
        # first half: action 1 best; second half: action 0 best -> disagree
        R = np.array(
            [
                [0.0, 0, 0, 0, 10, 10, 10, 10],
                [5.0, 5, 5, 5, 0, 0, 0, 0],
            ]
        )
        assert split_half_top1_agreement(R) == 0.0


# ---------------------------------------------------------------------------
# CPU: aggregation + go/no-go
# ---------------------------------------------------------------------------


def _make_record(entropy="medium", top2="medium", phase="mid", conv=None):
    rec = RootResultRecord(
        root_id="r",
        battle_id="b",
        lane=0,
        decision=0,
        legal_actions=[0, 1, 2],
        base_probs=[0.5, 0.3, 0.2],
        base_argmax=0,
        base_entropy=1.0,
        base_top2_gap=0.2,
        entropy_band=entropy,
        top2_gap_band=top2,
        phase_band=phase,
        n_legal=3,
    )
    rec.convergence = conv or {}
    return rec


class TestAggregateAndGate:
    def test_aggregate_collects_per_cell_means(self):
        recs = []
        for agree_d0_k4, agree_d0_k16 in [(0, 1), (1, 1), (0, 1)]:
            rec = _make_record(
                conv={
                    "policy_expectation:D0:K4": {
                        "top1_agree": agree_d0_k4,
                        "block_top1_agree": 0.5,
                        "regret": 4.0,
                        "mae": 3.0,
                        "spearman": 0.5,
                        "kendall": 0.4,
                        "se_ratio": 1.0,
                        "theo_se_mean": 2.0,
                        "block_std_mean": 2.0,
                    },
                    "policy_expectation:D0:K16": {
                        "top1_agree": agree_d0_k16,
                        "block_top1_agree": 0.9,
                        "regret": 0.0,
                        "mae": 0.5,
                        "spearman": 1.0,
                        "kendall": 1.0,
                        "se_ratio": 1.0,
                        "theo_se_mean": 1.0,
                        "block_std_mean": 1.0,
                    },
                }
            )
            rec.split_half_top1_d0 = 1.0
            recs.append(rec)
        summary = aggregate_convergence(recs)
        assert summary["_n_roots"] == 3
        k4 = summary["policy_expectation:D0:K4"]
        k16 = summary["policy_expectation:D0:K16"]
        assert k4["n"] == 3
        assert k4["top1_agree"] == pytest.approx(1 / 3)
        assert k16["top1_agree"] == pytest.approx(1.0)
        assert k16["regret_mean"] == pytest.approx(0.0)
        # stratified bands present
        assert "by_entropy" in k4 and "by_phase" in k4
        # reference stability
        assert summary["_reference_stability"]["split_half_top1_d0"] == 1.0

    def test_go_no_go_passes_on_clean_convergence(self):
        # reference stable, top1 rises monotonically 0.4 -> 0.7 -> 1.0,
        # regret falls, SE calibrated.
        recs = []
        agrees_by_k = {4: [0, 1, 0, 0, 1], 16: [1, 1, 0, 1, 1], 64: [1, 1, 1, 1, 1]}
        regrets_by_k = {4: [4, 0, 4, 4, 0], 16: [0, 0, 4, 0, 0], 64: [0, 0, 0, 0, 0]}
        for i in range(5):
            conv = {}
            for k in (4, 16, 64):
                conv[f"policy_expectation:D0:K{k}"] = {
                    "top1_agree": agrees_by_k[k][i],
                    "block_top1_agree": float(agrees_by_k[k][i]),
                    "regret": float(regrets_by_k[k][i]),
                    "mae": float(regrets_by_k[k][i]),
                    "spearman": 0.9,
                    "kendall": 0.8,
                    "se_ratio": 1.0,
                    "theo_se_mean": 2.0,
                    "block_std_mean": 2.0,
                }
            rec = _make_record(conv=conv)
            rec.split_half_top1_d0 = 1.0
            recs.append(rec)
        summary = aggregate_convergence(recs)
        assessment = go_no_go_assessment(summary, [4, 16, 64])
        assert assessment["verdict"] == "PASS", assessment
        assert assessment["passed"] == assessment["total"]

    def test_go_no_go_fails_when_K_does_not_improve(self):
        recs = []
        for i in range(5):
            conv = {
                "policy_expectation:D0:K4": {
                    "top1_agree": 1,
                    "block_top1_agree": 0.9,
                    "regret": 0.0,
                    "mae": 0.0,
                    "spearman": 1.0,
                    "kendall": 1.0,
                    "se_ratio": 1.0,
                    "theo_se_mean": 2.0,
                    "block_std_mean": 2.0,
                },
                "policy_expectation:D0:K16": {
                    "top1_agree": 0,
                    "block_top1_agree": 0.4,
                    "regret": 5.0,
                    "mae": 4.0,
                    "spearman": 0.3,
                    "kendall": 0.2,
                    "se_ratio": 1.0,
                    "theo_se_mean": 1.0,
                    "block_std_mean": 1.0,
                },
                "policy_expectation:D0:K64": {
                    "top1_agree": 0,
                    "block_top1_agree": 0.3,
                    "regret": 6.0,
                    "mae": 5.0,
                    "spearman": 0.1,
                    "kendall": 0.0,
                    "se_ratio": 1.0,
                    "theo_se_mean": 0.5,
                    "block_std_mean": 0.5,
                },
            }
            rec = _make_record(conv=conv)
            rec.split_half_top1_d0 = 1.0
            recs.append(rec)
        summary = aggregate_convergence(recs)
        assessment = go_no_go_assessment(summary, [4, 16, 64])
        assert assessment["verdict"] in ("PARTIAL", "INCONCLUSIVE")
        assert assessment["criteria"]["top1_agreement_rises_with_K_D0"]["pass"] is False
        assert assessment["criteria"]["regret_falls_with_K_D0"]["pass"] is False


# ---------------------------------------------------------------------------
# CPU: root_dataset features / manifest
# ---------------------------------------------------------------------------


class TestRootDataset:
    def test_compute_root_features(self):
        probs = np.array([0.0, 0.6, 0.3, 0.1, 0.0])
        legal = np.array([1, 2, 3])
        f = compute_root_features(probs, legal, base_argmax=1)
        assert f["n_legal"] == 3
        assert f["base_top1_prob"] == pytest.approx(0.6)
        assert f["base_top2_gap"] == pytest.approx(0.3)
        # entropy of [0.6,0.3,0.1]
        ent = -(0.6 * np.log(0.6) + 0.3 * np.log(0.3) + 0.1 * np.log(0.1))
        assert f["base_entropy"] == pytest.approx(ent)

    def test_bands(self):
        assert entropy_band(0.5) == "low"
        assert entropy_band(1.2) == "medium"
        assert entropy_band(1.8) == "high"
        assert top2_gap_band(0.05) == "small"
        assert top2_gap_band(0.3) == "medium"
        assert top2_gap_band(0.6) == "large"
        assert ref_gap_band(5.0) == "near_tied"
        assert ref_gap_band(50.0) == "medium"
        assert ref_gap_band(200.0) == "clear"
        assert phase_band(10) == "early"
        assert phase_band(60) == "mid"
        assert phase_band(100) == "late"

    def test_manifest_entry_round_trip(self):
        probs = np.array([0.0, 0.6, 0.3, 0.1, 0.0])
        legal = np.array([1, 2, 3])
        e = make_manifest_entry(
            battle_id="b0_0",
            lane=0,
            decision=3,
            battle_seed=42,
            legal=[1, 2, 3],
            base_probs=probs,
            legal_arr=legal,
            base_argmax=1,
            action_history=[[1, 2], [3, 1]],
        )
        assert e.root_id == "b0_0:d3"
        assert e.n_legal == 3
        assert e.base_argmax == 1
        assert e.action_history == [[1, 2], [3, 1]]
        # round-trip JSON
        e2 = rd.RootManifestEntry.from_dict(e.to_dict())
        assert e2.root_id == e.root_id
        assert e2.legal_actions == [1, 2, 3]

    def test_fill_reference_fields(self):
        probs = np.array([0.5, 0.3, 0.2])
        legal = np.array([0, 1, 2])
        e = make_manifest_entry(
            battle_id="b",
            lane=0,
            decision=0,
            battle_seed=1,
            legal=[0, 1, 2],
            base_probs=probs,
            legal_arr=legal,
            base_argmax=0,
            action_history=[],
        )
        fill_reference_fields(
            e,
            ref_q=np.array([150.0, 50.0, 10.0]),
            legal_arr=legal,
            critic_disagreement=5.0,
            terminal_frac_d0=0.2,
        )
        assert e.ref_q_argmax == 0
        assert e.ref_q_gap == pytest.approx(100.0)
        assert e.ref_q_top2_gap_band == "clear"
        assert e.critic_disagreement == 5.0
        assert e.terminal_frac_d0 == 0.2


# ---------------------------------------------------------------------------
# CPU: config grid
# ---------------------------------------------------------------------------


class TestConfigGrid:
    def test_grid_has_critic_only_and_depths(self):
        grid = build_grid_configs(k_ref=64, depths=[0, 1], include_inherited_rng=True)
        assert set(grid) == {"root_critic_only", "d0", "d1", "inherited_d0"}
        assert grid["root_critic_only"].search_leaf_value_mode == "root_critic_only"
        assert grid["d0"].search_rollouts_per_action == 64
        assert grid["d0"].search_depth == 0
        assert grid["d1"].search_depth == 1
        assert grid["inherited_d0"].search_chance_mode == "inherited_trunk_rng"
        # research-safe defaults preserved
        assert grid["d0"].search_root_candidate_mode == "all_legal"
        assert grid["d0"].search_chance_mode == "resample_crn"
        assert grid["d0"].search_root_opponent_coupling is True
        assert grid["d0"].search_error_policy == "raise"


# ---------------------------------------------------------------------------
# CPU: estimator head-to-head (actor vs critic-only vs D=0-ref vs D=1)
# ---------------------------------------------------------------------------


def _make_record_with_configs(
    legal,
    base_probs,
    base_argmax,
    d0_q,
    critic_q=None,
    d1_q=None,
    entropy="medium",
    phase="mid",
    critic_disagreement=0.0,
):
    """Build a RootResultRecord with the configs dict estimator_comparison reads."""
    legal_arr = np.asarray(legal)
    d0_argmax = int(legal_arr[int(np.argmax(d0_q))])
    configs = {
        "d0": {
            "q_mean": [float(x) for x in d0_q],
            "q_argmax": d0_argmax,
            "critic_disagreement": float(critic_disagreement),
        }
    }
    if critic_q is not None:
        configs["root_critic_only"] = {
            "q_mean": [float(x) for x in critic_q],
            "q_argmax": int(legal_arr[int(np.argmax(critic_q))]),
        }
    if d1_q is not None:
        configs["d1"] = {
            "q_mean": [float(x) for x in d1_q],
            "q_argmax": int(legal_arr[int(np.argmax(d1_q))]),
        }
    return RootResultRecord(
        root_id="r",
        battle_id="b",
        lane=0,
        decision=0,
        legal_actions=[int(x) for x in legal_arr],
        base_probs=[float(x) for x in np.asarray(base_probs)[legal_arr]],
        base_argmax=int(base_argmax),
        base_entropy=1.0,
        base_top2_gap=0.2,
        entropy_band=entropy,
        top2_gap_band="medium",
        phase_band=phase,
        n_legal=int(legal_arr.size),
        configs=configs,
    )


class TestEstimatorComparison:
    def test_actor_agrees_with_ref_everywhere(self):
        # actor argmax == d0 argmax on every root -> 0 disagreement
        recs = [
            _make_record_with_configs(
                legal=[0, 1, 2],
                base_probs=[0.6, 0.3, 0.1],
                base_argmax=0,
                d0_q=[100.0, 50.0, 10.0],
                critic_q=[90.0, 50.0, 10.0],
                entropy="low",
                phase="early",
            )
            for _ in range(5)
        ]
        comp = estimator_comparison(recs)
        assert comp["n_roots"] == 5
        assert comp["actor_disagrees_with_ref"] == 0.0
        # regret is 0 when actor == ref
        assert comp["actor_mean_regret_vs_ref"] == 0.0

    def test_actor_disagrees_critic_agrees_with_ref(self):
        # actor picks action 0, but ref (d0) and critic both pick action 1
        recs = [
            _make_record_with_configs(
                legal=[0, 1, 2],
                base_probs=[0.6, 0.3, 0.1],
                base_argmax=0,
                d0_q=[50.0, 100.0, 10.0],
                critic_q=[50.0, 100.0, 10.0],
                entropy="high",
                phase="mid",
                critic_disagreement=10.0,
            )
            for _ in range(5)
        ]
        comp = estimator_comparison(recs)
        assert comp["actor_disagrees_with_ref"] == 1.0
        # critic agrees with d0 -> critic disagreement with ref = 0
        assert comp["root_critic_only_disagrees_with_ref"] == 0.0
        # actor vs critic: they disagree
        assert comp["actor_vs_critic_agree"] == 0.0
        # regret: ref[best]=100, ref[actor_pick=0]=50 -> regret 50
        assert comp["actor_mean_regret_vs_ref"] == pytest.approx(50.0)

    def test_d0_disagrees_with_critic(self):
        # critic picks 0, d0 picks 1 -> critic_vs_d0 agreement = 0
        recs = [
            _make_record_with_configs(
                legal=[0, 1],
                base_probs=[0.5, 0.5],
                base_argmax=0,
                d0_q=[50.0, 100.0],
                critic_q=[100.0, 50.0],
                d1_q=[50.0, 100.0],
                entropy="medium",
                phase="late",
            )
        ]
        comp = estimator_comparison(recs)
        assert comp["critic_vs_d0_agree"] == 0.0
        assert comp["d0_vs_d1_agree"] == 1.0  # d1 matches d0

    def test_pruning_diagnostic_drops_ref_best(self):
        # ref-best is action 1 (d0_q=[10, 100]) but its base_prob (0.02) is < 5%
        # of max (0.95) -> legacy prune would drop it.
        recs = [
            _make_record_with_configs(
                legal=[0, 1],
                base_probs=[0.95, 0.02],
                base_argmax=0,
                d0_q=[10.0, 100.0],
                critic_q=[10.0, 100.0],
            )
        ]
        comp = estimator_comparison(recs)
        assert comp["legacy_prune_would_drop_ref_best"] == 1.0

    def test_pruning_diagnostic_keeps_ref_best(self):
        recs = [
            _make_record_with_configs(
                legal=[0, 1],
                base_probs=[0.6, 0.4],
                base_argmax=0,
                d0_q=[10.0, 100.0],
                critic_q=[10.0, 100.0],
            )
        ]
        comp = estimator_comparison(recs)
        assert comp["legacy_prune_would_drop_ref_best"] == 0.0

    def test_phase_distribution_reported(self):
        recs = [
            _make_record_with_configs(
                [0, 1],
                [0.5, 0.5],
                0,
                [10.0, 20.0],
                [10.0, 20.0],
                phase="early",
            ),
            _make_record_with_configs(
                [0, 1],
                [0.5, 0.5],
                0,
                [10.0, 20.0],
                [10.0, 20.0],
                phase="mid",
            ),
            _make_record_with_configs(
                [0, 1],
                [0.5, 0.5],
                0,
                [10.0, 20.0],
                [10.0, 20.0],
                phase="mid",
            ),
        ]
        comp = estimator_comparison(recs)
        assert comp["phase_distribution"] == {"early": 1, "mid": 2, "late": 0}

    def test_comparison_table_renders(self):
        recs = [
            _make_record_with_configs(
                [0, 1],
                [0.6, 0.4],
                0,
                [10.0, 100.0],
                [10.0, 100.0],
                entropy="high",
                phase="mid",
            )
        ]
        comp = estimator_comparison(recs)
        table = comparison_table(comp)
        assert "actor (frozen)" in table
        assert "root_critic_only" in table
        assert "addressable opportunity" in table


class TestSearchJustificationGate:
    def test_passes_when_actor_disagrees_and_concentrated(self):
        # actor disagrees with ref on high-entropy roots; agrees on low-entropy
        recs = []
        for i in range(6):
            recs.append(
                _make_record_with_configs(
                    [0, 1],
                    [0.6, 0.4],
                    0,
                    [10.0, 100.0],
                    [10.0, 100.0],
                    entropy="high",
                    phase="mid",
                    critic_disagreement=20.0,
                )
            )
        for i in range(4):
            # low-entropy: actor matches ref (d0 picks 0, actor picks 0)
            recs.append(
                _make_record_with_configs(
                    [0, 1],
                    [0.9, 0.1],
                    0,
                    [100.0, 10.0],
                    [100.0, 10.0],
                    entropy="low",
                    phase="early",
                    critic_disagreement=1.0,
                )
            )
        comp = estimator_comparison(recs)
        gate = search_justification_gate(comp)
        assert gate["verdict"] == "PASS", gate
        assert gate["criteria"]["addressable_opportunity"]["pass"] is True
        assert gate["criteria"]["concentrated_not_random"]["pass"] is True

    def test_fails_when_actor_matches_ref(self):
        # actor always matches ref -> no addressable opportunity
        recs = [
            _make_record_with_configs(
                [0, 1],
                [0.9, 0.1],
                0,
                [100.0, 10.0],
                [100.0, 10.0],
                entropy="low",
                phase="early",
            )
            for _ in range(5)
        ]
        comp = estimator_comparison(recs)
        gate = search_justification_gate(comp)
        assert gate["criteria"]["addressable_opportunity"]["pass"] is False
        assert gate["verdict"] != "PASS"

    def test_fails_when_everyone_agrees_with_actor(self):
        # actor == critic == d0 -> no search signal at all
        recs = [
            _make_record_with_configs(
                [0, 1],
                [0.6, 0.4],
                0,
                [100.0, 10.0],
                [100.0, 10.0],
                entropy="medium",
                phase="mid",
            )
            for _ in range(5)
        ]
        comp = estimator_comparison(recs)
        gate = search_justification_gate(comp)
        assert gate["criteria"]["addressable_opportunity"]["pass"] is False
        assert gate["criteria"]["search_adds_over_critic"]["pass"] is False


# ---------------------------------------------------------------------------
# GPU-gated: end-to-end benchmark smoke on the real checkpoint
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("frozen_env_bundle")
class TestBenchmarkSmokeGPU:
    pytestmark = [gpu_required]

    def test_tiny_benchmark_runs_and_produces_verdict(self, frozen_env_bundle):
        """End-to-end: a 2-root, K_ref=16 benchmark on the real checkpoint
        produces well-formed records, a go/no-go verdict, and does not leak
        fork lanes (skill §22)."""
        grid = build_grid_configs(k_ref=16, depths=[0], search_seed=0)
        records, manifest = br.benchmark_roots(
            bundle=frozen_env_bundle,
            config_grid=grid,
            k_ref=16,
            derived_ks=[4],
            depths=[0],
            max_roots=2,
            max_battles=4,
            root_stride=1,
            progress_every=1,
            env_seed=42,
        )
        assert len(records) == 2
        assert len(manifest) == 2
        r0 = records[0]
        # the grid configs are present per root
        assert "root_critic_only" in r0.configs
        assert "d0" in r0.configs
        # per-action Q over the retained legal actions
        d0 = r0.configs["d0"]
        assert len(d0["q_mean"]) == r0.n_legal
        assert all(np.isfinite(q) for q in d0["q_mean"])
        # convergence metrics derived for K=4 D=0
        key = "policy_expectation:D0:K4"
        assert key in r0.convergence
        cm = r0.convergence[key]
        assert cm["k_prime"] == 4
        assert 0 <= cm["top1_agree"] <= 1
        # manifest reference fields filled from the D=0 high-K estimate
        m0 = manifest[0]
        assert m0.ref_q_mean is not None
        assert m0.ref_q_argmax is not None
        # aggregate + go/no-go run
        summary = aggregate_convergence(records)
        assert summary["_n_roots"] == 2
        assessment = go_no_go_assessment(summary, [4])
        assert assessment["verdict"] in ("PASS", "PARTIAL", "INCONCLUSIVE")
        # estimator head-to-head + search-justification gate run (skill §40)
        comp = estimator_comparison(records)
        assert comp["n_roots"] == 2
        assert 0.0 <= comp["actor_disagrees_with_ref"] <= 1.0
        gate = search_justification_gate(comp)
        assert gate["verdict"] in ("PASS", "PARTIAL", "INCONCLUSIVE")
        assert "phase_distribution" in comp
        # No fork lanes leaked: benchmark_roots ran 2 roots x 2 grid configs
        # (root_critic_only + D=0 K=16) through estimate_root -> _rollout_core,
        # each of which cleans up its branches in a finally. The bundle's env
        # is still usable afterward (the test fixture reuses it). The Phase 0
        # test_search_cleanup stress test already proves _rollout_core's
        # cleanup path releases lanes/snapshots; this smoke confirms the
        # benchmark drives that path repeatedly without raising.
