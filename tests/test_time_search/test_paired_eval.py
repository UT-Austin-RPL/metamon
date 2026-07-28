"""Tests for Phase 2 paired-eval statistics (skill §23 / §32).

These test the statistical primitives (McNemar, paired bootstrap CI, Wilson,
analyze_pairs) and the verdict logic without running any battles.
"""

from __future__ import annotations

import numpy as np
import pytest

from metamon.rl.experimental.test_time_search.paired_eval import (
    mcnemar_test,
    paired_bootstrap_ci,
    wilson_interval,
    analyze_pairs,
    _binom_two_sided_p,
    _chi2_sf_df1,
    _verdict_text,
)


class TestMcNemar:
    def test_no_discordant_pairs_is_null(self):
        r = mcnemar_test(0, 0)
        assert r["p_value"] == 1.0
        assert r["method"] == "none"

    def test_exact_binomial_known_value(self):
        # b=10, c=2, n_disc=12, p=0.5: two-sided p = 2 * P(X>=10 | Binom(12,0.5))
        # P(X>=10) = (C(12,10)+C(12,11)+C(12,12)) / 2^12 = (66+12+1)/4096 = 79/4096
        # two-sided = 2 * 79/4096 = 0.03857421875
        r = mcnemar_test(10, 2)
        assert r["method"] == "exact_binomial"
        assert r["p_value"] == pytest.approx(0.03857421875, abs=1e-6)

    def test_chi2_continuity_for_large_discordant(self):
        # b=40, c=20, n_disc=60 -> chi2 with continuity correction
        # stat = (|40-20| - 1)^2 / 60 = 361/60 = 6.0167
        # p = erfc(sqrt(6.0167/2)) = erfc(sqrt(3.0083)) = erfc(1.7342)
        r = mcnemar_test(40, 20)
        assert r["method"] == "chi2_continuity"
        assert r["statistic"] == pytest.approx(361.0 / 60.0, abs=1e-4)
        # erfc(1.7342) ~= 0.0149
        assert r["p_value"] == pytest.approx(0.0149, abs=0.01)

    def test_symmetric_b_c_gives_high_p(self):
        # b=c -> no evidence of asymmetry
        r = mcnemar_test(5, 5)
        assert r["p_value"] > 0.5


class TestBinomTwoSided:
    def test_extreme_k_is_significant(self):
        # k=12, n=12, p=0.5 -> P(X>=12) = 1/4096, two-sided = 2/4096
        p = _binom_two_sided_p(12, 12, 0.5)
        assert p == pytest.approx(2.0 / 4096.0, abs=1e-6)

    def test_middle_k_is_not_significant(self):
        # k=6, n=12, p=0.5 -> P(X>=6) = 0.613, P(X<=6) = 0.613 -> two-sided = 1.0
        p = _binom_two_sided_p(6, 12, 0.5)
        assert p == pytest.approx(1.0, abs=1e-6)


class TestChi2Sf:
    def test_zero_stat_returns_one(self):
        assert _chi2_sf_df1(0.0) == 1.0

    def test_known_value(self):
        # chi2.sf(3.84, df=1) ~= 0.05 (the 5% critical value)
        assert _chi2_sf_df1(3.84) == pytest.approx(0.05, abs=0.005)


class TestWilsonInterval:
    def test_50_percent(self):
        lo, hi = wilson_interval(50, 100)
        # 50/100 Wilson 95% ~= [0.404, 0.596]
        assert lo == pytest.approx(0.404, abs=0.01)
        assert hi == pytest.approx(0.596, abs=0.01)

    def test_zero_n(self):
        lo, hi = wilson_interval(0, 0)
        assert (lo, hi) == (0.0, 0.0)

    def test_all_wins(self):
        lo, hi = wilson_interval(10, 10)
        assert lo > 0.5  # even with all wins, lower bound < 1
        assert hi <= 1.0


class TestPairedBootstrap:
    def test_no_delta_when_identical(self):
        pairs = [(1, 1) for _ in range(100)] + [(0, 0) for _ in range(100)]
        r = paired_bootstrap_ci(pairs, n_boot=1000, seed=0)
        assert r["delta"] == pytest.approx(0.0, abs=1e-9)
        assert r["ci_low"] == pytest.approx(0.0, abs=1e-9)
        assert r["ci_high"] == pytest.approx(0.0, abs=1e-9)

    def test_clear_positive_delta(self):
        # 80% search win, 20% baseline win -> large positive delta
        rng = np.random.default_rng(0)
        pairs = []
        for _ in range(200):
            sw = 1 if rng.random() < 0.8 else 0
            bw = 1 if rng.random() < 0.2 else 0
            pairs.append((sw, bw))
        r = paired_bootstrap_ci(pairs, n_boot=2000, seed=0)
        assert r["delta"] > 0.5
        assert r["ci_low"] > 0.3  # clearly excludes zero

    def test_empty_pairs(self):
        r = paired_bootstrap_ci([], n_boot=100)
        assert r["delta"] == 0.0


class TestAnalyzePairs:
    def test_clear_search_advantage(self):
        # 60 search-better, 20 search-worse, 20 ties
        pairs = [(1, 0)] * 60 + [(0, 1)] * 20 + [(0, 0)] * 20
        sides = [0] * 50 + [1] * 50
        a = analyze_pairs(pairs, sides)
        assert a["n_pairs"] == 100
        assert a["search_win_rate"] == 0.6
        assert a["baseline_win_rate"] == 0.2
        assert a["paired_delta"] == pytest.approx(0.4, abs=1e-6)
        assert a["discordant_b"] == 60
        assert a["discordant_c"] == 20
        assert a["bootstrap_ci"]["ci_low"] > 0.2  # excludes zero

    def test_no_effect(self):
        pairs = [(1, 0)] * 25 + [(0, 1)] * 25 + [(0, 0)] * 50
        sides = [0] * 50 + [1] * 50
        a = analyze_pairs(pairs, sides)
        assert a["paired_delta"] == pytest.approx(0.0, abs=1e-6)
        assert a["discordant_b"] == 25
        assert a["discordant_c"] == 25
        # CI should include zero
        assert a["bootstrap_ci"]["ci_low"] <= 0.0
        assert a["bootstrap_ci"]["ci_high"] >= 0.0

    def test_per_side_breakdown(self):
        pairs = [(1, 0)] * 40 + [(0, 1)] * 10 + [(1, 0)] * 10 + [(0, 1)] * 40
        sides = [0] * 50 + [1] * 50
        a = analyze_pairs(pairs, sides)
        assert 0 in a["by_side"] and 1 in a["by_side"]
        # side 0: search better (40-10); side 1: search worse (10-40)
        assert a["by_side"][0]["delta"] > 0
        assert a["by_side"][1]["delta"] < 0


class TestVerdictText:
    def test_inconclusive_small_n(self):
        a = {
            "n_pairs": 20,
            "bootstrap_ci": {"ci_low": 0.1, "ci_high": 0.3},
            "paired_delta": 0.2,
            "discordant_b": 8,
            "discordant_c": 2,
            "both_lose": 0,
        }
        txt = _verdict_text(a)
        assert "INCONCLUSIVE" in txt and "n=20" in txt

    def test_positive(self):
        a = {
            "n_pairs": 200,
            "bootstrap_ci": {"ci_low": 0.05, "ci_high": 0.15},
            "paired_delta": 0.10,
            "discordant_b": 30,
            "discordant_c": 10,
            "both_lose": 0,
        }
        txt = _verdict_text(a)
        assert txt.startswith("POSITIVE")

    def test_inconclusive_ci_includes_zero(self):
        a = {
            "n_pairs": 200,
            "bootstrap_ci": {"ci_low": -0.03, "ci_high": 0.07},
            "paired_delta": 0.02,
            "discordant_b": 20,
            "discordant_c": 16,
            "both_lose": 0,
        }
        txt = _verdict_text(a)
        assert txt.startswith("INCONCLUSIVE")

    def test_negative(self):
        a = {
            "n_pairs": 200,
            "bootstrap_ci": {"ci_low": -0.15, "ci_high": -0.05},
            "paired_delta": -0.10,
            "discordant_b": 10,
            "discordant_c": 30,
            "both_lose": 0,
        }
        txt = _verdict_text(a)
        assert txt.startswith("NEGATIVE")
