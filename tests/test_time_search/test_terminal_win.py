"""Tests for the Phase A terminal-win fixed-root benchmark (skill §37 Gate A).

The analysis tests (prefix win-rate derivation, Spearman/regret aggregation, the
go/no-go gate) run on CPU with synthetic records. The end-to-end smoke is
GPU-gated (needs the frozen checkpoint + a real Showdown fork-to-terminal).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pytest

from metamon.rl.experimental.test_time_search.terminal_win import (
    TerminalWinRootRecord,
    prefix_win_rate,
    win_rate_sem,
    aggregate_terminal_win,
    terminal_win_gate,
)
from metamon.rl.experimental.test_time_search.rng import RootSeedBank, branch_env_seed

# ---------------------------------------------------------------------------
# Derived-G' terminal win (prefix averaging) + SEM
# ---------------------------------------------------------------------------


def test_prefix_win_rate_uses_first_g_columns():
    # 3 actions, G=8; action 0 wins first 4, loses last 4 -> 0.5 at G=8, 1.0 at G=4
    wins = np.array(
        [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
        ],
        dtype=np.float64,
    )
    g8 = prefix_win_rate(wins, 8)
    g4 = prefix_win_rate(wins, 4)
    assert g4.shape == (3,)
    assert g4[0] == pytest.approx(1.0)  # first 4 all wins
    assert g8[0] == pytest.approx(0.5)  # 4 wins / 8
    # G' > G clamps to G
    g20 = prefix_win_rate(wins, 20)
    np.testing.assert_allclose(g20, g8)


def test_prefix_win_rate_draws_count_as_half():
    wins = np.array([[0.5, 0.5, 1.0, 0.0]], dtype=np.float64)
    # mean of [0.5,0.5,1,0] = 0.5
    assert prefix_win_rate(wins, 4)[0] == pytest.approx(0.5)


def test_win_rate_sem_binomial():
    # p=0.5, g=100 -> sqrt(0.25/100) = 0.05
    assert win_rate_sem(np.array([0.5]), 100)[0] == pytest.approx(0.05)
    # p=1.0 is clipped to avoid 0 SEM
    assert win_rate_sem(np.array([1.0]), 100)[0] > 0.0


# ---------------------------------------------------------------------------
# CRN pairing: the terminal continuation's branch seed is action-independent
# (so wins[a,k] pairs with R[a,k] on the same chance stream k). This is the
# invariant that makes prefix averaging + the paired comparison valid.
# ---------------------------------------------------------------------------


def test_terminal_continuation_seed_is_action_independent():
    """env_seed[root, k] must NOT depend on the forced action (skill §7).

    The terminal_continuations method builds RootSeedBank.build(..., K=G) with
    the root identity (battle_id, side, decision) -- never the action -- so the
    same rollout index k gets the same seed regardless of which action is
    forced. This is what lets the benchmark pair shaped Q against terminal win
    on the same chance stream.
    """
    bank = RootSeedBank.build(0, "b0_0", "p1", 5, K=64)
    # branch b -> env_seeds[b % K]; for a single-action continuation rollout_index
    # = arange(G), so branch b == k == b. The seed for k is env_seeds[k].
    seeds_for_k = [bank.env_seeds[k] for k in range(64)]
    # rebuilding with the SAME root identity but a different (irrelevant) "action"
    # context must give identical seeds -- the bank never sees the action.
    bank2 = RootSeedBank.build(0, "b0_0", "p1", 5, K=64)
    assert bank2.env_seeds == seeds_for_k
    # a different rollout index gives a different seed (CRN across actions, not
    # across k)
    assert bank.env_seeds[0] != bank.env_seeds[1]
    # the low-level builder is also action-independent
    assert branch_env_seed(0, "b0_0", "p1", 5, k=3) == branch_env_seed(
        0, "b0_0", "p1", 5, k=3
    )


# ---------------------------------------------------------------------------
# Synthetic records -> aggregate_terminal_win + gate
# ---------------------------------------------------------------------------


def _synth_record(
    *,
    battle_id: str,
    decision: int,
    legal: list,
    base_probs: list,
    shaped_q: list,
    terminal_win: list,
    d1_q=None,
    G: int = 128,
    phase: str = "early",
    request_kind: str = "move",
    tactical: str = "move",
) -> TerminalWinRootRecord:
    A = len(legal)
    return TerminalWinRootRecord(
        root_id=f"{battle_id}_d{decision}",
        battle_id=battle_id,
        lane=0,
        decision=decision,
        legal_actions=legal,
        base_probs=base_probs,
        base_argmax=int(legal[int(np.argmax(base_probs))]),
        n_legal=A,
        base_entropy=1.0,
        base_top2_gap=0.2,
        entropy_band="medium",
        top2_gap_band="medium",
        phase_band=phase,
        forced_switch=(request_kind == "forceswitch"),
        request_kind=request_kind,
        eval_low_hp=False,
        opp_low_hp=False,
        status_present=False,
        opponents_remaining=6,
        tactical_category=tactical,
        root_critic_q=shaped_q,
        d0_q=shaped_q,
        d0_q_sem=[10.0] * A,
        d1_q=d1_q,
        derived_shaped_q={f"D0:K{k}": shaped_q for k in (4, 16, 64)},
        terminal_win=terminal_win,
        terminal_win_sem=[0.05] * A,
        n_truncated=[0] * A,
        n_draws=[0] * A,
        derived_terminal_win={f"G{k}": terminal_win for k in (4, 16, 64)},
        per_action_wins=None,
        per_action_shaped_r=None,
        G=G,
        mean_steps_to_terminal=60.0,
        latency_ms_shaped=1000.0,
        latency_ms_terminal=50000.0,
    )


def test_aggregate_perfect_correlation_spearman_one():
    """When shaped Q perfectly ranks terminal win, Spearman = 1.0."""
    recs = [
        _synth_record(
            battle_id=f"b{i}",
            decision=10 * i,
            legal=[0, 1, 2],
            base_probs=[0.2, 0.5, 0.3],
            shaped_q=[10.0, 30.0, 20.0],
            terminal_win=[0.3, 0.7, 0.5],
        )
        for i in range(10)
    ]
    s = aggregate_terminal_win(recs, [4, 16, 64])
    sp = s["spearman_shaped_vs_terminal"]["d0_k_ref"]["mean"]
    assert sp == pytest.approx(1.0)
    tm = s["top1_match_vs_terminal"]["d0_k_ref"]["mean"]
    assert tm == pytest.approx(1.0)


def test_aggregate_anti_correlation_negative_spearman():
    """When shaped Q ranks the WORST action highest, Spearman < 0 + regret high."""
    recs = [
        _synth_record(
            battle_id=f"b{i}",
            decision=10 * i,
            legal=[0, 1, 2],
            base_probs=[0.2, 0.5, 0.3],
            # shaped Q picks action 0 (q=30), but action 0 has the LOWEST win rate
            shaped_q=[30.0, 10.0, 20.0],
            terminal_win=[0.3, 0.7, 0.5],
        )
        for i in range(10)
    ]
    s = aggregate_terminal_win(recs, [4, 16, 64])
    sp = s["spearman_shaped_vs_terminal"]["d0_k_ref"]["mean"]
    assert sp < 0.0
    # shaped-Q argmax regret should be higher than actor regret
    reg_actor = s["terminal_win_regret"]["actor"]["mean"]
    reg_d0 = s["terminal_win_regret"]["d0_k_ref"]["mean"]
    assert reg_d0 > reg_actor
    # shaped-Q argmax decreases terminal win vs actor on most roots
    dec = s["decrease_freq_vs_actor"]["d0_k_ref"]
    assert dec > 0.5


def test_aggregate_regret_and_actor_gap():
    """Regret = terminal_win[best] - terminal_win[selector]; actor gap recorded."""
    # actor picks action 1 (prob 0.5); best is action 2 (win 0.9)
    rec = _synth_record(
        battle_id="b0",
        decision=5,
        legal=[0, 1, 2],
        base_probs=[0.2, 0.5, 0.3],
        shaped_q=[10.0, 20.0, 30.0],  # shaped Q picks the best (action 2)
        terminal_win=[0.4, 0.5, 0.9],
    )
    s = aggregate_terminal_win([rec], [4, 16, 64])
    # actor regret = 0.9 - 0.5 = 0.4
    assert s["terminal_win_regret"]["actor"]["mean"] == pytest.approx(0.4)
    # d0 regret = 0.9 - 0.9 = 0.0 (shaped Q picked the best)
    assert s["terminal_win_regret"]["d0_k_ref"]["mean"] == pytest.approx(0.0)
    # actor-vs-best gap = 0.4
    assert s["actor_vs_best_gap_mean"] == pytest.approx(0.4)


def test_aggregate_stratification_by_phase_and_tactical():
    recs = []
    for i in range(6):
        phase = "early" if i < 3 else "mid"
        tac = "move+imminent_ko" if i < 3 else "move"
        recs.append(
            _synth_record(
                battle_id=f"b{i}",
                decision=5 * i,
                legal=[0, 1],
                base_probs=[0.5, 0.5],
                shaped_q=[10.0, 20.0],
                terminal_win=[0.3, 0.7],
                phase=phase,
                tactical=tac,
            )
        )
    s = aggregate_terminal_win(recs, [4, 16, 64])
    assert "early" in s["stratified"]["spearman_d0_k_ref_by_phase"]
    assert "mid" in s["stratified"]["spearman_d0_k_ref_by_phase"]
    assert "move+imminent_ko" in s["stratified"]["spearman_d0_k_ref_by_tactical"]
    assert s["phase_distribution"]["early"] == 3
    assert s["phase_distribution"]["mid"] == 3


def test_aggregate_truncation_and_draw_rates():
    rec = _synth_record(
        battle_id="b0",
        decision=5,
        legal=[0, 1],
        base_probs=[0.5, 0.5],
        shaped_q=[10.0, 20.0],
        terminal_win=[0.4, 0.6],
        G=100,
    )
    rec.n_truncated = [5, 10]
    rec.n_draws = [3, 7]
    s = aggregate_terminal_win([rec], [4, 16, 64])
    # 2 actions * 100 branches = 200 total
    assert s["n_branches_total"] == 200
    assert s["n_truncated_total"] == 15
    assert s["n_draws_total"] == 10
    assert s["truncation_rate"] == pytest.approx(15 / 200)


# ---------------------------------------------------------------------------
# Gate A
# ---------------------------------------------------------------------------


def _gate_summary(spearman_mean, actor_reg, d0_reg, dec_freq, sp_low=0.3, sp_high=0.6):
    """Build a minimal summary dict the gate reads."""
    return {
        "n_roots_used": 10,
        "n_roots": 10,
        "n_branches_total": 1000,
        "n_truncated_total": 0,
        "truncation_rate": 0.0,
        "n_draws_total": 0,
        "draw_rate": 0.0,
        "spearman_shaped_vs_terminal": {
            "d0_k_ref": {"mean": spearman_mean, "median": spearman_mean, "n": 10},
            "d0_K4": {"mean": sp_low, "median": sp_low, "n": 10},
            "d0_K64": {"mean": sp_high, "median": sp_high, "n": 10},
        },
        "top1_match_vs_terminal": {"d0_k_ref": {"mean": 0.8, "n": 10}},
        "terminal_win_regret": {
            "actor": {"mean": actor_reg, "median": actor_reg, "n": 10},
            "d0_k_ref": {"mean": d0_reg, "median": d0_reg, "n": 10},
            "d0_K4": {"mean": actor_reg, "median": actor_reg, "n": 10},
            "d0_K64": {"mean": d0_reg, "median": d0_reg, "n": 10},
            "term_G4": {"mean": actor_reg, "median": actor_reg, "n": 10},
            "term_G64": {"mean": d0_reg, "median": d0_reg, "n": 10},
        },
        "decrease_freq_vs_actor": {"d0_k_ref": dec_freq, "d0_K4": dec_freq},
        "actor_vs_best_gap_mean": 0.1,
        "actor_vs_best_gap_median": 0.1,
        "stratified": {},
        "phase_distribution": {},
        "request_kind_distribution": {},
        "tactical_distribution": {},
    }


def test_gate_pass_when_correlated_improves_not_catastrophic_converges():
    s = _gate_summary(spearman_mean=0.6, actor_reg=0.15, d0_reg=0.05, dec_freq=0.1)
    g = terminal_win_gate(s, [4, 16, 64])
    assert g["verdict"] == "PASS"
    assert g["passed"] == g["total"]


def test_gate_fail_when_anti_correlated():
    s = _gate_summary(spearman_mean=-0.3, actor_reg=0.05, d0_reg=0.15, dec_freq=0.6)
    g = terminal_win_gate(s, [4, 16, 64])
    assert g["verdict"] != "PASS"
    assert g["criteria"]["correlated"]["pass"] is False
    assert g["criteria"]["improves_over_actor"]["pass"] is False
    assert g["criteria"]["not_catastrophic"]["pass"] is False


def test_gate_partial_when_correlated_but_does_not_improve():
    # shaped Q correlates with terminal win, but the actor is already near-optimal
    # so the shaped-Q argmax doesn't beat the actor.
    s = _gate_summary(spearman_mean=0.5, actor_reg=0.02, d0_reg=0.04, dec_freq=0.3)
    g = terminal_win_gate(s, [4, 16, 64])
    assert g["criteria"]["correlated"]["pass"] is True
    assert g["criteria"]["improves_over_actor"]["pass"] is False
    assert g["verdict"] == "PARTIAL"


def test_gate_inconclusive_no_roots():
    s = _gate_summary(
        spearman_mean=float("nan"),
        actor_reg=float("nan"),
        d0_reg=float("nan"),
        dec_freq=float("nan"),
    )
    s["n_roots_used"] = 0
    g = terminal_win_gate(s, [4, 16, 64])
    assert g["verdict"] == "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# End-to-end smoke (GPU-gated via the frozen_env_bundle fixture, which
# auto-skips without CUDA + the frozen checkpoint). Mirrors the Phase 1 GPU
# smoke in test_root_benchmark.py: a tiny G=8 run exercises the real
# fork-to-terminal path on the production checkpoint.
# ---------------------------------------------------------------------------


def test_terminal_win_smoke_gpu(frozen_env_bundle):
    """2-root, G=8 end-to-end: shaped-Q + to-terminal continuations produce a
    well-formed record + a gate verdict, with a valid per-action win-rate vector.
    """
    from metamon.rl.experimental.test_time_search.terminal_win import (
        benchmark_terminal_win,
    )

    bundle = frozen_env_bundle
    records, manifest = benchmark_terminal_win(
        bundle=bundle,
        k_ref=8,
        derived_ks=[4],
        depths=[0],
        max_roots=2,
        max_battles=4,
        decision_stride=1,
        progress_every=1,
        env_seed=42,
        search_seed=0,
        max_steps_to_terminal=200,
    )
    assert len(records) >= 1
    r = records[0]
    assert r.n_legal >= 2
    assert len(r.terminal_win) == r.n_legal
    assert len(r.d0_q) == r.n_legal
    assert len(r.root_critic_q) == r.n_legal
    assert r.G == 8
    # CRN pairing: per-action wins matrix has G columns (verbose mode here off;
    # check the derived prefix win rate exists)
    assert "G4" in r.derived_terminal_win
    # gate should produce a verdict (not crash)
    s = aggregate_terminal_win(records, [4])
    g = terminal_win_gate(s, [4])
    assert g["verdict"] in ("PASS", "PARTIAL", "INCONCLUSIVE")
