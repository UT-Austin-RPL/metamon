"""Unit tests for PSRO-Lite (``metamon.rl.psro_lite``).

Covers the filename regex, the win-rate→weight transform, EMA, floor,
min-games gate, uniform fallback on empty/cold buffers, sidecar round-trip,
and the ``OpponentPoolConfig.set_weights`` sampling distribution.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from collections import Counter

import pytest

from metamon.rl.psro_lite import (
    PsroConfig,
    PsroLite,
    compute_prioritized_weights,
    match_agent_name,
    parse_trajectory_filename,
    read_sidecar,
    weight_entropy,
)
from metamon.rl.evaluate.opponent_pool import OpponentPoolConfig


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------


def _make_filename(fmt, opp_label, result, player="MMVecSD-12345678", bid="1234567890"):
    return (
        f"metamon-{fmt}-{bid}_Unrated_{player}_vs_{opp_label}_"
        f"01-02-2025-03:04:05_{result}.json.lz4"
    )


def test_parse_trajectory_filename_basic():
    fn = _make_filename("gen1ou", "TaurosV0-ckpt40-gl_05_26", "WIN")
    assert parse_trajectory_filename(fn) == ("TaurosV0-ckpt40-gl_05_26", "WIN")


def test_parse_trajectory_filename_loss():
    fn = _make_filename("gen1ou", "TaurosV0", "LOSS")
    assert parse_trajectory_filename(fn) == ("TaurosV0", "LOSS")


def test_parse_trajectory_filename_underscore_teamset():
    """Team sets like ``gl_05_26`` contain underscores — the regex must not split
    on them (it anchors on the fixed-shape timestamp)."""
    fn = _make_filename("gen1ou", "SomeAgent-ckpt5-t2.0-gl_05_26", "WIN")
    assert parse_trajectory_filename(fn) == ("SomeAgent-ckpt5-t2.0-gl_05_26", "WIN")


def test_parse_trajectory_filename_non_metamon_returns_none():
    # Human replay filenames have a different shape.
    assert parse_trajectory_filename("gen1ou-1234_1500_p1_vs_p2_01-02-2025_WIN.json") is None


def test_parse_trajectory_filename_json_no_lz4():
    fn = (
        f"metamon-gen1ou-1234567890_Unrated_MMVecSD-1_vs_OppA_"
        f"01-02-2025-03:04:05_WIN.json"
    )
    assert parse_trajectory_filename(fn) == ("OppA", "WIN")


def test_match_agent_name_exact():
    assert match_agent_name("TaurosV0", ["TaurosV0", "Other"]) == "TaurosV0"


def test_match_agent_name_prefix_with_suffix():
    # short_label = name-ckptN-tN-team_set
    assert match_agent_name("TaurosV0-ckpt40-gl_05_26", ["TaurosV0"]) == "TaurosV0"


def test_match_agent_name_longest_wins():
    """``TaurosV0-1`` (num_agents expansion) must win over ``TaurosV0``."""
    assert (
        match_agent_name("TaurosV0-1-ckpt40-gl_05_26", ["TaurosV0", "TaurosV0-1"])
        == "TaurosV0-1"
    )


def test_match_agent_name_no_false_prefix():
    """``TaurosV0X`` must NOT match ``TaurosV0`` (next char is not ``-``)."""
    assert match_agent_name("TaurosV0X-ckpt40", ["TaurosV0"]) is None


# ---------------------------------------------------------------------------
# compute_prioritized_weights
# ---------------------------------------------------------------------------


def _write_files(d, opp_counts, fmt="gen1ou"):
    """Write ``opp_counts = {label: (n_wins, n_losses)}`` dummy files into ``d``."""
    for label, (nw, nl) in opp_counts.items():
        for _ in range(nw):
            fn = _make_filename(fmt, label, "WIN", bid="".join(random.choices("0123456789", k=10)))
            with open(os.path.join(d, fn), "wb") as f:
                f.write(b"")
        for _ in range(nl):
            fn = _make_filename(fmt, label, "LOSS", bid="".join(random.choices("0123456789", k=10)))
            with open(os.path.join(d, fn), "wb") as f:
                f.write(b"")


def test_weights_concentrate_on_losing_opponent():
    """An opponent the learner loses to should get more weight than one it beats."""
    agents = ["AgentA", "AgentB"]
    with tempfile.TemporaryDirectory() as buf:
        fmt_dir = os.path.join(buf, "gen1ou")
        os.makedirs(fmt_dir)
        # AgentA: learner loses most (10 wins / 100 games → p≈0.11 → high exploit)
        # AgentB: learner wins most (90 wins / 100 games → p≈0.89 → low exploit)
        _write_files(fmt_dir, {"AgentA": (10, 90), "AgentB": (90, 10)})
        w, diag = compute_prioritized_weights(
            buffer_dir=buf,
            battle_format="gen1ou",
            agent_names=agents,
            window=0,
            min_games=20,
            temp=1.0,
            floor=0.05,
            ema=0.0,
        )
    assert w["AgentA"] > w["AgentB"], diag
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_weights_uniform_when_all_cold():
    agents = ["AgentA", "AgentB"]
    with tempfile.TemporaryDirectory() as buf:
        fmt_dir = os.path.join(buf, "gen1ou")
        os.makedirs(fmt_dir)
        _write_files(fmt_dir, {"AgentA": (1, 0)})  # below min_games=20
        w, diag = compute_prioritized_weights(
            buffer_dir=buf,
            battle_format="gen1ou",
            agent_names=agents,
            window=0,
            min_games=20,
            ema=0.0,
        )
    assert w["AgentA"] == pytest.approx(0.5)
    assert w["AgentB"] == pytest.approx(0.5)


def test_weights_uniform_empty_buffer():
    agents = ["AgentA", "AgentB", "AgentC"]
    with tempfile.TemporaryDirectory() as buf:
        w, diag = compute_prioritized_weights(
            buffer_dir=buf,
            battle_format="gen1ou",
            agent_names=agents,
            ema=0.0,
        )
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert all(v == pytest.approx(1.0 / 3) for v in w.values())


def test_floor_prevents_collapse():
    """Even with one opponent completely dominated, it keeps ``floor`` mass."""
    agents = ["AgentA", "AgentB"]
    with tempfile.TemporaryDirectory() as buf:
        fmt_dir = os.path.join(buf, "gen1ou")
        os.makedirs(fmt_dir)
        # AgentA: learner wins all → score 0 → floored
        # AgentB: learner loses all → max weight
        _write_files(fmt_dir, {"AgentA": (100, 0), "AgentB": (0, 100)})
        w, diag = compute_prioritized_weights(
            buffer_dir=buf,
            battle_format="gen1ou",
            agent_names=agents,
            window=0,
            min_games=20,
            temp=1.0,
            floor=0.3,
            ema=0.0,
        )
    # AgentA gets floor, AgentB gets the rest.
    assert w["AgentA"] >= 0.1  # floor is relative after normalization
    assert w["AgentB"] > w["AgentA"]


def test_ema_blends_with_prev():
    """EMA: W_t = β·new + (1-β)·prev, normalized."""
    agents = ["AgentA", "AgentB"]
    with tempfile.TemporaryDirectory() as buf:
        fmt_dir = os.path.join(buf, "gen1ou")
        os.makedirs(fmt_dir)
        _write_files(fmt_dir, {"AgentA": (10, 90), "AgentB": (90, 10)})
        prev = {"AgentA": 0.5, "AgentB": 0.5}
        w, diag = compute_prioritized_weights(
            buffer_dir=buf,
            battle_format="gen1ou",
            agent_names=agents,
            window=0,
            min_games=20,
            temp=1.0,
            floor=0.05,
            ema=0.9,  # heavy weight on prev → closer to uniform
            prev_weights=prev,
        )
        w_no_ema, _ = compute_prioritized_weights(
            buffer_dir=buf,
            battle_format="gen1ou",
            agent_names=agents,
            window=0,
            min_games=20,
            temp=1.0,
            floor=0.05,
            ema=0.0,
            prev_weights=prev,
        )
    # With heavy EMA toward uniform, the gap should shrink vs no-EMA.
    gap_ema = w["AgentA"] - w["AgentB"]
    gap_no = w_no_ema["AgentA"] - w_no_ema["AgentB"]
    assert abs(gap_ema) < abs(gap_no)


def test_window_respects_most_recent():
    """Only the ``window`` most-recent files are scored (by mtime)."""
    agents = ["AgentA", "AgentB"]
    with tempfile.TemporaryDirectory() as buf:
        fmt_dir = os.path.join(buf, "gen1ou")
        os.makedirs(fmt_dir)
        # Old games vs AgentA (learner wins), new games vs AgentB (learner loses).
        _write_files(fmt_dir, {"AgentA": (100, 0)})
        # Backdate AgentA files.
        import time as _time

        old_t = _time.time() - 100000
        for fn in os.listdir(fmt_dir):
            os.utime(os.path.join(fmt_dir, fn), (old_t, old_t))
        _write_files(fmt_dir, {"AgentB": (0, 100)})
        # window large enough → sees both; window small → sees only AgentB.
        w_all, _ = compute_prioritized_weights(
            buffer_dir=buf,
            battle_format="gen1ou",
            agent_names=agents,
            window=0,
            min_games=20,
            ema=0.0,
        )
        w_win, diag_win = compute_prioritized_weights(
            buffer_dir=buf,
            battle_format="gen1ou",
            agent_names=agents,
            window=100,
            min_games=20,
            ema=0.0,
        )
    # Full window: AgentA dominated (floored), AgentB gets high weight.
    assert w_all["AgentB"] > w_all["AgentA"]
    # Small window: only AgentB games visible; AgentA cold (n=0 < min_games).
    assert diag_win["AgentA"]["n"] == 0
    assert diag_win["AgentB"]["n"] == 100


def test_diagnostics_report_counts_and_winrates():
    agents = ["AgentA"]
    with tempfile.TemporaryDirectory() as buf:
        fmt_dir = os.path.join(buf, "gen1ou")
        os.makedirs(fmt_dir)
        _write_files(fmt_dir, {"AgentA": (30, 70)})
        _, diag = compute_prioritized_weights(
            buffer_dir=buf,
            battle_format="gen1ou",
            agent_names=agents,
            window=0,
            min_games=20,
            ema=0.0,
        )
    assert diag["AgentA"]["n"] == 100
    assert diag["AgentA"]["win_rate"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Sidecar reader / writer
# ---------------------------------------------------------------------------


def test_sidecar_round_trip():
    with tempfile.TemporaryDirectory() as buf:
        path = os.path.join(buf, "gen1ou", "meta_weights.json")
        cfg = PsroConfig(
            buffer_dir=buf,
            battle_format="gen1ou",
            agent_names=["AgentA", "AgentB"],
        )
        psro = PsroLite(config=cfg)
        # Write a fake buffer so step() has something to score.
        fmt_dir = os.path.join(buf, "gen1ou")
        os.makedirs(fmt_dir)
        _write_files(fmt_dir, {"AgentA": (10, 90), "AgentB": (90, 10)})
        weights, diag = psro.step(epoch=0)
        assert os.path.exists(path)
        with open(path) as f:
            disk = json.load(f)
        assert set(disk.keys()) == {"AgentA", "AgentB"}
        assert abs(disk["AgentA"] + disk["AgentB"] - 1.0) < 1e-9
        assert diag["_sidecar_write_ok"] is True


def test_read_sidecar_caches_by_mtime():
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"AgentA": 0.7, "AgentB": 0.3}, f)
        path = f.name
    try:
        w1, m1 = read_sidecar(path, None)
        assert w1 == {"AgentA": 0.7, "AgentB": 0.3}
        # Same mtime → cached (returns None weights).
        w2, m2 = read_sidecar(path, m1)
        assert w2 is None and m2 == m1
        # Rewrite → new mtime → re-read.
        import time as _time

        _time.sleep(0.01)
        with open(path, "w") as f2:
            json.dump({"AgentA": 0.1, "AgentB": 0.9}, f2)
        w3, m3 = read_sidecar(path, m1)
        assert w3 == {"AgentA": 0.1, "AgentB": 0.9}
    finally:
        os.remove(path)


def test_read_sidecar_missing_file_returns_none():
    w, m = read_sidecar("/nonexistent/path/meta_weights.json", None)
    assert w is None and m is None


def test_read_sidecar_bad_json_returns_none():
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write("{not valid json")
        path = f.name
    try:
        w, m = read_sidecar(path, None)
        assert w is None
    finally:
        os.remove(path)


def test_weight_entropy_uniform():
    h = weight_entropy({"a": 0.5, "b": 0.5})
    assert h == pytest.approx(__import__("math").log(2))


def test_weight_entropy_collapsed():
    h = weight_entropy({"a": 1.0, "b": 0.0})
    assert h == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# OpponentPoolConfig.set_weights
# ---------------------------------------------------------------------------


def _dummy_pool(agents):
    rows = [(name, {"model_name": name, "team_set": "competitive"}) for name in agents]
    return OpponentPoolConfig(agents=rows, battle_format="gen1ou", rng=random.Random(0))


def test_pool_set_weights_samples_distribution():
    pool = _dummy_pool(["AgentA", "AgentB"])
    pool.set_weights([0.9, 0.1])
    counts = Counter(pool.sample_opponent().name for _ in range(10000))
    # 0.9 / 0.1 split with 10k samples → within 3%.
    assert counts["AgentA"] / 10000 == pytest.approx(0.9, abs=0.03)
    assert counts["AgentB"] / 10000 == pytest.approx(0.1, abs=0.03)


def test_pool_uniform_when_weights_none():
    pool = _dummy_pool(["AgentA", "AgentB", "AgentC"])
    pool.set_weights(None)
    counts = Counter(pool.sample_opponent().name for _ in range(9000))
    for name in ("AgentA", "AgentB", "AgentC"):
        assert counts[name] / 9000 == pytest.approx(1 / 3, abs=0.03)


def test_pool_set_weights_bad_length_raises():
    pool = _dummy_pool(["AgentA", "AgentB"])
    with pytest.raises(ValueError):
        pool.set_weights([1.0])


def test_pool_set_weights_all_zero_falls_back_uniform():
    pool = _dummy_pool(["AgentA", "AgentB"])
    pool.set_weights([0.0, 0.0])
    assert pool.weights is None


def test_pool_set_weights_negative_falls_back_uniform():
    pool = _dummy_pool(["AgentA", "AgentB"])
    pool.set_weights([-1.0, 2.0])
    assert pool.weights is None


# ---------------------------------------------------------------------------
# ConfigBatchedOpponent sidecar read
# ---------------------------------------------------------------------------


def test_config_batched_opponent_reads_sidecar():
    """``ConfigBatchedOpponent._maybe_refresh_weights`` parses the sidecar and
    applies weights to the pool (without loading any NN models)."""
    import torch

    from metamon.env.vectorized.opponent import ConfigBatchedOpponent

    pool = _dummy_pool(["AgentA", "AgentB"])
    with tempfile.TemporaryDirectory() as buf:
        sidecar = os.path.join(buf, "meta_weights.json")
        with open(sidecar, "w") as f:
            json.dump({"AgentA": 0.9, "AgentB": 0.1}, f)
        opp = ConfigBatchedOpponent(
            config=pool,
            num_lanes=1,
            device=torch.device("cpu"),
            weights_path=sidecar,
        )
        # No weights yet (nothing read).
        assert opp.config.weights is None
        opp._maybe_refresh_weights()
        assert opp.config.weights is not None
        assert opp.config.weights == pytest.approx([0.9, 0.1], abs=1e-6)
        # Second call is cached by mtime (no re-read needed).
        opp._maybe_refresh_weights()


def test_config_batched_opponent_uniform_when_no_sidecar():
    import torch

    from metamon.env.vectorized.opponent import ConfigBatchedOpponent

    pool = _dummy_pool(["AgentA", "AgentB"])
    opp = ConfigBatchedOpponent(
        config=pool,
        num_lanes=1,
        device=torch.device("cpu"),
        weights_path="/nonexistent/meta_weights.json",
    )
    opp._maybe_refresh_weights()
    assert opp.config.weights is None  # uniform fallback


# ---------------------------------------------------------------------------
# Cold-fallback confidence weighting (no more uniform spike for dominated
# opponents whose rolling-window count dips below min_games)
# ---------------------------------------------------------------------------

def test_dominated_cold_opponent_stays_at_floor():
    """A dominated opponent with n < min_games must NOT spike to 1/n_agents.

    This is the regression test for the original oscillation: dominated → floor
    → rarely sampled → n dips below min_games → cold fallback snapped weight to
    the uniform share → spike → dominated again. The confidence-weighted fix
    keeps cold dominated opponents at the floor.
    """
    agents = ["AgentA", "AgentB", "AgentC", "AgentD"]
    with tempfile.TemporaryDirectory() as buf:
        fmt_dir = os.path.join(buf, "gen1ou")
        os.makedirs(fmt_dir)
        # AgentA: dominated but cold (5 games, all learner wins → n=5 < 20).
        _write_files(fmt_dir, {"AgentA": (5, 0)})
        w, diag = compute_prioritized_weights(
            buffer_dir=buf,
            battle_format="gen1ou",
            agent_names=agents,
            window=0,
            min_games=20,
            temp=1.0,
            floor=0.05,
            ema=0.0,
        )
    # AgentA's raw weight is the floor (0.05), not 1/n_agents (0.25).
    assert diag["AgentA"]["raw_weight"] == pytest.approx(0.05)
    # After normalization AgentA must not exceed the uniform share.
    assert w["AgentA"] <= 1.0 / len(agents) + 1e-9
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_novelty_bonus_boosts_never_played():
    """With novelty_gamma > 0, a never-played opponent gets a small floor+ bump
    that decays as games accrue — without snapping to the uniform share."""
    agents = ["AgentA", "AgentB"]
    with tempfile.TemporaryDirectory() as buf:
        fmt_dir = os.path.join(buf, "gen1ou")
        os.makedirs(fmt_dir)
        _write_files(fmt_dir, {"AgentA": (0, 0)})  # never played
        w0, diag0 = compute_prioritized_weights(
            buffer_dir=buf, battle_format="gen1ou", agent_names=agents,
            window=0, min_games=20, ema=0.0, floor=0.05, novelty_gamma=0.0,
        )
        w_g, diag_g = compute_prioritized_weights(
            buffer_dir=buf, battle_format="gen1ou", agent_names=agents,
            window=0, min_games=20, ema=0.0, floor=0.05, novelty_gamma=0.5,
        )
    # Without novelty: both cold → floor → uniform.
    assert w0["AgentA"] == pytest.approx(0.5)
    # With novelty: AgentA (n=0) gets floor + γ/γ0 > AgentB (n=0, same bonus).
    # Both get the same novelty bonus here (both n=0), so still uniform — but the
    # raw weight is above the floor. Use a one-agent-above case:
    assert diag_g["AgentA"]["raw_weight"] > 0.05


def test_cap_ratio_bounds_raw_weight():
    """cap_ratio hard-bounds each raw weight to R*floor."""
    agents = ["AgentA", "AgentB"]
    with tempfile.TemporaryDirectory() as buf:
        fmt_dir = os.path.join(buf, "gen1ou")
        os.makedirs(fmt_dir)
        # AgentB: learner loses all → max score. floor=0.05, cap R=4 → max 0.20.
        _write_files(fmt_dir, {"AgentA": (100, 0), "AgentB": (0, 100)})
        w, diag = compute_prioritized_weights(
            buffer_dir=buf, battle_format="gen1ou", agent_names=agents,
            window=0, min_games=20, ema=0.0, floor=0.05, cap_ratio=4.0,
        )
    assert diag["AgentB"]["raw_weight"] <= 4.0 * 0.05 + 1e-9


# ---------------------------------------------------------------------------
# Quota-based sampling in ConfigBatchedOpponent
# ---------------------------------------------------------------------------

def _dummy_pool_multi(agents):
    rows = [(name, {"model_name": name, "team_set": "competitive"}) for name in agents]
    return OpponentPoolConfig(agents=rows, battle_format="gen1ou", rng=random.Random(0))


def test_pool_sample_opponent_for_agent():
    pool = _dummy_pool_multi(["AgentA", "AgentB"])
    spec = pool.sample_opponent_for_agent("AgentB")
    assert spec.name == "AgentB"
    with pytest.raises(KeyError):
        pool.sample_opponent_for_agent("Nope")


def test_quota_guarantees_representation():
    """Over a rolling window, every agent gets >= quota_min_assignments picks,
    even when the PSRO weights try to starve the dominated ones."""
    import torch

    from metamon.env.vectorized.opponent import ConfigBatchedOpponent

    agents = ["AgentA", "AgentB", "AgentC", "AgentD"]
    pool = _dummy_pool_multi(agents)
    opp = ConfigBatchedOpponent(
        config=pool,
        num_lanes=8,
        device=torch.device("cpu"),
        # Starve everyone but AgentA via the sidecar weights.
        weights_path=None,
        quota_min_games=16,  # → 2 assignments (16/8 lanes)
        quota_window=64,
    )
    # Simulate the PSRO weights concentrating 97% on AgentA.
    pool.set_weights([0.97, 0.01, 0.01, 0.01])
    picks = Counter()
    for _ in range(opp._quota_window):
        spec = opp._sample_with_quota()
        picks[spec.name] += 1
    min_a = opp._quota_min_assignments
    for nm in agents:
        assert picks[nm] >= min_a, (nm, picks[nm], min_a)
    # The surplus still lets AgentA collect extra picks (it has the high weight).
    assert picks["AgentA"] > min_a


def test_quota_disabled_when_min_games_zero():
    """quota_min_games <= 0 disables the quota → pure weighted sampling."""
    import torch

    from metamon.env.vectorized.opponent import ConfigBatchedOpponent

    pool = _dummy_pool_multi(["AgentA", "AgentB"])
    opp = ConfigBatchedOpponent(
        config=pool,
        num_lanes=8,
        device=torch.device("cpu"),
        quota_min_games=0,
        quota_window=64,
    )
    assert opp._quota_min_assignments == 0
    pool.set_weights([0.99, 0.01])
    picks = Counter(opp._sample_with_quota().name for _ in range(200))
    # AgentB can be completely starved (no quota guarantee).
    assert picks["AgentA"] > picks["AgentB"]


def test_quota_infeasible_falls_back_to_weighted():
    """If n_agents * min > window, the quota can't be satisfied → weighted sample."""
    import torch

    from metamon.env.vectorized.opponent import ConfigBatchedOpponent

    agents = ["A", "B", "C", "D"]
    pool = _dummy_pool_multi(agents)
    # min_games=80 / 8 lanes = 10 assignments; 4 agents * 10 = 40 > window=16.
    opp = ConfigBatchedOpponent(
        config=pool,
        num_lanes=8,
        device=torch.device("cpu"),
        quota_min_games=80,
        quota_window=16,
    )
    assert opp._quota_min_assignments == 10
    pool.set_weights([0.97, 0.01, 0.01, 0.01])
    picks = Counter(opp._sample_with_quota().name for _ in range(200))
    # Infeasible → falls back to weighted → dominated agents can be starved.
    assert picks["A"] > picks["D"]
    assert picks["D"] < 10  # well below the infeasible quota target
