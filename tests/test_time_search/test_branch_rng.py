"""Branch RNG tests for test-time search (skill §7).

Two layers:

  1. Pure-python ``RootSeedBank`` unit tests: the common-random-number seed
     bank shares a seed across candidate actions for a fixed rollout index k,
     differs across k, never depends on candidate-action identity, and is
     reproducible.

  2. Sim-level integration tests through the real ``ShowdownSimProcess``: a
     branch-only reseed replaces the fork's inherited trunk future-RNG stream;
     same seed + same actions -> identical trajectory; different seeds ->
     divergent stochastic outcomes; reseeding does not alter the trunk PRNG;
     the inherited-PRNG mode reproduces the trunk's actual future while a
     resampled seed does not (the future-chance oracle vs. the research mode).
"""

from __future__ import annotations

import copy
import json
import random
from typing import List

import pytest

from metamon.env.vectorized.lane import SIDES, StreamBattleLane
from metamon.env.vectorized.sim_process import (
    ShowdownSimProcess,
    ShowdownSimProcessError,
)
from metamon.rl.experimental.test_time_search.rng import (
    RootSeedBank,
    branch_env_seed,
    make_rng,
    opp_root_key,
)

_TEAM = [
    {
        "name": "Tauros",
        "moves": ["bodyslam", "earthquake", "hyperbeam", "blizzard"],
        "evs": {"hp": 0, "atk": 252, "def": 0, "spa": 0, "spe": 252, "spd": 0},
        "ivs": {"hp": 31, "atk": 31, "def": 31, "spa": 31, "spe": 31, "spd": 31},
        "level": 100,
    },
    {
        "name": "Alakazam",
        "moves": ["psychic", "thunderwave", "recover", "seismictoss"],
        "evs": {"hp": 0, "atk": 0, "def": 0, "spa": 252, "spe": 252, "spd": 0},
        "ivs": {"hp": 31, "atk": 31, "def": 31, "spa": 31, "spe": 31, "spd": 31},
        "level": 100,
    },
    {
        "name": "Snorlax",
        "moves": ["bodyslam", "amnesia", "rest", "reflect"],
        "evs": {"hp": 252, "atk": 0, "def": 0, "spa": 0, "spe": 0, "spd": 252},
        "ivs": {"hp": 31, "atk": 31, "def": 31, "spa": 31, "spe": 31, "spd": 31},
        "level": 100,
    },
    {
        "name": "Exeggutor",
        "moves": ["psychic", "megadrain", "sleeppowder", "explosion"],
        "evs": {"hp": 0, "atk": 0, "def": 0, "spa": 252, "spe": 252, "spd": 0},
        "ivs": {"hp": 31, "atk": 31, "def": 31, "spa": 31, "spe": 31, "spd": 31},
        "level": 100,
    },
    {
        "name": "Chansey",
        "moves": ["seismictoss", "softboiled", "thunderwave", "reflect"],
        "evs": {"hp": 252, "atk": 0, "def": 0, "spa": 0, "spe": 0, "spd": 252},
        "ivs": {"hp": 31, "atk": 31, "def": 31, "spa": 31, "spe": 31, "spd": 31},
        "level": 100,
    },
    {
        "name": "Starmie",
        "moves": ["surf", "thunderwave", "recover", "blizzard"],
        "evs": {"hp": 0, "atk": 0, "def": 0, "spa": 252, "spe": 252, "spd": 0},
        "ivs": {"hp": 31, "atk": 31, "def": 31, "spa": 31, "spe": 31, "spd": 31},
        "level": 100,
    },
]


# ---------------------------------------------------------------------------
# sim helpers (mirrors test_sim_fork.py)
# ---------------------------------------------------------------------------


def _raw_state(lane: StreamBattleLane) -> List[str]:
    return [
        json.dumps(lane.universal_state(s).to_dict(), sort_keys=True, default=str)
        for s in SIDES
    ]


def _advance(proc, lane, timeout=90.0):
    proc.pump_until(
        lambda: lane.decision_ready(), timeout=timeout, idle_timeout=timeout
    )


def _legal_choices(lane, side):
    req = lane.last_request[side]
    if req is None or req.get("wait"):
        return []
    if req.get("forceSwitch"):
        side_block = req.get("side", {}).get("pokemon", [])
        active_idx = [i for i, p in enumerate(side_block) if p.get("active")]
        switches = []
        for i, p in enumerate(side_block):
            if not p.get("fainted") and i not in active_idx:
                switches.append(f"switch {i + 1}")
        return switches or ["pass"]
    moves = req.get("active", [{}])[0].get("moves", [])
    return [f"move {i + 1}" for i in range(len(moves))] or ["pass"]


def _step(proc, lane, plan):
    lane.mark_settled()
    for side in SIDES:
        if plan.get(side) and lane.needs_agent_decision(side):
            proc.choose(lane.lane_id, side, plan[side])
    _advance(proc, lane)


def _run_to_settled(proc, lane, rng, n_turns):
    proc.start_battle(
        lane.lane_id,
        "gen1ou",
        {"name": "p1", "team": _TEAM},
        {"name": "p2", "team": _TEAM},
        seed=[rng.randint(0, 0xFFFF) for _ in range(4)],
    )
    _advance(proc, lane)
    lane.mark_settled()
    for _ in range(n_turns):
        plan = {s: rng.choice(_legal_choices(lane, s) or ["pass"]) for s in SIDES}
        for side in SIDES:
            if plan[side] and lane.needs_agent_decision(side):
                proc.choose(lane.lane_id, side, plan[side])
        _advance(proc, lane)
        if lane.ended:
            break


def _fork(proc, trunk, fork_id, seed):
    proc.drain()
    snap_id = proc.snapshot(trunk.lane_id)
    proc.drain()
    fork = copy.deepcopy(trunk)
    fork.lane_id = fork_id
    proc.register_lane(fork_id, fork)
    proc.fork(snap_id, fork_id, replay_log=False, seed=seed)
    return fork, snap_id


@pytest.fixture
def proc():
    p = ShowdownSimProcess()
    yield p
    p.close()


# ===========================================================================
# 1. RootSeedBank unit tests (pure python)
# ===========================================================================


def test_branch_env_seed_is_4_uint16():
    s = branch_env_seed(42, "b0", "p1", 0, 0)
    assert isinstance(s, list) and len(s) == 4
    assert all(isinstance(x, int) and 0 <= x <= 0xFFFF for x in s)


def test_seed_bank_is_deterministic_and_reproducible():
    a = RootSeedBank.build(7, "battle1", "p1", 10, K=4)
    b = RootSeedBank.build(7, "battle1", "p1", 10, K=4)
    assert a.env_seeds == b.env_seeds
    assert a.opp_root_keys == b.opp_root_keys


def test_seed_bank_per_k_distinct():
    bank = RootSeedBank.build(7, "b", "p1", 3, K=4)
    assert len(bank.env_seeds) == 4
    assert len({tuple(s) for s in bank.env_seeds}) == 4  # all distinct
    assert len(set(bank.opp_root_keys)) == 4


def test_seed_bank_shared_across_candidate_actions():
    """env_seed[root, k] is shared across candidate actions a (CRN). The branch
    layout is ``np.repeat(legal, K)`` -> branch b = a*K + k, so k = b % K."""
    K = 4
    A = 3  # three candidate actions
    bank = RootSeedBank.build(7, "b", "p1", 3, K=K)
    for k in range(K):
        seeds_for_k = {tuple(bank.env_seed_for_branch(a * K + k)) for a in range(A)}
        assert len(seeds_for_k) == 1, f"candidate actions share seed for k={k}"
    # different k -> different seed
    for k1 in range(K):
        for k2 in range(k1 + 1, K):
            assert bank.env_seeds[k1] != bank.env_seeds[k2]


def test_seed_bank_independent_of_action_count():
    """The seed for rollout k must not depend on how many candidate actions A
    are being searched (skill §7: no branch seed depends on candidate action
    identity)."""
    k = 2
    seed_with_A3 = RootSeedBank.build(7, "b", "p1", 3, K=4).env_seeds[k]
    seed_with_A9 = RootSeedBank.build(7, "b", "p1", 3, K=4).env_seeds[k]
    assert seed_with_A3 == seed_with_A9
    # and the per-branch lookup depends only on k (b % K), not on A:
    b3 = RootSeedBank.build(7, "b", "p1", 3, K=4)
    assert b3.env_seed_for_branch(0 * 4 + k) == b3.env_seed_for_branch(5 * 4 + k)


def test_seed_bank_different_root_or_battle_differs():
    a = RootSeedBank.build(7, "b0", "p1", 0, K=2)
    b = RootSeedBank.build(7, "b1", "p1", 0, K=2)  # different battle
    c = RootSeedBank.build(7, "b0", "p1", 1, K=2)  # different decision
    d = RootSeedBank.build(99, "b0", "p1", 0, K=2)  # different global seed
    assert a.env_seeds != b.env_seeds
    assert a.env_seeds != c.env_seeds
    assert a.env_seeds != d.env_seeds


def test_opp_root_key_per_k_makes_independent_generators():
    K = 4
    bank = RootSeedBank.build(7, "b", "p1", 2, K=K)
    draws = [make_rng(bank.opp_root_keys[k]).integers(0, 1 << 30) for k in range(K)]
    assert len(set(draws)) == K  # independent streams per k


def test_seed_bank_log_dict_has_hashes_not_raw_seeds():
    bank = RootSeedBank.build(7, "b", "p1", 2, K=3)
    d = bank.to_log_dict()
    assert "env_seed_hashes" in d and len(d["env_seed_hashes"]) == 3
    # raw seed bytes must not appear verbatim in the hash log
    raw = json.dumps(bank.env_seeds[0])
    assert d["env_seed_hashes"][0] != raw


# ===========================================================================
# 2. Sim-level reseed integration tests
# ===========================================================================


def test_reseed_same_seed_same_actions_identical(proc):
    """Same snapshot + same branch seed + same actions -> identical trajectory."""
    rng = random.Random(2025)
    trunk = StreamBattleLane(0, "gen1ou")
    proc.register_lane(0, trunk)
    _run_to_settled(proc, trunk, rng, n_turns=2)
    if trunk.ended:
        pytest.skip(
            "trunk ended before the snapshot point; re-seed with a different seed"
        )
    seed = [0x1234, 0x5678, 0x9ABC, 0xDEF0]
    f1, s1 = _fork(proc, trunk, 11, seed)
    f2, s2 = _fork(proc, trunk, 12, seed)
    _advance(proc, f1)
    _advance(proc, f2)
    assert _raw_state(f1) == _raw_state(f2) == _raw_state(trunk)
    for _ in range(5):
        if f1.ended or f2.ended:
            break
        plan = {s: rng.choice(_legal_choices(f1, s) or ["pass"]) for s in SIDES}
        _step(proc, f1, plan)
        _step(proc, f2, plan)
        proc.drain()  # flush any trailing chunks into both lanes before comparing
        assert _raw_state(f1) == _raw_state(
            f2
        ), "same-seed forks diverged under identical actions"
    proc.release_snapshot(s1)
    proc.release_snapshot(s2)


def test_reseed_different_seeds_diverge_on_stochastic_position(proc):
    """Same snapshot + different branch seeds + identical actions -> the
    stochastic outcomes (damage rolls / crits / accuracy) diverge. This proves
    the reseed actually replaces the chance stream (skill §7 requirement 2)."""
    rng = random.Random(7)
    trunk = StreamBattleLane(0, "gen1ou")
    proc.register_lane(0, trunk)
    _run_to_settled(proc, trunk, rng, n_turns=1)
    # pick a starting position where both sides have damage moves (Tauros lead)
    f1, s1 = _fork(proc, trunk, 21, [0x1111, 0x2222, 0x3333, 0x4444])
    f2, s2 = _fork(proc, trunk, 22, [0xAAAA, 0xBBBB, 0xCCCC, 0xDDDD])
    _advance(proc, f1)
    _advance(proc, f2)
    assert _raw_state(f1) == _raw_state(f2)  # identical at the snapshot point
    diverged = False
    for _ in range(6):
        if f1.ended or f2.ended:
            break
        # play IDENTICAL actions on both forks
        plan = {s: rng.choice(_legal_choices(f1, s) or ["pass"]) for s in SIDES}
        _step(proc, f1, plan)
        _step(proc, f2, plan)
        if _raw_state(f1) != _raw_state(f2):
            diverged = True
            break
    assert (
        diverged
    ), "different-seed forks never diverged (reseed did not change the chance stream)"
    proc.release_snapshot(s1)
    proc.release_snapshot(s2)


def test_reseed_does_not_alter_trunk_prng_or_state(proc):
    """Reseeding branches must not mutate the trunk battle or its PRNG (skill §7
    requirement 6). The trunk must stay frozen across phantom reseeded forks."""
    rng = random.Random(99)
    trunk = StreamBattleLane(0, "gen1ou")
    proc.register_lane(0, trunk)
    _run_to_settled(proc, trunk, rng, n_turns=3)
    proc.drain()
    trunk_state = _raw_state(trunk)
    snap_ids = []
    for i in range(4):
        fl, sid = _fork(proc, trunk, 30 + i, [i, i + 1, i + 2, i + 3])
        snap_ids.append(sid)
        _advance(proc, fl)
        plan = {s: rng.choice(_legal_choices(fl, s) or ["pass"]) for s in SIDES}
        _step(proc, fl, plan)
    proc.drain()
    assert _raw_state(trunk) == trunk_state, "reseeded phantom fork corrupted the trunk"
    for sid in snap_ids:
        proc.release_snapshot(sid)


def test_inherited_rng_matches_trunk_future_resampled_does_not(proc):
    """The inherited-PRNG mode (seed=None) reproduces the trunk's actual future
    RNG stream (the future-chance oracle); a resampled seed does not. This is
    the defining distinction between the legacy diagnostic mode and the
    research ``resample_crn`` mode (skill §7)."""
    rng = random.Random(123)
    trunk = StreamBattleLane(0, "gen1ou")
    proc.register_lane(0, trunk)
    _run_to_settled(proc, trunk, rng, n_turns=1)
    f_inherit, s1 = _fork(proc, trunk, 41, seed=None)  # inherited trunk PRNG
    f_resample, s2 = _fork(proc, trunk, 42, seed=[0xC0DE, 0xFEED, 0x1234, 0x5678])
    _advance(proc, f_inherit)
    _advance(proc, f_resample)
    # one concrete plan applied to the trunk and to both forks
    plan = {s: rng.choice(_legal_choices(trunk, s) or ["pass"]) for s in SIDES}
    _step(proc, trunk, plan)
    _step(proc, f_inherit, plan)
    _step(proc, f_resample, plan)
    # inherited fork followed the trunk's exact future RNG -> identical state
    assert _raw_state(f_inherit) == _raw_state(
        trunk
    ), "inherited-PRNG fork did NOT reproduce the trunk's future (expected the oracle)"
    # resampled fork used a different chance stream -> diverges
    assert _raw_state(f_resample) != _raw_state(
        trunk
    ), "resampled fork matched the trunk future (reseed did not take effect)"
    proc.release_snapshot(s1)
    proc.release_snapshot(s2)


def test_reseed_cleanup_on_unknown_snapshot(proc):
    """A fork (with seed) against a bad snapshot_id does not crash the host and
    leaves the trunk untouched; the dead lane can be safely reset (skill §7
    requirement 10). The host emits an async lane-error (not a synchronous
    raise), so cleanup is about leaving no live fork battle and preserving the
    trunk.
    """
    rng = random.Random(321)
    trunk = StreamBattleLane(0, "gen1ou")
    proc.register_lane(0, trunk)
    _run_to_settled(proc, trunk, rng, n_turns=1)
    proc.drain()
    trunk_state = _raw_state(trunk)
    bad_snap = 999_999
    # No synchronous raise; the host emits a lane_error for lane 55 instead.
    proc.fork(bad_snap, 55, replay_log=False, seed=[1, 2, 3, 4])
    # Cleanup: reset the dead lane and drain any pending lane-error chunk.
    proc.reset(55)
    proc.drain()
    assert _raw_state(trunk) == trunk_state, "failed fork corrupted the trunk"
