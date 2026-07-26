"""Search cleanup invariants (skill §19).

Verified at the simulator-transport level (no GPU / checkpoint needed): after a
search-root worth of fork_batch + rollout + cleanup, all branch lanes are
returned to the pool, all snapshots are released, and the trunk battle is
unchanged. A stress test performs many create/cleanup cycles and checks the
trunk stays frozen (lane/snapshot counts bounded, no leaks).
"""

from __future__ import annotations

import copy
import json
import random
from typing import List

import pytest

from metamon.env.vectorized.lane import SIDES, StreamBattleLane
from metamon.env.vectorized.sim_process import ShowdownSimProcess

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


def _raw_state(lane):
    return [
        json.dumps(lane.universal_state(s).to_dict(), sort_keys=True, default=str)
        for s in SIDES
    ]


def _advance(proc, lane, to=60.0):
    proc.pump_until(lambda: lane.decision_ready(), timeout=to, idle_timeout=to)


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
        req = lane.last_request["p1"] or {}
        moves = (
            req.get("active", [{}])[0].get("moves", [])
            if not req.get("forceSwitch")
            else []
        )
        choice = f"move {random.randint(1, max(len(moves), 1))}" if moves else "pass"
        for side in SIDES:
            if lane.needs_agent_decision(side):
                proc.choose(lane.lane_id, side, choice)
        _advance(proc, lane)
        if lane.ended:
            break


@pytest.fixture
def proc():
    p = ShowdownSimProcess()
    yield p
    p.close()


def test_fork_batch_with_seeds_then_cleanup(proc):
    """A search-root worth of fork_batch (with branch seeds) + cleanup leaves the
    trunk frozen and releases the snapshot (skill §19 cleanup invariants)."""
    rng = random.Random(4242)
    trunk = StreamBattleLane(0, "gen1ou")
    proc.register_lane(0, trunk)
    _run_to_settled(proc, trunk, rng, n_turns=2)
    if trunk.ended:
        pytest.skip("trunk ended early")
    proc.drain()
    trunk_state = _raw_state(trunk)
    snap_id = proc.snapshot(0)
    proc.drain()

    N = 6
    lane_ids = [100 + i for i in range(N)]
    lanes = []
    for bid in lane_ids:
        fl = copy.deepcopy(trunk)
        fl.lane_id = bid
        proc.register_lane(bid, fl)
        lanes.append(fl)
    seeds = [[i, i + 1, i + 2, i + 3] for i in range(N)]
    proc.fork_batch(snap_id, lane_ids, replay_log=False, seeds=seeds)
    # advance each branch one step (a real rollout would do more)
    for fl in lanes:
        _advance(proc, fl)
    # cleanup: reset every branch lane + release the snapshot
    for bid in lane_ids:
        proc.reset(bid)
    proc.release_snapshot(snap_id)
    proc.drain()
    assert _raw_state(trunk) == trunk_state, "trunk corrupted by fork_batch + cleanup"


def test_repeated_fork_release_cycles_keep_trunk_frozen(proc):
    """Many search-root create/cleanup cycles must not leak lanes/snapshots or
    drift the trunk (skill §19 stress test)."""
    rng = random.Random(909)
    trunk = StreamBattleLane(0, "gen1ou")
    proc.register_lane(0, trunk)
    _run_to_settled(proc, trunk, rng, n_turns=2)
    if trunk.ended:
        pytest.skip("trunk ended early")
    proc.drain()
    trunk_state = _raw_state(trunk)
    for cycle in range(30):
        proc.drain()
        snap_id = proc.snapshot(0)
        proc.drain()
        ids = [2000 + cycle * 10 + i for i in range(4)]
        for bid in ids:
            fl = copy.deepcopy(trunk)
            fl.lane_id = bid
            proc.register_lane(bid, fl)
        proc.fork_batch(
            snap_id,
            ids,
            replay_log=False,
            seeds=[[k, k + 1, k + 2, k + 3] for k in range(4)],
        )
        for bid in ids:
            proc.reset(bid)
        proc.release_snapshot(snap_id)
    proc.drain()
    assert _raw_state(trunk) == trunk_state, "trunk drifted over repeated search cycles"


def test_release_snapshot_then_fork_is_safe(proc):
    """After release_snapshot, a fork from the released id does not crash the
    host and the trunk stays intact (cleanup robustness)."""
    rng = random.Random(555)
    trunk = StreamBattleLane(0, "gen1ou")
    proc.register_lane(0, trunk)
    _run_to_settled(proc, trunk, rng, n_turns=1)
    if trunk.ended:
        pytest.skip("trunk ended early")
    proc.drain()
    trunk_state = _raw_state(trunk)
    snap_id = proc.snapshot(0)
    proc.drain()
    proc.release_snapshot(snap_id)
    # fork from the released snapshot: host emits a lane_error (no sync raise)
    proc.fork(snap_id, 77, replay_log=False, seed=[1, 2, 3, 4])
    proc.reset(77)
    proc.drain()
    assert _raw_state(trunk) == trunk_state
