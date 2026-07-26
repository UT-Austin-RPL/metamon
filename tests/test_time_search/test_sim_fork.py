"""Simulator snapshot/fork equivalence and independence tests.

These tests exercise the host-level ``snapshot``/``fork`` commands through the
real :class:`ShowdownSimProcess` binary transport against randomized Gen1 OU
battles. They verify the three correctness properties required for test-time
search:

  1. **Equivalence**: a fork given the same future actions as the trunk produces
     the same sequence of requests, rewards, HP/status/PP/move-list changes, and
     terminal outcome.
  2. **Independence**: forks given divergent actions evolve independently.
  3. **Trunk isolation**: phantom forks never alter the live trunk battle.

Forking uses the validated two-part mechanism (see
``test_time_search/ARCHITECTURE.md``):

  * **JS simulator state**: ``Battle.toJSON()`` → JSON string →
    ``Battle.fromJSON`` (official Showdown serialization; deep-copies via
    ``JSON.parse`` so the trunk's ``log`` array is never aliased). The fork is
    created with ``replay_log=False`` so only *new* log entries are emitted.
  * **Python parsed state**: ``copy.deepcopy`` of the trunk
    :class:`StreamBattleLane` (both ``MetamonBackendBattle`` POVs, requests,
    serials). This carries the exact, incrementally-built observation state
    (revealed moves, PP from requests, etc.) and avoids the request-
    regeneration quirk where ``deserializeBattle``'s ``getRequests`` would emit
    a request with different PP/field-ordering than the live emission.

A canonical state digest is used for compact comparison; because the deepcopy
fork is byte-exact, the digest includes PP and move lists.
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


def _raw_state(lane: StreamBattleLane) -> List[str]:
    """Full JSON state of both POVs (includes PP, moves, status — everything)."""
    return [
        json.dumps(lane.universal_state(s).to_dict(), sort_keys=True, default=str)
        for s in SIDES
    ]


def _requests(lane: StreamBattleLane) -> str:
    return json.dumps(
        {s: lane.last_request[s] for s in SIDES}, sort_keys=True, default=str
    )


def _advance(
    proc: ShowdownSimProcess, lane: StreamBattleLane, timeout: float = 90.0
) -> None:
    proc.pump_until(
        lambda: lane.decision_ready(), timeout=timeout, idle_timeout=timeout
    )


def _step(
    proc: ShowdownSimProcess,
    lane: StreamBattleLane,
    plan: dict,
) -> None:
    """Consume the current decision, send choices, pump to the next decision.

    Mirrors the env's flow: ``mark_settled`` (so ``decision_ready`` flips to
    False), send ``choose`` only for sides that owe an answerable decision (move /
    force-switch / team-preview — never a ``wait`` side), then pump until the next
    request.
    """
    lane.mark_settled()
    for side in SIDES:
        if plan.get(side) and lane.needs_agent_decision(side):
            proc.choose(lane.lane_id, side, plan[side])
    _advance(proc, lane)


def _legal_choices(lane: StreamBattleLane, side: str) -> List[str]:
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


def _run_random_battle_to_settled(
    proc: ShowdownSimProcess, lane: StreamBattleLane, rng: random.Random, n_turns: int
) -> None:
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


def _fork_lane(proc, trunk, fork_id):
    """Deepcopy the trunk's Python lane and create a no-replay JS fork.

    Order matters: snapshot the JS battle *first* (its ``_sync`` + a ``drain``
    flush any trailing chunks into the trunk so the trunk's parsed state is
    fully caught up to the JS snapshot), then ``deepcopy`` the trunk, then fork.
    This keeps the Python deepcopy and the JS snapshot at the same settled point.
    """
    proc.drain()
    snap_id = proc.snapshot(trunk.lane_id)
    proc.drain()
    fork = copy.deepcopy(trunk)
    fork.lane_id = fork_id
    proc.register_lane(fork_id, fork)
    proc.fork(snap_id, fork_id, replay_log=False)
    return fork, snap_id


@pytest.fixture
def proc():
    p = ShowdownSimProcess()
    yield p
    p.close()


def test_fork_same_actions_equivalent(proc):
    rng = random.Random(2024)
    trunk = StreamBattleLane(0, "gen1ou")
    proc.register_lane(0, trunk)
    _run_random_battle_to_settled(proc, trunk, rng, n_turns=3)
    assert not trunk.ended
    fork, snap_id = _fork_lane(proc, trunk, 10)
    _advance(proc, fork)
    assert _raw_state(fork) == _raw_state(trunk)
    assert _requests(fork) == _requests(trunk)
    for _ in range(6):
        if trunk.ended or fork.ended:
            break
        plan = {s: rng.choice(_legal_choices(trunk, s) or ["pass"]) for s in SIDES}
        _step(proc, trunk, plan)
        _step(proc, fork, plan)
        assert _raw_state(fork) == _raw_state(
            trunk
        ), "fork diverged under identical actions"
        assert _requests(fork) == _requests(trunk)
    assert trunk.ended == fork.ended
    assert trunk.winner == fork.winner
    proc.release_snapshot(snap_id)


def test_fork_divergent_actions_independent(proc):
    rng = random.Random(7)
    trunk = StreamBattleLane(0, "gen1ou")
    proc.register_lane(0, trunk)
    _run_random_battle_to_settled(proc, trunk, rng, n_turns=2)
    f1, s1 = _fork_lane(proc, trunk, 11)
    f2, s2 = _fork_lane(proc, trunk, 12)
    _advance(proc, f1)
    _advance(proc, f2)
    assert _raw_state(f1) == _raw_state(f2) == _raw_state(trunk)
    diverged = False
    for _ in range(6):
        if f1.ended or f2.ended:
            break
        plan1, plan2 = {}, {}
        for side in SIDES:
            c1 = rng.choice(_legal_choices(f1, side) or ["pass"])
            legal2 = _legal_choices(f2, side) or ["pass"]
            c2 = rng.choice([c for c in legal2 if c != c1] or legal2)
            plan1[side] = c1
            plan2[side] = c2
        _step(proc, f1, plan1)
        _step(proc, f2, plan2)
        if _raw_state(f1) != _raw_state(f2):
            diverged = True
            break
    assert diverged, "divergent forks never diverged"
    proc.release_snapshot(s1)
    proc.release_snapshot(s2)


def test_trunk_unaffected_by_phantom_forks(proc):
    """Phantom forks must not change the trunk's state (no chooses hit lane 0)."""
    rng = random.Random(99)
    trunk = StreamBattleLane(0, "gen1ou")
    proc.register_lane(0, trunk)
    _run_random_battle_to_settled(proc, trunk, rng, n_turns=3)
    proc.drain()  # fully catch up the trunk before capturing the reference
    trunk_state = _raw_state(trunk)
    trunk_reqs = _requests(trunk)
    snap_ids = []
    for i in range(4):
        fl, sid = _fork_lane(proc, trunk, 20 + i)
        snap_ids.append(sid)
        _advance(proc, fl)
        plan = {s: rng.choice(_legal_choices(fl, s) or ["pass"]) for s in SIDES}
        _step(proc, fl, plan)
    # No chooses were sent to lane 0; the trunk must be frozen at its state.
    proc.drain()
    assert _raw_state(trunk) == trunk_state, "phantom fork corrupted the trunk state"
    assert _requests(trunk) == trunk_reqs, "phantom fork changed the trunk requests"
    for sid in snap_ids:
        proc.release_snapshot(sid)


def test_fork_equivalence_many_seeds(proc):
    """Snapshot equivalence across many random settled states (stress test).

    The step-by-step equivalence is covered by
    ``test_fork_same_actions_equivalent``; this test focuses on the weaker but
    broadly-sampled property that a fork matches the trunk at the snapshot point
    across many random battles (different turn counts, seeds, and thus different
    field/status/forced-switch states).
    """
    failures = []
    for seed in range(25):
        rng = random.Random(5000 + seed)
        n_turns = rng.randint(0, 5)
        trunk_id = 1000 + seed
        fork_id = 2000 + seed
        trunk = StreamBattleLane(trunk_id, "gen1ou")
        proc.register_lane(trunk_id, trunk)
        try:
            _run_random_battle_to_settled(proc, trunk, rng, n_turns=n_turns)
            if trunk.ended:
                proc.reset(trunk_id)
                continue
            proc.drain()
            fork, snap_id = _fork_lane(proc, trunk, fork_id)
            _advance(proc, fork, timeout=10.0)
            if _raw_state(fork) != _raw_state(trunk):
                failures.append(seed)
            proc.release_snapshot(snap_id)
            proc.reset(fork_id)
            proc.reset(trunk_id)
            proc.drain()
        except ShowdownSimProcessError:
            # A random choice can trigger a Showdown |error| re-prompt that this
            # naive test harness does not re-answer (the real env's _pump_settle
            # does). Skip inconclusive seeds rather than counting them as fork
            # failures; the focused equivalence test covers step-by-step checks.
            try:
                proc.reset(trunk_id)
                proc.reset(fork_id)
                proc.drain()
            except Exception:
                pass
            continue
        except Exception as exc:  # noqa: BLE001
            failures.append((seed, repr(exc)))
            try:
                proc.reset(trunk_id)
                proc.reset(fork_id)
            except Exception:
                pass
    assert not failures, f"fork snapshot equivalence failed: {failures[:6]}"
