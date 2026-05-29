"""Throwaway diagnostic: verify the two-perspective mirror property of obs.

For a given battle state, the UniversalState built from side 0's perspective
should be the mirror of the one built from side 1's: side-0's `player_*` fields
must equal side-1's `opponent_*` fields and vice versa. Any field that fails to
mirror is an observation asymmetry that handicaps one side's policy.
"""

import os
import random
from pathlib import Path

POKEPY_ENGINE = Path(__file__).resolve().parents[1] / "pokepy-engine"
os.environ.setdefault(
    "POKEPY_DATA_PATH", str(POKEPY_ENGINE / "pokepy" / "data" / "extracted")
)

import random as _random

from metamon.interface import DefaultActionSpace
from metamon.env import get_metamon_teams
from metamon.env.pokepy_battle.vector_env import _BattleLane
from metamon.env.pokepy_battle.team_adapter import team_set_to_pokepy_dict
from metamon.env.pokepy_battle.state_adapter import pokepy_state_to_universal
from metamon.env.pokepy_battle.action_adapter import (
    legal_action_indices,
    universal_action_to_pokepy,
)
from pokepy.data.loader import (
    load_game_data,
    load_id_mappings,
    load_move_effect_data,
)


def rand_action(action_space, lane, gd, mp, side, rng):
    u = pokepy_state_to_universal(
        lane.state, gd, mp, format_str=lane.battle_format, player_side=side
    )
    legal = legal_action_indices(action_space, u, lane.state, gd, mp, player_side=side)
    if not legal:
        return 0, False
    ua = action_space.agent_output_to_action(u, int(rng.choice(legal)))
    try:
        return universal_action_to_pokepy(ua, u, lane.state, mp, player_side=side)
    except (ValueError, IndexError):
        return 0, False


def compare(d0: dict, d1: dict):
    """d0 = side0 perspective dict, d1 = side1 perspective dict.

    Mirror property: d0['player_*'] == d1['opponent_*'] and
    d0['opponent_*'] == d1['player_*']. Returns list of mismatched keys.
    """
    mismatches = []
    for k0, v0 in d0.items():
        if k0.startswith("player_"):
            k1 = "opponent_" + k0[len("player_") :]
        elif k0.startswith("opponent_"):
            k1 = "player_" + k0[len("opponent_") :]
        elif k0 in ("battle_won", "battle_lost"):
            # these intentionally flip; check d0.won == d1.lost
            continue
        else:
            # symmetric/global fields: must be equal across perspectives
            k1 = k0
        if k1 not in d1:
            mismatches.append((k0, "MISSING_IN_D1", k1))
            continue
        if repr(v0) != repr(d1[k1]):
            mismatches.append((k0, repr(v0)[:80], repr(d1[k1])[:80]))
    return mismatches


import copy


def main():
    gd = load_game_data()
    mp = load_id_mappings()
    me = load_move_effect_data()
    team_set = get_metamon_teams("gen9ou", "competitive")
    rng = _random.Random(7)

    # IDENTICAL teams on both sides at turn 0 -> perfectly symmetric position.
    # Every field of the side-0 and side-1 perspectives must then be EQUAL.
    mismatches = {}
    n_states = 0
    for trial in range(12):
        lane = _BattleLane(gd, mp, me, "gen9ou")
        t = team_set_to_pokepy_dict(team_set, mappings=mp)
        t0 = copy.deepcopy(t)
        t1 = copy.deepcopy(t)
        lane.reset(t0, t1, rng.randint(0, 2**31 - 1))
        u0 = pokepy_state_to_universal(
            lane.state, gd, mp, format_str="gen9ou", player_side=0
        )
        u1 = pokepy_state_to_universal(
            lane.state, gd, mp, format_str="gen9ou", player_side=1
        )
        d0, d1 = u0.to_dict(), u1.to_dict()
        n_states += 1
        for k in d0:
            if repr(d0[k]) != repr(d1.get(k)):
                mismatches.setdefault(k, [])
                if len(mismatches[k]) < 2:
                    mismatches[k].append((repr(d0[k])[:120], repr(d1.get(k))[:120]))

    print(f"checked {n_states} turn-0 identical-team states for side0==side1 equality")
    if not mismatches:
        print(
            "PASS: side0 and side1 perspectives are identical in a symmetric position"
        )
    else:
        print("FIELDS THAT DIFFER (should be identical in a mirror position):")
        for k, examples in mismatches.items():
            print(f"  {k}:")
            for a, b in examples:
                print(f"      side0={a}")
                print(f"      side1={b}")

    # ---- action mapping + legal mask symmetry in a symmetric position ----
    from metamon.env.pokepy_battle.action_adapter import (
        build_illegal_actions_mask,
        legal_action_indices as _legal,
        universal_action_to_pokepy as _u2p,
    )

    action_space = DefaultActionSpace()
    action_mismatch = {}
    n_act = 0
    for trial in range(12):
        lane = _BattleLane(gd, mp, me, "gen9ou")
        t = team_set_to_pokepy_dict(team_set, mappings=mp)
        lane.reset(copy.deepcopy(t), copy.deepcopy(t), rng.randint(0, 2**31 - 1))
        u0 = pokepy_state_to_universal(
            lane.state, gd, mp, format_str="gen9ou", player_side=0
        )
        u1 = pokepy_state_to_universal(
            lane.state, gd, mp, format_str="gen9ou", player_side=1
        )
        n_act += 1
        legal0 = _legal(action_space, u0, lane.state, gd, mp, player_side=0)
        legal1 = _legal(action_space, u1, lane.state, gd, mp, player_side=1)
        if legal0 != legal1:
            action_mismatch.setdefault("legal_set", [])
            if len(action_mismatch["legal_set"]) < 3:
                action_mismatch["legal_set"].append((legal0, legal1))
        for idx in range(action_space.gym_space.n):
            a0 = action_space.agent_output_to_action(u0, idx)
            a1 = action_space.agent_output_to_action(u1, idx)
            try:
                p0 = _u2p(a0, u0, lane.state, mp, player_side=0)
            except Exception as e:
                p0 = ("ERR", str(e))
            try:
                p1 = _u2p(a1, u1, lane.state, mp, player_side=1)
            except Exception as e:
                p1 = ("ERR", str(e))
            if repr(p0) != repr(p1):
                action_mismatch.setdefault(f"idx{idx}", [])
                if len(action_mismatch[f"idx{idx}"]) < 2:
                    action_mismatch[f"idx{idx}"].append((repr(p0), repr(p1)))

    print(f"\nchecked {n_act} symmetric states for action-mapping symmetry")
    if not action_mismatch:
        print(
            "PASS: legal sets and action->pokepy mappings identical for side0 and side1"
        )
    else:
        print("ACTION-MAPPING DIFFERENCES (side0 vs side1):")
        for k, examples in action_mismatch.items():
            print(f"  {k}:")
            for a, b in examples:
                print(f"      side0={a}")
                print(f"      side1={b}")


if __name__ == "__main__":
    main()
