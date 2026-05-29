"""Throwaway diagnostic: random-legal-action self-play to isolate engine side bias.

Both sides pick uniformly at random among their *legal* actions (computed by the
exact same adapter code the env uses). This removes all policy/observation
quality from the equation, so any win-rate gap between physical side 0 and side 1
is attributable to the engine (or the legal-action computation), not the NN obs.
"""

import os
import random
from pathlib import Path

POKEPY_ENGINE = Path(__file__).resolve().parents[1] / "pokepy-engine"
os.environ.setdefault(
    "POKEPY_DATA_PATH", str(POKEPY_ENGINE / "pokepy" / "data" / "extracted")
)

import numpy as np

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


def random_legal_pokepy_action(action_space, lane, game_data, mappings, side, rng):
    universal = pokepy_state_to_universal(
        lane.state, game_data, mappings, format_str=lane.battle_format, player_side=side
    )
    legal = legal_action_indices(
        action_space, universal, lane.state, game_data, mappings, player_side=side
    )
    if not legal:
        return 0, False
    idx = rng.choice(legal)
    ua = action_space.agent_output_to_action(universal, int(idx))
    try:
        return universal_action_to_pokepy(
            ua, universal, lane.state, mappings, player_side=side
        )
    except (ValueError, IndexError):
        return 0, False


def main():
    n_battles = int(os.environ.get("DIAG_N", "300"))
    turn_limit = 200
    battle_format = "gen9ou"

    game_data = load_game_data()
    mappings = load_id_mappings()
    move_effects = load_move_effect_data()
    action_space = DefaultActionSpace()
    team_set = get_metamon_teams(battle_format, "competitive")
    team_set_opp = team_set  # independent yields per battle anyway
    rng = random.Random(0)

    wins = {0: 0, 1: 0, "draw": 0, "timeout": 0}
    for b in range(n_battles):
        lane = _BattleLane(game_data, mappings, move_effects, battle_format)
        team0 = team_set_to_pokepy_dict(team_set, mappings=mappings)
        team1 = team_set_to_pokepy_dict(team_set_opp, mappings=mappings)
        seed = rng.randint(0, 2**31 - 1)
        lane.reset(team0, team1, seed)
        done = False
        hit_limit = False
        while not done and not hit_limit:
            a0, t0 = random_legal_pokepy_action(
                action_space, lane, game_data, mappings, 0, rng
            )
            a1, t1 = random_legal_pokepy_action(
                action_space, lane, game_data, mappings, 1, rng
            )
            done, _ = lane.step(a0, a1, tera0=t0, tera1=t1)
            hit_limit = lane.turn_counter > turn_limit
        w = int(lane.state.winner)
        if hit_limit and w < 0:
            wins["timeout"] += 1
        elif w == 0:
            wins[0] += 1
        elif w == 1:
            wins[1] += 1
        else:
            wins["draw"] += 1
        if (b + 1) % 25 == 0:
            tot = b + 1
            decided = wins[0] + wins[1]
            s0 = wins[0] / decided if decided else float("nan")
            print(
                f"n={tot} side0={wins[0]} side1={wins[1]} draw={wins['draw']} "
                f"timeout={wins['timeout']} side0_share_of_decided={s0:.3f}",
                flush=True,
            )

    tot = n_battles
    decided = wins[0] + wins[1]
    print("=" * 60)
    print(f"FINAL n={tot} {wins}")
    if decided:
        print(f"side0 share of decided battles = {wins[0]/decided:.4f}")


if __name__ == "__main__":
    main()
