"""throwaway: drive battles through StreamBattleLane using legal random actions"""

import random

from sim_process import ShowdownSimProcess
from lane import (
    StreamBattleLane,
    KIND_MOVE,
    KIND_FORCESWITCH,
    KIND_TEAMPREVIEW,
)
from action_adapter import action_idx_to_choice, DEFAULT_CHOICE

from metamon.interface import DefaultActionSpace


def main():
    proc = ShowdownSimProcess()
    asp = DefaultActionSpace()
    n_lanes = 3
    lanes = {}
    for k in range(n_lanes):
        lane = StreamBattleLane(k, "gen9randombattle")
        lanes[k] = lane
        proc.register_lane(k, lane)
        proc.start_battle(k, "gen9randombattle", p1={"name": "p1"}, p2={"name": "p2"})

    def all_ready():
        return all(l.decision_ready() for l in lanes.values())

    proc.pump_until(all_ready, timeout=20.0)

    finished = 0
    for step in range(400):
        active = [k for k, l in lanes.items() if not l.ended]
        if not active:
            break
        for k in active:
            lane = lanes[k]
            for side in ("p1", "p2"):
                kind = lane.request_kind(side)
                if kind in (KIND_MOVE, KIND_FORCESWITCH):
                    state = lane.universal_state(side)
                    legal = lane.legal_action_indices(side, asp, state)
                    battle = lane.battle(side)
                    choice = None
                    if legal:
                        idx = random.choice(legal)
                        choice = action_idx_to_choice(idx, battle)
                    proc.choose(k, side, choice or DEFAULT_CHOICE)
                elif kind == KIND_TEAMPREVIEW:
                    proc.choose(k, side, DEFAULT_CHOICE)
            lane.mark_settled()

        proc.pump_until(all_ready, timeout=20.0)

    for k, l in lanes.items():
        s = l.universal_state("p1")
        print(
            f"lane {k}: ended={l.ended} winner={l.winner} p1_won={s.battle_won} p1_lost={s.battle_lost}"
        )
        if l.ended:
            finished += 1
    print(f"finished {finished}/{n_lanes} lanes")
    proc.close()


if __name__ == "__main__":
    main()
