"""Decisive symmetric self-play: drive BOTH physical sides with the exact same
inference code (no AMAGO on one side, manual on the other). Identical obs space
handling, per-side hidden state, per-side rl2, per-side time_idx.

If side 0 still wins ~60%, the asymmetry is intrinsic to the engine or the
state->universal conversion (side-dependent). If ~50%, the asymmetry lives in
the eval harness (AMAGO eval-agent path vs our opponent path).
"""

import argparse
import copy
import random

import numpy as np
import torch

from metamon.env import get_metamon_teams
from metamon.env.pokepy_battle.vector_env import _BattleLane
from metamon.env.pokepy_battle.team_adapter import team_set_to_pokepy_dict
from metamon.env.pokepy_battle.state_adapter import pokepy_state_to_universal
from metamon.env.pokepy_battle.action_adapter import (
    build_illegal_actions_mask,
    universal_action_to_pokepy,
)
from metamon.rl.pretrained import Kakuna
from pokepy.data.loader import (
    load_game_data,
    load_id_mappings,
    load_move_effect_data,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battles", type=int, default=128)
    ap.add_argument("--turn_limit", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--identical_teams", action="store_true")
    args = ap.parse_args()

    gd = load_game_data()
    mp = load_id_mappings()
    me = load_move_effect_data()
    team_set = get_metamon_teams("gen9ou", "competitive")

    model = Kakuna()
    agent = model.initialize_agent(log=False)
    policy = agent.policy.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    obs_space = model.observation_space
    action_space = model.action_space
    reward_fn = model.reward_function
    n_actions = action_space.gym_space.n

    rng = random.Random(args.seed)

    def obs_to_torch(obs):
        return {
            k: torch.from_numpy(v).to(device).unsqueeze(0).unsqueeze(1)
            for k, v in obs.items()
        }

    def fresh_rl2():
        return torch.zeros(1, 1, n_actions + 1, device=device)

    wins = {0: 0, 1: 0, -1: 0}
    for battle in range(args.battles):
        lane = _BattleLane(gd, mp, me, "gen9ou")
        t0 = team_set_to_pokepy_dict(team_set, mappings=mp)
        if args.identical_teams:
            t1 = copy.deepcopy(t0)
        else:
            t1 = team_set_to_pokepy_dict(team_set, mappings=mp)
        lane.reset(t0, t1, rng.randint(0, 2**31 - 1))

        hs = {s: policy.traj_encoder.init_hidden_state(1, device) for s in (0, 1)}
        os_ = {s: copy.deepcopy(obs_space) for s in (0, 1)}
        os_[0].reset()
        os_[1].reset()
        rl2 = {s: fresh_rl2() for s in (0, 1)}
        steps = {s: 0 for s in (0, 1)}

        while True:
            prev_u = {}
            poke = {}
            act_idx = {}
            for side in (0, 1):
                u = pokepy_state_to_universal(
                    lane.state, gd, mp, format_str="gen9ou", player_side=side
                )
                prev_u[side] = u
                obs = os_[side].state_to_obs(u)
                obs["illegal_actions"] = build_illegal_actions_mask(
                    action_space, u, lane.state, gd, mp, player_side=side
                )
                ti = torch.tensor([[[steps[side]]]], device=device, dtype=torch.long)
                with torch.no_grad():
                    a, hs[side] = policy.get_actions(
                        obs=obs_to_torch(obs),
                        rl2s=rl2[side],
                        time_idxs=ti,
                        hidden_state=hs[side],
                        sample=True,
                    )
                idx = int(a.squeeze().cpu().item())
                act_idx[side] = idx
                ua = action_space.agent_output_to_action(u, idx)
                try:
                    poke[side] = universal_action_to_pokepy(
                        ua, u, lane.state, mp, player_side=side
                    )
                except (ValueError, IndexError):
                    poke[side] = (0, False)
                steps[side] += 1

            done, _ = lane.step(
                poke[0][0], poke[1][0], tera0=poke[0][1], tera1=poke[1][1]
            )

            for side in (0, 1):
                new_u = pokepy_state_to_universal(
                    lane.state, gd, mp, format_str="gen9ou", player_side=side
                )
                r = float(reward_fn(prev_u[side], new_u))
                rl2[side] = fresh_rl2()
                rl2[side][0, 0, 0] = r
                if 0 <= act_idx[side] < n_actions:
                    rl2[side][0, 0, 1 + act_idx[side]] = 1.0

            if done or lane.turn_counter > args.turn_limit:
                w = int(lane.state.winner)
                wins[w if w in wins else -1] += 1
                break

        if (battle + 1) % 16 == 0:
            dec = wins[0] + wins[1]
            share = wins[0] / dec if dec else float("nan")
            print(
                f"[{battle + 1}/{args.battles}] side0={wins[0]} side1={wins[1]} "
                f"draw={wins[-1]} side0_share={share:.3f}",
                flush=True,
            )

    dec = wins[0] + wins[1]
    print("FINAL", wins)
    if dec:
        print(f"side0 share of decided = {wins[0] / dec:.4f}")


if __name__ == "__main__":
    main()
