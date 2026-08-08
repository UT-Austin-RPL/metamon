"""Build a cache of per-turn model evals + metadata for squirtle ladder trajectories.

Writes:
  {out}/battles.parquet        one row per battle (metadata)
  {out}/turns.parquet          one row per turn (eval curve, pokemon remaining, etc.)
  {out}/values.npz             raw per-turn v_s/q_sa/advantage matrices (B, T, G)
  {out}/meta.json              build info

Usage: python build_cache.py --out CACHE_DIR [--limit N] [--device cuda|cpu]
"""

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("METAMON_CACHE_DIR", os.path.expanduser("~/metamon_cache"))

from amago.loading import Batch
from metamon.data.parsed_replay_dset import MetamonDataset
from metamon.rl.metamon_to_amago import MetamonAMAGODataset
from metamon.rl.pretrained import get_pretrained_model

TRAJ_ROOT = os.path.expanduser("~/metamon/trajectories/squirtle")
FMT = "gen1ou"
SMOG_TEAM_DIR = os.path.expanduser("~/metamon_cache/teams/smog_ladder/gen1ou")

GAMMA_MAIN = 6  # gamma = 0.999


def norm_move(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def parse_team_file(path):
    """Return dict species -> set(moves) for a showdown team file."""
    content = open(path).read()
    mon = None
    team = {}
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if " - " in line or line.startswith("- "):
            move = line.lstrip("- ").strip()
            if mon is not None:
                team[mon].add(norm_move(move))
        elif not line.startswith(
            ("EVs:", "IVs:", "Nature", "Ability", "Item", "Level")
        ):
            m = re.match(r"^([A-Za-z][A-Za-z0-9'\- ]*?)(?:\s*@.*)?$", line)
            if m:
                mon = m.group(1).strip().lower()
                team[mon] = set()
    return team


def load_smog_teams():
    teams = {}
    for f in sorted(os.listdir(SMOG_TEAM_DIR)):
        if f.endswith("_team"):
            teams[f.replace(".gen1ou_team", "")] = parse_team_file(
                os.path.join(SMOG_TEAM_DIR, f)
            )
    return teams


def match_team(player_roster, smog_teams):
    """player_roster: dict species -> set(moves). Return team name or None."""
    # species-only match first
    sp_set = set(player_roster)
    cands = []
    for name, t in smog_teams.items():
        if set(t) == sp_set:
            cands.append(name)
    if len(cands) == 1:
        return cands[0]
    if len(cands) > 1:
        # disambiguate by per-species move sets
        best, best_score = None, -1
        for name in cands:
            t = smog_teams[name]
            score = 0
            for sp, moves in player_roster.items():
                if sp in t:
                    score += len(moves & t[sp])
            if score > best_score:
                best, best_score = name, score
        return best
    return None


def collect_roster(state_dict):
    """species -> set(moves) from a saved state dict (player's active + switch mons)."""
    roster = {}
    active = state_dict["player_active_pokemon"]
    roster[active["base_species"]] = {norm_move(m["name"]) for m in active["moves"]}
    for sw in state_dict["available_switches"]:
        roster[sw["base_species"]] = {norm_move(m["name"]) for m in sw["moves"]}
    return roster


def player_remaining(state_dict, original_species):
    alive = {state_dict["player_active_pokemon"]["base_species"]}
    alive.update(s["base_species"] for s in state_dict["available_switches"])
    return len(original_species & alive)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=os.path.expanduser("~/metamon/trajectories/squirtle/eval_cache"),
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    agent_maker = get_pretrained_model("squirtle")
    print("loading squirtle agent (latest checkpoint)...")
    experiment = agent_maker.initialize_agent(checkpoint=-1, log=False)
    agent = experiment.policy.to(device)
    agent.eval()

    dset = MetamonDataset(
        dset_root=TRAJ_ROOT,
        observation_space=agent_maker.observation_space,
        action_space=agent_maker.action_space,
        reward_function=agent_maker.reward_function,
        formats=[FMT],
        verbose=False,
        write_index_cache=False,
    )
    amago_dset = MetamonAMAGODataset(dset)
    smog_teams = load_smog_teams()
    print("smog teams:", list(smog_teams))

    files = dset.filenames
    if args.limit:
        files = files[: args.limit]

    battle_rows = []
    turn_rows = []
    values = {"v_s": [], "q_sa": [], "advantage": []}
    n_fail = 0

    with torch.no_grad():
        for i, path in enumerate(files):
            base = os.path.basename(path)
            try:
                raw = dset.load_filename(path)
                rldata = amago_dset._process_data(raw)
                batch = Batch(
                    obs={k: v.unsqueeze(0).to(device) for k, v in rldata.obs.items()},
                    rl2s=rldata.rl2s.unsqueeze(0).to(device),
                    rews=rldata.rews.unsqueeze(0).to(device),
                    dones=rldata.dones.unsqueeze(0).to(device),
                    actions=rldata.actions.unsqueeze(0).to(device),
                    time_idxs=rldata.time_idxs.unsqueeze(0).to(device),
                )
                vals = agent.get_values(batch)
                # v_s: (1, T-1, G, 1)
                v_s = vals["v_s"].cpu().numpy()[0, :, :, 0]  # (T-1, G)
                q_sa = vals["q_sa"].cpu().numpy()[0, :, :, 0]  # (T-1, G)
                adv = vals["advantage"].cpu().numpy()[0, :, :, 0]  # (T-1, G)

                # Load raw states for turn metadata
                data = dset._load_json(path)
                states = data["states"]
                actions = data["actions"]  # last is -1
                roster0 = collect_roster(states[0])
                team = match_team(roster0, smog_teams)
                orig_species = set(roster0)

                pat = re.compile(
                    r"_vs_(?P<opp>.+?)_\d{2}-\d{2}-\d{4}-\d{2}:\d{2}:\d{2}(?:_ts-[A-Za-z0-9_.]+)?_(?P<result>WIN|LOSS)\.json\.lz4$"
                )
                m = pat.search(base)
                opponent = m.group("opp") if m else "unknown"
                result = m.group("result") if m else "?"

                T = len(v_s)  # real turns
                p_remaining = [player_remaining(s, orig_species) for s in states[:T]]
                o_remaining = [s["opponents_remaining"] for s in states[:T]]
                active_sp = [
                    s["player_active_pokemon"]["base_species"] for s in states[:T]
                ]
                opp_sp = [
                    s["opponent_active_pokemon"]["base_species"] for s in states[:T]
                ]
                hp = [s["player_active_pokemon"]["hp_pct"] for s in states[:T]]

                battle_rows.append(
                    {
                        "file": base,
                        "battle_id": base.split("-")[2],
                        "opponent": opponent,
                        "result": result,
                        "team": team,
                        "roster": ",".join(sorted(orig_species)),
                        "n_turns": int(T),
                        "v0": float(v_s[0, GAMMA_MAIN]),
                        "v_final": float(v_s[-1, GAMMA_MAIN]),
                        "main_gamma": float(agent.gammas[GAMMA_MAIN].item()),
                    }
                )
                values["v_s"].append(v_s)
                values["q_sa"].append(q_sa)
                values["advantage"].append(adv)

                for t in range(T):
                    turn_rows.append(
                        {
                            "file": base,
                            "opponent": opponent,
                            "result": result,
                            "team": team,
                            "turn": t,
                            "v_main": float(v_s[t, GAMMA_MAIN]),
                            "v_mean": float(v_s[t].mean()),
                            "q_sa_main": float(q_sa[t, GAMMA_MAIN]),
                            "adv_main": float(adv[t, GAMMA_MAIN]),
                            "player_remaining": (
                                p_remaining[t] if t < len(p_remaining) else None
                            ),
                            "opp_remaining": (
                                o_remaining[t] if t < len(o_remaining) else None
                            ),
                            "active": active_sp[t] if t < len(active_sp) else None,
                            "opp_active": opp_sp[t] if t < len(opp_sp) else None,
                            "hp_pct": hp[t] if t < len(hp) else None,
                            "action": int(actions[t]) if t < len(actions) else None,
                        }
                    )
            except Exception as e:
                n_fail += 1
                print(f"  FAIL {base}: {type(e).__name__}: {e}")
            if (i + 1) % 50 == 0:
                print(f"  processed {i+1}/{len(files)}")

    os.makedirs(args.out, exist_ok=True)
    battles = pd.DataFrame(battle_rows)
    turns = pd.DataFrame(turn_rows)
    battles.to_parquet(os.path.join(args.out, "battles.parquet"))
    turns.to_parquet(os.path.join(args.out, "turns.parquet"))
    maxT = max(len(v) for v in values["v_s"]) if values["v_s"] else 0
    G = values["v_s"][0].shape[1] if values["v_s"] else 0
    pad = lambda arr: np.stack([np.pad(a, ((0, maxT - len(a)), (0, 0))) for a in arr])
    np.savez_compressed(
        os.path.join(args.out, "values.npz"),
        v_s=pad(values["v_s"]),
        q_sa=pad(values["q_sa"]),
        advantage=pad(values["advantage"]),
        lengths=np.array([len(v) for v in values["v_s"]]),
        gammas=agent.gammas.cpu().numpy(),
        main_gamma_idx=GAMMA_MAIN,
    )
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(
            {
                "traj_root": TRAJ_ROOT,
                "format": FMT,
                "n_battles": len(battles),
                "n_fail": n_fail,
                "max_turns": maxT,
                "n_gammas": G,
                "main_gamma_idx": GAMMA_MAIN,
                "model": "squirtle (mini_online_smogon_v0 latest)",
            },
            f,
            indent=2,
        )
    print(f"wrote {len(battles)} battles, {len(turns)} turns to {args.out}")
    print(battles.groupby(["team", "result"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
