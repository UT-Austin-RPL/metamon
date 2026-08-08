"""Compute per-turn model value estimates for squirtle ladder trajectories.

Usage: python compute_evals.py [--limit N] [--out CACHE.parquet]
"""

import argparse
import json
import os

import numpy as np
import torch

os.environ.setdefault("METAMON_CACHE_DIR", os.path.expanduser("~/metamon_cache"))

from amago.loading import Batch

from metamon.data.parsed_replay_dset import MetamonDataset
from metamon.rl.metamon_to_amago import MetamonAMAGODataset
from metamon.rl.pretrained import get_pretrained_model

TRAJ_ROOT = os.path.expanduser("~/metamon/trajectories/squirtle")
FMT = "gen1ou"


def build_batch_from_rldata(rldata, device):
    batch = Batch(
        obs={k: v.unsqueeze(0).to(device) for k, v in rldata.obs.items()},
        rl2s=rldata.rl2s.unsqueeze(0).to(device),
        rews=rldata.rews.unsqueeze(0).to(device),
        dones=rldata.dones.unsqueeze(0).to(device),
        actions=rldata.actions.unsqueeze(0).to(device),
        time_idxs=rldata.time_idxs.unsqueeze(0).to(device),
    )
    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    agent_maker = get_pretrained_model("squirtle")
    print("loading squirtle agent (latest checkpoint)...")
    experiment = agent_maker.initialize_agent(checkpoint=-1, log=False)
    agent = experiment.policy.to(device)
    agent.eval()
    print("agent loaded")

    dset = MetamonDataset(
        dset_root=TRAJ_ROOT,
        observation_space=agent_maker.observation_space,
        action_space=agent_maker.action_space,
        reward_function=agent_maker.reward_function,
        formats=[FMT],
        verbose=False,
        write_index_cache=False,
    )
    print("n files:", len(dset))
    amago_dset = MetamonAMAGODataset(dset)

    files = dset.filenames
    if args.limit:
        files = files[: args.limit]

    results = []
    with torch.no_grad():
        for i, path in enumerate(files):
            try:
                raw = dset.load_filename(path)
                rldata = amago_dset._process_data(raw)
                batch = build_batch_from_rldata(rldata, device)
                vals = agent.get_values(batch)
                v_s = (
                    vals["v_s"].squeeze(0).squeeze(-1).squeeze(-1)
                )  # (T-1, G) -> take?
                # From MultiTaskAgent get_values: v_s (B, L-1, G, 1); gamma dim G — use single gamma?
                q_sa = vals["q_sa"].squeeze(0).squeeze(-1).squeeze(-1)
                adv = vals["advantage"].squeeze(0).squeeze(-1).squeeze(-1)
                results.append(
                    {
                        "file": os.path.basename(path),
                        "v_s": v_s.cpu().numpy(),
                        "q_sa": q_sa.cpu().numpy(),
                        "advantage": adv.cpu().numpy(),
                    }
                )
                if i % 20 == 0:
                    print(
                        f"  {i}: {os.path.basename(path)[:60]} len={len(v_s)} v_s0={v_s[0].item():.3f}"
                    )
            except Exception as e:
                print(f"  FAIL {os.path.basename(path)}: {type(e).__name__}: {e}")

    print(f"done: {len(results)}/{len(files)}")
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        np.savez_compressed(
            args.out, **{f"r{i}": r["v_s"] for i, r in enumerate(results)}
        )
        with open(args.out + ".meta.json", "w") as f:
            json.dump(
                [
                    {
                        k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in r.items()
                        if k != "v_s" and k != "q_sa" and k != "advantage"
                    }
                    for r in results
                ],
                f,
            )
    print(results[0]["v_s"][:10])


if __name__ == "__main__":
    main()
