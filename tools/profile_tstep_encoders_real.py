#!/usr/bin/env python
import argparse
import os
import sys
import time
from typing import Dict, List

# # Allow running this script without installing the package.
# REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if REPO_ROOT not in sys.path:
#     sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import gymnasium as gym

from metamon.interface import (
    get_observation_space,
    TokenizedObservationSpace,
    DefaultActionSpace,
    DefaultShapedReward,
)
from metamon.tokenizer import get_tokenizer
from metamon.data import ParsedReplayDataset
from metamon.rl.metamon_to_amago import (
    MetamonAMAGODataset,
    MetamonTstepEncoder,
    MetamonPerceiverTstepEncoder,
    MetamonGroupedTstepEncoder,
)
from amago.loading import RLData_pad_collate


def _make_rl2_space(action_space: gym.Space) -> gym.spaces.Box:
    if isinstance(action_space, gym.spaces.Discrete):
        action_shape = action_space.n
    else:
        action_shape = action_space.shape[-1]
    return gym.spaces.Box(
        shape=(action_shape + 1,),
        dtype=np.float32,
        low=float("-inf"),
        high=float("inf"),
    )


def _sample_batch(dset: MetamonAMAGODataset, batch_size: int, seq_len: int):
    samples = []
    for _ in range(batch_size):
        rl_data = dset.sample_random_trajectory()
        rl_data = rl_data.random_slice(length=seq_len - 1)
        samples.append(rl_data)
    return RLData_pad_collate(samples)


def _profile_encoder(
    name: str,
    encoder: torch.nn.Module,
    batch,
    device: torch.device,
    steps: int,
):
    encoder.train()
    batch = batch.to(device)

    # warmup
    for _ in range(3):
        emb = encoder(obs=batch.obs, rl2s=batch.rl2s)
        loss = emb.mean()
        encoder.zero_grad(set_to_none=True)
        loss.backward()

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    timings_ms: List[float] = []
    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=steps, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        total_steps = 1 + 1 + steps
        for _ in range(total_steps):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            emb = encoder(obs=batch.obs, rl2s=batch.rl2s)
            loss = emb.mean()
            encoder.zero_grad(set_to_none=True)
            loss.backward()

            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            timings_ms.append((end - start) * 1000.0)
            prof.step()

    # drop wait + warmup from timings
    timings_ms = timings_ms[2:]
    return prof, timings_ms


def _timing_stats(timings_ms: List[float]) -> Dict[str, float]:
    arr = np.array(timings_ms, dtype=np.float32)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Profile Metamon timestep encoders on real replay inputs."
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--format", type=str, default="gen9ou")
    parser.add_argument("--dset-root", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--profile-steps", type=int, default=3)
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = get_tokenizer("DefaultObservationSpace-v1")
    action_space = DefaultActionSpace()
    rl2_space = _make_rl2_space(action_space.gym_space)

    def make_dset(obs_space_name: str):
        obs_space = get_observation_space(obs_space_name)
        tok_obs_space = TokenizedObservationSpace(obs_space, tokenizer)
        parsed = ParsedReplayDataset(
            observation_space=tok_obs_space,
            action_space=action_space,
            reward_function=DefaultShapedReward(),
            dset_root=args.dset_root,
            formats=[args.format],
            max_seq_len=args.seq_len - 1,
            shuffle=True,
            use_cached_filenames=True,
        )
        return MetamonAMAGODataset(parsed_replay_dset=parsed)

    encoders = []
    # flat encoders
    flat_dset = make_dset("DefaultObservationSpace")
    encoders.append(
        (
            "MetamonTstepEncoder",
            MetamonTstepEncoder(flat_dset.parsed_replay_dset.observation_space.gym_space, rl2_space, tokenizer),
            flat_dset,
        )
    )
    encoders.append(
        (
            "MetamonPerceiverTstepEncoder",
            MetamonPerceiverTstepEncoder(flat_dset.parsed_replay_dset.observation_space.gym_space, rl2_space, tokenizer),
            flat_dset,
        )
    )

    # grouped encoder
    grouped_dset = make_dset("GroupedObservationSpace")
    encoders.append(
        (
            "MetamonGroupedTstepEncoder",
            MetamonGroupedTstepEncoder(grouped_dset.parsed_replay_dset.observation_space.gym_space, rl2_space, tokenizer),
            grouped_dset,
        )
    )

    use_wandb = args.wandb_project is not None and args.wandb_entity is not None
    if use_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=vars(args),
        )

    for name, encoder, dset in encoders:
        batch = _sample_batch(dset, args.batch_size, args.seq_len)
        if batch.rl2s is None:
            raise RuntimeError("RL2 inputs were not generated. This should not happen.")

        prof, timings_ms = _profile_encoder(
            name=name,
            encoder=encoder.to(device),
            batch=batch,
            device=device,
            steps=args.profile_steps,
        )
        stats = _timing_stats(timings_ms)

        profile_dir = os.path.join(
            os.getcwd(), "profiling", f"{name}_{int(time.time())}"
        )
        os.makedirs(profile_dir, exist_ok=True)
        trace_path = os.path.join(profile_dir, "trace.json")
        prof.export_chrome_trace(trace_path)

        print(f"{name} timing (ms): {stats}")
        print(f"{name} trace: {trace_path}")

        if use_wandb:
            import wandb

            wandb.log({
                f"profile/{name}/mean_ms": stats["mean_ms"],
                f"profile/{name}/median_ms": stats["median_ms"],
                f"profile/{name}/p95_ms": stats["p95_ms"],
            })
            art = wandb.Artifact(
                name=f"torch-profiler-{wandb.run.id}-{name}",
                type="profile",
                metadata={
                    "encoder": name,
                    "batch_size": args.batch_size,
                    "seq_len": args.seq_len,
                    "device": args.device,
                },
            )
            art.add_file(trace_path, name="trace.json")
            wandb.log_artifact(art)

    if use_wandb:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
