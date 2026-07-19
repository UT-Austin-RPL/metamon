#!/usr/bin/env python3
"""Consolidate the SmallG1Online online-RL FIFO buffers into one self-play subset.

The online RL runs (metamon.rl.online_rl) each wrote the trainee's eval-side
trajectories into a per-run FIFO buffer of ~300k ``gen1ou/*.json.lz4`` files.
This script packages all four production lineages into a single ``smallg1-online``
self-play subset that can be referenced from a dataset config's ``self_play:``
block (exactly like ``pac-tauros``) and mixed into the OFFLINE weight of a future
online run.

Lineages consolidated (all gen1ou, ~1.22M trajectories total):
  - V0        mini_online_v1            (from-scratch, gl_05_26 opponents)
  - V1 / V1_5 mini_online_g1_expert_v2   (expert teams)
  - V2        smallg1online_v2          (expert_curated 50-team)
  - V3        smallg1online_v3          (expert_curated 44-team)

Output layout (mirrors pac-tauros so the loader treats it identically):
  {METAMON_CACHE_DIR}/self-play/smallg1-online/
    gen1ou.tar                 # members named gen1ou/<battle>.json.lz4
    gen1ou.tar.index.sqlite    # ratarmount index (built on first load / --index)
    gen1ou.tar.index.txt       # cached member-name list (built on first load)

Why a tar: the four buffers hold ~1.22M tiny files. A flat directory of that many
inodes is brutal on the NFS metadata server; the pac-* subsets all ship as a
single tar + sqlite index for O(1) random access, so we match that.

Local disk is typically full on this box, so the tar is written directly to NFS
(the cache dir) and the source files are only READ -- nothing is staged locally.

Throughput: packaging is dominated by small-file open/read latency (V0 lives on
NFS; ~0.1 MB/s single-threaded => ~9h). This is latency-bound, not
bandwidth-bound, so we PARALLELIZE: each buffer's file list is split into shards,
shards are tarred concurrently into part files, and the parts are concatenated
into the final ``gen1ou.tar``. With --jobs 16 this drops to well under an hour.

Usage:
  # Build the tar in parallel (default). Refuses to clobber unless --force.
  python -m scripts.build_smallg1_online_dataset --jobs 16

  # Build tar, then eagerly build the sqlite index and verify N random loads.
  python -m scripts.build_smallg1_online_dataset --jobs 16 --index --verify 25

  # Just (re)build the index / verify an already-built tar.
  python -m scripts.build_smallg1_online_dataset --skip-tar --index --verify 25
"""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from metamon.config import METAMON_CACHE_DIR

SUBSET = "smallg1-online"
FORMAT = "gen1ou"
FILES_PER_SHARD = 20_000  # ~15 shards for a 300k-file buffer


@dataclass(frozen=True)
class Source:
    label: str  # human-facing lineage label
    models: str  # which registered pretrained models this buffer produced
    parent: Path  # dir that CONTAINS the ``gen1ou/`` buffer subdir


# NOTE: `parent` must contain a ``gen1ou/`` subdir; we tar that subdir so archive
# members are named ``gen1ou/<battle>.json.lz4``. V0 lives on NFS; V1-V3 on local.
SOURCES: list[Source] = [
    Source(
        "V0",
        "SmallG1OnlineV0",
        Path("/mnt/nfs_client/jake/metamon_scratchpad/mini_online_v1_buffer"),
    ),
    Source(
        "V1/V1_5",
        "SmallG1OnlineV1, SmallG1OnlineV1_5",
        Path("/home/jake/metamon_local/mini_online_g1_expert_v2_buffer"),
    ),
    Source(
        "V2",
        "SmallG1OnlineV2",
        Path("/home/jake/metamon_local/smallg1online_v2_buffer"),
    ),
    Source(
        "V3",
        "SmallG1OnlineV3",
        Path("/home/jake/metamon_local/smallg1online_v3_buffer"),
    ),
]


def _fmt_count(n: int) -> str:
    return f"{n:,}"


def _list_traj(fmt_dir: Path) -> list[str]:
    """List *.json.lz4 basenames in a buffer's gen1ou dir (excludes *.tmp)."""
    names = []
    with os.scandir(fmt_dir) as it:
        for e in it:
            if e.name.endswith(".json.lz4"):
                names.append(e.name)
    return names


def preflight(only_labels: str | None) -> Path:
    if METAMON_CACHE_DIR is None:
        sys.exit("METAMON_CACHE_DIR is not set; cannot resolve the cache location.")
    out_dir = Path(METAMON_CACHE_DIR) / "self-play" / SUBSET
    missing = []
    for src in _select_sources(only_labels):
        fmt_dir = src.parent / FORMAT
        if not fmt_dir.is_dir():
            missing.append(str(fmt_dir))
    if missing:
        sys.exit("Missing source buffer dirs:\n  " + "\n  ".join(missing))
    return out_dir


@dataclass
class _Shard:
    src: Source
    idx: int  # shard index within source
    list_path: Path  # file listing gen1ou/<name> members (relative to src.parent)
    part_path: Path  # output part tar
    count: int


def _plan_shards(
    work_dir: Path, sources: list[Source]
) -> tuple[list[_Shard], int]:
    """Scan the given sources, write per-shard file lists, return the plan."""
    shards: list[_Shard] = []
    grand_total = 0
    for src in sources:
        fmt_dir = src.parent / FORMAT
        names = _list_traj(fmt_dir)
        grand_total += len(names)
        n_shards = max(1, math.ceil(len(names) / FILES_PER_SHARD))
        # even split
        per = math.ceil(len(names) / n_shards)
        print(
            f"  {src.label:<8} {_fmt_count(len(names)):>10} files "
            f"-> {n_shards} shard(s) ({src.models})"
        )
        for i in range(n_shards):
            chunk = names[i * per : (i + 1) * per]
            if not chunk:
                continue
            safe = src.label.replace("/", "-")
            list_path = work_dir / f"list_{safe}_{i:03d}.txt"
            with open(list_path, "w") as f:
                for name in chunk:
                    # paths relative to src.parent so members are gen1ou/<name>
                    f.write(f"{FORMAT}/{name}\n")
            part_path = work_dir / f"part_{safe}_{i:03d}.tar"
            shards.append(_Shard(src, i, list_path, part_path, len(chunk)))
    return shards, grand_total


def _tar_shard(shard: _Shard) -> tuple[_Shard, float]:
    t0 = time.time()
    cmd = [
        "tar",
        "-C",
        str(shard.src.parent),
        "-cf",
        str(shard.part_path),
        "-T",
        str(shard.list_path),
    ]
    subprocess.run(cmd, check=True)
    return shard, time.time() - t0


def _select_sources(only_labels: str | None) -> list[Source]:
    if not only_labels:
        return list(SOURCES)
    wanted = {s.strip() for s in only_labels.split(",") if s.strip()}
    chosen = [s for s in SOURCES if s.label in wanted]
    unknown = wanted - {s.label for s in chosen}
    if unknown:
        sys.exit(
            f"Unknown --only-labels {sorted(unknown)}; valid: "
            f"{[s.label for s in SOURCES]}"
        )
    return chosen


def build_parts(out_dir: Path, jobs: int, only_labels: str | None) -> None:
    """Phase 1: tar the selected buffers into part tars in the work dir.

    Does NOT concatenate. Existing part tars for the SAME labels are cleared, but
    parts from other labels are left in place so client and NFS-side runs can
    populate ``_build_parts/`` independently before a single concat pass.
    """
    work_dir = out_dir / "_build_parts"
    work_dir.mkdir(parents=True, exist_ok=True)
    sources = _select_sources(only_labels)

    # clear stale parts/lists for just these labels
    for src in sources:
        safe = src.label.replace("/", "-")
        for p in work_dir.glob(f"part_{safe}_*.tar"):
            p.unlink()
        for p in work_dir.glob(f"list_{safe}_*.txt"):
            p.unlink()

    t0 = time.time()
    print(f"Planning shards for: {[s.label for s in sources]}")
    shards, grand_total = _plan_shards(work_dir, sources)
    print(
        f"\n{len(shards)} shards, ~{_fmt_count(grand_total)} files; "
        f"tarring with {jobs} parallel jobs...\n"
    )
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
        futures = {ex.submit(_tar_shard, s): s for s in shards}
        for fut in concurrent.futures.as_completed(futures):
            shard, dt = fut.result()
            done += 1
            safe = shard.src.label.replace("/", "-")
            print(
                f"  [{done}/{len(shards)}] {safe}_{shard.idx:03d} "
                f"({_fmt_count(shard.count)} files) in {dt:.0f}s"
            )
    # drop the list files for these labels now that their parts are built
    for src in sources:
        safe = src.label.replace("/", "-")
        for p in work_dir.glob(f"list_{safe}_*.txt"):
            p.unlink()
    print(
        f"\nBuilt {len(shards)} part tars (~{_fmt_count(grand_total)} files) "
        f"in {time.time() - t0:.0f}s -> {work_dir}"
    )


def concat_parts(out_dir: Path, force: bool) -> Path:
    """Phase 2: concatenate every ``part_*.tar`` in the work dir into the final.

    Renames the first part to be the final archive, then appends the rest with
    ``tar --concatenate`` (avoids re-copying the first). ratarmount handles the
    duplicate ``gen1ou/`` dir entries across concatenated parts.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tar_path = out_dir / f"{FORMAT}.tar"
    if tar_path.exists():
        if not force:
            sys.exit(f"{tar_path} already exists; pass --force to overwrite.")
        tar_path.unlink()
        for suffix in (".index.sqlite", ".index.txt"):
            stale = out_dir / f"{FORMAT}.tar{suffix}"
            if stale.exists():
                stale.unlink()

    work_dir = out_dir / "_build_parts"
    parts = sorted(work_dir.glob("part_*.tar"))
    if not parts:
        sys.exit(f"No part tars found in {work_dir}; run --phase parts first.")

    print(f"Concatenating {len(parts)} parts into {tar_path.name}...")
    tc = time.time()
    os.replace(parts[0], tar_path)
    rest = [str(p) for p in parts[1:]]
    BATCH = 32
    for i in range(0, len(rest), BATCH):
        subprocess.run(
            ["tar", "--concatenate", "-f", str(tar_path), *rest[i : i + BATCH]],
            check=True,
        )
    shutil.rmtree(work_dir, ignore_errors=True)
    size_gb = tar_path.stat().st_size / 1e9
    print(
        f"  concatenated in {time.time() - tc:.0f}s -> {tar_path} ({size_gb:.1f} GB)"
    )
    return tar_path


def build_tar(out_dir: Path, force: bool, jobs: int, only_labels: str | None) -> Path:
    """phase=all: build all selected parts then concatenate (single-machine)."""
    tar_path = out_dir / f"{FORMAT}.tar"
    if tar_path.exists() and not force:
        sys.exit(
            f"{tar_path} already exists. Re-run with --force, or --skip-tar to reuse."
        )
    build_parts(out_dir, jobs=jobs, only_labels=only_labels)
    return concat_parts(out_dir, force=force)


def build_index_and_verify(n_verify: int) -> None:
    """Instantiate the subset via the normal loader path.

    Opening the tar triggers ratarmount to build the sqlite index (cached), and
    the first refresh_files() writes the .txt member cache. Loading a few random
    trajectories confirms the tar is readable end-to-end under the subset name.
    """
    from metamon.data import SelfPlayDataset
    from metamon.interface import (
        DefaultShapedReward,
        DefaultActionSpace,
        TokenizedObservationSpace,
        get_observation_space,
    )
    from metamon.tokenizer import get_tokenizer

    print("\nOpening subset via SelfPlayDataset (builds sqlite index on first open)...")
    obs_space = TokenizedObservationSpace(
        get_observation_space("DefaultObservationSpace"),
        tokenizer=get_tokenizer("DefaultObservationSpace-v1"),
    )
    t0 = time.time()
    dset = SelfPlayDataset(
        subset=SUBSET,
        observation_space=obs_space,
        action_space=DefaultActionSpace(),
        reward_function=DefaultShapedReward(),
        formats=[FORMAT],
        verbose=True,
    )
    print(
        f"Indexed {_fmt_count(len(dset))} trajectories in {time.time() - t0:.0f}s"
    )

    if n_verify > 0:
        print(f"\nVerifying {n_verify} random trajectory loads...")
        ok = 0
        for _ in range(n_verify):
            try:
                obs, actions, rewards, dones = dset.random_sample()
                assert len(rewards) > 0 and dones[-1]
                ok += 1
            except Exception as e:  # noqa: BLE001
                print(f"  FAILED to load a sample: {e}")
        print(f"  {ok}/{n_verify} samples loaded cleanly")
        if ok != n_verify:
            sys.exit("Verification failed: some samples did not load.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="Rebuild tar if it exists.")
    ap.add_argument(
        "--jobs",
        type=int,
        default=16,
        help="Parallel tar workers (latency-bound; 16 is a good default).",
    )
    ap.add_argument(
        "--phase",
        choices=["all", "parts", "concat"],
        default="all",
        help=(
            "all: build parts + concatenate (single machine). "
            "parts: only build part tars into _build_parts/. "
            "concat: only concatenate existing parts into the final tar. "
            "Use parts/concat to split work across machines (e.g. build local "
            "buffers here, build the NFS-resident buffer + concat on the NFS host)."
        ),
    )
    ap.add_argument(
        "--only-labels",
        type=str,
        default=None,
        help="Comma list of lineage labels to include (e.g. 'V1/V1_5,V2,V3').",
    )
    ap.add_argument(
        "--skip-tar", action="store_true", help="Reuse existing tar; skip build step."
    )
    ap.add_argument(
        "--index",
        action="store_true",
        help="Eagerly build the sqlite index (open the subset once).",
    )
    ap.add_argument(
        "--verify",
        type=int,
        default=0,
        metavar="N",
        help="Load N random trajectories to sanity-check the tar (implies --index).",
    )
    args = ap.parse_args()

    out_dir = preflight(args.only_labels if args.phase != "concat" else "")
    print(f"Target subset dir: {out_dir}\n")

    if args.skip_tar:
        tar_path = out_dir / f"{FORMAT}.tar"
        if not tar_path.exists():
            sys.exit(f"--skip-tar but {tar_path} does not exist.")
        print(f"--skip-tar: reusing {tar_path}")
    elif args.phase == "parts":
        build_parts(out_dir, jobs=args.jobs, only_labels=args.only_labels)
    elif args.phase == "concat":
        concat_parts(out_dir, force=args.force)
    else:
        build_tar(out_dir, force=args.force, jobs=args.jobs, only_labels=args.only_labels)

    if args.index or args.verify:
        build_index_and_verify(args.verify)

    print("\nDone. Reference it from a dataset config as:")
    print("  self_play:\n    smallg1-online: <weight>")


if __name__ == "__main__":
    main()
