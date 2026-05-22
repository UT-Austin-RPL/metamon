import os
import json
import datetime
import shutil
import tarfile
from collections import defaultdict

from huggingface_hub import hf_hub_download

import metamon
from metamon.config import SUPPORTED_BATTLE_FORMATS, METAMON_CACHE_DIR

SELF_PLAY_SUBSETS = ["pac-base", "pac-exploratory", "pac-tauros"]
SELF_PLAY_FORMATS = [
    "gen1ou",
    "gen2ou",
    "gen3ou",
    "gen4ou",
    "gen9ou",
]  # OU formats available for self-play
SELF_PLAY_SUBSET_FORMATS = {
    "pac-base": SELF_PLAY_FORMATS,
    "pac-exploratory": SELF_PLAY_FORMATS,
    "pac-tauros": ["gen1ou"],
}

# Replay-derived team sets on HF are published for OU tiers only (gen1-4ou, gen9ou).
REPLAY_DERIVED_OU_ONLY_TEAM_SETS = frozenset(
    {
        "paper_replays",
        "modern_replays",
        "modern_replays_v2",
        "gl_05_26",
        "hl_05_26",
    }
)


def get_self_play_formats(subset: str) -> list[str]:
    """Formats published on HF for a self-play subset."""
    if subset not in SELF_PLAY_SUBSET_FORMATS:
        raise ValueError(
            f"Invalid subset: {subset}. Must be one of {list(SELF_PLAY_SUBSET_FORMATS)}"
        )
    return SELF_PLAY_SUBSET_FORMATS[subset]


def iter_self_play_downloads(
    subsets: list[str] | None = None,
    formats: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Resolve (subset, format) pairs to download for self-play data."""
    selected_subsets = subsets or SELF_PLAY_SUBSETS
    unknown = [subset for subset in selected_subsets if subset not in SELF_PLAY_SUBSETS]
    if unknown:
        raise ValueError(
            f"Invalid subset(s): {unknown}. Must be one of {SELF_PLAY_SUBSETS}"
        )

    downloads: list[tuple[str, str]] = []
    for subset in selected_subsets:
        available_formats = get_self_play_formats(subset)
        requested_formats = formats or available_formats
        for battle_format in requested_formats:
            if battle_format not in available_formats:
                print(
                    f"Skipping {subset}/{battle_format}: "
                    f"not published for this subset (available: {available_formats})"
                )
                continue
            downloads.append((subset, battle_format))
    return downloads


if METAMON_CACHE_DIR is not None:
    VERSION_REFERENCE_PATH = os.path.join(METAMON_CACHE_DIR, "version_reference.json")
else:
    VERSION_REFERENCE_PATH = None

LATEST_RAW_REPLAY_REVISION = "v6"
LATEST_PARSED_REPLAY_REVISION = "v6"
LATEST_TEAMS_REVISION = "v5"
LATEST_USAGE_STATS_REVISION = "v5"


def _update_version_reference(key: str, name: str, version: str):
    """Maintains a version_reference.json file in the METAMON_CACHE_DIR.

    Records the version of each dataset that is currently active.
    """
    if VERSION_REFERENCE_PATH is None:
        return

    version_reference = defaultdict(dict)
    if os.path.exists(VERSION_REFERENCE_PATH):
        with open(VERSION_REFERENCE_PATH, "r") as f:
            existing_version_reference = json.load(f)
        version_reference.update(existing_version_reference)

    version_reference[key][
        name
    ] = f"version {version}, downloaded {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    with open(VERSION_REFERENCE_PATH, "w") as f:
        json.dump(dict(version_reference), f)


def get_active_dataset_versions() -> dict:
    """Get the current version of a dataset."""
    with open(VERSION_REFERENCE_PATH, "r") as f:
        version_reference = json.load(f)
    return version_reference


def download_parsed_replays(
    battle_format: str,
    version: str = LATEST_PARSED_REPLAY_REVISION,
    force_download: bool = False,
) -> str:
    """Download the parsed replays for a given battle format.

    Args:
        battle_format: Showdown battle format (e.g. "gen1ou")
        version: Version of the dataset to download. Corresponds to revisions on the
            Hugging Face Hub. Defaults to the latest version.
        force_download: If True, download the dataset even if a previous version
            already exists in the cache.

    Returns:
        The path to the dataset on disk.
    """
    if METAMON_CACHE_DIR is None:
        raise ValueError("METAMON_CACHE_DIR environment variable is not set")
    parsed_replay_dir = os.path.join(METAMON_CACHE_DIR, "parsed-replays")
    tar_path = os.path.join(parsed_replay_dir, f"{battle_format}.tar.gz")
    out_path = os.path.join(parsed_replay_dir, battle_format)
    if os.path.exists(out_path):
        if not force_download:
            return out_path
        print(f"Clearing existing dataset at {out_path}...")
        shutil.rmtree(out_path)
    hf_hub_download(
        cache_dir=os.path.join(METAMON_CACHE_DIR, "parsed-replays"),
        repo_id="jakegrigsby/metamon-parsed-replays",
        filename=f"{battle_format}.tar.gz",
        local_dir=os.path.join(METAMON_CACHE_DIR, "parsed-replays"),
        revision=version,
        repo_type="dataset",
    )
    with tarfile.open(tar_path) as tar:
        print(f"Extracting {tar_path}...")
        if version > "v1":
            extract_path = parsed_replay_dir
        else:
            extract_path = out_path
        tar.extractall(path=extract_path)
    os.remove(tar_path)
    _update_version_reference("parsed-replays", battle_format, version)
    return out_path


def download_teams(
    battle_format: str,
    set_name: str,
    version: str = LATEST_TEAMS_REVISION,
    force_download: bool = False,
) -> str:
    """Download the teams for a given battle format and set name.

    Args:
        battle_format: Showdown battle format (e.g. "gen1ou")
        set_name: Name of the team set to download.
        version: Version of the dataset to download. Corresponds to revisions on the
            Hugging Face Hub. Defaults to the latest version.
        force_download: If True, download the dataset even if a previous version
            already exists in the cache.

    Returns:
        The path to the dataset on disk.
    """
    if METAMON_CACHE_DIR is None:
        raise ValueError("METAMON_CACHE_DIR environment variable is not set")

    teams_dir = os.path.join(METAMON_CACHE_DIR, "teams", set_name)
    tar_path = os.path.join(teams_dir, f"{battle_format}.tar.gz")
    extract_path = os.path.join(teams_dir, battle_format)
    if os.path.exists(extract_path):
        if not force_download:
            return extract_path
        print(f"Clearing existing dataset at {extract_path}...")
        shutil.rmtree(extract_path)
    hf_hub_download(
        cache_dir=os.path.join(METAMON_CACHE_DIR, "teams", set_name),
        repo_id="jakegrigsby/metamon-teams",
        filename=f"{set_name}/{battle_format}.tar.gz",
        local_dir=os.path.join(METAMON_CACHE_DIR, "teams"),
        revision=version,
        repo_type="dataset",
    )
    with tarfile.open(tar_path) as tar:
        print(f"Extracting {tar_path}...")
        tar.extractall(path=os.path.dirname(extract_path))
    os.remove(tar_path)
    _update_version_reference("teams", f"{set_name}/{battle_format}", version)
    return extract_path


def download_replay_stats(
    version: str = LATEST_PARSED_REPLAY_REVISION, force_download: bool = False
) -> str:
    """Download the "replay stats" for a given version.

    Replay stats are json statistics generated from the revealed teams of the current
    replay dataset. They are used to predict team sets.

    Args:
        version: Version of the dataset to download. Corresponds to revisions on the
            Hugging Face Hub. Defaults to the latest version.
        force_download: If True, download the dataset even if a previous version
            already exists in the cache.

    Returns:
        The path to the dataset on disk.
    """
    replay_stats_dir = download_parsed_replays("replay_stats", version, force_download)
    return replay_stats_dir


def download_revealed_teams(
    version: str = LATEST_PARSED_REPLAY_REVISION, force_download: bool = False
) -> str:
    return download_parsed_replays("revealed_teams", version, force_download)


def download_raw_replays(version: str = LATEST_RAW_REPLAY_REVISION) -> str:
    """Download the "raw" (unprocessed) replays.

    We maintain a dataset of replays downloaded from Pokémon Showdown for convenience.
    Our versions are also stripped of player usernames and in-game chat logs.

    Args:
        version: Version of the dataset to download. Corresponds to revisions / git tags
            on the Hugging Face Hub. Defaults to the latest version.

    Returns:
        The path to the dataset on disk.
    """
    if METAMON_CACHE_DIR is None:
        raise ValueError("METAMON_CACHE_DIR environment variable is not set")
    metamon.data.raw_replay_util.process_dataset(
        dataset_id="jakegrigsby/metamon-raw-replays",
        output_dir=os.path.join(METAMON_CACHE_DIR, "raw-replays"),
        revision=version,
    )
    _update_version_reference("raw-replays", "raw-replays", version)
    return os.path.join(METAMON_CACHE_DIR, "raw-replays")


def download_self_play_data(
    subset: str,
    battle_format: str,
    version: str = "main",
    force_download: bool = False,
    extract: bool = False,
) -> str:
    """Download self-play data from the metamon-parsed-pile dataset.

    Args:
        subset: The subset to download. Options: "pac-base", "pac-exploratory", "pac-tauros"
        battle_format: Showdown battle format (e.g. "gen1ou")
        version: Version/revision of the dataset to download. Defaults to "main".
        force_download: If True, download the dataset even if a previous version
            already exists in the cache.
        extract: If True, extract all files from the tar archive (slow, uses many inodes).
            If False (default), keep data in .tar format for direct reading.

    Returns:
        The path to the .tar file (if extract=False) or extracted directory (if extract=True).
    """
    if METAMON_CACHE_DIR is None:
        raise ValueError("METAMON_CACHE_DIR environment variable is not set")
    if subset not in SELF_PLAY_SUBSETS:
        raise ValueError(
            f"Invalid subset: {subset}. Must be one of {SELF_PLAY_SUBSETS}"
        )

    self_play_dir = os.path.join(METAMON_CACHE_DIR, "self-play", subset)
    tar_lz4_path = os.path.join(self_play_dir, f"{battle_format}.tar.lz4")
    tar_path = os.path.join(self_play_dir, f"{battle_format}.tar")
    extracted_path = os.path.join(self_play_dir, battle_format)

    # Determine output path based on extract flag
    out_path = extracted_path if extract else tar_path

    if os.path.exists(out_path):
        if not force_download:
            return out_path
        if extract:
            print(f"Clearing existing dataset at {out_path}...")
            shutil.rmtree(out_path)
        else:
            os.remove(out_path)

    hf_hub_download(
        cache_dir=self_play_dir,
        repo_id="jakegrigsby/metamon-parsed-pile",
        filename=f"{subset}/{battle_format}.tar.lz4",
        local_dir=os.path.join(METAMON_CACHE_DIR, "self-play"),
        revision=version,
        repo_type="dataset",
    )

    # Download pre-built SQLite index (skips expensive index build)
    sqlite_index_path = os.path.join(self_play_dir, f"{battle_format}.tar.index.sqlite")
    if not os.path.exists(sqlite_index_path) or force_download:
        try:
            hf_hub_download(
                cache_dir=self_play_dir,
                repo_id="jakegrigsby/metamon-parsed-pile",
                filename=f"{subset}/{battle_format}.tar.index.sqlite",
                local_dir=os.path.join(METAMON_CACHE_DIR, "self-play"),
                revision=version,
                repo_type="dataset",
            )
            print(f"Downloaded pre-built index: {sqlite_index_path}")
        except Exception as e:
            print(
                f"Note: Pre-built index not available, will be built on first load ({e})"
            )

    # Decompress .tar.lz4 -> .tar
    import lz4.frame
    from tqdm import tqdm

    compressed_size = os.path.getsize(tar_lz4_path)
    print(f"Decompressing {tar_lz4_path} ({compressed_size / 1e9:.1f}GB compressed)...")

    with lz4.frame.open(tar_lz4_path, "rb") as lz4_file:
        with open(tar_path, "wb") as tar_file:
            # Stream in chunks to handle large files
            bytes_written = 0
            with tqdm(unit="B", unit_scale=True, desc="Decompressing") as pbar:
                while True:
                    chunk = lz4_file.read(64 * 1024 * 1024)  # 64MB chunks
                    if not chunk:
                        break
                    tar_file.write(chunk)
                    bytes_written += len(chunk)
                    pbar.update(len(chunk))

    os.remove(tar_lz4_path)

    if extract:
        print(f"Extracting {tar_path} (this may take a while for large datasets)...")
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=self_play_dir)
        os.remove(tar_path)

    _update_version_reference("self-play", f"{subset}/{battle_format}", version)
    return out_path


def download_usage_stats(
    gen: int,
    version: str = LATEST_USAGE_STATS_REVISION,
    force_download: bool = False,
) -> str:
    """Download the usage stats for a given battle format and year/month.

    Usage stats are cheatsheet conversions of the raw Smogon data released for
    evaluating rule changes and metagame trends. They help us predict missing information
    based on team construction trends at the time the battle was played.

    Args:
        gen: Generation of the usage stats to download (e.g. 1 for Gen 1)
        version: Version of the dataset to download. Corresponds to revisions on the
            Hugging Face Hub. Defaults to the latest version.
        force_download: If True, download the dataset even if a previous version
            already exists in the cache.

    Returns:
        The path to the dataset on disk.
    """
    if METAMON_CACHE_DIR is None:
        raise ValueError("METAMON_CACHE_DIR environment variable is not set")

    usage_stats_dir = os.path.join(METAMON_CACHE_DIR, "usage-stats")
    movesets_path = os.path.join(usage_stats_dir, "movesets_data")
    checks_path = os.path.join(usage_stats_dir, "checks_data")
    movesets_tar_path = os.path.join(movesets_path, f"gen{gen}.tar.gz")
    checks_tar_path = os.path.join(checks_path, f"gen{gen}.tar.gz")
    movesets_extract_path = os.path.join(movesets_path, f"gen{gen}")
    checks_extract_path = os.path.join(checks_path, f"gen{gen}")

    def _download_and_extract(tar_path, extract_path):
        if os.path.exists(extract_path):
            if not force_download:
                return extract_path
            print(f"Clearing existing dataset at {extract_path}...")
            shutil.rmtree(extract_path)
        repo_folder = os.path.basename(os.path.dirname(tar_path))
        hf_hub_download(
            cache_dir=None,
            repo_id="jakegrigsby/metamon-usage-stats",
            filename=f"{repo_folder}/{os.path.basename(tar_path)}",
            local_dir=usage_stats_dir,
            revision=version,
            repo_type="dataset",
        )
        with tarfile.open(tar_path) as tar:
            print(f"Extracting {tar_path}...")
            tar.extractall(path=os.path.dirname(extract_path))
        os.remove(tar_path)

    _download_and_extract(movesets_tar_path, movesets_extract_path)
    _download_and_extract(checks_tar_path, checks_extract_path)
    return usage_stats_dir


def print_version_tree(version_dict: dict, indent: int = 0):
    for key, value in sorted(version_dict.items()):
        if isinstance(value, dict):
            print(" " * indent + f"{key}:")
            print_version_tree(value, indent + 4)
        else:
            print(" " * indent + f"{key}: {value}")


if __name__ == "__main__":
    import argparse
    from termcolor import colored

    parser = argparse.ArgumentParser(
        description=f"""
Metamon Dataset Downloader

This tool downloads and manages Metamon datasets from Hugging Face Hub.
Available datasets include:
    - raw-replays: Unprocessed Showdown replays (stripped of usernames/chat)
    - parsed-replays: RL-compatible version of replays with reconstructed player actions  
    - revealed-teams: Teams that were revealed during replay battles
    - replay-stats: Statistics generated from revealed teams
    - usage-stats: Team composition stats from Showdown
    - teams: Various team sets (competitive, paper_variety, paper_replays, modern_replays)

Examples:
    # Download all team files for Gen 1-4 OU
    python -m metamon.data.download teams --formats gen1ou gen2ou gen3ou gen4ou

    # Download parsed replays for Gen 1 UU  
    python -m metamon.data.download parsed-replays --formats gen1uu

    # Download (anonymized) Showdown replay logs (all formats)
    python -m metamon.data.download raw-replays

    # Download self-play datasets (pac-base, pac-exploratory, pac-tauros)
    python -m metamon.data.download self-play --formats gen1ou gen9ou

    # Download only pac-tauros (defaults to gen1ou for that subset)
    python -m metamon.data.download self-play --subsets pac-tauros

Note: Requires METAMON_CACHE_DIR environment variable to be set.

The cache directory is currently: {colored(METAMON_CACHE_DIR or 'NOT SET', 'red')}
For current dataset versions, see `get_active_dataset_versions()` or run:
    python -m metamon.download check-versions
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=[
            "raw-replays",
            "parsed-replays",
            "self-play",
            "revealed-teams",
            "replay-stats",
            "teams",
            "usage-stats",
            "check-versions",
        ],
        help="""
`check-versions`: Display the current versions of the datasets.

Dataset to download:
    raw-replays: Unprocessed Showdown replays (stripped of usernames/chat)
    parsed-replays: RL-compatible version of replays with reconstructed player actions
    self-play: Self-play battle data (pac-base, pac-exploratory, pac-tauros subsets)
    revealed-teams: Teams that were revealed during battles
    replay-stats: Statistics generated from revealed teams. Used to predict team sets.
    teams: Various team sets (competitive, paper_variety, paper_replays)
""",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        type=str,
        default=None,
        help="""
Battle formats to download. Defaults depend on dataset type:
  - parsed-replays, teams, usage-stats: All Gen 1-4 formats (OU, UU, NU, Ubers) + Gen 9 OU
  - self-play: gen1ou, gen2ou, gen3ou, gen4ou, gen9ou (only OU available; defaults depend on subset)
Examples:
    --formats gen1ou gen2ou    # Only Gen 1-2 OU
    --formats gen3uu gen4uu    # Only Gen 3-4 UU
""",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        type=str,
        choices=SELF_PLAY_SUBSETS,
        default=None,
        help="""
Self-play subsets to download. Defaults to all subsets (pac-base, pac-exploratory, pac-tauros).
When --formats is omitted, each subset uses its published formats (e.g. pac-tauros -> gen1ou only).
Examples:
    --subsets pac-tauros
    --subsets pac-base pac-exploratory --formats gen1ou
""",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="""
Specific version to download. Defaults to latest version.
Available versions:
    raw-replays: v1, v2, v3, v4, v5, v6
    parsed-replays: v0 (deprecated) v1, v2, v3-beta, v3, v4, v5, v6
    teams: v0, v1, v2, v3, v4, v5
    usage-stats: v0, v1, v2, v3, v4, v5
    
    The huggingface READMEs have changelogs.
""",
    )
    args = parser.parse_args()

    if args.dataset == "raw-replays":
        version = args.version or LATEST_RAW_REPLAY_REVISION
        download_raw_replays(version=version)
    elif args.dataset == "parsed-replays":
        version = args.version or LATEST_PARSED_REPLAY_REVISION
        formats = args.formats or SUPPORTED_BATTLE_FORMATS
        for format in formats:
            download_parsed_replays(format, version=version, force_download=True)
    elif args.dataset == "self-play":
        version = args.version or "main"
        downloads = iter_self_play_downloads(
            subsets=args.subsets,
            formats=args.formats,
        )
        if not downloads:
            raise ValueError(
                "No self-play downloads matched the requested subsets/formats"
            )
        print(f"Downloading self-play data: {downloads}")
        for subset, battle_format in downloads:
            print(f"\nDownloading {subset}/{battle_format}...")
            download_self_play_data(
                subset, battle_format, version=version, force_download=True
            )
    elif args.dataset == "revealed-teams":
        version = args.version or LATEST_PARSED_REPLAY_REVISION
        download_revealed_teams(version=version, force_download=True)
    elif args.dataset == "replay-stats":
        version = args.version or LATEST_PARSED_REPLAY_REVISION
        download_replay_stats(version=version, force_download=True)
    elif args.dataset == "usage-stats":
        version = args.version or LATEST_USAGE_STATS_REVISION
        formats = args.formats or SUPPORTED_BATTLE_FORMATS
        generations = set(metamon.backend.format_to_gen(format) for format in formats)
        for gen in generations:
            download_usage_stats(gen=gen, version=version, force_download=True)
    elif args.dataset == "teams":
        version = args.version or LATEST_TEAMS_REVISION
        formats = args.formats or SUPPORTED_BATTLE_FORMATS
        set_names = ["competitive", "paper_variety", "paper_replays"]
        if version > "v0":
            set_names += ["modern_replays", "modern_replays_v2"]
        if version > "v4":
            set_names += ["gl_05_26", "hl_05_26"]
        for set_name in set_names:
            for format in formats:
                if "ou" not in format and set_name in REPLAY_DERIVED_OU_ONLY_TEAM_SETS:
                    continue
                if "gen9" in format and "paper" in set_name:
                    # gen 9 was not supported
                    continue
                download_teams(
                    battle_format=format,
                    set_name=set_name,
                    version=version,
                    force_download=True,
                )
    elif args.dataset == "check-versions":
        print(colored("\nActive dataset versions:", "red"))
        print(f"Cache Dir: {METAMON_CACHE_DIR}\n")
        print_version_tree(get_active_dataset_versions())
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
