"""
PyTorch datasets for loading parsed Pokémon battle trajectories.

Classes:
    MetamonDataset: Base class for custom/local datasets.
    ParsedReplayDataset: Human replays from HuggingFace (flat directories).
    SelfPlayDataset: Self-play data from HuggingFace (tar archives).

Storage formats supported:
    - Flat directories: {format}/*.json[.lz4]
    - Tar archives: {format}.tar (O(1) access via ratarmountcore SQLite index)
"""

import os
import json
import random
import csv
import copy
from typing import Optional, Dict, Tuple, List, Any
from datetime import datetime
from collections import defaultdict

from torch.utils.data import Dataset
import lz4.frame
import numpy as np
import tqdm
from ratarmountcore.SQLiteIndexedTarFsspec import SQLiteIndexedTarFileSystem

import metamon
from metamon.interface import (
    ObservationSpace,
    RewardFunction,
    UniversalState,
    ActionSpace,
    UniversalAction,
)
from metamon.data.download import (
    download_parsed_replays,
    download_self_play_data,
    SELF_PLAY_SUBSETS,
    SELF_PLAY_FORMATS,
    METAMON_CACHE_DIR,
)


class MetamonDataset(Dataset):
    """Base dataset class for loading parsed Pokémon battle trajectories.

    Parsed replays are records of Pokémon Showdown battles converted to the partially
    observed point-of-view of a single player, matching the problem our agents face in
    the RL environment. They are created by the `metamon.backend.replay_parser` module.

    This class auto-detects whether data is stored as:
        - Flat directories: {format}/*.json or {format}/*.json.lz4
        - Tar archives: {format}.tar (uses ratarmountcore for O(1) random access)

    Use MetamonDataset directly for local/custom datasets. For official HuggingFace
    datasets, use the subclasses:
        - ParsedReplayDataset: Human replays from jakegrigsby/metamon-parsed-replays
        - SelfPlayDataset: Self-play data from jakegrigsby/metamon-parsed-pile

    Args:
        dset_root: Root directory containing format subdirs or tar files.
        observation_space: Observation space for converting states to observations.
        action_space: Action space for converting actions to agent outputs.
        reward_function: Reward function for computing rewards from state transitions.
        formats: List of battle formats to load (e.g., ["gen1ou", "gen9ou"]).
        wins_losses_both: Filter by outcome: "wins", "losses", or "both".
        min_rating: Minimum ELO rating filter (unrated battles default to 1000).
        max_rating: Maximum ELO rating filter.
        min_date: Minimum battle date filter.
        max_date: Maximum battle date filter.
        max_seq_len: Maximum trajectory length (randomly sliced if exceeded).
        verbose: Print progress information.
        shuffle: Shuffle the filename list.
        use_cached_filenames: Use cached index files for faster startup.

    Returns (from __getitem__):
        nested_obs: Dict of lists of numpy arrays for each observation key.
        actions: Dict with keys "chosen" (list), "legal" (list of sets), "missing" (list of bools).
        rewards: numpy array of shape (seq_len,).
        dones: numpy array of shape (seq_len,).

    Note:
        Missing actions (actions["missing"][i] == True) occur when player choices are not
        revealed in replay logs (e.g., paralysis, sleep, flinch decisions).
    """

    # Prefix for tar-backed filenames to distinguish from disk files
    TAR_PREFIX = "tar://"

    def __init__(
        self,
        dset_root: str,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        reward_function: RewardFunction,
        formats: List[str],
        wins_losses_both: str = "both",
        min_rating: Optional[int] = None,
        max_rating: Optional[int] = None,
        min_date: Optional[datetime] = None,
        max_date: Optional[datetime] = None,
        max_seq_len: Optional[int] = None,
        verbose: bool = False,
        shuffle: bool = False,
        use_cached_filenames: bool = False,
    ):
        assert os.path.exists(dset_root), f"Dataset root not found: {dset_root}"

        self.dset_root = dset_root
        self.observation_space = copy.deepcopy(observation_space)
        self.action_space = copy.deepcopy(action_space)
        self.reward_function = copy.deepcopy(reward_function)
        self.formats = formats
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.min_date = min_date
        self.max_date = max_date
        self.wins_losses_both = wins_losses_both
        self.max_seq_len = max_seq_len
        self.verbose = verbose
        self.shuffle = shuffle
        self.use_cached_filenames = use_cached_filenames

        self.index_path = os.path.join(self.dset_root, "index.csv")

        self._tar_files: Dict[str, SQLiteIndexedTarFileSystem] = {}
        self._tar_paths: Dict[str, str] = {}
        self._format_is_tar: Dict[str, bool] = {}
        self._owner_pid: int = os.getpid()  # Track PID for fork-safety

        self._detect_formats()
        self.refresh_files()

    ######################
    ## Format Detection ##
    ######################

    def _detect_formats(self):
        """Detect whether each format is stored as flat directory or tar archive."""
        available_formats = []

        for format_name in self.formats:
            tar_path = os.path.join(self.dset_root, f"{format_name}.tar")
            dir_path = os.path.join(self.dset_root, format_name)

            if os.path.exists(tar_path):
                # TAR ARCHIVE: gen1ou.tar
                self._format_is_tar[format_name] = True
                self._tar_paths[format_name] = tar_path
                available_formats.append(format_name)
                if self.verbose:
                    print(f"Detected tar archive for {format_name}")

            elif os.path.isdir(dir_path):
                # FLAT DIRECTORY: gen1ou/
                self._format_is_tar[format_name] = False
                available_formats.append(format_name)
                if self.verbose:
                    print(f"Detected flat directory for {format_name}")

            else:
                if self.verbose:
                    print(f"Skipping {format_name}: no data found")

        self.formats = available_formats

    #########################################
    ## Tar Archive Handling (ratarmountcore) #
    #########################################

    def _get_tar(self, format_name: str) -> SQLiteIndexedTarFileSystem:
        current_pid = os.getpid()
        is_worker = current_pid != self._owner_pid
        if is_worker:
            self._tar_files.clear()
            self._owner_pid = current_pid

        if format_name not in self._tar_files:
            if self.verbose and not is_worker:
                print(f"Opening {format_name}.tar...")
            self._tar_files[format_name] = SQLiteIndexedTarFileSystem(
                self._tar_paths[format_name],
                printDebug=-1,
            )
        return self._tar_files[format_name]

    def _get_tar_index_path(self, format_name: str) -> str:
        """Get path to our cached filename list for a tar archive."""
        return os.path.join(self.dset_root, f"{format_name}.tar.index.txt")

    def _index_tar(self, format_name: str) -> List[str]:
        """TAR: List all json files in archive using ratarmountcore.

        This opens the tar and builds the SQLite index if not present.
        Also caches filename list to .txt for use_cached_filenames=True.
        """
        fs = self._get_tar(format_name)

        # List files in the format directory within the tar
        try:
            files = fs.ls(f"/{format_name}", detail=False)
        except FileNotFoundError:
            files = fs.ls("/", detail=False)

        # Filter to json files and strip leading slash
        member_names = [
            f.lstrip("/") for f in files if f.endswith((".json", ".json.lz4"))
        ]

        # Cache the filename list for use_cached_filenames
        index_path = self._get_tar_index_path(format_name)
        with open(index_path, "w") as f:
            for name in member_names:
                f.write(name + "\n")

        if self.verbose:
            print(f"Found {len(member_names)} files in {format_name}.tar")

        return member_names

    def _load_tar_index(self, format_name: str) -> List[str]:
        """TAR: Load cached filename list from .txt file."""
        index_path = self._get_tar_index_path(format_name)
        with open(index_path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    #############################
    ## Flat Directory Handling ##
    #############################

    def _index_directory(self, format_name: str) -> List[str]:
        """DIRECTORY: Scan directory for json files."""
        format_dir = os.path.join(self.dset_root, format_name)
        try:
            files = os.listdir(format_dir)
        except (OSError, PermissionError) as e:
            if self.verbose:
                print(f"  Warning: Could not read {format_dir}: {e}")
            return []

        return [
            os.path.join(format_name, f)
            for f in files
            if f.endswith((".json", ".json.lz4"))
        ]

    ########################
    ## Filename Filtering ##
    ########################

    def _filter_filename(self, filename: str, format_name: str) -> bool:
        """Apply rating, date, and win/loss filters to a filename."""
        # Parse filename: battle_id_rating_p1_vs_p2_date_result.json
        name_without_ext = (
            filename[:-9] if filename.endswith(".json.lz4") else filename[:-5]
        )
        parts = name_without_ext.split("_")

        if len(parts) == 7:
            battle_id, rating_str, p1, _, p2, date_str, result = parts
        elif len(parts) == 8:
            battle_id, rating_str, p1a, p1b, _, p2, date_str, result = parts
        else:
            return False

        # Validate format in battle_id
        if (
            format_name
            not in battle_id.replace("[", "").replace("]", "").replace(" ", "").lower()
        ):
            return False

        # Result filter
        if self.wins_losses_both == "wins" and result != "WIN":
            return False
        if self.wins_losses_both == "losses" and result != "LOSS":
            return False

        # Rating filter
        if self.min_rating is not None or self.max_rating is not None:
            try:
                rating = int(rating_str)
            except ValueError:
                rating = 1000
            if self.min_rating and rating < self.min_rating:
                return False
            if self.max_rating and rating > self.max_rating:
                return False

        # Date filter
        if self.min_date is not None or self.max_date is not None:
            try:
                date = self._parse_date(date_str)
                if self.min_date and date < self.min_date:
                    return False
                if self.max_date and date > self.max_date:
                    return False
            except ValueError:
                return False

        return True

    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string from filename."""
        try:
            return datetime.strptime(date_str, "%m-%d-%Y")
        except ValueError:
            return datetime.strptime(date_str, "%m-%d-%Y-%H:%M:%S")

    ###################
    ## File Indexing ##
    ###################

    def refresh_files(self):
        """Build the list of files to load, applying filters."""
        self.filenames = []

        # Check if we need to rebuild directory index.csv
        has_directory_formats = any(
            not self._format_is_tar.get(fmt, False) for fmt in self.formats
        )
        will_rebuild_dir_index = has_directory_formats and (
            not self.use_cached_filenames or not os.path.exists(self.index_path)
        )
        if will_rebuild_dir_index and os.path.exists(self.index_path):
            os.remove(self.index_path)  # Clear stale index before rebuilding

        for format_name in self.formats:
            if self._format_is_tar.get(format_name, False):
                self._refresh_tar_format(format_name)
            else:
                self._refresh_directory_format(format_name)

        if self.verbose:
            print(f"Total: {len(self.filenames)} battles after filtering")

        if self.shuffle:
            random.shuffle(self.filenames)

    def _refresh_tar_format(self, format_name: str):
        """TAR: Index and filter files from a tar archive."""
        index_path = self._get_tar_index_path(format_name)

        # Get file list (from .txt cache or fresh scan)
        if self.use_cached_filenames and os.path.exists(index_path):
            if self.verbose:
                print(f"Loading cached tar index from {index_path}")
            member_names = self._load_tar_index(format_name)
        else:
            # This will open tar, build SQLite index if needed, and cache .txt
            member_names = self._index_tar(format_name)

        # Filter and add with TAR_PREFIX
        iterator = (
            tqdm.tqdm(member_names, desc=f"Filtering {format_name}", colour="green")
            if self.verbose
            else member_names
        )
        for member_name in iterator:
            if self._filter_filename(os.path.basename(member_name), format_name):
                self.filenames.append(f"{self.TAR_PREFIX}{format_name}/{member_name}")

    def _refresh_directory_format(self, format_name: str):
        """DIRECTORY: Index and filter files from a flat directory."""
        # Get file list (from cache or fresh scan)
        if self.use_cached_filenames and os.path.exists(self.index_path):
            with open(self.index_path, "r") as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                rel_paths = [
                    row[0] for row in reader if row[0].startswith(format_name + os.sep)
                ]
            if self.verbose:
                print(f"Loaded {len(rel_paths)} files from index.csv for {format_name}")
        else:
            rel_paths = self._index_directory(format_name)
            if self.verbose:
                print(f"Indexed {len(rel_paths)} files from {format_name}/")
            # Write to index.csv cache (append if exists, create with header if not)
            write_header = not os.path.exists(self.index_path)
            with open(self.index_path, "a") as f:
                if write_header:
                    f.write("filename\n")
                for rel_path in rel_paths:
                    f.write(f"{rel_path}\n")

        # Filter and add as absolute paths
        iterator = (
            tqdm.tqdm(rel_paths, desc=f"Filtering {format_name}", colour="green")
            if self.verbose
            else rel_paths
        )
        for rel_path in iterator:
            if self._filter_filename(os.path.basename(rel_path), format_name):
                self.filenames.append(os.path.join(self.dset_root, rel_path))

    ##################
    ## Data Loading ##
    ##################

    def _load_json(self, filename: str) -> dict:
        """Load JSON data from either tar archive or disk file."""
        if filename.startswith(self.TAR_PREFIX):
            return self._load_json_from_tar(filename)
        else:
            return self._load_json_from_disk(filename)

    def _load_json_from_tar(self, filename: str) -> dict:
        """TAR: Read file using ratarmountcore (O(1) random access)."""
        path = filename[len(self.TAR_PREFIX) :]
        format_name, member_name = path.split("/", 1)
        fs = self._get_tar(format_name)

        data = fs.cat("/" + member_name)
        if member_name.endswith(".lz4"):
            data = lz4.frame.decompress(data)
        return json.loads(data.decode("utf-8"))

    def _load_json_from_disk(self, filename: str) -> dict:
        """DIRECTORY: Read file from disk."""
        if filename.endswith(".json.lz4"):
            with lz4.frame.open(filename, "rb") as f:
                return json.loads(f.read().decode("utf-8"))
        else:
            with open(filename, "r") as f:
                return json.load(f)

    def load_filename(self, filename: str):
        """Load and process a single battle trajectory."""
        data = self._load_json(filename)
        states = [UniversalState.from_dict(s) for s in data["states"]]

        # Build observations
        self.observation_space.reset()
        obs = [self.observation_space.state_to_obs(s) for s in states]
        nested_obs = defaultdict(list)
        for o in obs:
            for k, v in o.items():
                nested_obs[k].append(v)

        # Build actions
        action_infos = {"chosen": [], "legal": [], "missing": []}
        for s, a_idx in zip(states, data["actions"][:-1]):
            universal_action = UniversalAction(action_idx=a_idx)
            action_infos["chosen"].append(
                self.action_space.action_to_agent_output(s, universal_action)
            )
            action_infos["legal"].append(
                set(
                    self.action_space.action_to_agent_output(s, l)
                    for l in UniversalAction.maybe_valid_actions(s)
                )
            )
            action_infos["missing"].append(universal_action.missing)

        # Build rewards and dones
        rewards = np.array(
            [
                self.reward_function(s_t, s_t1)
                for s_t, s_t1 in zip(states[:-1], states[1:])
            ],
            dtype=np.float32,
        )
        dones = np.zeros_like(rewards, dtype=bool)
        dones[-1] = True

        # Random slice if max_seq_len specified
        if self.max_seq_len is not None:
            start = random.randint(
                0, max(len(action_infos["chosen"]) - self.max_seq_len, 0)
            )
            end = start + self.max_seq_len
            nested_obs = {k: v[start : end + 1] for k, v in nested_obs.items()}
            action_infos = {k: v[start:end] for k, v in action_infos.items()}
            rewards = rewards[start:end]
            dones = dones[start:end]

        return dict(nested_obs), action_infos, rewards, dones

    ###############################
    ## PyTorch Dataset Interface ##
    ###############################

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i) -> Tuple[Dict, Dict, np.ndarray, np.ndarray]:
        return self.load_filename(self.filenames[i])

    def random_sample(self):
        return self.load_filename(random.choice(self.filenames))


class ParsedReplayDataset(MetamonDataset):
    """Human replay dataset from jakegrigsby/metamon-parsed-replays.

    Auto-downloads from HuggingFace to {$METAMON_CACHE_DIR}/parsed-replays.

    See MetamonDataset for full argument documentation.
    """

    def __init__(
        self,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        reward_function: RewardFunction,
        dset_root: Optional[str] = None,
        formats: Optional[List[str]] = None,
        wins_losses_both: str = "both",
        min_rating: Optional[int] = None,
        max_rating: Optional[int] = None,
        min_date: Optional[datetime] = None,
        max_date: Optional[datetime] = None,
        max_seq_len: Optional[int] = None,
        verbose: bool = False,
        shuffle: bool = False,
        use_cached_filenames: bool = False,
    ):
        formats = formats or metamon.SUPPORTED_BATTLE_FORMATS

        if dset_root is None:
            for format_name in formats:
                path = download_parsed_replays(format_name)
            dset_root = os.path.dirname(path)

        super().__init__(
            dset_root=dset_root,
            observation_space=observation_space,
            action_space=action_space,
            reward_function=reward_function,
            formats=formats,
            wins_losses_both=wins_losses_both,
            min_rating=min_rating,
            max_rating=max_rating,
            min_date=min_date,
            max_date=max_date,
            max_seq_len=max_seq_len,
            verbose=verbose,
            shuffle=shuffle,
            use_cached_filenames=use_cached_filenames,
        )


class SelfPlayDataset(MetamonDataset):
    """Self-play dataset from jakegrigsby/metamon-parsed-pile.

    Auto-downloads from HuggingFace to {$METAMON_CACHE_DIR}/self-play/{subset}.

    Args:
        subset: Which self-play subset to load:
            - "pac-base": 11M trajectories from PokéAgent Challenge training
            - "pac-exploratory": 7M trajectories from higher-temperature sampling.
        formats: Defaults to SELF_PLAY_FORMATS (gen1-4ou, gen9ou).

    See MetamonDataset for remaining argument documentation.
    """

    def __init__(
        self,
        subset: str,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        reward_function: RewardFunction,
        formats: Optional[List[str]] = None,
        wins_losses_both: str = "both",
        min_date: Optional[datetime] = None,
        max_date: Optional[datetime] = None,
        max_seq_len: Optional[int] = None,
        verbose: bool = False,
        shuffle: bool = False,
        use_cached_filenames: bool = False,
    ):
        if subset not in SELF_PLAY_SUBSETS:
            raise ValueError(
                f"Invalid subset: {subset}. Must be one of {SELF_PLAY_SUBSETS}"
            )

        self.subset = subset
        formats = formats or SELF_PLAY_FORMATS

        # Download tar files (without extracting)
        for format_name in formats:
            download_self_play_data(subset, format_name, extract=False)
        dset_root = os.path.join(METAMON_CACHE_DIR, "self-play", subset)

        super().__init__(
            dset_root=dset_root,
            observation_space=observation_space,
            action_space=action_space,
            reward_function=reward_function,
            formats=formats,
            wins_losses_both=wins_losses_both,
            min_date=min_date,
            max_date=max_date,
            max_seq_len=max_seq_len,
            verbose=verbose,
            shuffle=shuffle,
            use_cached_filenames=use_cached_filenames,
        )


if __name__ == "__main__":
    from argparse import ArgumentParser
    from metamon.interface import (
        DefaultShapedReward,
        get_observation_space,
        TokenizedObservationSpace,
        DefaultActionSpace,
    )
    from metamon.tokenizer import get_tokenizer

    parser = ArgumentParser()
    parser.add_argument("--dset_root", type=str, default=None)
    parser.add_argument("--formats", type=str, default=None, nargs="+")
    parser.add_argument("--obs_space", type=str, default="DefaultObservationSpace")
    args = parser.parse_args()

    dset = ParsedReplayDataset(
        dset_root=args.dset_root,
        observation_space=TokenizedObservationSpace(
            get_observation_space(args.obs_space),
            tokenizer=get_tokenizer("DefaultObservationSpace-v1"),
        ),
        action_space=DefaultActionSpace(),
        reward_function=DefaultShapedReward(),
        formats=args.formats,
        verbose=True,
        shuffle=True,
        use_cached_filenames=True,
    )
    for i in tqdm.tqdm(range(len(dset))):
        obs, actions, rewards, dones = dset[i]
