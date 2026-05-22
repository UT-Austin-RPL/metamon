"""YAML-based dataset configuration for metamon RL training.

A DatasetConfig captures the full composition of a training dataset:
replay weights, self-play subsets, and custom replay directories.

For iterative finetuning, configs can reference a previous iteration's
config via ``prev_dataset``, automatically flattening the chain and
computing annealing schedules for smooth data transitions.

Example base config (training from scratch)::

    # self_play_dset.yaml
    replay_weight: 0.05
    self_play:
      pac-base: 0.6
      pac-exploratory: 0.35
      # pac-tauros: 0.25  # gen1ou only; add when finetuning Tauros line
    # formats omitted → all metamon battle formats

Example iterative config (finetuning to a new iteration)::

    # my_iter1.yaml
    replay_weight: 0.05
    prev_dataset: self_play_dset.yaml
    prev_weight: 0.55
    custom_replays:
      - dir: /path/to/new_pile
        weight: 0.40
    anneal_epochs: 5
"""

import collections
import os
from dataclasses import dataclass
from typing import Optional

import yaml

import amago
from amago.loading import _DatasetStatus

import metamon
from metamon.data import ParsedReplayDataset, SelfPlayDataset, MetamonDataset
from metamon.interface import TokenizedObservationSpace, ActionSpace, RewardFunction
from metamon.rl.metamon_to_amago import MetamonAMAGODataset

DATASET_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs", "datasets")


@dataclass
class CustomReplaySource:
    dir: str
    weight: float


@dataclass
class DatasetConfig:
    """Declarative specification of a training dataset."""

    replay_weight: float = 0.05
    self_play: Optional[dict[str, float]] = None
    custom_replays: Optional[list[CustomReplaySource]] = None
    prev_dataset: Optional[str] = None
    prev_weight: Optional[float] = None
    anneal_epochs: Optional[int] = None
    formats: Optional[list[str]] = None


@dataclass
class _ResolvedEntry:
    """Single dataset source after flattening prev_dataset references."""

    dataset_type: str  # "self_play" or "custom_replay"
    identifier: str  # subset name or directory path
    weight: float
    is_new: bool  # True = new data for this iteration (annealed from 0)


@dataclass
class ResolvedDatasetConfig:
    """Fully resolved (flattened) dataset specification."""

    replay_weight: float
    entries: list[_ResolvedEntry]
    anneal_epochs: Optional[int]
    formats: Optional[list[str]]


def _resolve_config_path(path: str) -> str:
    """Resolve a config path: absolute paths pass through, relative paths
    are looked up in the ``configs/datasets/`` directory."""
    if os.path.isabs(path):
        return path
    candidate = os.path.join(DATASET_CONFIG_DIR, path)
    if os.path.exists(candidate):
        return candidate
    if os.path.exists(path):
        return os.path.abspath(path)
    return candidate


def load_dataset_config(path: str) -> DatasetConfig:
    """Load a DatasetConfig from a YAML file."""
    resolved_path = _resolve_config_path(path)
    with open(resolved_path) as f:
        raw = yaml.safe_load(f)

    custom_replays = None
    if "custom_replays" in raw and raw["custom_replays"]:
        custom_replays = [CustomReplaySource(**cr) for cr in raw["custom_replays"]]

    return DatasetConfig(
        replay_weight=raw.get("replay_weight", 0.05),
        self_play=raw.get("self_play"),
        custom_replays=custom_replays,
        prev_dataset=raw.get("prev_dataset"),
        prev_weight=raw.get("prev_weight"),
        anneal_epochs=raw.get("anneal_epochs"),
        formats=raw.get("formats"),
    )


def save_dataset_config(config: DatasetConfig, path: str) -> None:
    """Write a DatasetConfig to a YAML file."""
    data: dict = {"replay_weight": config.replay_weight}
    if config.self_play:
        data["self_play"] = config.self_play
    if config.custom_replays:
        data["custom_replays"] = [
            {"dir": cr.dir, "weight": cr.weight} for cr in config.custom_replays
        ]
    if config.prev_dataset is not None:
        data["prev_dataset"] = config.prev_dataset
    if config.prev_weight is not None:
        data["prev_weight"] = config.prev_weight
    if config.anneal_epochs is not None:
        data["anneal_epochs"] = config.anneal_epochs
    if config.formats is not None:
        data["formats"] = config.formats

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def resolve_dataset_config(config: DatasetConfig) -> ResolvedDatasetConfig:
    """Recursively flatten ``prev_dataset`` references into a flat entry list.

    When ``prev_dataset`` is set:
      - Load and resolve the referenced config.
      - Scale all of its entries by ``prev_weight``, preserving relative
        proportions within the previous iteration's non-replay data.
      - Tag them as ``is_new=False`` (existing data).
      - ``custom_replays`` on the current config become new data
        (``is_new=True``).

    When ``prev_dataset`` is *not* set (base config):
      - ``self_play`` and ``custom_replays`` are direct entries, all
        tagged ``is_new=False``.
    """
    entries: list[_ResolvedEntry] = []
    resolved_formats = config.formats

    if config.prev_dataset is not None:
        assert (
            config.prev_weight is not None
        ), "prev_weight is required when prev_dataset is set"
        prev_config = load_dataset_config(config.prev_dataset)
        prev_resolved = resolve_dataset_config(prev_config)

        if resolved_formats is None:
            resolved_formats = prev_resolved.formats

        prev_entries = prev_resolved.entries
        total_prev = sum(e.weight for e in prev_entries)

        if total_prev > 0:
            for e in prev_entries:
                entries.append(
                    _ResolvedEntry(
                        dataset_type=e.dataset_type,
                        identifier=e.identifier,
                        weight=config.prev_weight * (e.weight / total_prev),
                        is_new=False,
                    )
                )

        if config.custom_replays:
            for cr in config.custom_replays:
                entries.append(
                    _ResolvedEntry(
                        dataset_type="custom_replay",
                        identifier=cr.dir,
                        weight=cr.weight,
                        is_new=True,
                    )
                )
    else:
        if config.self_play:
            for subset, weight in config.self_play.items():
                entries.append(
                    _ResolvedEntry(
                        dataset_type="self_play",
                        identifier=subset,
                        weight=weight,
                        is_new=False,
                    )
                )
        if config.custom_replays:
            for cr in config.custom_replays:
                entries.append(
                    _ResolvedEntry(
                        dataset_type="custom_replay",
                        identifier=cr.dir,
                        weight=cr.weight,
                        is_new=False,
                    )
                )

    return ResolvedDatasetConfig(
        replay_weight=config.replay_weight,
        entries=entries,
        anneal_epochs=config.anneal_epochs,
        formats=resolved_formats,
    )


def flatten_config(config: DatasetConfig) -> DatasetConfig:
    """Resolve a config and convert it back to a flat DatasetConfig.

    The returned config has no ``prev_dataset`` / ``prev_weight`` /
    ``anneal_epochs`` -- all entries are inlined with their final effective
    weights.  Saving this to the checkpoint directory means later iterations
    can reference it without recursive resolution.
    """
    resolved = resolve_dataset_config(config)
    self_play: dict[str, float] = {}
    custom_replays: list[CustomReplaySource] = []

    for entry in resolved.entries:
        if entry.dataset_type == "self_play":
            self_play[entry.identifier] = entry.weight
        elif entry.dataset_type == "custom_replay":
            custom_replays.append(
                CustomReplaySource(dir=entry.identifier, weight=entry.weight)
            )

    return DatasetConfig(
        replay_weight=resolved.replay_weight,
        self_play=self_play or None,
        custom_replays=custom_replays or None,
        formats=resolved.formats,
    )


class TransitionMixtureOfDatasets(amago.loading.MixtureOfDatasets):
    """MixtureOfDatasets with per-dataset initial/final weight control.

    Used for iterative finetuning: old datasets start at inflated weights
    (filling the budget vacated by new datasets at weight 0) and linearly
    decrease to their target.  New datasets ramp from 0 to their target.

    For base configs (no annealing) this behaves identically to the
    standard ``MixtureOfDatasets``.
    """

    def __init__(
        self,
        datasets: list,
        initial_weights: list[float],
        final_weights: list[float],
        anneal_epochs: int,
        dset_name: Optional[str] = None,
    ):
        super().__init__(
            datasets=datasets,
            sampling_weights=final_weights,
            smooth_sudden_starts=anneal_epochs,
            dset_name=dset_name,
        )
        self._initial_weights = initial_weights
        self._final_weights = final_weights

    def configure_from_experiment(self, experiment):
        amago.loading.RLDataset.configure_from_experiment(self, experiment)
        for d in self.all_datasets:
            d.configure_from_experiment(experiment)

        self._dsets_status = []
        for d, iw, fw in zip(
            self.all_datasets, self._initial_weights, self._final_weights
        ):
            self._dsets_status.append(
                _DatasetStatus(
                    dataset=d,
                    initial_weight=iw,
                    final_weight=fw,
                    epoch_ready=0,
                )
            )
        self.update_dset_weights(0)
        self._sampling_metrics = collections.defaultdict(int)

    def update_dset_weights(self, epoch: int):
        """Bidirectional linear anneal that clamps correctly for both
        increasing (new data: 0 -> target) and decreasing (old data:
        inflated -> target) weight schedules."""
        self.check_configured()

        self._available_datasets = []
        for status in self._dsets_status:
            if self.smooth_sudden_starts is None:
                current_weight = status.final_weight
            else:
                m = (
                    status.final_weight - status.initial_weight
                ) / self.smooth_sudden_starts
                x = epoch - status.epoch_ready + 1
                raw = m * x + status.initial_weight
                lo = min(status.initial_weight, status.final_weight)
                hi = max(status.initial_weight, status.final_weight)
                current_weight = max(lo, min(hi, raw))
            self._available_datasets.append((status.dataset, current_weight))


def config_from_args(
    replay_weight: float = 1.0,
    self_play_subsets: Optional[list[str]] = None,
    self_play_weights: Optional[list[float]] = None,
    custom_replay_dir: Optional[str] = None,
    custom_replay_weight: float = 0.25,
    formats: Optional[list[str]] = None,
) -> DatasetConfig:
    """Build a DatasetConfig from individual keyword arguments (backward compat)."""
    self_play = None
    if self_play_subsets is not None:
        if self_play_weights is None:
            self_play_weights = [1.0] * len(self_play_subsets)
        elif len(self_play_weights) != len(self_play_subsets):
            raise ValueError(
                f"self_play_weights ({len(self_play_weights)}) must match "
                f"self_play_subsets ({len(self_play_subsets)})"
            )
        self_play = dict(zip(self_play_subsets, self_play_weights))

    custom_replays = None
    if custom_replay_dir is not None and custom_replay_weight > 0:
        custom_replays = [
            CustomReplaySource(dir=custom_replay_dir, weight=custom_replay_weight)
        ]

    return DatasetConfig(
        replay_weight=replay_weight,
        self_play=self_play,
        custom_replays=custom_replays,
        formats=formats,
    )


def build_dataset(
    config: DatasetConfig,
    obs_space: TokenizedObservationSpace,
    action_space: ActionSpace,
    reward_function: RewardFunction,
    verbose: bool = True,
    use_cached_filenames: bool = True,
    parsed_replay_dir: Optional[str] = None,
    split: Optional[str] = None,
    test_fraction: float = 0.1,
    split_seed: int = 42,
) -> amago.loading.RLDataset:
    """Construct an AMAGO dataset from a :class:`DatasetConfig`.

    Resolves the config (flattening ``prev_dataset`` references), creates
    the individual PyTorch datasets, and wraps them in a
    ``MixtureOfDatasets`` (or ``TransitionMixtureOfDatasets`` when
    annealing is requested).
    """
    resolved = resolve_dataset_config(config)
    formats = resolved.formats or metamon.config.SUPPORTED_BATTLE_FORMATS

    dset_kwargs = {
        "observation_space": obs_space,
        "action_space": action_space,
        "reward_function": reward_function,
        "max_seq_len": None,
        "formats": formats,
        "verbose": verbose,
        "use_cached_filenames": use_cached_filenames,
        "split": split,
        "test_fraction": test_fraction,
        "split_seed": split_seed,
    }

    datasets = []
    final_weights = []
    is_new_flags = []
    dataset_info = []

    # 1. Parsed Replays (human battles)
    if resolved.replay_weight > 0:
        parsed_dset = ParsedReplayDataset(dset_root=parsed_replay_dir, **dset_kwargs)
        datasets.append(
            MetamonAMAGODataset(
                dset_name="Parsed Replays (Human)",
                parsed_replay_dset=parsed_dset,
            )
        )
        final_weights.append(resolved.replay_weight)
        is_new_flags.append(False)
        dataset_info.append(
            ("Parsed Replays (Human)", len(parsed_dset), resolved.replay_weight)
        )

    # 2. Resolved entries (self-play + custom replays, possibly from prev configs)
    from metamon.data.download import SELF_PLAY_FORMATS, get_self_play_formats

    selfplay_formats = [f for f in formats if f in SELF_PLAY_FORMATS]
    selfplay_dset_kwargs = {**dset_kwargs, "formats": selfplay_formats}

    for entry in resolved.entries:
        if entry.dataset_type == "self_play":
            subset_formats = [
                f
                for f in selfplay_formats
                if f in get_self_play_formats(entry.identifier)
            ]
            if not subset_formats:
                if verbose:
                    print(
                        f"Skipping self-play subset {entry.identifier!r}: "
                        f"no formats overlap with {selfplay_formats}"
                    )
                continue
            sp_dset = SelfPlayDataset(
                subset=entry.identifier,
                **{**selfplay_dset_kwargs, "formats": subset_formats},
            )
            name = f"Self-Play ({entry.identifier})"
            datasets.append(
                MetamonAMAGODataset(dset_name=name, parsed_replay_dset=sp_dset)
            )
            final_weights.append(entry.weight)
            is_new_flags.append(entry.is_new)
            dataset_info.append((name, len(sp_dset), entry.weight))

        elif entry.dataset_type == "custom_replay":
            cr_dset = MetamonDataset(dset_root=entry.identifier, **dset_kwargs)
            label = os.path.basename(entry.identifier.rstrip("/"))
            name = f"Custom Replays ({label})"
            datasets.append(
                MetamonAMAGODataset(dset_name=name, parsed_replay_dset=cr_dset)
            )
            final_weights.append(entry.weight)
            is_new_flags.append(entry.is_new)
            dataset_info.append((name, len(cr_dset), entry.weight))

    if not datasets:
        raise ValueError("No datasets configured! Check your dataset config YAML.")

    # Renormalize
    total_weight = sum(final_weights)
    norm_weights = [w / total_weight for w in final_weights]

    if verbose:
        print("\n" + "=" * 70)
        print("TRAINING DATASET SUMMARY")
        print("=" * 70)
        print(f"{'Dataset':<40} {'Files':>10} {'Weight':>8} {'Norm':>8}")
        print("-" * 70)
        total_files = 0
        for (name, num_files, raw_weight), nw in zip(dataset_info, norm_weights):
            total_files += num_files
            print(f"{name:<40} {num_files:>10,} {raw_weight:>8.3f} {nw:>7.1%}")
        print("-" * 70)
        print(f"{'TOTAL':<40} {total_files:>10,} {total_weight:>8.3f} {'100.0%':>8}")
        print("=" * 70 + "\n")

    # Build the mixture
    if len(datasets) == 1:
        return datasets[0]

    has_new = any(is_new_flags)
    replay_idx = 0 if resolved.replay_weight > 0 else None
    if has_new and resolved.anneal_epochs is not None:
        # Full-transition annealing:
        #   - Replay weight stays constant
        #   - Old datasets start at inflated weights (filling budget vacated by new=0)
        #   - New datasets ramp from 0 to target
        replay_norm = norm_weights[replay_idx] if replay_idx is not None else 0.0
        old_nonreplay_total = sum(
            w
            for i, (w, is_new) in enumerate(zip(norm_weights, is_new_flags))
            if not is_new and i != replay_idx
        )
        nonreplay_budget = 1.0 - replay_norm

        initial_weights = []
        for i, (nw, is_new) in enumerate(zip(norm_weights, is_new_flags)):
            if is_new:
                initial_weights.append(0.0)
            elif i == replay_idx:
                initial_weights.append(replay_norm)
            else:
                initial_weights.append(
                    nonreplay_budget * (nw / old_nonreplay_total)
                    if old_nonreplay_total > 0
                    else nw
                )

        if verbose:
            print(
                f"  Annealing over {resolved.anneal_epochs} epochs: "
                f"old data {sum(iw for iw, f in zip(initial_weights, is_new_flags) if not f):.1%}"
                f" → {sum(nw for nw, f in zip(norm_weights, is_new_flags) if not f):.1%}"
            )
            print(
                f"  New data 0.0%"
                f" → {sum(nw for nw, f in zip(norm_weights, is_new_flags) if f):.1%}\n"
            )

        return TransitionMixtureOfDatasets(
            datasets=datasets,
            initial_weights=initial_weights,
            final_weights=norm_weights,
            anneal_epochs=resolved.anneal_epochs,
        )
    else:
        return amago.loading.MixtureOfDatasets(
            datasets=datasets,
            sampling_weights=norm_weights,
        )
