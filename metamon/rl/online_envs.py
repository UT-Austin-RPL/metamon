"""Env / dataset concerns for online RL, split out of :mod:`metamon.rl.online_rl`.

Contains:
- Resolution helpers (dataset config, training-state epoch, checkpoint paths,
  val opponent config).
- :class:`StatsDropoutObservationSpace` — online-only stat dropout to match the
  offline data's ``MISSING_STAT`` distribution.
- :func:`build_online_mixture_dataset` — offline replay mix + FIFO buffer.
- Env factories (:func:`make_collect_train_env`, :func:`make_val_env`) and the
  split-role configurator :func:`apply_async_mode`.
"""

from __future__ import annotations

import copy
import os
import random
from functools import partial
from typing import Callable, Optional

import amago

import metamon
from metamon.data import MetamonDataset
from metamon.env import get_metamon_team_set_or_mix
from metamon.interface import (
    ObservationSpace,
    UniversalPokemon,
    UniversalState,
)
from metamon.rl.dataset_config import (
    DATASET_CONFIG_DIR,
    build_dataset,
    load_dataset_config,
)
from metamon.rl.metamon_to_amago import (
    MetamonFIFODataset,
    make_metamon_env,
    make_placeholder_env,
)
from metamon.rl.online_schedule import (
    ScheduleState,
    resolve_train_team_set,
)

# Defaults re-exported so online_rl.py keeps a single source of truth.
OPPONENT_POOL_CONFIG_DIR = os.path.join(
    os.path.dirname(__file__), "configs", "opponent_pools"
)
TRAINING_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs", "training")
ONLINE_RL_TRAIN_GIN = os.path.join(TRAINING_CONFIG_DIR, "online_rl.gin")
DEFAULT_TRAIN_POOL = os.path.join(OPPONENT_POOL_CONFIG_DIR, "hl_gen1ou.yaml")
DEFAULT_BATTLE_FORMAT = "gen1ou"
DEFAULT_TRAIN_TEAM_SET = "gl_05_26"
DEFAULT_VAL_TEAM_SET = "competitive"


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def resolve_dataset_config_path(path: str) -> str:
    if os.path.isabs(path) and os.path.exists(path):
        return path
    candidate = os.path.join(DATASET_CONFIG_DIR, path)
    if os.path.exists(candidate):
        return candidate
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"Dataset config not found: {path}")


def latest_training_state_epoch(ckpt_dir: str, run_name: str) -> int:
    """Newest epoch with a full accelerate state under ckpts/training_states/."""
    ts_dir = os.path.join(ckpt_dir, "training_states")
    prefix = f"{run_name}_epoch_"
    epochs = []
    if os.path.isdir(ts_dir):
        for name in os.listdir(ts_dir):
            if name.startswith(prefix):
                try:
                    epochs.append(int(name[len(prefix) :]))
                except ValueError:
                    pass
    if not epochs:
        raise FileNotFoundError(
            f"No full accelerate states found under {ts_dir} "
            f"(expected dirs like '{prefix}<N>')."
        )
    return max(epochs)


def resolve_checkpoint_path(args, pretrained) -> str:
    """Return the path to the policy weights file to load."""
    if args.prev_run_dir is not None:
        assert (
            args.prev_run_name is not None
        ), "--prev_run_name required with --prev_run_dir"
        assert (
            args.prev_checkpoint is not None
        ), "--prev_checkpoint required with --prev_run_dir"
        return os.path.join(
            args.prev_run_dir,
            args.prev_run_name,
            "ckpts",
            "policy_weights",
            f"policy_epoch_{args.prev_checkpoint}.pt",
        )
    ckpt = args.base_checkpoint or pretrained.default_checkpoint
    return pretrained.get_path_to_checkpoint(ckpt)


def resolve_train_gin_path(pretrained, train_gin_config: Optional[str]) -> str:
    if train_gin_config is None:
        return pretrained.train_gin_config_path
    return os.path.join(metamon.rl.TRAINING_CONFIG_DIR, train_gin_config)


def resolve_val_opponent_config(
    *,
    val_pool_path: Optional[str],
    val_opponent: Optional[str],
    base_model: str,
    battle_format: str,
):
    """Return kwargs for ``make_metamon_env`` opponent configuration."""
    if val_pool_path is not None:
        return {"opponent_config_path": val_pool_path}
    from metamon.rl.evaluate.opponent_pool import load_simple_opponent_pool

    opponent = val_opponent or base_model
    return {
        "opponent_config": load_simple_opponent_pool(
            opponent_agent=opponent,
            battle_format=battle_format,
            team_set="competitive",
            checkpoint=None,
            temperature=1.0,
        )
    }


# ---------------------------------------------------------------------------
# Stats dropout (online FIFO only)
# ---------------------------------------------------------------------------


class StatsDropoutObservationSpace(ObservationSpace):
    """Online-only wrapper that randomly hides computed battle stats.

    Newly collected self-play battles carry real per-Pokemon computed stats
    (``*_stat``), but the large offline replay dataset stores them as
    ``UniversalPokemon.MISSING_STAT`` (-1). To keep the online FIFO buffer's
    stat distribution compatible with the offline data -- and to stop the model
    from over-relying on a feature that is absent in most of its training data --
    we randomly drop the computed stats from sampled online trajectories before
    they are encoded.

    The dice are rolled once per trajectory in ``reset()`` (``MetamonDataset``
    calls ``reset()`` once and then ``state_to_obs`` per timestep), so a sampled
    trajectory is dropped as a whole, mirroring how real data either has or
    lacks computed stats for an entire battle.

    Applied ONLY to the FIFO (online) dataset's observation space; offline data
    and live collection / validation keep their real stats.
    """

    STAT_NAMES = ("hp", "atk", "def", "spa", "spd", "spe")

    def __init__(self, base_obs_space: ObservationSpace, dropout_prob: float):
        self.base_obs_space = base_obs_space
        self.dropout_prob = float(dropout_prob)
        self._drop_this_traj = False
        super().__init__()

    def reset(self):
        self.base_obs_space.reset()
        self._drop_this_traj = (
            self.dropout_prob > 0.0 and random.random() < self.dropout_prob
        )

    @property
    def gym_space(self):
        return self.base_obs_space.gym_space

    @property
    def tokenizable(self):
        return self.base_obs_space.tokenizable

    @property
    def tokenizer(self):
        # delegate so downstream code that reads obs_space.tokenizer still works
        return self.base_obs_space.tokenizer

    def _drop_stats(self, pokemon: UniversalPokemon) -> None:
        for stat in self.STAT_NAMES:
            setattr(pokemon, f"{stat}_stat", UniversalPokemon.MISSING_STAT)

    def state_to_obs(self, state: UniversalState):
        if self._drop_this_traj:
            state = copy.deepcopy(state)
            self._drop_stats(state.player_active_pokemon)
            self._drop_stats(state.opponent_active_pokemon)
            for pokemon in state.available_switches:
                self._drop_stats(pokemon)
        return self.base_obs_space.state_to_obs(state)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def build_online_mixture_dataset(
    *,
    pretrained,
    buffer_dir: str,
    dataset_config_path: str,
    online_weight: float,
    dset_max_size: int,
    dset_min_size: int,
    online_anneal_epochs: int,
    battle_format: str,
    reward_function,
    stats_dropout_prob: float = 0.0,
    opponent_weight_provider: Optional["Callable[[], dict[str, float]]"] = None,
    teamset_weights: Optional[dict[str, float]] = None,
    default_teamset_weight: float = 1.0,
):
    """Offline replay mix + FIFO buffer of online-collected trajectories."""
    config = load_dataset_config(dataset_config_path)
    formats = config.formats or [battle_format]
    fifo_root = os.path.abspath(buffer_dir)
    os.makedirs(os.path.join(fifo_root, battle_format), exist_ok=True)
    # Offline data already has stats == -1; only the online FIFO buffer needs
    # stat dropout to match that distribution.
    fifo_obs_space = pretrained.observation_space
    if stats_dropout_prob > 0.0:
        fifo_obs_space = StatsDropoutObservationSpace(
            base_obs_space=fifo_obs_space, dropout_prob=stats_dropout_prob
        )
    fifo_metamon = MetamonDataset(
        dset_root=fifo_root,
        observation_space=fifo_obs_space,
        action_space=pretrained.action_space,
        reward_function=reward_function,
        formats=formats,
        shuffle=True,
        verbose=False,
        write_index_cache=False,
    )
    fifo = MetamonFIFODataset(
        parsed_replay_dset=fifo_metamon,
        dset_max_size=dset_max_size,
        dset_min_size=dset_min_size,
        dset_name="Online FIFO Buffer",
        opponent_weight_provider=opponent_weight_provider,
        teamset_weights=teamset_weights,
        default_teamset_weight=default_teamset_weight,
    )
    offline_weight = 1.0 - online_weight
    if offline_weight <= 0:
        return fifo
    offline = build_dataset(
        config=config,
        obs_space=pretrained.observation_space,
        action_space=pretrained.action_space,
        reward_function=reward_function,
    )
    if online_weight <= 0:
        return offline
    # Always anneal online from 0 even when the FIFO is already full at startup.
    # Without explicit initial_sampling_weights, AMAGO's legacy path sets
    # initial_weight=final_weight for ready datasets and the ramp is a no-op.
    return amago.loading.MixtureOfDatasets(
        datasets=[fifo, offline],
        sampling_weights=[online_weight, offline_weight],
        initial_sampling_weights=[0.0, offline_weight],
        smooth_sudden_starts=online_anneal_epochs,
        dset_name="Online + Offline Mixture",
    )


# ---------------------------------------------------------------------------
# Env factories
# ---------------------------------------------------------------------------


def make_collect_train_env(
    pretrained,
    *,
    battle_format: str,
    reward_function,
    opponent_config_path: str,
    buffer_dir: str,
    lanes: int,
    n_workers: int,
    seed: Optional[int],
    team_set_name: str = DEFAULT_TRAIN_TEAM_SET,
    team_mix_spec: Optional[str] = None,
    schedule_state: Optional[ScheduleState] = None,
    opponent_weights_path: Optional[str] = None,
    opponent_quota_min_games: Optional[int] = None,
    opponent_quota_window: int = 128,
):
    """Build the collector (self-play) env factory.

    When ``schedule_state`` is set, the player team set is schedule-aware
    (epoch-driven mix) and the opponent pool receives the same schedule +
    EpochRef so its ``"@schedule"`` agents follow the curriculum too. Otherwise
    falls back to the static team set / mix spec.
    """
    team_set = resolve_train_team_set(
        battle_format,
        team_set_name=team_set_name,
        team_mix_spec=team_mix_spec,
        schedule_state=schedule_state,
    )
    extra: dict = {}
    if schedule_state is not None:
        extra["opponent_team_schedule"] = schedule_state.schedule
        extra["opponent_epoch_ref"] = schedule_state.epoch_ref
    return partial(
        make_metamon_env,
        battle_format=battle_format,
        observation_space=pretrained.observation_space,
        action_space=pretrained.action_space,
        reward_function=reward_function,
        team_set=team_set,
        opponent_config_path=opponent_config_path,
        opponent_weights_path=opponent_weights_path,
        opponent_quota_min_games=opponent_quota_min_games,
        opponent_quota_window=opponent_quota_window,
        batched_envs=lanes,
        n_workers=n_workers,
        opponent_sample=True,
        save_trajectories_to=buffer_dir,
        save_results_to=buffer_dir,
        seed=seed,
        **extra,
    )


def apply_async_mode(
    experiment,
    mode: str,
    *,
    val_timesteps: int,
    val_interval: int,
    epochs: int,
    eval_during_training: bool = False,
):
    """Configure distributed roles (AMAGO only ships collect/learn helpers)."""
    if mode == "collect":
        experiment = amago.cli_utils.make_experiment_collect_only(experiment)
        experiment.val_timesteps_per_epoch = 0
        experiment.val_interval = None
        return experiment
    if mode == "learn":
        experiment = amago.cli_utils.make_experiment_learn_only(experiment)
        if eval_during_training and val_timesteps > 0:
            # Keep periodic validation enabled so the learner pauses every
            # `val_interval` epochs to evaluate the in-memory policy vs the val
            # opponent (TaurosV0 by default) and logs win rate to wandb.
            # always_load_latest stays False (set by make_experiment_learn_only),
            # so val uses the current learned weights, not a reloaded snapshot.
            experiment.val_timesteps_per_epoch = val_timesteps
            experiment.val_interval = val_interval
        else:
            experiment.val_timesteps_per_epoch = 0
            experiment.val_interval = None
        return experiment
    if mode == "validate":
        experiment.start_collecting_at_epoch = float("inf")
        experiment.train_timesteps_per_epoch = 0
        experiment.start_learning_at_epoch = float("inf")
        experiment.train_batches_per_epoch = 0
        experiment.ckpt_interval = None
        experiment.always_save_latest = False
        experiment.always_load_latest = True
        experiment.epochs = max(epochs, 1_000_000)
        experiment.has_dset_edit_rights = False
        experiment.val_timesteps_per_epoch = val_timesteps
        experiment.val_interval = val_interval
        experiment.init_dsets()
        return experiment
    return experiment


def make_val_env(
    pretrained,
    *,
    battle_format: str,
    reward_function,
    val_opponent_kwargs: dict,
    lanes: int,
    n_workers: int,
    seed: Optional[int],
    team_set_name: str = DEFAULT_VAL_TEAM_SET,
    team_mix_spec: Optional[str] = None,
):
    team_set = (
        get_metamon_team_set_or_mix(battle_format, team_mix_spec)
        if team_mix_spec
        else get_metamon_team_set_or_mix(battle_format, team_set_name)
    )
    return partial(
        make_metamon_env,
        battle_format=battle_format,
        observation_space=pretrained.observation_space,
        action_space=pretrained.action_space,
        reward_function=reward_function,
        team_set=team_set,
        batched_envs=lanes,
        n_workers=n_workers,
        opponent_sample=True,
        seed=seed,
        **val_opponent_kwargs,
    )
