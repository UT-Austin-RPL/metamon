"""Online RL finetuning from a registered pretrained model.

Architecture, observation/action spaces, tokenizer, and default train gin
come from ``--base_model`` (same registry as :mod:`metamon.rl.finetune`).

Collects self-play trajectories into a shared FIFO buffer (``MetamonFIFODataset``)
and trains on a mixture of online data + offline replays / self-play datasets.

Launch patterns
---------------

**Learner** (4 GPUs via accelerate; gradient updates only)::

    accelerate launch -m metamon.rl.online_rl \\
        --mode learn --run_name my_run --save_dir /path/to/ckpts \\
        --base_model TaurosV0 --buffer_dir /path/to/buffer \\
        --dataset_config online_selfplay.yaml --log

**Validator** (single process; reloads ``latest/policy.pt`` each epoch)::

    python -m metamon.rl.online_rl \\
        --mode validate --run_name my_run --save_dir /path/to/ckpts \\
        --base_model TaurosV0 --buffer_dir /path/to/buffer --lanes 32 --log

**Collectors** (never run validation; placeholder val env only)::

    python -m metamon.rl.online_rl \\
        --mode collect --run_name my_run --save_dir /path/to/ckpts \\
        --base_model TaurosV0 --buffer_dir /path/to/buffer --lanes 256

**Single-process smoke test**::

    python -m metamon.rl.online_rl \\
        --mode both --run_name smoke --save_dir /tmp/online_smoke \\
        --base_model TaurosV0 --buffer_dir /tmp/online_smoke/buffer --lanes 2 \\
        --epochs 1 --train_timesteps_per_epoch 5 --steps_per_epoch 2 \\
        --dset_min_size 0 --val_timesteps 10
"""

from __future__ import annotations

import os
import copy
import random
from functools import partial
from typing import Optional, Callable

import gin
import amago
import wandb

import metamon
from metamon.data import MetamonDataset
from metamon.env import get_metamon_team_set_or_mix
from metamon.interface import (
    ObservationSpace,
    UniversalPokemon,
    UniversalState,
    get_reward_function,
    get_reward_function_names,
)
from metamon.rl.dataset_config import (
    DATASET_CONFIG_DIR,
    build_dataset,
    flatten_config,
    load_dataset_config,
    save_dataset_config,
)
from metamon.rl.metamon_to_amago import (
    MetamonFIFODataset,
    MetamonOnlineExperiment,
    make_metamon_env,
    make_placeholder_env,
    mirror_online_experiment_gin_bindings,
)
from metamon.rl.psro_lite import PsroConfig, read_sidecar
from metamon.rl.pretrained import (
    get_pretrained_model,
    get_pretrained_model_names,
)

# W&B defaults — override via METAMON_WANDB_PROJECT / METAMON_WANDB_ENTITY env vars
# (same env vars already used by make_placeholder_experiment for opponents).
WANDB_PROJECT = os.environ.get("METAMON_WANDB_PROJECT", "online-metamon")
WANDB_ENTITY = os.environ.get("METAMON_WANDB_ENTITY", "ut-austin-rpl-metamon")

OPPONENT_POOL_CONFIG_DIR = os.path.join(
    os.path.dirname(__file__), "configs", "opponent_pools"
)
TRAINING_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs", "training")
ONLINE_RL_TRAIN_GIN = os.path.join(TRAINING_CONFIG_DIR, "online_rl.gin")
DEFAULT_TRAIN_POOL = os.path.join(OPPONENT_POOL_CONFIG_DIR, "hl_gen1ou.yaml")
DEFAULT_BATTLE_FORMAT = "gen1ou"
DEFAULT_TRAIN_TEAM_SET = "gl_05_26"
DEFAULT_VAL_TEAM_SET = "competitive"


def add_cli(parser):
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["collect", "learn", "validate", "both"],
        help="collect: rollouts; learn: grad updates; validate: val only; both: smoke test.",
    )
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Shared checkpoint root (ckpt_base_dir). Also used for run metadata.",
    )
    parser.add_argument(
        "--buffer_dir",
        type=str,
        required=True,
        help="Shared directory for online-collected trajectories "
        "(written to {buffer_dir}/{format}/).",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        choices=get_pretrained_model_names(),
        help="Registered pretrained model (architecture, spaces, default train gin).",
    )
    parser.add_argument(
        "--base_checkpoint",
        type=int,
        default=None,
        help="HF checkpoint epoch of the base model (ignored when --prev_run_dir is set). "
        "Defaults to the model's default checkpoint.",
    )
    parser.add_argument(
        "--from_scratch",
        action="store_true",
        help="Train the --base_model architecture from random initialization: skip "
        "loading any pretrained/base checkpoint weights. Still uses the base model's "
        "arch gin, train gin, and spaces, plus the same offline mix (--dataset_config) "
        "and online buffer (--buffer_dir). The learner trains random weights; collectors "
        "and validators bootstrap from the learner's rolling latest/policy.pt. Cannot be "
        "combined with --prev_run_dir.",
    )
    parser.add_argument(
        "--resume_training_state",
        action="store_true",
        help="Resume the SAME run (--run_name/--save_dir) from a full accelerate "
        "training state (model + optimizer + scheduler + RNG), not just policy weights. "
        "Loads training_states/<run_name>_epoch_<N> (newest unless --resume_epoch is "
        "given) and continues through --epochs. Relaunch the learner with the same "
        "accelerate (GPU) config used originally. Incompatible with --from_scratch and "
        "--prev_run_dir.",
    )
    parser.add_argument(
        "--resume_epoch",
        type=int,
        default=None,
        help="Epoch whose full accelerate state to resume (default: newest saved). "
        "Only used with --resume_training_state.",
    )
    parser.add_argument(
        "--prev_run_dir",
        type=str,
        default=None,
        help="--save_dir of a prior online RL run to continue from.",
    )
    parser.add_argument(
        "--prev_run_name",
        type=str,
        default=None,
        help="--run_name of the prior online RL run.",
    )
    parser.add_argument(
        "--prev_checkpoint",
        type=int,
        default=None,
        help="Checkpoint epoch to load from the prior run. Required with --prev_run_dir.",
    )
    parser.add_argument(
        "--train_gin_config",
        type=str,
        default=None,
        help="Training gin config basename under configs/training/. "
        "Defaults to the base model's train_gin_config.",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="online_selfplay.yaml",
        help="Offline dataset mix YAML (human replays + self-play subsets).",
    )
    parser.add_argument(
        "--online_weight",
        type=float,
        default=0.40,
        help="Sampling weight for the online FIFO buffer (default 0.40 → 60%% offline).",
    )
    parser.add_argument(
        "--dset_max_size",
        type=int,
        default=300_000,
        help="FIFO evicts oldest replays once it exceeds this many files.",
    )
    parser.add_argument(
        "--dset_min_size",
        type=int,
        default=5000,
        help="FIFO must hold more than this many replay files before online sampling "
        "begins (default 5000 → needs 5001 files).",
    )
    parser.add_argument(
        "--online_anneal_epochs",
        type=int,
        default=20,
        help="Learner epochs to linearly ramp online FIFO sampling weight from 0 to "
        "--online_weight after the buffer becomes ready (also applies on restart when "
        "the buffer is already full).",
    )
    parser.add_argument(
        "--stats_dropout_prob",
        type=float,
        default=0.0,
        help="Probability of dropping computed battle stats (*_stat -> MISSING) on a "
        "sampled ONLINE FIFO trajectory, so the online buffer's stat distribution "
        "matches the offline data (which has stats == -1). Only meaningful with a "
        "computed-stats observation space (e.g. GroupedStatsObservationSpace). "
        "0.0 disables.",
    )
    parser.add_argument("--lanes", type=int, default=32)
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument(
        "--train_pool",
        type=str,
        default=DEFAULT_TRAIN_POOL,
        help="Opponent pool YAML for training collection.",
    )
    parser.add_argument(
        "--val_pool",
        type=str,
        default=None,
        help="Opponent pool YAML for validation. If omitted, uses --val_opponent.",
    )
    parser.add_argument(
        "--val_opponent",
        type=str,
        default=None,
        choices=get_pretrained_model_names(),
        help="Validation opponent model name when --val_pool is omitted. "
        "Defaults to --base_model.",
    )
    parser.add_argument(
        "--battle_format",
        type=str,
        default=None,
        help="Showdown battle format (e.g. gen1ou). Defaults to the first "
        "format in --dataset_config, else gen1ou.",
    )
    parser.add_argument(
        "--train_team_set",
        type=str,
        default=DEFAULT_TRAIN_TEAM_SET,
        help=f"Team set for collection (self-play) envs. Default: "
        f"{DEFAULT_TRAIN_TEAM_SET}.",
    )
    parser.add_argument(
        "--train_team_mix",
        type=str,
        default=None,
        help="Weighted mix spec for collection (self-play) envs, overriding "
        "--train_team_set when set. Format: "
        "'set_name:weight,set_name:weight,...' "
        "e.g. 'gl_05_26:0.45,smogon_pass2:0.35,smogon_pass2_selected:0.20'. "
        "Weights need not sum to 1.",
    )
    parser.add_argument(
        "--val_team_set",
        type=str,
        default=DEFAULT_VAL_TEAM_SET,
        help=f"Team set for validation envs. Default: {DEFAULT_VAL_TEAM_SET} "
        f"(gen9ou runs typically use gl_05_26).",
    )
    parser.add_argument(
        "--val_team_mix",
        type=str,
        default=None,
        help="Weighted mix spec for validation envs, overriding --val_team_set "
        "when set. Same format as --train_team_mix.",
    )
    parser.add_argument(
        "--reward_function",
        type=str,
        default=None,
        choices=get_reward_function_names(),
        help="Override reward function (defaults to the base model's reward).",
    )
    parser.add_argument("--temp_low", type=float, default=1.0)
    parser.add_argument("--temp_high", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--train_timesteps_per_epoch", type=int, default=500)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)
    parser.add_argument("--batch_size_per_gpu", type=int, default=14)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Peak AdamW LR (default 8e-5 from online_rl.gin).",
    )
    parser.add_argument(
        "--lr_warmup_epochs",
        type=float,
        default=20.0,
        help="Linear LR warmup length in training-epoch units "
        "(warmup_steps = this × steps_per_epoch × grad_accum).",
    )
    parser.add_argument(
        "--seq_floor_warmup_epochs",
        type=float,
        default=None,
        help="Epochs to ramp the ISAdvantageFilter sequence floor from 1.0 "
        "(filter off) down to seq_floor, in training-epoch units. Defaults to "
        "--lr_warmup_epochs. Increase to delay sequence-level filtering on a "
        "cold start while the critic adapts to the online distribution.",
    )
    parser.add_argument(
        "--val_timesteps",
        type=int,
        default=1000,
        help="Validation env steps per epoch (--mode validate, both, or learn "
        "with --eval_during_training). Ignored for collect/learn otherwise.",
    )
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument(
        "--eval_during_training",
        action="store_true",
        help="In --mode learn, periodically pause grad updates to run validation "
        "vs the val opponent (TaurosV0 by default, or --val_pool/--val_opponent) "
        "every --val_interval epochs for --val_timesteps env steps, logging win "
        "rate to wandb under the 'val/' panel. Builds and keeps the val envs "
        "alive for the whole run (CPU-sim overhead only on val epochs; no GPU "
        "cost beyond the learner's own forward passes). No-op in other modes.",
    )
    parser.add_argument("--ckpt_interval", type=int, default=10)
    parser.add_argument("--dloader_workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log", action="store_true")
    # PSRO-Lite: prioritized opponent sampling — all default-off (current
    # uniform behavior identical unless explicitly enabled). See
    # docs/psro_lite_plan.md.
    parser.add_argument(
        "--psro_weighting",
        action="store_true",
        help="Enable PSRO-Lite prioritized opponent sampling at collection time "
        "(writes meta_weights.json sidecar from the collector). No-op before "
        "--psro_start_epoch.",
    )
    parser.add_argument(
        "--psro_start_epoch",
        type=int,
        default=0,
        help="First epoch to write/apply PSRO-Lite weights (default 0 = always; "
        "the live run sets 1000 for a mid-run switchover).",
    )
    parser.add_argument(
        "--psro_temp",
        type=float,
        default=1.0,
        help="Prioritization temperature τ; small ⇒ sharp, large ⇒ uniform.",
    )
    parser.add_argument(
        "--psro_floor",
        type=float,
        default=0.05,
        help="Per-opponent diversity floor (non-zero mass for every opponent).",
    )
    parser.add_argument(
        "--psro_min_games",
        type=int,
        default=20,
        help="Minimum games vs an opponent before weighting it (else uniform fallback).",
    )
    parser.add_argument(
        "--psro_window",
        type=int,
        default=50000,
        help="Number of most-recent buffer files to score (rolling window).",
    )
    parser.add_argument(
        "--psro_update_interval",
        type=int,
        default=5,
        help="Epochs between PSRO-Lite weight updates (one forced update on the "
        "start epoch itself).",
    )
    parser.add_argument(
        "--psro_ema",
        type=float,
        default=0.7,
        help="EMA smoothing factor β for weights across updates (0=no smoothing).",
    )
    parser.add_argument(
        "--psro_solver",
        type=str,
        default="prioritized",
        choices=["prioritized", "nash"],
        help="Weight solver. 'prioritized' (PFSP-style) is v1; 'nash' is reserved "
        "for v3 (requires pool-vs-pool eval).",
    )
    parser.add_argument(
        "--psro_fifo_reweight",
        action="store_true",
        help="Per-trajectory FIFO reweighting: the learner's online 40%% mixture "
        "samples files in proportion to the current per-opponent weight instead "
        "of uniformly (fixes buffer lag at a mid-run switchover).",
    )
    parser.add_argument(
        "--psro_buffer_trim",
        type=int,
        default=None,
        help="If set, evict the FIFO down to this many files once at "
        "--psro_start_epoch to accelerate turnover of the uniform-sampled "
        "backlog (e.g. 50000).",
    )
    # Diversification quota: guarantees every pool agent a minimum number of
    # games over a rolling window so dominated, ladder-strong policies never
    # fall to ~0 games played (which previously triggered the cold-fallback
    # weight spike). The PSRO-Lite weights then tilt the *surplus* (window
    # slots beyond all quotas) toward weaker matchups.
    parser.add_argument(
        "--psro_quota_min_games",
        type=int,
        default=0,
        help="Per-agent minimum games over the rolling --psro_quota_window. "
        "0 disables the quota (pure weighted sampling). One configure() call "
        "assigns one shared opponent to all lanes for a battle, so the quota "
        "is enforced in units of ceil(min_games / lanes) assignments. Default "
        "0; the launch scripts set this when PSRO is on.",
    )
    parser.add_argument(
        "--psro_quota_window",
        type=int,
        default=128,
        help="Rolling window (in env reset / configure() calls) over which the "
        "per-agent quota is enforced. Must be >= n_agents * ceil(min_games / "
        "lanes) or the quota is infeasible and falls back to weighted sampling.",
    )
    parser.add_argument(
        "--psro_novelty",
        type=float,
        default=0.0,
        help="Decaying novelty bonus γ added to each opponent's raw weight: "
        "γ/(n+γ0). 0 (default) disables — the collection quota is the primary "
        "exploration mechanism. Set >0 to give genuinely novel opponents a "
        "small, n-decaying bump on top of the floor.",
    )
    parser.add_argument(
        "--psro_cap",
        type=float,
        default=None,
        help="Weight-ratio cap R: hard-bounds each raw weight to R*floor as a "
        "safety net against solver spikes. None (default) disables.",
    )
    return parser


def _resolve_dataset_config_path(path: str) -> str:
    if os.path.isabs(path) and os.path.exists(path):
        return path
    candidate = os.path.join(DATASET_CONFIG_DIR, path)
    if os.path.exists(candidate):
        return candidate
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"Dataset config not found: {path}")


def _latest_training_state_epoch(ckpt_dir: str, run_name: str) -> int:
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


def _resolve_checkpoint_path(args, pretrained) -> str:
    """Return the path to the policy weights file to load."""
    if args.prev_run_dir is not None:
        assert args.prev_run_name is not None, "--prev_run_name required with --prev_run_dir"
        assert args.prev_checkpoint is not None, "--prev_checkpoint required with --prev_run_dir"
        return os.path.join(
            args.prev_run_dir,
            args.prev_run_name,
            "ckpts",
            "policy_weights",
            f"policy_epoch_{args.prev_checkpoint}.pt",
        )
    ckpt = args.base_checkpoint or pretrained.default_checkpoint
    return pretrained.get_path_to_checkpoint(ckpt)


def _resolve_train_gin_path(pretrained, train_gin_config: Optional[str]) -> str:
    if train_gin_config is None:
        return pretrained.train_gin_config_path
    return os.path.join(metamon.rl.TRAINING_CONFIG_DIR, train_gin_config)


def _resolve_val_opponent_config(
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


def _psro_sidecar_path(buffer_dir: str, battle_format: str) -> str:
    return os.path.join(os.path.abspath(buffer_dir), battle_format, "meta_weights.json")


def _load_pool_agent_names(
    opponent_config_path: str, battle_format: str
) -> list[str]:
    """Return the row names (``agents[i][0]``) of the training opponent pool."""
    from metamon.rl.evaluate.opponent_pool import load_opponent_pool

    pool = load_opponent_pool(opponent_config_path, battle_format=battle_format)
    return [row[0] for row in pool.agents]


def _make_psro_config(args, *, battle_format: str) -> Optional[PsroConfig]:
    """Build a ``PsroConfig`` from CLI args, or ``None`` if PSRO-Lite is off."""
    if not args.psro_weighting:
        return None
    agent_names = _load_pool_agent_names(args.train_pool, battle_format)
    return PsroConfig(
        buffer_dir=args.buffer_dir,
        battle_format=battle_format,
        agent_names=agent_names,
        start_epoch=args.psro_start_epoch,
        update_interval=args.psro_update_interval,
        window=args.psro_window,
        min_games=args.psro_min_games,
        temp=args.psro_temp,
        floor=args.psro_floor,
        ema=args.psro_ema,
        solver=args.psro_solver,
        fifo_reweight=args.psro_fifo_reweight,
        buffer_trim=args.psro_buffer_trim,
        novelty_gamma=args.psro_novelty,
        cap_ratio=args.psro_cap,
    )


def _make_opponent_weight_provider(
    sidecar_path: str,
):
    """Return a callable that reads the PSRO-Lite sidecar (cached by mtime)."""
    state = {"mtime": None, "weights": None}

    def provider() -> dict[str, float]:
        weights, mtime = read_sidecar(sidecar_path, state["mtime"])
        if weights is not None:
            state["weights"] = weights
            state["mtime"] = mtime
        return state["weights"] or {}

    return provider


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


def _make_collect_train_env(
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
    opponent_weights_path: Optional[str] = None,
    opponent_quota_min_games: Optional[int] = None,
    opponent_quota_window: int = 128,
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
    )


def _apply_async_mode(
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


def _make_val_env(
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


def create_online_experiment(
    *,
    mode: str,
    run_name: str,
    save_dir: str,
    pretrained,
    train_gin_config_path: str,
    amago_dataset,
    battle_format: str,
    reward_function,
    opponent_config_path: str,
    val_opponent_kwargs: dict,
    buffer_dir: str,
    lanes: int,
    n_workers: int,
    train_team_set: str = DEFAULT_TRAIN_TEAM_SET,
    val_team_set: str = DEFAULT_VAL_TEAM_SET,
    train_team_mix: Optional[str] = None,
    val_team_mix: Optional[str] = None,
    temp_low: float,
    temp_high: float,
    epochs: int,
    train_timesteps_per_epoch: int,
    steps_per_epoch: int,
    batch_size_per_gpu: int,
    grad_accum: int,
    learning_rate: Optional[float],
    lr_warmup_epochs: float,
    seq_floor_warmup_epochs: Optional[float],
    val_timesteps: int,
    val_interval: int,
    ckpt_interval: int,
    eval_during_training: bool,
    dloader_workers: int,
    seed: Optional[int],
    log: bool,
    psro_config: Optional["PsroConfig"] = None,
    opponent_weights_path: Optional[str] = None,
    opponent_quota_min_games: Optional[int] = None,
    opponent_quota_window: int = 128,
):
    config = {
        "MetamonTstepEncoder.tokenizer": pretrained.observation_space.tokenizer,
        "MetamonPerceiverTstepEncoder.tokenizer": pretrained.observation_space.tokenizer,
        "MetamonGroupedTstepEncoderV2.tokenizer": pretrained.observation_space.tokenizer,
        "MetamonDiscrete.temperature": 1.0,
    }
    if pretrained.gin_overrides:
        config.update(pretrained.gin_overrides)
    if learning_rate is not None:
        config["MetamonAMAGOExperiment.learning_rate"] = learning_rate
    gin_files = [
        pretrained.model_gin_config_path,
        train_gin_config_path,
        ONLINE_RL_TRAIN_GIN,
    ]
    amago.cli_utils.use_config(config, gin_files, finalize=False)
    lr_warmup_steps = int(round(steps_per_epoch * grad_accum * lr_warmup_epochs))
    gin.bind_parameter("MetamonAMAGOExperiment.lr_warmup_steps", lr_warmup_steps)
    # Sequence-floor warmup is independent of LR warmup so a cold start can keep
    # the ISAdvantageFilter near-off longer. Falls back to lr_warmup_epochs.
    seq_warmup_epochs = (
        seq_floor_warmup_epochs
        if seq_floor_warmup_epochs is not None
        else lr_warmup_epochs
    )
    seq_floor_warmup_steps = int(round(steps_per_epoch * grad_accum * seq_warmup_epochs))
    try:
        gin.bind_parameter(
            "custom_agent.ISAdvantageFilter.seq_floor_warmup_steps",
            seq_floor_warmup_steps,
        )
    except ValueError:
        pass
    # These computed bindings are not part of ``config``/``gin_files``; record them
    # so MetamonOnlineExperiment._reload_gin can restore them after it clears the
    # opponent-polluted gin scope during collection (see _reload_gin docstring).
    gin_extra_bindings = {
        "MetamonAMAGOExperiment.lr_warmup_steps": lr_warmup_steps,
        "custom_agent.ISAdvantageFilter.seq_floor_warmup_steps": seq_floor_warmup_steps,
    }
    mirror_online_experiment_gin_bindings()
    gin.finalize()

    collects = mode in ("collect", "both")
    learns = mode in ("learn", "both")
    validates = mode == "validate"
    runs_val = (
        validates
        or (mode == "both" and val_timesteps > 0)
        or (mode == "learn" and eval_during_training and val_timesteps > 0)
    )

    if collects:
        make_train_env = _make_collect_train_env(
            pretrained,
            battle_format=battle_format,
            reward_function=reward_function,
            opponent_config_path=opponent_config_path,
            buffer_dir=buffer_dir,
            lanes=lanes,
            n_workers=n_workers,
            seed=seed,
            team_set_name=train_team_set,
            team_mix_spec=train_team_mix,
            opponent_weights_path=opponent_weights_path,
            opponent_quota_min_games=opponent_quota_min_games,
            opponent_quota_window=opponent_quota_window,
        )
        parallel_actors = lanes
        effective_train_timesteps = train_timesteps_per_epoch
    elif validates:
        make_train_env = partial(
            make_placeholder_env,
            pretrained.observation_space,
            pretrained.action_space,
        )
        parallel_actors = lanes
        effective_train_timesteps = 0
    else:
        make_train_env = partial(
            make_placeholder_env,
            pretrained.observation_space,
            pretrained.action_space,
        )
        parallel_actors = 1
        effective_train_timesteps = 0

    if runs_val and val_timesteps > 0:
        make_val_env = _make_val_env(
            pretrained,
            battle_format=battle_format,
            reward_function=reward_function,
            val_opponent_kwargs=val_opponent_kwargs,
            lanes=lanes,
            n_workers=n_workers,
            seed=seed,
            team_set_name=val_team_set,
            team_mix_spec=val_team_mix,
        )
        effective_val_timesteps = val_timesteps
        effective_val_interval = val_interval
    else:
        make_val_env = partial(
            make_placeholder_env,
            pretrained.observation_space,
            pretrained.action_space,
        )
        effective_val_timesteps = 0
        effective_val_interval = None

    experiment = MetamonOnlineExperiment(
        run_name=run_name,
        ckpt_base_dir=save_dir,
        dataset=amago_dataset,
        make_train_env=make_train_env,
        make_val_env=make_val_env,
        val_timesteps_per_epoch=effective_val_timesteps,
        env_mode="already_vectorized",
        parallel_actors=parallel_actors,
        exploration_wrapper_type=None,
        sample_actions_train=True,
        sample_actions_val=True,
        force_reset_train_envs_every=1 if collects else None,
        log_to_wandb=log,
        wandb_project=WANDB_PROJECT,
        wandb_entity=WANDB_ENTITY,
        verbose=True,
        padded_sampling="none",
        dloader_workers=dloader_workers,
        traj_save_len=int(1e10),
        stagger_traj_file_lengths=False,
        epochs=epochs,
        start_learning_at_epoch=0,
        start_collecting_at_epoch=0,
        train_timesteps_per_epoch=effective_train_timesteps,
        train_batches_per_epoch=steps_per_epoch * grad_accum if learns else 0,
        val_interval=effective_val_interval,
        ckpt_interval=ckpt_interval if learns else None,
        always_save_latest=True,
        always_load_latest=mode in ("collect", "validate"),
        batch_size=batch_size_per_gpu,
        batches_per_update=grad_accum,
        mixed_precision="no",
        temp_low=temp_low,
        temp_high=temp_high,
        gin_config=config,
        gin_config_files=gin_files,
        gin_extra_bindings=gin_extra_bindings,
        psro_config=psro_config,
    )
    if mode == "both":
        experiment = amago.cli_utils.switch_async_mode(experiment, mode)
    else:
        experiment = _apply_async_mode(
            experiment,
            mode,
            val_timesteps=effective_val_timesteps,
            val_interval=effective_val_interval or 1,
            epochs=epochs,
            eval_during_training=eval_during_training,
        )
    return experiment


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Online RL finetuning from a registered pretrained model."
    )
    add_cli(parser)
    args = parser.parse_args()

    metamon.print_banner()
    is_continuation = args.prev_run_dir is not None
    if args.from_scratch and is_continuation:
        raise ValueError(
            "--from_scratch cannot be combined with --prev_run_dir "
            "(continuation resumes saved weights, which contradicts random init)."
        )
    if args.resume_training_state and (args.from_scratch or is_continuation):
        raise ValueError(
            "--resume_training_state resumes THIS run's full accelerate state and is "
            "incompatible with --from_scratch and --prev_run_dir."
        )
    if args.resume_training_state:
        print(
            f"  Online RL (resume state): {args.run_name}  →  continue to "
            f"{args.epochs} epochs"
        )
    elif is_continuation:
        print(
            f"  Online RL (cont): {args.prev_run_name} @ epoch {args.prev_checkpoint}"
            f"  →  {args.run_name}"
        )
    elif args.from_scratch:
        print(
            f"  Online RL (scratch): {args.base_model} arch, random init "
            f"  →  {args.run_name}"
        )
    else:
        print(f"  Online RL (init): {args.base_model}  →  {args.run_name}")
    print(f"  mode={args.mode}  |  Buffer: {args.buffer_dir}")
    print(f"  Checkpoints: {args.save_dir}")
    if args.mode in ("learn", "both"):
        warmup_steps = int(
            round(args.steps_per_epoch * args.grad_accum * args.lr_warmup_epochs)
        )
        lr_note = (
            f"{args.learning_rate:g}"
            if args.learning_rate is not None
            else "8e-5 (online_rl.gin)"
        )
        print(
            f"  LR: {lr_note}  |  warmup {warmup_steps} grad steps "
            f"({args.lr_warmup_epochs:g} train epochs, linear 0→peak)"
        )
    if args.log:
        print(f"  W&B: {WANDB_ENTITY}/{WANDB_PROJECT}")
    print()

    pretrained = get_pretrained_model(args.base_model)
    train_gin_config_path = _resolve_train_gin_path(pretrained, args.train_gin_config)
    reward_function = (
        get_reward_function(args.reward_function)
        if args.reward_function is not None
        else pretrained.reward_function
    )

    dataset_config_path = _resolve_dataset_config_path(args.dataset_config)
    dataset_config = load_dataset_config(dataset_config_path)
    battle_format = (
        args.battle_format
        or (dataset_config.formats[0] if dataset_config.formats else None)
        or DEFAULT_BATTLE_FORMAT
    )
    formats = dataset_config.formats or [battle_format]
    val_opponent_kwargs = _resolve_val_opponent_config(
        val_pool_path=args.val_pool,
        val_opponent=args.val_opponent,
        base_model=args.base_model,
        battle_format=battle_format,
    )

    psro_config = _make_psro_config(args, battle_format=battle_format)
    # The collector writes the sidecar; both the collector env and the learner's
    # FIFO sampler read it. ``opponent_weights_path`` is only meaningful for the
    # collector env; ``opponent_weight_provider`` is only meaningful for the
    # learner's FIFO dataset. Both fall back to uniform when the sidecar is
    # absent (e.g. before psro_start_epoch).
    sidecar_path = _psro_sidecar_path(args.buffer_dir, battle_format)
    opponent_weights_path = sidecar_path if args.psro_weighting else None
    opponent_weight_provider = (
        _make_opponent_weight_provider(sidecar_path)
        if args.psro_fifo_reweight
        else None
    )
    # Quota-based diversification: guarantee every pool agent a minimum number
    # of games over a rolling window so dominated, ladder-strong policies never
    # fall to ~0 games played. Meaningful for collect/both (the collector env
    # draws opponents); no-op for learn/validate (no pool sampling). Defaults to
    # off (0); the launch scripts enable it alongside --psro_weighting.
    opponent_quota_min_games = (
        args.psro_quota_min_games if args.psro_quota_min_games > 0 else None
    )
    opponent_quota_window = args.psro_quota_window
    if psro_config is not None:
        print(f"  PSRO-Lite: ON (start_epoch={psro_config.start_epoch}, "
              f"solver={psro_config.solver}, temp={psro_config.temp}, "
              f"floor={psro_config.floor}, min_games={psro_config.min_games}, "
              f"window={psro_config.window}, update_interval={psro_config.update_interval}, "
              f"ema={psro_config.ema})")
        print(f"  PSRO agents: {psro_config.agent_names}")
        if args.psro_fifo_reweight:
            print(f"  PSRO FIFO reweighting: ON (sidecar={sidecar_path})")
        if args.psro_buffer_trim is not None:
            print(f"  PSRO buffer trim: {args.psro_buffer_trim} at start epoch")
    if opponent_quota_min_games is not None:
        print(f"  PSRO quota: min {opponent_quota_min_games} games/agent over "
              f"window={opponent_quota_window} resets")

    if args.mode == "collect":
        fifo_root = os.path.abspath(args.buffer_dir)
        os.makedirs(os.path.join(fifo_root, battle_format), exist_ok=True)
        fifo_metamon = MetamonDataset(
            dset_root=fifo_root,
            observation_space=pretrained.observation_space,
            action_space=pretrained.action_space,
            reward_function=reward_function,
            formats=formats,
            shuffle=True,
            verbose=False,
            write_index_cache=False,
        )
        amago_dataset = MetamonFIFODataset(
            parsed_replay_dset=fifo_metamon,
            dset_max_size=args.dset_max_size,
            dset_min_size=args.dset_min_size,
            dset_name="Online FIFO Buffer",
            opponent_weight_provider=opponent_weight_provider,
        )
    elif args.mode == "validate":
        amago_dataset = amago.loading.DoNothingDataset()
    else:
        amago_dataset = build_online_mixture_dataset(
            pretrained=pretrained,
            buffer_dir=args.buffer_dir,
            dataset_config_path=dataset_config_path,
            online_weight=args.online_weight,
            dset_max_size=args.dset_max_size,
            dset_min_size=args.dset_min_size,
            online_anneal_epochs=args.online_anneal_epochs,
            battle_format=battle_format,
            reward_function=reward_function,
            stats_dropout_prob=args.stats_dropout_prob,
            opponent_weight_provider=opponent_weight_provider,
        )

    config_save_path = os.path.join(args.save_dir, args.run_name, "dataset_config.yaml")
    save_dataset_config(flatten_config(dataset_config), config_save_path)
    print(f"  Offline dataset config saved to: {config_save_path}\n")

    experiment = create_online_experiment(
        mode=args.mode,
        run_name=args.run_name,
        save_dir=args.save_dir,
        pretrained=pretrained,
        train_gin_config_path=train_gin_config_path,
        amago_dataset=amago_dataset,
        battle_format=battle_format,
        reward_function=reward_function,
        opponent_config_path=args.train_pool,
        val_opponent_kwargs=val_opponent_kwargs,
        buffer_dir=args.buffer_dir,
        lanes=args.lanes,
        n_workers=args.n_workers,
        train_team_set=args.train_team_set,
        val_team_set=args.val_team_set,
        train_team_mix=args.train_team_mix,
        val_team_mix=args.val_team_mix,
        temp_low=args.temp_low,
        temp_high=args.temp_high,
        epochs=args.epochs,
        train_timesteps_per_epoch=args.train_timesteps_per_epoch,
        steps_per_epoch=args.steps_per_epoch,
        batch_size_per_gpu=args.batch_size_per_gpu,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_warmup_epochs=args.lr_warmup_epochs,
        seq_floor_warmup_epochs=args.seq_floor_warmup_epochs,
        val_timesteps=args.val_timesteps,
        val_interval=args.val_interval,
        ckpt_interval=args.ckpt_interval,
        eval_during_training=args.eval_during_training,
        dloader_workers=args.dloader_workers,
        seed=args.seed,
        log=args.log,
        psro_config=psro_config,
        opponent_weights_path=opponent_weights_path,
        opponent_quota_min_games=opponent_quota_min_games,
        opponent_quota_window=opponent_quota_window,
    )

    experiment.start()

    if args.resume_training_state:
        # True resume of THIS run: restore model + optimizer + scheduler + RNG from a
        # full accelerate state, then continue. learn() picks up at self.epoch.
        resume_epoch = (
            args.resume_epoch
            if args.resume_epoch is not None
            else _latest_training_state_epoch(experiment.ckpt_dir, args.run_name)
        )
        print(f"  Resuming full accelerate training state from epoch {resume_epoch} ...")
        experiment.load_checkpoint(resume_epoch, resume_training_state=True)
        print(
            f"  Resumed at epoch {experiment.epoch}; continuing to {args.epochs} "
            f"({max(args.epochs - experiment.epoch, 0)} epochs / "
            f"{max(args.epochs - experiment.epoch, 0) * args.steps_per_epoch * args.grad_accum:,} "
            f"grad steps remaining)."
        )
    elif args.from_scratch:
        # Random init from gin; do not load any pretrained/base weights. The learner
        # trains these random weights and publishes latest/policy.pt; collectors and
        # validators (always_load_latest=True) sync to it once available, and
        # read_latest_policy is a no-op until then (no deadlock with dset_min_size).
        print("  From scratch: random initialization (no checkpoint loaded).")
    else:
        ckpt_path = _resolve_checkpoint_path(args, pretrained)
        print(f"  Loading weights from: {ckpt_path}")
        experiment.load_checkpoint_from_path(ckpt_path, is_accelerate_state=False)

    experiment.learn()
    if args.log:
        wandb.finish()
