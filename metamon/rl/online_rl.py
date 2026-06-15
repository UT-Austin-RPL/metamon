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
from functools import partial
from typing import Optional

import gin
import amago
import wandb

import metamon
from metamon.data import MetamonDataset
from metamon.env import get_metamon_teams
from metamon.interface import get_reward_function, get_reward_function_names
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
from metamon.rl.pretrained import (
    get_pretrained_model,
    get_pretrained_model_names,
)

WANDB_PROJECT = "online-metamon"
WANDB_ENTITY = "ut-austin-rpl-metamon"

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
        help="Validation env steps per epoch (--mode validate, or both). "
        "Ignored for collect/learn.",
    )
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--ckpt_interval", type=int, default=10)
    parser.add_argument("--dloader_workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log", action="store_true")
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
):
    """Offline replay mix + FIFO buffer of online-collected trajectories."""
    config = load_dataset_config(dataset_config_path)
    formats = config.formats or [battle_format]
    fifo_root = os.path.abspath(buffer_dir)
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
    fifo = MetamonFIFODataset(
        parsed_replay_dset=fifo_metamon,
        dset_max_size=dset_max_size,
        dset_min_size=dset_min_size,
        dset_name="Online FIFO Buffer",
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
):
    team_set = get_metamon_teams(battle_format, DEFAULT_TRAIN_TEAM_SET)
    return partial(
        make_metamon_env,
        battle_format=battle_format,
        observation_space=pretrained.observation_space,
        action_space=pretrained.action_space,
        reward_function=reward_function,
        team_set=team_set,
        opponent_config_path=opponent_config_path,
        batched_envs=lanes,
        n_workers=n_workers,
        opponent_sample=True,
        save_trajectories_to=buffer_dir,
        seed=seed,
    )


def _apply_async_mode(
    experiment,
    mode: str,
    *,
    val_timesteps: int,
    val_interval: int,
    epochs: int,
):
    """Configure distributed roles (AMAGO only ships collect/learn helpers)."""
    if mode == "collect":
        experiment = amago.cli_utils.make_experiment_collect_only(experiment)
        experiment.val_timesteps_per_epoch = 0
        experiment.val_interval = None
        return experiment
    if mode == "learn":
        experiment = amago.cli_utils.make_experiment_learn_only(experiment)
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
):
    team_set = get_metamon_teams(battle_format, DEFAULT_VAL_TEAM_SET)
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
    dloader_workers: int,
    seed: Optional[int],
    log: bool,
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
    runs_val = validates or (mode == "both" and val_timesteps > 0)

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
        traj_save_len=1e10,
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
        dloader_workers=args.dloader_workers,
        seed=args.seed,
        log=args.log,
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
