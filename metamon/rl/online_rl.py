"""Online RL finetuning from a registered pretrained model.

Architecture, observation/action spaces, tokenizer, and default train gin
come from ``--base_model`` (same registry as :mod:`metamon.rl.finetune`).

Collects self-play trajectories into a shared FIFO buffer (``MetamonFIFODataset``)
and trains on a mixture of online data + offline replays / self-play datasets.

Three roles that can run as one process (``--mode both``) or as separate
processes (``--mode collect|learn|validate``) sharing a checkpoint dir + buffer:

**Learner** (grad updates only; writes ``ckpts/latest/policy.pt`` each epoch)::

    python -m metamon.rl.online_rl \\
        --mode learn --run_name my_run --save_dir /path/to/ckpts \\
        --base_model TaurosV0 --buffer_dir /path/to/buffer --log

**Collector** (self-play rollouts into the FIFO buffer; syncs to
``latest/policy.pt`` each epoch; writes the PSRO-Lite sidecar)::

    python -m metamon.rl.online_rl \\
        --mode collect --run_name my_run --save_dir /path/to/ckpts \\
        --base_model TaurosV0 --buffer_dir /path/to/buffer --lanes 256

**Validator** (evaluates ``latest/policy.pt`` vs the val opponent each epoch)::

    python -m metamon.rl.online_rl \\
        --mode validate --run_name my_run --save_dir /path/to/ckpts \\
        --base_model TaurosV0 --buffer_dir /path/to/buffer --lanes 32 --log

**Single-process smoke test**::

    python -m metamon.rl.online_rl \\
        --mode both --run_name smoke --save_dir /tmp/online_smoke \\
        --base_model TaurosV0 --buffer_dir /tmp/online_smoke/buffer --lanes 2 \\
        --epochs 1 --train_timesteps_per_epoch 5 --steps_per_epoch 2 \\
        --dset_min_size 0 --val_timesteps 10

See ``.pi/skills/online-training/SKILL.md`` for the split-layout launch recipe
and resume/recovery flow. Concerns are split across companion modules:

- :mod:`metamon.rl.online_psro`  — PSRO-Lite CLI flags, config, sidecar helpers.
- :mod:`metamon.rl.online_schedule` — team-mix schedule (epoch-driven curriculum).
- :mod:`metamon.rl.online_envs`  — env factories, dataset builder, stat dropout,
  resolution helpers.
"""

from __future__ import annotations

import os
from functools import partial
from typing import Optional

import gin
import amago
import wandb

import metamon
from metamon.interface import get_reward_function, get_reward_function_names
from metamon.data import MetamonDataset
from metamon.rl.dataset_config import (
    flatten_config,
    load_dataset_config,
    save_dataset_config,
)
from metamon.rl.metamon_to_amago import (
    MetamonFIFODataset,
    MetamonOnlineExperiment,
    make_placeholder_env,
    mirror_online_experiment_gin_bindings,
)
from metamon.rl.pretrained import (
    get_pretrained_model,
    get_pretrained_model_names,
)
from metamon.rl.psro_lite import PsroConfig

# Split-out concerns.
from metamon.rl.online_envs import (
    DEFAULT_BATTLE_FORMAT,
    DEFAULT_TRAIN_POOL,
    DEFAULT_TRAIN_TEAM_SET,
    DEFAULT_VAL_TEAM_SET,
    ONLINE_RL_TRAIN_GIN,
    TRAINING_CONFIG_DIR,
    StatsDropoutObservationSpace,  # re-exported for backwards compat
    apply_async_mode,
    build_online_mixture_dataset,
    latest_training_state_epoch,
    make_collect_train_env,
    make_val_env,
    resolve_checkpoint_path,
    resolve_dataset_config_path,
    resolve_train_gin_path,
    resolve_val_opponent_config,
)
from metamon.rl.online_psro import (
    add_psro_cli_args,
    log_psro_status,
    make_opponent_weight_provider,
    make_psro_config_from_args,
    psro_sidecar_path,
    resolve_quota,
)
from metamon.rl.online_schedule import (
    ScheduleState,
    add_schedule_cli_args,
    log_schedule_start,
    make_schedule_state,
)


def _parse_teamset_weights(spec: Optional[str]) -> Optional[dict[str, float]]:
    """Parse a ``'set:mult,set:mult,...'`` spec into a ``{set: multiplier}`` dict.

    Returns ``None`` when ``spec`` is falsy (teamset up-sampling disabled).
    Mirrors the ``parse_team_mix_spec`` format but the numbers are multipliers
    (not normalized sampling weights).", """
    if not spec:
        return None
    out: dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"--fifo_teamset_weights: bad entry {part!r} (expected 'set:mult')")
        name, _, val = part.partition(":")
        name = name.strip()
        try:
            mult = float(val.strip())
        except ValueError:
            raise ValueError(f"--fifo_teamset_weights: bad multiplier {val!r} for {name!r}")
        if mult <= 0.0 or mult != mult:  # NaN
            raise ValueError(f"--fifo_teamset_weights: multiplier for {name!r} must be positive")
        out[name] = mult
    return out or None

# W&B defaults — override via METAMON_WANDB_PROJECT / METAMON_WANDB_ENTITY env vars
# (same env vars already used by make_placeholder_experiment for opponents).
WANDB_PROJECT = os.environ.get("METAMON_WANDB_PROJECT", "online-metamon")
WANDB_ENTITY = os.environ.get("METAMON_WANDB_ENTITY", "ut-austin-rpl-metamon")


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

    # Learner FIFO teamset up-sampling: shift the online mix toward trajectories
    # where the learner drew a specific teamset (parsed from the _ts- filename
    # token). Aggressively up-weight under-represented compositions (e.g. smogon)
    # so the policy trains on them more intensively. Independent of PSRO; composes
    # multiplicatively with --psro_fifo_reweight when both are set.
    parser.add_argument(
        "--fifo_teamset_weights",
        type=str,
        default=None,
        help="Up-sample online FIFO trajectories by learner teamset. Format: "
        "'set:multiplier,set:multiplier,...' e.g. "
        "'smogon_pass2:4.0,smogon_pass2_selected:4.0'. Files whose teamset isn't "
        "listed (or has no _ts- token) get --fifo_teamset_default_weight. "
        "Composes multiplicatively with --psro_fifo_reweight. Default None = disabled.",
    )
    parser.add_argument(
        "--fifo_teamset_default_weight",
        type=float,
        default=1.0,
        help="Weight multiplier for FIFO files whose learner teamset is not in "
        "--fifo_teamset_weights (including pre-token / _unknown files). Default 1.0.",
    )

    # PSRO-Lite (default-off) + team-mix schedule (optional, required for
    # "@schedule" opponent pools).
    add_psro_cli_args(parser)
    add_schedule_cli_args(parser)
    return parser


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
    train_team_schedule_state: Optional[ScheduleState] = None,
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
    psro_config: Optional[PsroConfig] = None,
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
        make_train_env = make_collect_train_env(
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
            schedule_state=train_team_schedule_state,
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
        make_val_env_fn = make_val_env(
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
        make_val_env_fn = partial(
            make_placeholder_env,
            pretrained.observation_space,
            pretrained.action_space,
        )
        effective_val_timesteps = 0
        effective_val_interval = None

    # The shared EpochRef (if a schedule is set) is stored on the experiment so
    # it can bump it each collection cycle, advancing the curriculum.
    epoch_ref = train_team_schedule_state.epoch_ref if train_team_schedule_state else None

    experiment = MetamonOnlineExperiment(
        run_name=run_name,
        ckpt_base_dir=save_dir,
        dataset=amago_dataset,
        make_train_env=make_train_env,
        make_val_env=make_val_env_fn,
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
        epoch_ref=epoch_ref,
    )
    if mode == "both":
        experiment = amago.cli_utils.switch_async_mode(experiment, mode)
    else:
        experiment = apply_async_mode(
            experiment,
            mode,
            val_timesteps=effective_val_timesteps,
            val_interval=effective_val_interval or 1,
            epochs=epochs,
            eval_during_training=eval_during_training,
        )
    return experiment


def run_online_rl(args) -> None:
    """Entry point: resolve config, build dataset + experiment, start training.

    Handles the three start modes: ``--resume_training_state`` (reload this
    run's full accelerate state and continue), ``--from_scratch`` (random init),
    and continuation/init (load a base or prior-run policy checkpoint). Bumps
    the shared schedule ``EpochRef`` to the resumed epoch on resume so the
    curriculum picks up at the right phase.
    """
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
    train_gin_config_path = resolve_train_gin_path(pretrained, args.train_gin_config)
    reward_function = (
        get_reward_function(args.reward_function)
        if args.reward_function is not None
        else pretrained.reward_function
    )

    dataset_config_path = resolve_dataset_config_path(args.dataset_config)
    dataset_config = load_dataset_config(dataset_config_path)
    battle_format = (
        args.battle_format
        or (dataset_config.formats[0] if dataset_config.formats else None)
        or DEFAULT_BATTLE_FORMAT
    )
    formats = dataset_config.formats or [battle_format]
    val_opponent_kwargs = resolve_val_opponent_config(
        val_pool_path=args.val_pool,
        val_opponent=args.val_opponent,
        base_model=args.base_model,
        battle_format=battle_format,
    )

    psro_config = make_psro_config_from_args(args, battle_format=battle_format)
    # The collector writes the sidecar; both the collector env and the learner's
    # FIFO sampler read it. ``opponent_weights_path`` is only meaningful for the
    # collector env; ``opponent_weight_provider`` is only meaningful for the
    # learner's FIFO dataset. Both fall back to uniform when the sidecar is
    # absent (e.g. before psro_start_epoch).
    sidecar_path = psro_sidecar_path(args.buffer_dir, battle_format)
    opponent_weights_path = sidecar_path if args.psro_weighting else None
    opponent_weight_provider = (
        make_opponent_weight_provider(sidecar_path)
        if args.psro_fifo_reweight
        else None
    )
    opponent_quota_min_games, opponent_quota_window = resolve_quota(args)
    log_psro_status(
        psro_config,
        sidecar_path=sidecar_path,
        fifo_reweight=args.psro_fifo_reweight,
        buffer_trim=args.psro_buffer_trim,
        quota_min_games=opponent_quota_min_games,
        quota_window=opponent_quota_window,
    )

    # Learner FIFO teamset up-sampling (independent of PSRO). Aggressively shifts
    # the learner's online 40% mix toward trajectories where the learner drew the
    # named teamsets, so the policy trains on those compositions more intensively.
    teamset_weights = _parse_teamset_weights(args.fifo_teamset_weights)
    if teamset_weights is not None:
        print(f"  FIFO teamset up-sampling: {teamset_weights} "
              f"(default={args.fifo_teamset_default_weight})")

    # Team-mix schedule: required when the training pool uses "@schedule"; the
    # collector's player team set and the pool's "@schedule" agents both follow
    # it via a shared EpochRef (bumped each collection cycle by the experiment).
    # Only meaningful for collect/both (the collector builds train envs); the
    # learner and validator receive None so they don't load the YAML pointlessly.
    schedule_state = (
        make_schedule_state(args.train_team_schedule)
        if args.mode in ("collect", "both")
        else None
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
            opponent_weight_provider=opponent_weight_provider,
            teamset_weights=teamset_weights,
            default_teamset_weight=args.fifo_teamset_default_weight,
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
            teamset_weights=teamset_weights,
            default_teamset_weight=args.fifo_teamset_default_weight,
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
        train_team_schedule_state=schedule_state,
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

    resume_epoch = None
    if args.resume_training_state:
        # True resume of THIS run: restore model + optimizer + scheduler + RNG from a
        # full accelerate state, then continue. learn() picks up at self.epoch.
        resume_epoch = (
            args.resume_epoch
            if args.resume_epoch is not None
            else latest_training_state_epoch(experiment.ckpt_dir, args.run_name)
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
        ckpt_path = resolve_checkpoint_path(args, pretrained)
        print(f"  Loading weights from: {ckpt_path}")
        experiment.load_checkpoint_from_path(ckpt_path, is_accelerate_state=False)

    # Bump the shared schedule EpochRef to the current (possibly resumed) epoch
    # so the curriculum picks up at the right phase. The experiment bumps it
    # again at the start of each collection cycle thereafter.
    if schedule_state is not None:
        start_epoch = resume_epoch if resume_epoch is not None else experiment.epoch
        schedule_state.epoch_ref.epoch = start_epoch
        log_schedule_start(schedule_state, resume_epoch=resume_epoch)

    experiment.learn()
    if args.log:
        wandb.finish()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Online RL finetuning from a registered pretrained model."
    )
    add_cli(parser)
    args = parser.parse_args()
    run_online_rl(args)
