import os
from functools import partial
from typing import List, Optional

import wandb

import amago

import metamon
import metamon.rl
from metamon.env import get_metamon_teams
from metamon.interface import (
    TokenizedObservationSpace,
    ActionSpace,
    RewardFunction,
)
from metamon.tokenizer import get_tokenizer
from metamon.rl.metamon_to_amago import (
    MetamonAMAGOExperiment,
    MetamonAMAGODataset,
    make_baseline_env,
    make_placeholder_env,
)
from metamon.rl.dataset_config import (
    load_dataset_config,
    save_dataset_config,
    flatten_config,
    build_dataset,
)
from metamon import baselines

WANDB_PROJECT = os.environ.get("METAMON_WANDB_PROJECT")
WANDB_ENTITY = os.environ.get("METAMON_WANDB_ENTITY")
EVAL_OPPONENTS = [
    baselines.heuristic.basic.PokeEnvHeuristic,
    baselines.heuristic.basic.Gen1BossAI,
    baselines.heuristic.basic.Grunt,
    baselines.heuristic.basic.GymLeader,
    baselines.heuristic.kaizo.EmeraldKaizo,
]


def add_cli(parser):
    parser.add_argument(
        "--run_name",
        required=True,
        help="Give the run a name to identify logs and checkpoints.",
    )
    parser.add_argument(
        "--obs_space",
        type=str,
        default="TeamPreviewObservationSpace",
        help="See the README for a description of the different observation spaces.",
    )
    parser.add_argument(
        "--reward_function",
        type=str,
        default="DefaultShapedReward",
        help="See the README for a description of the different reward functions.",
    )
    parser.add_argument(
        "--action_space",
        type=str,
        default="DefaultActionSpace",
        help="See the README for a description of the different action spaces.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Path to save checkpoints. Find checkpoints under {save_dir}/{run_name}/ckpts/",
    )
    parser.add_argument(
        "--ckpt",
        type=int,
        default=None,
        help="Resume training from an existing run with this run_name. Provide the epoch checkpoint to load.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to train for. In offline RL model, an epoch is an arbitrary interval (here: 25k) of training steps on a fixed dataset.",
    )
    parser.add_argument(
        "--ckpt_interval",
        type=int,
        default=2,
        help="Save a checkpoint every N epochs.",
    )
    parser.add_argument(
        "--batch_size_per_gpu",
        type=int,
        default=12,
        help="Batch size per GPU. Total batch size is batch_size_per_gpu * num_gpus.",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=1,
        help="Number of gradient accumulations per update.",
    )
    parser.add_argument(
        "--model_gin_config",
        type=str,
        required=True,
        help="Path to a gin config file that edits the model architecture. See provided rl/configs/models/",
    )
    parser.add_argument(
        "--train_gin_config",
        type=str,
        required=True,
        help="Path to a gin config file that edits the training or hparams. See provided rl/configs/training/",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="DefaultObservationSpace-v1",
        help="The tokenizer to use for the text observation space. See metamon.tokenizer for options.",
    )
    parser.add_argument(
        "--dloader_workers",
        type=int,
        default=10,
        help="Number of workers for the data loader.",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="Path to a dataset config YAML file. See metamon/rl/configs/datasets/ for examples.",
    )
    parser.add_argument(
        "--async_env_mp_context",
        type=str,
        default="forkserver",
        help="Async environment setup method. Options: 'forkserver' (recommended, fast), 'fork' (fastest but unsafe with threads), 'spawn' (slowest but safest). Use 'spawn' only if others hang.",
    )
    parser.add_argument(
        "--eval_gens",
        type=int,
        nargs="*",
        default=[1, 2, 3, 4, 9],
        help="Generations (of OU) to play against heuristics between training epochs. Win rates usually saturate at 90%%+ quickly, so this is mostly a sanity-check. Reduce gens to save time on launch! Use `--eval_gens` (no arguments) to disable evaluation.",
    )
    parser.add_argument("--log", action="store_true", help="Log to wandb.")
    return parser


def create_offline_rl_trainer(
    ckpt_dir: str,
    run_name: str,
    model_gin_config: str,
    train_gin_config: str,
    obs_space: TokenizedObservationSpace,
    action_space: ActionSpace,
    reward_function: RewardFunction,
    amago_dataset: amago.loading.Dataset,
    eval_gens: List[int],
    async_env_mp_context: str = "spawn",
    val_timesteps_per_epoch: int = 300,
    dloader_workers: int = 8,
    epochs: int = 40,
    grad_accum: int = 1,
    steps_per_epoch: int = 25_000,
    batch_size_per_gpu: int = 16,
    log: bool = False,
    wandb_project: str = WANDB_PROJECT,
    wandb_entity: str = WANDB_ENTITY,
    manual_gin_overrides: Optional[dict] = None,
    ckpt_interval: int = 2,
):
    """
    Convenience function that creates an AMAGO experiment with default arguments
    set for offline RL in metamon.
    """
    # configuration
    config = {
        "MetamonTstepEncoder.tokenizer": obs_space.tokenizer,
        "MetamonPerceiverTstepEncoder.tokenizer": obs_space.tokenizer,
        "MetamonGroupedTstepEncoderV2.tokenizer": obs_space.tokenizer,
    }
    if manual_gin_overrides is not None:
        config.update(manual_gin_overrides)
    model_config_path = os.path.join(metamon.rl.MODEL_CONFIG_DIR, model_gin_config)
    training_config_path = os.path.join(
        metamon.rl.TRAINING_CONFIG_DIR, train_gin_config
    )
    amago.cli_utils.use_config(config, [model_config_path, training_config_path])

    # validation environments (evaluated throughout training)
    if eval_gens:
        make_envs = [
            partial(
                make_baseline_env,
                battle_format=f"gen{gen}ou",
                observation_space=obs_space,
                action_space=action_space,
                reward_function=reward_function,
                team_set=get_metamon_teams(f"gen{gen}ou", "competitive"),
                opponent_type=opponent,
            )
            for gen in set(eval_gens)
            for opponent in EVAL_OPPONENTS
        ]
    else:
        # turn off eval envs during training (do evals separately).
        make_envs = [partial(make_placeholder_env, obs_space, action_space)]
        val_timesteps_per_epoch = 0

    experiment = MetamonAMAGOExperiment(
        ## required ##
        run_name=run_name,
        ckpt_base_dir=ckpt_dir,
        dataset=amago_dataset,
        val_timesteps_per_epoch=val_timesteps_per_epoch,
        ## environment ##
        make_train_env=partial(make_placeholder_env, obs_space, action_space),
        make_val_env=make_envs,
        env_mode="async",
        async_env_mp_context=async_env_mp_context,
        parallel_actors=len(make_envs),
        exploration_wrapper_type=None,
        sample_actions_train=True,
        sample_actions_val=True,
        force_reset_train_envs_every=None,
        ## logging ##
        log_to_wandb=log,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        verbose=True,
        log_interval=300,
        ## replay ##
        padded_sampling="none",
        dloader_workers=dloader_workers,
        ## learning schedule ##
        epochs=epochs,
        start_learning_at_epoch=0,
        start_collecting_at_epoch=float("inf"),
        train_timesteps_per_epoch=0,
        train_batches_per_epoch=steps_per_epoch * grad_accum,
        val_interval=1,
        ckpt_interval=ckpt_interval,
        ## optimization ##
        batch_size=batch_size_per_gpu,
        batches_per_update=grad_accum,
        mixed_precision="no",
    )
    return experiment


if __name__ == "__main__":
    from argparse import ArgumentParser
    from metamon.interface import (
        get_observation_space,
        get_reward_function,
        get_action_space,
    )

    parser = ArgumentParser(
        description="Train a Metamon RL agent from scratch using offline RL on parsed replay data. "
        "This script trains new models using imitation learning or reinforcement learning objectives "
        "on the dataset of human Pokémon battles (& an optional custom dataset of self-play data you've collected)."
    )
    add_cli(parser)
    args = parser.parse_args()

    metamon.print_banner()
    print(
        f"  Run: {args.run_name}  |  Model: {args.model_gin_config}  |  Training: {args.train_gin_config}"
    )
    print(f"  Dataset config: {args.dataset_config}")
    print()

    # agent input/output/rewards
    obs_space = TokenizedObservationSpace(
        get_observation_space(args.obs_space), get_tokenizer(args.tokenizer)
    )
    reward_function = get_reward_function(args.reward_function)
    action_space = get_action_space(args.action_space)

    # load dataset config and build dataset
    dataset_config = load_dataset_config(args.dataset_config)
    amago_dataset = build_dataset(
        config=dataset_config,
        obs_space=obs_space,
        action_space=action_space,
        reward_function=reward_function,
    )

    # auto-save effective config to checkpoint directory
    config_save_path = os.path.join(args.save_dir, args.run_name, "dataset_config.yaml")
    save_dataset_config(flatten_config(dataset_config), config_save_path)
    print(f"  Dataset config saved to: {config_save_path}\n")

    # quick-setup for an offline RL experiment
    experiment = create_offline_rl_trainer(
        ckpt_dir=args.save_dir,
        run_name=args.run_name,
        model_gin_config=args.model_gin_config,
        train_gin_config=args.train_gin_config,
        obs_space=obs_space,
        action_space=action_space,
        reward_function=reward_function,
        amago_dataset=amago_dataset,
        eval_gens=args.eval_gens,
        async_env_mp_context=args.async_env_mp_context,
        dloader_workers=args.dloader_workers,
        epochs=args.epochs,
        grad_accum=args.grad_accum,
        batch_size_per_gpu=args.batch_size_per_gpu,
        log=args.log,
        wandb_project=WANDB_PROJECT,
        wandb_entity=WANDB_ENTITY,
        ckpt_interval=args.ckpt_interval,
    )
    experiment.start()
    if args.ckpt is not None:
        experiment.load_checkpoint(args.ckpt)
    experiment.learn()
    wandb.finish()
