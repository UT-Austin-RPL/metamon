import wandb

from metamon.rl.train import (
    create_offline_dataset,
    create_offline_rl_trainer,
    WANDB_PROJECT,
    WANDB_ENTITY,
)
from metamon.rl.pretrained import get_pretrained_model_names, get_pretrained_model
from metamon.interface import get_reward_function_names, get_reward_function


def add_cli(parser):
    parser.add_argument(
        "--run_name",
        required=True,
        help="Give the run a name to identify logs and checkpoints.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Path to save your custom checkpoints. Find checkpoints under save_dir/run_name/ckpts/",
    )
    parser.add_argument(
        "--finetune_from_model",
        type=str,
        required=True,
        choices=get_pretrained_model_names(),
        help="Name of a pretrained model to finetune from.",
    )
    parser.add_argument(
        "--finetune_from_checkpoint",
        type=int,
        default=None,
        help="Checkpoint number to finetune from. You can find a full list on HuggingFace: jakegrigsby/metamon. Most models have checkpoints in range(2, 42, 2). Defaults to the default evaluation checkpoint of the base model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to finetune for. In offline RL mode, an epoch is an arbitrary interval (here: 25k) of training steps on a fixed dataset.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=25_000,
        help="Number of training steps to perform per epoch. Convention is 25k, but you may want to go shorter if finetuning on a small dataset.",
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
        "--ckpt_interval",
        type=int,
        default=2,
        help="Save checkpoints every N epochs. Set to 1 to save all epochs (useful for EMA experiments). Default is 2.",
    )
    parser.add_argument(
        "--train_gin_config",
        type=str,
        default=None,
        help="Path to a gin config file that edits the training parameters. Note that when finetuning, you are not able to change settings that impact the model architecture (e.g., cannot switch an IL model to an RL update). Defaults to the same config as the base model.",
    )
    parser.add_argument(
        "--reward_function",
        type=str,
        default=None,
        choices=get_reward_function_names(),
        help="Defaults to the same reward function as the base model. See the README for a description of each reward function, or create your own!",
    )
    parser.add_argument(
        "--dloader_workers",
        type=int,
        default=10,
        help="Number of workers for the data loader.",
    )
    parser.add_argument(
        "--parsed_replay_dir",
        type=str,
        default=None,
        help="Path to the parsed replay directory. Defaults to the official huggingface version.",
    )
    parser.add_argument(
        "--custom_replay_dir",
        type=str,
        default=None,
        help="Path to an optional second parsed replay dataset (e.g., self-play data you've collected).",
    )
    parser.add_argument(
        "--custom_replay_sample_weight",
        type=float,
        default=0.25,
        help="[0, 1] portion of each batch to sample from the custom dataset (if provided).",
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
        help="Generations (of OU) to play against heuristics between training epochs. Win rates usually saturate at 90\%+ quickly, so this is mostly a sanity-check. Reduce gens to save time on launch! Use `--eval_gens` (no arguments) to disable evaluation.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=None,
        help="Showdown battle formats to include in the dataset. Defaults to all supported formats.",
    )
    parser.add_argument("--log", action="store_true", help="Log to wandb.")
    return parser


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Finetune a pretrained Metamon model from HuggingFace (jakegrigsby/metamon). "
        "This script allows you to continue training any of the published pretrained models "
        "on additional data or with modified training parameters. The model architecture "
        "and base configuration are inherited from the chosen pretrained model."
    )
    add_cli(parser)
    args = parser.parse_args()

    pretrained = get_pretrained_model(args.finetune_from_model)
    # create the dataset we'll be finetuning on
    amago_dataset = create_offline_dataset(
        obs_space=pretrained.observation_space,
        action_space=pretrained.action_space,
        reward_function=pretrained.reward_function,
        parsed_replay_dir=args.parsed_replay_dir,
        custom_replay_dir=args.custom_replay_dir,
        custom_replay_sample_weight=args.custom_replay_sample_weight,
        formats=args.formats,
    )
    if args.reward_function is not None:
        # custom reward function
        reward_function = get_reward_function(args.reward_function)
    else:
        # use the base reward function
        reward_function = pretrained.reward_function
    # create a new policy that matches the pretrained policy's architecture
    experiment = create_offline_rl_trainer(
        ckpt_dir=args.save_dir,
        run_name=args.run_name,
        model_gin_config=pretrained.model_gin_config_path,
        train_gin_config=args.train_gin_config or pretrained.train_gin_config_path,
        obs_space=pretrained.observation_space,
        action_space=pretrained.action_space,
        reward_function=reward_function,
        amago_dataset=amago_dataset,
        eval_gens=args.eval_gens,
        async_env_mp_context=args.async_env_mp_context,
        dloader_workers=args.dloader_workers,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        grad_accum=args.grad_accum,
        batch_size_per_gpu=args.batch_size_per_gpu,
        ckpt_interval=args.ckpt_interval,
        log=args.log,
        wandb_project=WANDB_PROJECT,
        wandb_entity=WANDB_ENTITY,
        manual_gin_overrides=pretrained.gin_overrides,
    )
    experiment.start()
    # load the pretrained checkpoint
    checkpoint = args.finetune_from_checkpoint or pretrained.default_checkpoint
    start_checkpoint = pretrained.get_path_to_checkpoint(checkpoint)
    experiment.load_checkpoint_from_path(
        start_checkpoint,
        is_accelerate_state=False,
    )
    # Update reference policy for dynamic damping (if enabled)
    if hasattr(experiment, 'update_reference_policy'):
        experiment.update_reference_policy()
    # finetune!
    experiment.learn()
    wandb.finish()
