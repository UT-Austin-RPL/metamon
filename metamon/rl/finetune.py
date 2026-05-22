"""Iterative finetuning with MetamonFinetuneAgent (tortoise EMA + IS correction).

Two independent knobs (example: finetuning ``Kakuna``):

- **Data** — ``--dataset_config`` YAML (``replay_weight``, ``self_play``,
  ``custom_replays``, optional ``prev_dataset`` / ``anneal_epochs``).
- **Weights** — omit ``--prev_run_*`` to start from HuggingFace; set
  ``--prev_run_dir/name/checkpoint`` to continue a prior finetune run.

``--base_model`` is always required (architecture, obs/action spaces, tokenizer).
See ``metamon.rl.dataset_config`` and ``metamon/rl/configs/datasets/`` for YAML
details. The repo ships ``self_play_dset.yaml`` as a starting mix.

Typical loop: finetune from HF → collect self-play → finetune again with new
YAML + ``--prev_run_*``.

**Iter 1 — HuggingFace weights**::

    python -m metamon.rl.finetune \\
        --run_name kakuna_iter1 --save_dir /path/to/ckpts \\
        --base_model Kakuna --dataset_config self_play_dset.yaml --log

**Collect self-play** (point ``custom_replays[].dir`` at the output pile in the
next YAML)::

    python -m metamon.rl.evaluate.ladder_self_play \\
        --config metamon/rl/evaluate/ladder_self_play/example_config.yaml \\
        --format gen1ou --gpus 0 1 2 3 \\
        --save_trajectories_to /path/to/kakuna_iter2_pile

**Iter 2+ — local weights + new YAML**::

    python -m metamon.rl.finetune \\
        --run_name kakuna_iter2 --save_dir /path/to/ckpts \\
        --base_model Kakuna \\
        --prev_run_dir /path/to/ckpts --prev_run_name kakuna_iter1 \\
        --prev_checkpoint 10 --dataset_config my_iter2.yaml --log

Example ``my_iter2.yaml`` (save under ``metamon/rl/configs/datasets/`` or pass
an absolute path)::

    replay_weight: 0.05
    prev_dataset: /path/to/ckpts/kakuna_iter1/dataset_config.yaml
    prev_weight: 0.75
    custom_replays:
      - dir: /path/to/kakuna_iter2_pile
        weight: 0.20
    anneal_epochs: 10
    formats:
      - gen1ou

Each run saves a flattened config to
``{save_dir}/{run_name}/dataset_config.yaml`` — reuse that path as ``prev_dataset``
on the next iteration.
"""

import os

import wandb

import metamon
from metamon.rl.train import (
    create_offline_rl_trainer,
    WANDB_PROJECT,
    WANDB_ENTITY,
)
from metamon.rl.dataset_config import (
    load_dataset_config,
    save_dataset_config,
    flatten_config,
    build_dataset,
)
from metamon.rl.pretrained import get_pretrained_model_names, get_pretrained_model
from metamon.interface import get_reward_function_names, get_reward_function


def add_cli(parser):
    # Identity
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    # Base model (architecture source)
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        choices=get_pretrained_model_names(),
        help="Registered pretrained model that defines the architecture.",
    )
    parser.add_argument(
        "--base_checkpoint",
        type=int,
        default=None,
        help="Checkpoint epoch of the base model. Only used on the first "
        "iteration (ignored when --prev_run_dir is set). "
        "Defaults to the model's default checkpoint.",
    )

    # Previous iteration (optional -- omit for the first round)
    parser.add_argument(
        "--prev_run_dir",
        type=str,
        default=None,
        help="--save_dir of the previous finetuning iteration. "
        "If omitted, initialises from --base_model directly.",
    )
    parser.add_argument(
        "--prev_run_name",
        type=str,
        default=None,
        help="--run_name of the previous finetuning iteration.",
    )
    parser.add_argument(
        "--prev_checkpoint",
        type=int,
        default=None,
        help="Checkpoint epoch to load from the previous iteration. "
        "Required when --prev_run_dir is set.",
    )

    # Training
    parser.add_argument("--train_gin_config", type=str, default="finetune.gin")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--steps_per_epoch", type=int, default=1_000)
    parser.add_argument("--batch_size_per_gpu", type=int, default=12)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--ckpt_interval", type=int, default=10)

    # Data
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="Path to a dataset config YAML file. Can include prev_dataset / "
        "anneal_epochs for iterative dataset transitions.",
    )

    # Reward
    parser.add_argument(
        "--reward_function",
        type=str,
        default=None,
        choices=get_reward_function_names(),
    )

    # Eval / infra
    parser.add_argument("--dloader_workers", type=int, default=10)
    parser.add_argument("--async_env_mp_context", type=str, default="forkserver")
    parser.add_argument(
        "--eval_gens",
        type=int,
        nargs="*",
        default=[1, 2, 3, 4, 9],
    )
    parser.add_argument("--log", action="store_true")
    return parser


def _resolve_checkpoint_path(args, pretrained):
    """Return the path to the policy weights file to load."""
    if args.prev_run_dir is not None:
        assert (
            args.prev_run_name is not None
        ), "--prev_run_name is required when --prev_run_dir is set"
        assert (
            args.prev_checkpoint is not None
        ), "--prev_checkpoint is required when --prev_run_dir is set"
        return os.path.join(
            args.prev_run_dir,
            args.prev_run_name,
            "ckpts",
            "policy_weights",
            f"policy_epoch_{args.prev_checkpoint}.pt",
        )
    ckpt = args.base_checkpoint or pretrained.default_checkpoint
    return pretrained.get_path_to_checkpoint(ckpt)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Iterative finetuning with MetamonFinetuneAgent."
    )
    add_cli(parser)
    args = parser.parse_args()

    metamon.print_banner()
    is_continuation = args.prev_run_dir is not None
    if is_continuation:
        print(
            f"  Finetune (iter): {args.prev_run_name} @ epoch {args.prev_checkpoint}"
            f"  →  {args.run_name}"
        )
    else:
        print(f"  Finetune (init): {args.base_model}  →  {args.run_name}")
    print(f"  Dataset config: {args.dataset_config}")
    print()

    pretrained = get_pretrained_model(args.base_model)

    dataset_config = load_dataset_config(args.dataset_config)

    # Auto-fill prev_dataset from the base model when the config declares
    # prev_weight (iterative finetuning) but omits prev_dataset.
    if dataset_config.prev_weight is not None and dataset_config.prev_dataset is None:
        if pretrained.dataset_config is None:
            raise ValueError(
                f"Base model '{args.base_model}' has no known dataset_config. "
                f"Set prev_dataset explicitly in your dataset config YAML."
            )
        dataset_config.prev_dataset = pretrained.dataset_config
        print(
            f"  prev_dataset inferred from {args.base_model}: "
            f"{pretrained.dataset_config}"
        )

    amago_dataset = build_dataset(
        config=dataset_config,
        obs_space=pretrained.observation_space,
        action_space=pretrained.action_space,
        reward_function=pretrained.reward_function,
    )

    # auto-save effective config to checkpoint directory
    config_save_path = os.path.join(args.save_dir, args.run_name, "dataset_config.yaml")
    save_dataset_config(flatten_config(dataset_config), config_save_path)
    print(f"  Dataset config saved to: {config_save_path}\n")

    reward_function = (
        get_reward_function(args.reward_function)
        if args.reward_function is not None
        else pretrained.reward_function
    )

    experiment = create_offline_rl_trainer(
        ckpt_dir=args.save_dir,
        run_name=args.run_name,
        model_gin_config=pretrained.model_gin_config_path,
        train_gin_config=args.train_gin_config,
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
        log=args.log,
        wandb_project=WANDB_PROJECT,
        wandb_entity=WANDB_ENTITY,
        manual_gin_overrides=pretrained.gin_overrides,
        ckpt_interval=args.ckpt_interval,
    )

    experiment.start()

    ckpt_path = _resolve_checkpoint_path(args, pretrained)
    print(f"  Loading weights from: {ckpt_path}")
    experiment.load_checkpoint_from_path(ckpt_path, is_accelerate_state=False)

    experiment.learn()
    wandb.finish()
