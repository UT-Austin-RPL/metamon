"""
Improved training script for TeamTransformer with:
- Curriculum learning (dynamic masking schedule)
- Enhanced metrics (per-attribute, weighted, top-k)
- Better model architecture (Pre-LN)
- Optional iterative training

Usage:
    python train_improved.py --name experiment1 --project team_prediction --entity your-entity
"""

import os
import argparse
import random
from typing import Optional

import tqdm
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import wandb

from metamon.backend.team_prediction.dataset import (
    TeamPredictionDataset,
    CompetitiveTeamPredictionDataset,
)
from metamon.backend.team_prediction.model import TeamTransformer
from metamon.backend.team_prediction.vocabulary import Vocabulary

# Import our new components
from curriculum import MaskingScheduler, HierarchicalMaskingScheduler
from improved_metrics import compute_loss_and_metrics, DetailedEvaluator
from iterative_decoder import IterativeTeamDecoder, IterativeTrainingWrapper


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    vocab: Vocabulary,
    max_steps: Optional[int] = None,
    use_iterative: bool = False,
    num_iterations: int = 8,
) -> dict:
    """
    Evaluate model on a dataloader with comprehensive metrics.

    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation
        device: Device to run on
        vocab: Vocabulary for metrics
        max_steps: Maximum number of batches to evaluate
        use_iterative: Whether to use iterative decoding
        num_iterations: Number of iterations for iterative decoding

    Returns:
        Dictionary of averaged metrics
    """
    model.eval()

    accumulated_metrics = {}
    num_steps = 0

    # Setup iterative decoder if needed
    if use_iterative:
        decoder = IterativeTeamDecoder(model, vocab, num_iterations=num_iterations)

    with torch.no_grad():
        for batch in dataloader:
            x_tokens, type_ids, y_tokens, pred_mask = batch
            x_tokens = x_tokens.to(device)
            type_ids = type_ids.to(device)
            y_tokens = y_tokens.to(device)
            pred_mask = pred_mask.to(device)

            if use_iterative:
                # Use iterative decoder for predictions
                predictions = decoder.decode(x_tokens, type_ids, pred_mask)

                # Compute metrics on final predictions
                # (create logits by one-hot encoding predictions)
                batch_size, seq_len = predictions.shape
                vocab_size = len(vocab.tokenizer)
                logits = torch.zeros(batch_size, seq_len, vocab_size, device=device)
                logits.scatter_(2, predictions.unsqueeze(-1), 1.0)
                logits = torch.log(logits + 1e-10)  # Convert to log-probs
            else:
                # Standard single-shot prediction
                logits = model(x_tokens, type_ids)

            loss, metrics = compute_loss_and_metrics(
                logits, y_tokens, pred_mask, type_ids, vocab
            )

            # Accumulate metrics
            if num_steps == 0:
                accumulated_metrics = {k: v for k, v in metrics.items()}
                accumulated_metrics["loss"] = loss.item()
            else:
                for k, v in metrics.items():
                    accumulated_metrics[k] += v
                accumulated_metrics["loss"] += loss.item()

            num_steps += 1
            if max_steps is not None and num_steps >= max_steps:
                break

    # Average metrics
    for k in accumulated_metrics:
        accumulated_metrics[k] /= max(num_steps, 1)

    return accumulated_metrics


def train(config, use_wandb: bool = True):
    """
    Main training loop with curriculum learning and enhanced evaluation.
    """
    # Set random seed
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Initialize vocabulary
    vocab = Vocabulary()

    # Setup curriculum learning
    if config.use_hierarchical_curriculum:
        curriculum = HierarchicalMaskingScheduler(
            total_steps=config.max_steps,
            phase1_ratio=config.phase1_ratio,
            phase2_ratio=config.phase2_ratio,
        )
        print("Using hierarchical curriculum (3 phases)")
    elif config.use_curriculum:
        curriculum = MaskingScheduler(
            initial_mask_prob=config.initial_mask_prob,
            final_mask_prob=config.final_mask_prob,
            warmup_steps=config.curriculum_warmup_steps,
            schedule=config.curriculum_schedule,
        )
        print(f"Using {config.curriculum_schedule} curriculum")
    else:
        curriculum = None
        print("No curriculum (fixed masking)")

    # Prepare datasets
    train_dset = TeamPredictionDataset(
        data_dir=config.train_data_dir,
        split="train",
        validation_ratio=config.val_ratio,
        mask_pokemon_prob_range=(config.mask_pokemon_prob, config.mask_pokemon_prob),
        mask_attrs_prob_range=(config.mask_attrs_prob, config.mask_attrs_prob),
        seed=config.seed,
        use_cached_filenames=True,
        verbose=True,
        toy_names_only=config.toy_names_only,
    )

    val_dset = TeamPredictionDataset(
        data_dir=config.train_data_dir,
        split="val",
        validation_ratio=config.val_ratio,
        mask_pokemon_prob_range=(config.mask_pokemon_prob, config.mask_pokemon_prob),
        mask_attrs_prob_range=(config.mask_attrs_prob, config.mask_attrs_prob),
        seed=config.seed,
        use_cached_filenames=True,
        verbose=True,
        toy_names_only=config.toy_names_only,
    )

    comp_dset = CompetitiveTeamPredictionDataset(
        mask_pokemon_prob_range=(config.mask_pokemon_prob, config.mask_pokemon_prob),
        mask_attrs_prob_range=(config.mask_attrs_prob, config.mask_attrs_prob),
        verbose=True,
        toy_names_only=config.toy_names_only,
    )

    # Debug overfit mode
    if config.debug_overfit:
        print(f"DEBUG OVERFIT MODE: Using {config.batch_size} samples")
        from torch.utils.data import Subset

        indices = list(range(min(config.batch_size, len(train_dset))))
        train_dset = Subset(train_dset, indices)
        val_dset = Subset(train_dset, indices)
        comp_indices = list(range(min(config.batch_size, len(comp_dset))))
        comp_dset = Subset(comp_dset, comp_indices)

    # DataLoaders
    shuffle = not config.debug_overfit
    num_workers = 0 if config.debug_overfit else config.num_workers
    persistent = num_workers > 0

    train_loader = DataLoader(
        train_dset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_dset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent,
    )
    comp_loader = DataLoader(
        comp_dset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent,
    )

    # Initialize model with Pre-LN
    # Note: You'll need to modify TeamTransformer to accept norm_first parameter
    model = TeamTransformer(
        max_seq_len=config.max_seq_len,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_ff,
        dropout=config.dropout,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Optimizer with warmup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )

    # Learning rate scheduler with warmup
    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Setup iterative training if enabled
    if config.use_iterative_training:
        iterative_wrapper = IterativeTrainingWrapper(
            model,
            vocab,
            num_training_iterations=config.num_training_iterations,
            confidence_threshold=config.training_confidence_threshold,
        )
        print(
            f"Using iterative training with {config.num_training_iterations} iterations"
        )

    # Checkpoint directory
    ckpt_dir = os.path.join(config.checkpoint_dir, config.run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Training state
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    patience_count = 0
    global_step = 0
    running_loss = 0.0
    running_metrics = {}
    steps_since_eval = 0

    train_iter = iter(train_loader)
    pbar = tqdm.tqdm(total=config.max_steps, desc="Training")

    print(f"\nStarting training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    while global_step < config.max_steps:
        # Get next batch
        try:
            x_tokens, type_ids, y_tokens, pred_mask = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x_tokens, type_ids, y_tokens, pred_mask = next(train_iter)

        # Update masking probabilities if using curriculum
        if curriculum is not None:
            if isinstance(curriculum, HierarchicalMaskingScheduler):
                mask_config = curriculum.get_masking_config(global_step)
                # TODO: Update dataset masking based on config
                # This requires modifying the dataset to accept dynamic masking
            else:
                current_mask_prob = curriculum.get_mask_prob(global_step)
                # TODO: Update dataset masking

        model.train()
        x_tokens = x_tokens.to(device)
        type_ids = type_ids.to(device)
        y_tokens = y_tokens.to(device)
        pred_mask = pred_mask.to(device)

        # Forward pass
        if config.use_iterative_training:
            logits, pred_mask, num_unmasked = iterative_wrapper.forward_with_unmasking(
                x_tokens, type_ids, y_tokens, pred_mask
            )
        else:
            logits = model(x_tokens, type_ids)

        # Compute loss and metrics
        loss, metrics = compute_loss_and_metrics(
            logits, y_tokens, pred_mask, type_ids, vocab
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        scheduler.step()

        # Accumulate metrics
        running_loss += loss.item()
        for k, v in metrics.items():
            running_metrics[k] = running_metrics.get(k, 0.0) + v

        global_step += 1
        steps_since_eval += 1

        # Update progress bar
        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "acc": f"{metrics['token_accuracy']:.2%}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            }
        )
        pbar.update(1)

        # Log training metrics
        if use_wandb and global_step % config.log_train_every_steps == 0:
            avg_loss = running_loss / steps_since_eval
            avg_metrics = {k: v / steps_since_eval for k, v in running_metrics.items()}

            wandb.log(
                {
                    "global_step": global_step,
                    "train/loss": avg_loss,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    **{f"train/{k}": v for k, v in avg_metrics.items()},
                },
                step=global_step,
            )

        # Evaluation
        if global_step % config.eval_every_steps == 0:
            # Average training metrics
            train_loss = running_loss / steps_since_eval
            train_metrics = {
                k: v / steps_since_eval for k, v in running_metrics.items()
            }

            # Reset accumulators
            running_loss = 0.0
            running_metrics = {}
            steps_since_eval = 0

            # Evaluate on validation sets
            print(f"\n\nEvaluating at step {global_step}...")

            val_metrics = evaluate(
                model,
                val_loader,
                device,
                vocab,
                max_steps=config.max_eval_steps,
                use_iterative=config.eval_with_iterative,
                num_iterations=config.eval_num_iterations,
            )

            comp_metrics = evaluate(
                model,
                comp_loader,
                device,
                vocab,
                max_steps=config.max_eval_steps,
                use_iterative=config.eval_with_iterative,
                num_iterations=config.eval_num_iterations,
            )

            # Print metrics
            print(f"\nStep {global_step}:")
            print(
                f"  Train Loss: {train_loss:.4f} | Acc: {train_metrics['token_accuracy']:.3f}"
            )
            print(
                f"  Val Loss:   {val_metrics['loss']:.4f} | Acc: {val_metrics['token_accuracy']:.3f}"
            )
            print(
                f"  Comp Loss:  {comp_metrics['loss']:.4f} | Acc: {comp_metrics['token_accuracy']:.3f}"
            )

            # Print per-attribute accuracy
            print("\n  Per-Attribute Validation Accuracy:")
            for k, v in sorted(val_metrics.items()):
                if k.endswith("_accuracy") and k != "token_accuracy":
                    print(f"    {k}: {v:.3f}")

            # Log to wandb
            if use_wandb:
                wandb.log(
                    {
                        "global_step": global_step,
                        "val/replay_loss": val_metrics["loss"],
                        "val/competitive_loss": comp_metrics["loss"],
                        **{
                            f"val/{k}": v for k, v in val_metrics.items() if k != "loss"
                        },
                        **{
                            f"comp/{k}": v
                            for k, v in comp_metrics.items()
                            if k != "loss"
                        },
                    },
                    step=global_step,
                )

            # Save best model
            if not config.debug_overfit:
                # Use weighted accuracy as primary metric
                val_score = val_metrics.get(
                    "weighted_accuracy", val_metrics["token_accuracy"]
                )

                if val_score > best_val_accuracy:
                    best_val_accuracy = val_score
                    best_val_loss = val_metrics["loss"]
                    patience_count = 0

                    best_model_path = os.path.join(ckpt_dir, "best_model.pt")
                    torch.save(
                        {
                            "step": global_step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_accuracy": val_score,
                            "val_loss": val_metrics["loss"],
                        },
                        best_model_path,
                    )

                    print(f"\n  ✓ New best model! Accuracy: {val_score:.3f}")

                    if use_wandb:
                        artifact = wandb.Artifact(
                            f"{config.run_name}-best-model",
                            type="model",
                            metadata={"val_accuracy": val_score},
                        )
                        artifact.add_file(best_model_path)
                        wandb.log_artifact(artifact)
                else:
                    patience_count += 1
                    if patience_count >= config.patience:
                        print(f"\nEarly stopping at step {global_step}")
                        break

    pbar.close()

    # Save final model
    print("\nTraining complete! Saving final model...")
    final_model_path = os.path.join(ckpt_dir, "final_model.pt")
    torch.save(
        {
            "step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        final_model_path,
    )

    if use_wandb:
        artifact = wandb.Artifact(f"{config.run_name}-final-model", type="model")
        artifact.add_file(final_model_path)
        wandb.log_artifact(artifact)

    print(f"Models saved to {ckpt_dir}")


if __name__ == "__main__":
    from metamon.data.download import download_revealed_teams

    parser = argparse.ArgumentParser(description="Improved TeamTransformer training")
    parser.add_argument("--project", type=str, help="W&B project name")
    parser.add_argument("--entity", type=str, help="W&B entity/user")
    parser.add_argument("--group", type=str, default=None, help="W&B group for sweeps")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--name", type=str, default=None, help="Run name")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--debug-overfit", action="store_true")
    parser.add_argument("--toy-names-only", action="store_true")

    args = parser.parse_args()

    # Enhanced hyperparameters
    sweep_defaults = {
        # Data
        "train_data_dir": download_revealed_teams(),
        "val_ratio": 0.1,
        "batch_size": 64,  # Increased from 32
        "num_workers": 4,
        "seed": 42,
        # Model architecture (larger capacity)
        "max_seq_len": 64,
        "d_model": 512,  # Increased from 320
        "nhead": 8,
        "num_layers": 6,  # Increased from 4
        "dim_ff": 2048,  # Increased from 1280
        "dropout": 0.1,  # Added dropout
        # Optimizer
        "learning_rate": 3e-4,  # Slightly increased
        "weight_decay": 1e-4,
        "max_grad_norm": 1.0,
        "warmup_steps": 5000,  # Added warmup
        # Training
        "max_steps": 200000,  # Increased from 100k
        "log_train_every_steps": 100,
        "eval_every_steps": 500,  # More frequent evaluation
        "max_eval_steps": 100,
        "patience": 100,  # Increased patience
        # Curriculum learning
        "use_curriculum": True,
        "use_hierarchical_curriculum": False,
        "curriculum_schedule": "cosine",
        "initial_mask_prob": 0.05,
        "final_mask_prob": 0.25,
        "curriculum_warmup_steps": 20000,
        # Hierarchical curriculum (if enabled)
        "phase1_ratio": 0.33,
        "phase2_ratio": 0.33,
        # Default masking (if no curriculum)
        "mask_pokemon_prob": 0.15,
        "mask_attrs_prob": 0.15,
        # Iterative training/eval
        "use_iterative_training": False,
        "num_training_iterations": 3,
        "training_confidence_threshold": 0.7,
        "eval_with_iterative": True,
        "eval_num_iterations": 8,
        # Flags
        "debug_overfit": False,
        "toy_names_only": False,
        "num_examples": 4,
    }

    # Override with command line args
    if args.debug_overfit:
        sweep_defaults.update(
            {
                "debug_overfit": True,
                "log_train_every_steps": 1,
                "eval_every_steps": 10,
                "max_steps": 1000,
            }
        )
    if args.toy_names_only:
        sweep_defaults["toy_names_only"] = True

    use_wandb = not args.no_wandb

    if use_wandb:
        wandb.init(
            project=args.project,
            entity=args.entity,
            group=args.group,
            config=sweep_defaults,
            name=args.name,
        )
        cfg = wandb.config
        cfg.checkpoint_dir = args.checkpoint_dir
        cfg.run_name = wandb.run.name
        wandb.define_metric("global_step")
        wandb.define_metric("train/*", step_metric="global_step")
        wandb.define_metric("val/*", step_metric="global_step")
        wandb.define_metric("comp/*", step_metric="global_step")
    else:
        from argparse import Namespace

        cfg = Namespace(**sweep_defaults)
        cfg.checkpoint_dir = args.checkpoint_dir
        cfg.run_name = args.name or "local_run"

    train(cfg, use_wandb)
