"""
Model-based team prediction began as part of the changes that became version 1.0.
However, we already added an improved ReplayPredictor, and the need for the further
(learned) improvements is unclear at this time. Therefore work on team prediction
training is on hold and this script is mostly untested/TODO.

05/13/2025
"""

import os
import argparse
import html
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
from metamon.backend.team_prediction.team import TeamSet
from metamon.tokenizer import UNKNOWN_TOKEN


def compute_loss_and_accuracy(
    logits: torch.Tensor, y_tokens: torch.Tensor, pred_mask: torch.Tensor
) -> tuple[torch.Tensor, float]:
    """
    Computes cross-entropy loss and accuracy. Only masked positions are used for loss/accuracy.
    Returns: (loss, accuracy)
    """
    B, L, V = logits.shape
    loss = F.cross_entropy(
        logits.view(-1, V),
        y_tokens.view(-1),
        reduction="none",
        ignore_index=UNKNOWN_TOKEN,
    )
    num_preds = max(pred_mask.sum().item(), 1)
    loss = (loss * pred_mask.view(-1)).sum() / num_preds
    preds = logits.argmax(dim=-1)
    correct = ((preds == y_tokens) * pred_mask).sum().item()
    accuracy = correct / num_preds
    return loss, accuracy


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_steps: Optional[int] = None,
) -> tuple[float, float]:
    """
    Evaluate model on a dataloader. Returns (avg_loss, avg_accuracy).
    If max_steps is provided, only evaluate on that many batches.
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_steps = 0
    with torch.no_grad():
        for batch in dataloader:
            x_tokens, type_ids, y_tokens, pred_mask = batch
            x_tokens = x_tokens.to(device)
            type_ids = type_ids.to(device)
            y_tokens = y_tokens.to(device)
            pred_mask = pred_mask.to(device)
            logits = model(x_tokens, type_ids)
            loss, acc = compute_loss_and_accuracy(logits, y_tokens, pred_mask)
            total_loss += loss.item()
            total_acc += acc
            num_steps += 1
            if max_steps is not None and num_steps >= max_steps:
                break
    avg_loss = total_loss / max(num_steps, 1)
    avg_acc = total_acc / max(num_steps, 1)
    return avg_loss, avg_acc


def log_example_predictions(
    model: nn.Module,
    vocab: Vocabulary,
    x_tokens: torch.Tensor,
    type_ids: torch.Tensor,
    y_tokens: torch.Tensor,
    pred_masks: torch.Tensor,
    device: torch.device,
    num_examples: int,
    use_wandb: bool,
    epoch: int,
):
    """
    Log example predictions to wandb or print to console.
    """
    model.eval()
    x_tokens = x_tokens.to(device)
    type_ids = type_ids.to(device)
    pred_masks = pred_masks.to(device)
    logits = model(x_tokens, type_ids)
    probs = torch.softmax(logits, dim=-1)
    filt = vocab.filter_probs(probs, type_ids)
    bs, seq_len, vs = filt.shape
    flat = filt.view(-1, vs)
    sampled = torch.multinomial(flat, 1).view(bs, seq_len)
    # Use sampled predictions where pred_mask is True, otherwise keep input tokens
    merged = torch.where(pred_masks, sampled, x_tokens).cpu()

    table = wandb.Table(columns=["input", "predicted", "ground_truth"])
    for i in range(min(bs, num_examples)):
        x_seq = vocab.ints_to_pokeset_seq(x_tokens[i].cpu().tolist())
        pred_seq = vocab.ints_to_pokeset_seq(merged[i].tolist())
        true_seq = vocab.ints_to_pokeset_seq(y_tokens[i].tolist())
        mask = pred_masks[i].cpu()

        # Build HTML strings for wandb (colors render in tables)
        x_parts = []
        for x, m in zip(x_seq, mask):
            x_escaped = html.escape(x)
            if m:
                x_parts.append(
                    f'<span style="color: green; font-weight: bold">{x_escaped}</span>'
                )
            else:
                x_parts.append(x_escaped)
        x_str_html = " ".join(x_parts)

        pred_parts = []
        true_parts = []
        for p, t, m in zip(pred_seq, true_seq, mask):
            p_escaped = html.escape(p)
            t_escaped = html.escape(t)
            if m:
                # Blue if correct, red if wrong
                color = "blue" if p == t else "red"
                pred_parts.append(
                    f'<span style="color: {color}; font-weight: bold">{p_escaped}</span>'
                )
                true_parts.append(
                    f'<span style="color: {color}; font-weight: bold">{t_escaped}</span>'
                )
            else:
                pred_parts.append(p_escaped)
                true_parts.append(t_escaped)
        pred_str_html = " ".join(pred_parts)
        true_str_html = " ".join(true_parts)

        table.add_data(
            wandb.Html(f"<b>Input:</b><br>{x_str_html}"),
            wandb.Html(f"<b>Predicted:</b><br>{pred_str_html}"),
            wandb.Html(f"<b>Ground truth:</b><br>{true_str_html}"),
        )

    if use_wandb:
        wandb.log({"val/example_predictions": table}, step=epoch)
    else:
        # Console output with ANSI colors
        print(f"Examples at step {epoch}:")
        for i in range(min(bs, num_examples)):
            x_seq = vocab.ints_to_pokeset_seq(x_tokens[i].cpu().tolist())
            pred_seq = vocab.ints_to_pokeset_seq(merged[i].tolist())
            true_seq = vocab.ints_to_pokeset_seq(y_tokens[i].tolist())
            mask = pred_masks[i].cpu()

            x_str = " ".join(
                f"\033[92m{x}\033[0m" if m else x for x, m in zip(x_seq, mask)
            )
            pred_parts = []
            true_parts = []
            for p, t, m in zip(pred_seq, true_seq, mask):
                if m:
                    color = "\033[94m" if p == t else "\033[91m"
                    pred_parts.append(f"{color}{p}\033[0m")
                    true_parts.append(f"{color}{t}\033[0m")
                else:
                    pred_parts.append(p)
                    true_parts.append(t)

            print("---")
            print(f"Input: {x_str}")
            print(f"Predicted: {' '.join(pred_parts)}")
            print(f"Ground truth: {' '.join(true_parts)}")


def train(config, use_wandb: bool = True):
    # config: hyperparameters namespace or wandb.config

    # Set random seed
    random.seed(config.seed)
    torch.manual_seed(config.seed)

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

    # Debug overfit mode: use tiny subset (one batch), same data for train/val, no shuffle
    if config.debug_overfit:
        print(f"DEBUG OVERFIT MODE: Using {config.batch_size} samples (one batch)")
        from torch.utils.data import Subset

        indices = list(range(min(config.batch_size, len(train_dset))))
        train_dset = Subset(train_dset, indices)
        val_dset = Subset(train_dset, indices)  # Same data!
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

    # Initialize model
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
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    vocab = Vocabulary()

    ckpt_dir = os.path.join(config.checkpoint_dir, config.run_name)
    artifact_dir = os.path.join(ckpt_dir, "artifacts")
    os.makedirs(artifact_dir, exist_ok=True)

    best_val_loss = float("inf")
    patience_count = 0
    global_step = 0
    running_loss = 0.0
    running_acc = 0.0
    steps_since_eval = 0

    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    pbar = tqdm.tqdm(total=config.max_steps, desc="Training")

    while global_step < config.max_steps:
        try:
            x_tokens, type_ids, y_tokens, pred_mask = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x_tokens, type_ids, y_tokens, pred_mask = next(train_iter)

        model.train()
        x_tokens = x_tokens.to(device)
        type_ids = type_ids.to(device)
        y_tokens = y_tokens.to(device)
        pred_mask = pred_mask.to(device)
        logits = model(x_tokens, type_ids)
        loss, acc = compute_loss_and_accuracy(logits, y_tokens, pred_mask)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()

        running_loss += loss.item()
        running_acc += acc
        global_step += 1
        steps_since_eval += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.2%}"})
        pbar.update(1)

        if use_wandb and global_step % config.log_train_every_steps == 0:
            avg_train_loss = running_loss / steps_since_eval
            avg_train_acc = running_acc / steps_since_eval
            wandb.log(
                {
                    "global_step": global_step,
                    "train/loss": avg_train_loss,
                    "train/accuracy": avg_train_acc,
                },
                step=global_step,
            )

        if global_step % config.eval_every_steps == 0:
            train_loss = running_loss / steps_since_eval
            train_acc = running_acc / steps_since_eval
            running_loss = 0.0
            running_acc = 0.0
            steps_since_eval = 0

            val_loss, val_acc = evaluate(
                model, val_loader, device, max_steps=config.max_eval_steps
            )
            comp_loss, comp_acc = evaluate(
                model, comp_loader, device, max_steps=config.max_eval_steps
            )

            metrics = {
                "train": {"loss": train_loss, "accuracy": train_acc},
                "val": {
                    "replay_loss": val_loss,
                    "replay_accuracy": val_acc,
                    "competitive_loss": comp_loss,
                    "competitive_accuracy": comp_acc,
                },
            }
            if use_wandb:
                wandb_metrics = {"global_step": global_step}
                for split, split_metrics in metrics.items():
                    for metric_name, value in split_metrics.items():
                        wandb_metrics[f"{split}/{metric_name}"] = value
                wandb.log(wandb_metrics, step=global_step)
            else:
                print(f"\nStep {global_step}")
                print(
                    f"Train       - loss: {train_loss:.4f}, accuracy: {train_acc:.4f}"
                )
                print(f"Replay Val  - loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")
                print(
                    f"Competitive - loss: {comp_loss:.4f}, accuracy: {comp_acc:.4f}\n"
                )

            try:
                example_batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                example_batch = next(val_iter)
            x_tokens, type_ids, y_tokens, pred_masks = example_batch
            log_example_predictions(
                model=model,
                vocab=vocab,
                x_tokens=x_tokens,
                type_ids=type_ids,
                y_tokens=y_tokens,
                pred_masks=pred_masks,
                device=device,
                num_examples=config.num_examples,
                use_wandb=use_wandb,
                epoch=global_step,
            )

            if not config.debug_overfit:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_count = 0
                    best_model = os.path.join(ckpt_dir, "best_model.pt")
                    torch.save(model.state_dict(), best_model)
                    print(f"New best model saved to {best_model}")
                    if use_wandb:
                        artifact = wandb.Artifact(
                            f"{config.run_name}-best-model", type="model"
                        )
                        artifact.add_file(best_model)
                        wandb.log_artifact(artifact)
                else:
                    patience_count += 1
                    if patience_count >= config.patience:
                        print(f"Early stopping at step {global_step}")
                        break

    pbar.close()

    print("Training complete, saving final model...")

    final_model = os.path.join(ckpt_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model)
    print(f"Final model saved to {final_model}")

    if use_wandb:
        print("Uploading model artifact to wandb...")
        artifact = wandb.Artifact(f"{config.run_name}-final-model", type="model")
        artifact.add_file(final_model)
        wandb.log_artifact(artifact)
        print("Done uploading artifact.")


if __name__ == "__main__":
    from metamon.data.download import download_revealed_teams

    parser = argparse.ArgumentParser(
        description="Train TeamTransformer with optional W&B"
    )
    parser.add_argument("--project", type=str, help="W&B project name")
    parser.add_argument("--entity", type=str, help="W&B entity/user")
    parser.add_argument(
        "--group", type=str, default=None, help="W&B group name for sweeps"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging and print to console instead",
    )
    parser.add_argument(
        "--name", type=str, default=None, help="Run name to use for checkpoints and W&B"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--debug-overfit",
        action="store_true",
        help="Debug mode: overfit to a small number of samples (same for train/val)",
    )
    parser.add_argument(
        "--toy-names-only",
        action="store_true",
        help="Toy mode: only mask Pokemon names, keep everything else revealed",
    )
    args = parser.parse_args()

    # Default hyperparameters
    sweep_defaults = {
        "train_data_dir": download_revealed_teams(),
        "val_ratio": 0.1,
        "batch_size": 32,
        "num_workers": 4,
        "mask_pokemon_prob": 0.1,
        "mask_attrs_prob": 0.1,
        "seed": 42,
        "max_seq_len": 64,
        "d_model": 300,
        "nhead": 8,
        "num_layers": 4,
        "dim_ff": 1200,
        "dropout": 0.0,
        "learning_rate": 1e-4,
        "max_grad_norm": 1.0,
        "max_steps": 100000,
        "log_train_every_steps": 100,
        "eval_every_steps": 1000,
        "max_eval_steps": 100,
        "patience": 25,
        "weight_decay": 1e-4,
        "num_examples": 4,
        "debug_overfit": False,
        "toy_names_only": False,
    }

    if args.debug_overfit:
        sweep_defaults["debug_overfit"] = True
        sweep_defaults["log_train_every_steps"] = 1
        sweep_defaults["eval_every_steps"] = 10
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
    else:
        from argparse import Namespace

        cfg = Namespace(**sweep_defaults)
        cfg.checkpoint_dir = args.checkpoint_dir
        cfg.run_name = args.name or "local_run"

    train(cfg, use_wandb)
