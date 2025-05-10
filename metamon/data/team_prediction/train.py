import os
import argparse
import random
import re

import tqdm
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import wandb

from metamon.data.team_prediction.dataset import (
    TeamPredictionDataset,
    CompetitiveTeamPredictionDataset,
)
from metamon.data.team_prediction.model import TeamTransformer
from metamon.data.team_prediction.vocabulary import Vocabulary
from metamon.data.team_prediction.team import TeamSet


def compute_loss_and_accuracy(
    logits: torch.Tensor, y_tokens: torch.Tensor, pred_mask: torch.Tensor
) -> tuple[torch.Tensor, float]:
    """
    Computes cross-entropy loss and accuracy. Only masked positions are used for loss/accuracy.
    Returns: (loss, accuracy)
    """
    B, L, V = logits.shape
    loss = F.cross_entropy(logits.view(-1, V), y_tokens.view(-1), reduction="none")
    num_preds = max(pred_mask.sum().item(), 1)
    loss = (loss * pred_mask.view(-1)).sum() / num_preds
    preds = logits.argmax(dim=-1)
    correct = ((preds == y_tokens) * pred_mask).sum().item()
    accuracy = correct / num_preds
    return loss, accuracy


def evaluate(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> tuple[float, float]:
    """
    Evaluate model on a dataloader. Returns (avg_loss, avg_accuracy).
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Evaluating"):
            x_tokens, type_ids, y_tokens, pred_mask = batch
            x_tokens = x_tokens.to(device)
            type_ids = type_ids.to(device)
            y_tokens = y_tokens.to(device)
            pred_mask = pred_mask.to(device)
            logits = model(x_tokens, type_ids)
            loss, acc = compute_loss_and_accuracy(logits, y_tokens, pred_mask)
            total_loss += loss.item()
            total_acc += acc
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    return avg_loss, avg_acc


def wandb_to_console_color(text: str) -> str:
    # Replace :blue[], :red[], :green[] with ANSI codes
    text = re.sub(r":blue\[(.*?)\]", r"\033[94m\1\033[0m", text)
    text = re.sub(r":red\[(.*?)\]", r"\033[91m\1\033[0m", text)
    text = re.sub(r":green\[(.*?)\]", r"\033[92m\1\033[0m", text)
    return text


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
        mask = pred_masks[i]
        x_str = " ".join(f":green[{x}]" if m else x for x, m in zip(x_seq, mask))
        pred_str = []
        true_str = []
        for p, t, m in zip(pred_seq, true_seq, mask):
            if m:
                color = ":blue[" if p == t else ":red["
                end = "]"
            else:
                color = ""
                end = ""
            pred_str.append(f"{color}{p}{end}")
            true_str.append(f"{color}{t}{end}")
        pred_str = " ".join(pred_str)
        true_str = " ".join(true_str)

        table.add_data(
            f"**Input**:\n{x_str}",
            f"**Predicted**:\n{pred_str}",
            f"**Ground truth**:\n{true_str}",
        )
    if use_wandb:
        wandb.log({"val/example_predictions": table}, step=epoch)
    else:
        print(f"Examples at epoch {epoch}:")
        for i in range(min(bs, num_examples)):
            print("---")
            # Use the same strings as above, but convert color markup to ANSI
            x_str_console = wandb_to_console_color(table.data[i][0].split("\n", 1)[1])
            pred_str_console = wandb_to_console_color(
                table.data[i][1].split("\n", 1)[1]
            )
            true_str_console = wandb_to_console_color(
                table.data[i][2].split("\n", 1)[1]
            )
            print(f"**Input**:\n{x_str_console}")
            print(f"**Predicted**:\n{pred_str_console}")
            print(f"**Ground truth**:\n{true_str_console}")


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
    )
    val_dset = TeamPredictionDataset(
        data_dir=config.train_data_dir,
        split="val",
        validation_ratio=config.val_ratio,
        mask_pokemon_prob_range=(config.mask_pokemon_prob, config.mask_pokemon_prob),
        mask_attrs_prob_range=(config.mask_attrs_prob, config.mask_attrs_prob),
        seed=config.seed,
    )
    comp_dset = CompetitiveTeamPredictionDataset(
        mask_pokemon_prob_range=(config.mask_pokemon_prob, config.mask_pokemon_prob),
        mask_attrs_prob_range=(config.mask_attrs_prob, config.mask_attrs_prob),
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    comp_loader = DataLoader(
        comp_dset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
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

    # Create checkpoint directory (per run) and artifacts subdir
    ckpt_dir = os.path.join(config.checkpoint_dir, config.run_name)
    artifact_dir = os.path.join(ckpt_dir, "artifacts")
    os.makedirs(artifact_dir, exist_ok=True)

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience_count = 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        for x_tokens, type_ids, y_tokens, pred_mask in tqdm.tqdm(
            train_loader, desc="Training"
        ):
            x_tokens = x_tokens.to(device)
            type_ids = type_ids.to(device)
            y_tokens = y_tokens.to(device)
            pred_mask = pred_mask.to(device)
            logits = model(x_tokens, type_ids)
            loss, acc = compute_loss_and_accuracy(logits, y_tokens, pred_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_acc += acc
        train_loss = running_loss / len(train_loader)
        train_acc = running_acc / len(train_loader)

        val_loss, val_acc = evaluate(model, val_loader, device)
        comp_loss, comp_acc = evaluate(model, comp_loader, device)

        # Log metrics for each dataset split
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
            # Log all metrics to wandb
            wandb_metrics = {}
            for split, split_metrics in metrics.items():
                for metric_name, value in split_metrics.items():
                    wandb_metrics[f"{split}/{metric_name}"] = value
            wandb.log(wandb_metrics, step=epoch)
        else:
            # Print metrics to console
            print(f"\nEpoch {epoch}")
            print(f"Train       - loss: {train_loss:.4f}, accuracy: {train_acc:.4f}")
            print(f"Replay Val  - loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")
            print(f"Competitive - loss: {comp_loss:.4f}, accuracy: {comp_acc:.4f}\n")

        example_batch = next(iter(val_loader))
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
            epoch=epoch,
        )

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            best_model = os.path.join(ckpt_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model)
            print(f"New best model saved to {best_model}")
            if use_wandb:
                # Log best checkpoint as Artifact
                artifact = wandb.Artifact(f"{config.run_name}-best-model", type="model")
                artifact.add_file(best_model)
                wandb.log_artifact(artifact)
        else:
            patience_count += 1
            if patience_count >= config.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Save final model
    final_model = os.path.join(ckpt_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model)
    if use_wandb:
        # Log final model as Artifact
        artifact = wandb.Artifact(f"{config.run_name}-final-model", type="model")
        artifact.add_file(final_model)
        wandb.log_artifact(artifact)
    else:
        print(f"Final model saved to {final_model}")


if __name__ == "__main__":
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
    args = parser.parse_args()

    # Default hyperparameters
    sweep_defaults = {
        "train_data_dir": "/mnt/data1/shared_pokemon_project/metamon_team_files",
        "val_ratio": 0.1,
        "batch_size": 8,
        "num_workers": 0,
        "mask_pokemon_prob": 0.1,
        "mask_attrs_prob": 0.1,
        "seed": 42,
        "max_seq_len": 64,
        "d_model": 256,
        "nhead": 4,
        "num_layers": 3,
        "dim_ff": 1024,
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "max_epochs": 1000,
        "patience": 5,
        "weight_decay": 1e-4,
        "num_examples": 4,
    }

    # Determine whether to use WandB
    use_wandb = not args.no_wandb
    if use_wandb:
        # Initialize WandB run
        wandb.init(
            project=args.project,
            entity=args.entity,
            group=args.group,
            config=sweep_defaults,
            name=args.name,
        )
        cfg = wandb.config
        # Override checkpoint dir & run name in config
        cfg.checkpoint_dir = args.checkpoint_dir
        cfg.run_name = wandb.run.name
    else:
        # Use local config namespace
        from argparse import Namespace

        cfg = Namespace(**sweep_defaults)
        cfg.checkpoint_dir = args.checkpoint_dir
        cfg.run_name = args.name or "local_run"

    # Start training
    train(cfg, use_wandb)
