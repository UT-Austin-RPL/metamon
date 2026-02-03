import os
import argparse
import html
import random
from typing import Optional
from dataclasses import dataclass

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
from metamon.backend.team_prediction.curriculum import (
    TeamMasker,
    NamesOnlyMasker,
    CurriculumMasker,
)
from metamon.backend.team_prediction.prediction_metrics import (
    compute_loss_and_metrics,
    EvaluationAccumulator,
)
from metamon.backend.team_prediction.iterative_decoder import (
    IterativeTeamDecoder,
    IterativeStatsAccumulator,
)


@dataclass
class EvalResults:
    oneshot_metrics: dict
    iterative_metrics: Optional[dict] = None
    examples: Optional[list] = None
    iter_stats: Optional[dict] = None
    mask_counts: Optional[list] = None


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    vocab: Vocabulary,
    max_steps: Optional[int] = None,
    include_iterative: bool = True,
    num_iterations: int = 8,
    num_examples: int = 0,
) -> EvalResults:
    model.eval()

    oneshot_accumulator = EvaluationAccumulator(vocab)
    iterative_accumulator = EvaluationAccumulator(vocab) if include_iterative else None
    iter_stats_accumulator = (
        IterativeStatsAccumulator(num_iterations) if include_iterative else None
    )
    val_mask_counts = []

    decoder = None
    if include_iterative:
        # deterministic for fair comparison with one-shot eval
        # eventually we should let the one-shot eval use the same samling strategy
        # as each iterative eval step.
        decoder = IterativeTeamDecoder(
            model, vocab, num_iterations=num_iterations, deterministic=True
        )

    # Collect batches
    batches = []
    num_steps = 0
    for batch in dataloader:
        batches.append(batch)
        num_steps += 1
        if max_steps is not None and num_steps >= max_steps:
            break

    examples = []

    desc = "Eval" if not include_iterative else "Eval (one-shot + iterative)"
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm.tqdm(batches, desc=desc, leave=False)):
            x_tokens, type_ids, y_tokens, pred_mask = batch
            x_tokens = x_tokens.to(device)
            type_ids = type_ids.to(device)
            y_tokens = y_tokens.to(device)
            pred_mask = pred_mask.to(device)

            val_mask_counts.extend(pred_mask.sum(dim=1).cpu().tolist())

            # one-shot eval
            logits = model(x_tokens, type_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                y_tokens.view(-1),
                reduction="none",
            )
            loss = (loss * pred_mask.view(-1)).sum() / max(pred_mask.sum().item(), 1)
            oneshot_accumulator.add_batch(
                logits, y_tokens, pred_mask, type_ids, x_tokens, loss=loss
            )
            probs = torch.softmax(logits, dim=-1)
            filt = vocab.filter_probs(probs, type_ids)
            # merge predictions onto input (only replace masked positions).
            # iterative decoder does this internally.
            oneshot_preds = x_tokens.clone()
            oneshot_preds[pred_mask] = filt.argmax(dim=-1)[pred_mask]

            # iterative eval
            iterative_preds = None
            if include_iterative and decoder is not None:
                iterative_preds, stats = decoder.decode(x_tokens, type_ids, pred_mask)
                iter_stats_accumulator.add_batch(stats)
                # placeholder logits
                vocab_size = len(vocab.tokenizer)
                iter_logits = torch.zeros(
                    iterative_preds.shape[0],
                    iterative_preds.shape[1],
                    vocab_size,
                    device=device,
                )
                iter_logits.scatter_(2, iterative_preds.unsqueeze(-1), 1.0)
                iterative_accumulator.add_batch(
                    iter_logits, y_tokens, pred_mask, type_ids, x_tokens
                )

            # save some predictions for fancy wandb example viz
            if batch_idx == 0 and num_examples > 0:
                for i in range(min(num_examples, x_tokens.shape[0])):
                    examples.append(
                        {
                            "input": x_tokens[i].cpu(),
                            "ground_truth": y_tokens[i].cpu(),
                            "oneshot_pred": oneshot_preds[i].cpu(),
                            "iterative_pred": (
                                iterative_preds[i].cpu()
                                if iterative_preds is not None
                                else None
                            ),
                            "mask": pred_mask[i].cpu(),
                        }
                    )

    # summarize metrics
    oneshot_metrics = oneshot_accumulator.compute_metrics()
    iterative_metrics = (
        iterative_accumulator.compute_metrics() if iterative_accumulator else None
    )
    iter_stats = (
        iter_stats_accumulator.compute_results() if iter_stats_accumulator else None
    )

    return EvalResults(
        oneshot_metrics=oneshot_metrics,
        iterative_metrics=iterative_metrics,
        examples=examples if num_examples > 0 else None,
        iter_stats=iter_stats,
        mask_counts=val_mask_counts,
    )


def log_example_predictions(
    examples: list,
    vocab: Vocabulary,
    step: int,
    include_iterative: bool = True,
):
    """
    Log example predictions to wandb with colored HTML output.

    Green: masked input tokens
    Blue: correct predictions
    Red: incorrect predictions
    """
    if not examples:
        return

    columns = ["input", "oneshot_pred", "ground_truth"]
    if include_iterative:
        columns = ["input", "oneshot_pred", "iterative_pred", "ground_truth"]

    table = wandb.Table(columns=columns)

    for ex in examples:
        x_seq = vocab.ints_to_pokeset_seq(ex["input"].tolist())
        oneshot_seq = vocab.ints_to_pokeset_seq(ex["oneshot_pred"].tolist())
        true_seq = vocab.ints_to_pokeset_seq(ex["ground_truth"].tolist())
        mask = ex["mask"]

        # Build HTML for input (green = masked)
        x_parts = []
        for x, m in zip(x_seq, mask):
            x_escaped = html.escape(x)
            if m:
                x_parts.append(
                    f'<span style="color: green; font-weight: bold">{x_escaped}</span>'
                )
            else:
                x_parts.append(x_escaped)
        x_html = " ".join(x_parts)

        # Build HTML for one-shot predictions (blue = correct, red = wrong)
        oneshot_parts = []
        for p, t, m in zip(oneshot_seq, true_seq, mask):
            p_escaped = html.escape(p)
            if m:
                color = "blue" if p == t else "red"
                oneshot_parts.append(
                    f'<span style="color: {color}; font-weight: bold">{p_escaped}</span>'
                )
            else:
                oneshot_parts.append(p_escaped)
        oneshot_html = " ".join(oneshot_parts)

        # Build HTML for ground truth
        true_parts = []
        for t, m in zip(true_seq, mask):
            t_escaped = html.escape(t)
            if m:
                true_parts.append(
                    f'<span style="color: purple; font-weight: bold">{t_escaped}</span>'
                )
            else:
                true_parts.append(t_escaped)
        true_html = " ".join(true_parts)

        if include_iterative and ex["iterative_pred"] is not None:
            iter_seq = vocab.ints_to_pokeset_seq(ex["iterative_pred"].tolist())
            iter_parts = []
            for p, t, m in zip(iter_seq, true_seq, mask):
                p_escaped = html.escape(p)
                if m:
                    color = "blue" if p == t else "red"
                    iter_parts.append(
                        f'<span style="color: {color}; font-weight: bold">{p_escaped}</span>'
                    )
                else:
                    iter_parts.append(p_escaped)
            iter_html = " ".join(iter_parts)

            table.add_data(
                wandb.Html(x_html),
                wandb.Html(oneshot_html),
                wandb.Html(iter_html),
                wandb.Html(true_html),
            )
        else:
            table.add_data(
                wandb.Html(x_html),
                wandb.Html(oneshot_html),
                wandb.Html(true_html),
            )

    wandb.log({"val/example_predictions": table}, step=step)


def train(config, use_wandb: bool = True):
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    vocab = Vocabulary()

    # maskers (create training examples)
    strategy = config.masking_strategy
    if strategy == "names_only":
        train_masker = NamesOnlyMasker(
            mask_all=False
        )  # Random 1-6 for context learning
        val_masker = NamesOnlyMasker(mask_all=True)  # All 6 for consistent eval
        print("Using NamesOnlyMasker (train: random 1-6, val: all 6)")
    elif strategy == "variable":
        train_masker = TeamMasker(
            pokemon_prob_range=(0.0, config.mask_pokemon_prob),
            attrs_prob_range=(0.0, config.mask_attrs_prob),
        )
        val_masker = TeamMasker(
            pokemon_prob_range=(config.mask_pokemon_prob, config.mask_pokemon_prob),
            attrs_prob_range=(config.mask_attrs_prob, config.mask_attrs_prob),
        )
        print(f"Using TeamMasker: {train_masker}")

    elif strategy == "curriculum":
        train_masker = CurriculumMasker(
            warmup_steps=config.curriculum_warmup_steps,
            pokemon_prob=config.mask_pokemon_prob,
            attrs_prob=config.mask_attrs_prob,
        )
        val_masker = TeamMasker(
            pokemon_prob_range=(config.mask_pokemon_prob, config.mask_pokemon_prob),
            attrs_prob_range=(config.mask_attrs_prob, config.mask_attrs_prob),
        )
        print(f"Using CurriculumMasker over {config.curriculum_warmup_steps} steps")

    else:
        raise ValueError(
            f"Unknown masking_strategy: {strategy}. Use 'names_only', 'variable', or 'curriculum'"
        )

    # datasets
    train_dset = TeamPredictionDataset(
        data_dir=config.train_data_dir,
        split="train",
        validation_ratio=config.val_ratio,
        seed=config.seed,
        use_cached_filenames=True,
        verbose=True,
        masker=train_masker,
    )

    val_dset = TeamPredictionDataset(
        data_dir=config.train_data_dir,
        split="val",
        validation_ratio=config.val_ratio,
        seed=config.seed,
        use_cached_filenames=True,
        verbose=True,
        masker=val_masker,
    )

    # dataset of complete (zero blank) teams.... but because they are really
    # forum sample teams, predicting them is trivial, and this isn't an ideal eval.
    comp_dset = CompetitiveTeamPredictionDataset(
        verbose=True,
        masker=val_masker,
    )

    if config.debug_overfit:
        print(f"DEBUG OVERFIT MODE: Using {config.batch_size} samples")
        from torch.utils.data import Subset

        indices = list(range(min(config.batch_size, len(train_dset))))
        train_dset = Subset(train_dset, indices)
        val_dset = Subset(train_dset, indices)
        comp_indices = list(range(min(config.batch_size, len(comp_dset))))
        comp_dset = Subset(comp_dset, comp_indices)

    # dataloaders
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

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )

    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ckpt_dir = os.path.join(config.checkpoint_dir, config.run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_val_accuracy = 0.0
    patience_count = 0
    global_step = 0
    running_loss = 0.0
    running_metrics = {}
    steps_since_eval = 0
    train_mask_counts = []
    train_iter = iter(train_loader)
    pbar = tqdm.tqdm(total=config.max_steps, desc="Training")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    while global_step < config.max_steps:
        try:
            x_tokens, type_ids, y_tokens, pred_mask = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x_tokens, type_ids, y_tokens, pred_mask = next(train_iter)

        train_masker.set_step(global_step)

        # training
        model.train()
        x_tokens = x_tokens.to(device)
        type_ids = type_ids.to(device)
        y_tokens = y_tokens.to(device)
        pred_mask = pred_mask.to(device)

        logits = model(x_tokens, type_ids)
        loss, metrics = compute_loss_and_metrics(
            logits, y_tokens, pred_mask, type_ids, vocab
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        scheduler.step()

        train_mask_counts.extend(pred_mask.sum(dim=1).cpu().tolist())
        running_loss += loss.item()
        for k, v in metrics.items():
            running_metrics[k] = running_metrics.get(k, 0.0) + v

        global_step += 1
        steps_since_eval += 1

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "acc": f"{metrics['token_accuracy']:.2%}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            }
        )
        pbar.update(1)

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

            if train_mask_counts:
                wandb.log(
                    {"train/num_blanks": wandb.Histogram(train_mask_counts)},
                    step=global_step,
                )
                train_mask_counts = []

            if global_step % config.eval_every_steps != 0:
                running_loss = 0.0
                running_metrics = {}
                steps_since_eval = 0

        # evaluation
        if global_step % config.eval_every_steps == 0:
            train_loss = running_loss / steps_since_eval
            train_metrics = {
                k: v / steps_since_eval for k, v in running_metrics.items()
            }

            running_loss = 0.0
            running_metrics = {}
            steps_since_eval = 0

            print(f"\n\nEvaluating at step {global_step}...")

            val_results = evaluate(
                model,
                val_loader,
                device,
                vocab,
                max_steps=config.max_eval_steps,
                include_iterative=config.eval_with_iterative,
                num_iterations=config.eval_num_iterations,
                num_examples=config.num_examples if use_wandb else 0,
            )

            comp_results = evaluate(
                model,
                comp_loader,
                device,
                vocab,
                max_steps=config.max_eval_steps,
                include_iterative=config.eval_with_iterative,
                num_iterations=config.eval_num_iterations,
                num_examples=0,
            )

            val_oneshot = val_results.oneshot_metrics
            val_iter = val_results.iterative_metrics
            comp_oneshot = comp_results.oneshot_metrics
            comp_iter = comp_results.iterative_metrics

            print(f"\nStep {global_step}:")
            print(
                f"  Train Loss: {train_loss:.4f} | Acc: {train_metrics['token_accuracy']:.3f}"
            )
            val_acc_str = f"{val_oneshot['token_accuracy']:.3f}"
            if val_iter:
                val_acc_str += f" (iter: {val_iter['token_accuracy']:.3f})"
            print(f"  Val Loss:   {val_oneshot['loss']:.4f} | Acc: {val_acc_str}")

            comp_acc_str = f"{comp_oneshot['token_accuracy']:.3f}"
            if comp_iter:
                comp_acc_str += f" (iter: {comp_iter['token_accuracy']:.3f})"
            print(f"  Comp Loss:  {comp_oneshot['loss']:.4f} | Acc: {comp_acc_str}")

            print("\n  Per-Generation Validation Accuracy:")
            for gen in range(1, 10):
                gen_key = f"gen{gen}_accuracy"
                count_key = f"gen{gen}_count"
                if gen_key in val_oneshot:
                    count = val_oneshot.get(count_key, 0)
                    iter_str = ""
                    if val_iter and gen_key in val_iter:
                        iter_str = f" (iter: {val_iter[gen_key]:.3f})"
                    print(
                        f"    Gen{gen}: {val_oneshot[gen_key]:.3f}{iter_str} (n={int(count)})"
                    )

            print("\n  Per-Attribute Validation Accuracy:")
            for k, v in sorted(val_oneshot.items()):
                if (
                    k.endswith("_accuracy")
                    and k != "token_accuracy"
                    and not k.startswith("gen")
                ):
                    print(f"    {k}: {v:.3f}")

            if use_wandb:
                log_dict = {
                    "global_step": global_step,
                    **{f"val/one_shot/{k}": v for k, v in val_oneshot.items()},
                    **{f"comp/one_shot/{k}": v for k, v in comp_oneshot.items()},
                }

                if val_iter:
                    log_dict.update(
                        {f"val/iterative/{k}": v for k, v in val_iter.items()}
                    )
                if comp_iter:
                    log_dict.update(
                        {f"comp/iterative/{k}": v for k, v in comp_iter.items()}
                    )

                if val_results.iter_stats:
                    stats = val_results.iter_stats
                    for i, (mask_ratio, frac) in enumerate(
                        zip(stats["mask_ratios"], stats["remaining_frac"])
                    ):
                        log_dict[f"val/iterative/iter_{i}_target_mask_ratio"] = (
                            mask_ratio
                        )
                        log_dict[f"val/iterative/iter_{i}_remaining_frac"] = frac

                wandb.log(log_dict, step=global_step)

                if val_results.examples:
                    log_example_predictions(
                        examples=val_results.examples,
                        vocab=vocab,
                        step=global_step,
                        include_iterative=config.eval_with_iterative,
                    )

                if val_results.iter_stats:
                    hist_dict = {}
                    for i, conf in enumerate(val_results.iter_stats["confidences"]):
                        if len(conf) > 0:
                            hist_dict[f"val/iterative/iter_{i}_confidences"] = (
                                wandb.Histogram(conf.numpy(), num_bins=50)
                            )
                    if hist_dict:
                        wandb.log(hist_dict, step=global_step)

                if val_results.mask_counts:
                    wandb.log(
                        {"val/num_blanks": wandb.Histogram(val_results.mask_counts)},
                        step=global_step,
                    )

            # checkpointing
            if not config.debug_overfit:
                # Use weighted accuracy as primary metric (prefer iterative if available)
                if val_iter:
                    val_score = val_iter.get(
                        "weighted_accuracy", val_iter["token_accuracy"]
                    )
                else:
                    val_score = val_oneshot.get(
                        "weighted_accuracy", val_oneshot["token_accuracy"]
                    )

                if val_score > best_val_accuracy:
                    # early stopping
                    best_val_accuracy = val_score
                    patience_count = 0

                    best_model_path = os.path.join(ckpt_dir, "best_model.pt")
                    torch.save(
                        {
                            "step": global_step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_accuracy": val_score,
                            "val_loss": val_oneshot["loss"],
                        },
                        best_model_path,
                    )

                    print(f"\nNew best model! Accuracy: {val_score:.3f}")

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

    sweep_defaults = {
        # dataset
        "train_data_dir": download_revealed_teams(),
        "val_ratio": 0.1,
        "batch_size": 64,
        "num_workers": 4,
        "seed": 42,
        # architecture
        "max_seq_len": 64,
        "d_model": 400,
        "nhead": 8,
        "num_layers": 8,
        "dim_ff": 1600,
        "dropout": 0.05,
        # training
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "max_grad_norm": 1.0,
        "warmup_steps": 5000,
        "max_steps": 5_000_000,
        "log_train_every_steps": 100,
        "eval_every_steps": 10_000,
        "max_eval_steps": 50,
        "patience": 500,
        # masking + curriculum params
        # "masking_strategy": "variable",
        "masking_strategy": "curriculum",
        "mask_pokemon_prob": 0.15,
        "mask_attrs_prob": 0.4,
        "curriculum_warmup_steps": 200_000,
        "eval_with_iterative": True,
        "eval_num_iterations": 8,
        "debug_overfit": False,
        "num_examples": 4,  # for wandb viz
    }

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
        sweep_defaults["masking_strategy"] = "names_only"

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
