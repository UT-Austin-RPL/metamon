import os
import argparse
import html
import random
import warnings
from collections import Counter
from typing import Optional
from dataclasses import dataclass

import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

from metamon.backend.team_prediction.dataset import (
    TeamPredictionDataset,
    ScoredTeamPredictionDataset,
)
from metamon.backend.team_prediction.prediction_model import (
    TeamPredictionModel,
    create_model,
)
from metamon.backend.team_prediction.vocabulary import Vocabulary
from metamon.backend.team_prediction.masking import (
    TeamMasker,
    NamesOnlyMasker,
    CurriculumMasker,
)
from metamon.backend.team_prediction.prediction_metrics import (
    compute_loss_and_metrics,
    EvaluationAccumulator,
    SemanticMetricsAccumulator,
)
from metamon.backend.team_prediction.team import Team2Seq, TeamSet, PokemonSet
from metamon.backend.team_prediction.iterative_decoder import IterativeStatsAccumulator


def create_demo_teams() -> list[TeamSet]:
    """
    Create demonstration teams for iterative decoding visualization.
    Each team has minimal information to show the full decoding process.
    """
    demos = []

    # Gen 1: Only Gengar name visible
    gen1_lead = PokemonSet(
        name="Gengar",
        gen=1,
        ability=PokemonSet.NO_ABILITY,
        item=PokemonSet.NO_ITEM,
        nature=PokemonSet.NO_NATURE,
        moves=[PokemonSet.MISSING_MOVE] * 4,
        evs=[252] * 6,
        ivs=[31] * 6,
        tera_type=PokemonSet.NO_TERA_TYPE,
    )
    demos.append(
        TeamSet(
            format="gen1ou",
            lead=gen1_lead,
            reserve=[PokemonSet.missing_pokemon(gen=1) for _ in range(5)],
        )
    )

    # Gen 2: Only Cloyster name visible
    gen2_lead = PokemonSet(
        name="Cloyster",
        gen=2,
        ability=PokemonSet.NO_ABILITY,
        item=PokemonSet.MISSING_ITEM,
        nature=PokemonSet.NO_NATURE,
        moves=[PokemonSet.MISSING_MOVE] * 4,
        evs=[252] * 6,
        ivs=[31] * 6,
        tera_type=PokemonSet.NO_TERA_TYPE,
    )
    demos.append(
        TeamSet(
            format="gen2ou",
            lead=gen2_lead,
            reserve=[PokemonSet.missing_pokemon(gen=2) for _ in range(5)],
        )
    )

    # Gen 3: Only Tyranitar name visible
    gen3_lead = PokemonSet(
        name="Tyranitar",
        gen=3,
        ability=PokemonSet.MISSING_ABILITY,
        item=PokemonSet.MISSING_ITEM,
        nature=PokemonSet.NO_NATURE,
        moves=[PokemonSet.MISSING_MOVE] * 4,
        evs=[252] * 6,
        ivs=[31] * 6,
        tera_type=PokemonSet.NO_TERA_TYPE,
    )
    demos.append(
        TeamSet(
            format="gen3ou",
            lead=gen3_lead,
            reserve=[PokemonSet.missing_pokemon(gen=3) for _ in range(5)],
        )
    )

    # Gen 4: Only Metagross name visible
    gen4_lead = PokemonSet(
        name="Metagross",
        gen=4,
        ability=PokemonSet.MISSING_ABILITY,
        item=PokemonSet.MISSING_ITEM,
        nature=PokemonSet.NO_NATURE,
        moves=[PokemonSet.MISSING_MOVE] * 4,
        evs=[252] * 6,
        ivs=[31] * 6,
        tera_type=PokemonSet.NO_TERA_TYPE,
    )
    demos.append(
        TeamSet(
            format="gen4ou",
            lead=gen4_lead,
            reserve=[PokemonSet.missing_pokemon(gen=4) for _ in range(5)],
        )
    )

    # Gen 9: All 6 names visible, everything else masked
    gen9_names = [
        "Gholdengo",
        "Darkrai",
        "Clefable",
        "Ting-Lu",
        "Dragonite",
        "Pecharunt",
    ]

    def gen9_pokemon(name: str) -> PokemonSet:
        return PokemonSet(
            name=name,
            gen=9,
            ability=PokemonSet.MISSING_ABILITY,
            item=PokemonSet.MISSING_ITEM,
            nature=PokemonSet.NO_NATURE,
            moves=[PokemonSet.MISSING_MOVE] * 4,
            evs=[252] * 6,
            ivs=[31] * 6,
            tera_type=PokemonSet.MISSING_TERA_TYPE,
        )

    demos.append(
        TeamSet(
            format="gen9ou",
            lead=gen9_pokemon(gen9_names[0]),
            reserve=[gen9_pokemon(n) for n in gen9_names[1:]],
        )
    )

    return demos


def log_demo_decoding(
    prediction_model: TeamPredictionModel,
    vocab: Vocabulary,
    step: int,
):
    """
    Log iterative decoding demonstrations showing progression for each generation.
    Highlights tokens committed at each iteration.
    """
    demos = create_demo_teams()

    # Build wandb table
    num_cols = prediction_model.iterative_decoder.num_iterations + 1
    columns = ["format", "input"] + [f"iter_{i}" for i in range(1, num_cols)]
    table = wandb.Table(columns=columns)

    for team in demos:
        # Run iterative decoding with token tracking
        _, stats = prediction_model.predict(team, return_stats=True)
        tokens_per_iter = stats.tokens_per_iter
        # Build row with format and each iteration
        row_data = [team.format]

        # Track token counts (handles duplicates like common moves on multiple pokemon)
        input_seq = vocab.ints_to_pokeset_seq(tokens_per_iter[0][0].tolist())
        previously_visible = Counter(tok for tok in input_seq if "$" not in tok)

        for iter_idx, tokens in enumerate(tokens_per_iter):
            seq = vocab.ints_to_pokeset_seq(tokens[0].tolist())  # batch dim 0
            currently_visible = Counter(tok for tok in seq if "$" not in tok)

            # Counter subtraction gives positive counts for newly added tokens
            # e.g., if "Earthquake" was 1, now 2, diff["Earthquake"] = 1
            newly_committed = currently_visible - previously_visible

            parts = []
            for tok_str in seq:
                tok_escaped = html.escape(tok_str)

                if newly_committed.get(tok_str, 0) > 0:
                    # Newly predicted this iteration - orange
                    parts.append(
                        f'<span style="color: darkorange; font-weight: bold">{tok_escaped}</span>'
                    )
                    newly_committed[tok_str] -= 1  # consume one highlight
                elif "$" in tok_str:
                    # Still masked - gray
                    parts.append(f'<span style="color: gray">{tok_escaped}</span>')
                else:
                    # Already visible (from input or previous iterations)
                    parts.append(tok_escaped)

            row_data.append(wandb.Html(" ".join(parts)))

            # Update for next iteration
            previously_visible = currently_visible

        table.add_data(*row_data)

    wandb.log({"demo_iterative_decoding": table}, step=step)


def log_demo_oneshot(
    prediction_model: TeamPredictionModel,
    vocab: Vocabulary,
    step: int,
):
    """
    Log one-shot predictions for demo teams.
    Shows input and single-pass prediction side by side.
    """
    demos = create_demo_teams()
    t2s = Team2Seq()

    columns = ["format", "input", "one_shot_prediction"]
    table = wandb.Table(columns=columns)

    for team in demos:
        # Encode the team
        x_tokens, type_ids, pred_mask = t2s.encode(team)
        x_tokens = x_tokens.unsqueeze(0).to(prediction_model.device)
        type_ids = type_ids.unsqueeze(0).to(prediction_model.device)
        pred_mask = pred_mask.unsqueeze(0).to(prediction_model.device)

        # Get one-shot prediction
        oneshot_preds = prediction_model.oneshot_forward(x_tokens, type_ids, pred_mask)

        # Convert to sequences
        input_seq = vocab.ints_to_pokeset_seq(x_tokens[0].tolist())
        pred_seq = vocab.ints_to_pokeset_seq(oneshot_preds[0].tolist())

        # Build input HTML (gray for masked)
        input_parts = []
        for tok_str in input_seq:
            tok_escaped = html.escape(tok_str)
            if "$" in tok_str:
                input_parts.append(f'<span style="color: gray">{tok_escaped}</span>')
            else:
                input_parts.append(tok_escaped)

        # Build prediction HTML (highlight predictions, i.e. positions that were masked in input)
        input_visible = set(i for i, tok in enumerate(input_seq) if "$" not in tok)
        pred_parts = []
        for i, tok_str in enumerate(pred_seq):
            tok_escaped = html.escape(tok_str)
            if i not in input_visible:
                # This was predicted (was masked in input)
                pred_parts.append(
                    f'<span style="color: darkorange; font-weight: bold">{tok_escaped}</span>'
                )
            else:
                pred_parts.append(tok_escaped)

        table.add_data(
            team.format,
            wandb.Html(" ".join(input_parts)),
            wandb.Html(" ".join(pred_parts)),
        )

    wandb.log({"demo_oneshot": table}, step=step)


@dataclass
class EvalResults:
    oneshot_metrics: dict  # Position-based metrics for one-shot
    oneshot_semantic_metrics: dict  # Semantic (set-based) metrics for one-shot
    iterative_metrics: Optional[dict] = None  # Position-based metrics for iterative
    iterative_semantic_metrics: Optional[dict] = None  # Semantic metrics for iterative
    examples: Optional[list] = None
    iter_stats: Optional[dict] = None
    mask_counts: Optional[list] = None
    revealed_counts: Optional[list] = None


def evaluate(
    prediction_model: TeamPredictionModel,
    dataloader: DataLoader,
    max_steps: Optional[int] = None,
    include_iterative: bool = True,
    num_examples: int = 0,
    desc: str = "Eval",
) -> EvalResults:
    prediction_model.eval()
    vocab = prediction_model.vocab
    device = prediction_model.device
    num_iterations = prediction_model.iterative_decoder.num_iterations

    t2s = Team2Seq()
    oneshot_accumulator = EvaluationAccumulator(vocab)
    oneshot_semantic_accumulator = SemanticMetricsAccumulator(vocab)
    iterative_accumulator = EvaluationAccumulator(vocab) if include_iterative else None
    iterative_semantic_accumulator = (
        SemanticMetricsAccumulator(vocab) if include_iterative else None
    )
    iter_stats_accumulator = (
        IterativeStatsAccumulator(num_iterations) if include_iterative else None
    )
    val_mask_counts = []
    val_revealed_counts = []

    # Collect batches
    batches = []
    num_steps = 0
    for batch in dataloader:
        batches.append(batch)
        num_steps += 1
        if max_steps is not None and num_steps >= max_steps:
            break

    examples = []

    full_desc = desc if not include_iterative else f"{desc} (one-shot + iterative)"
    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm.tqdm(batches, desc=full_desc, leave=False)
        ):
            x_tokens, type_ids, y_tokens, pred_mask = batch
            x_tokens = x_tokens.to(device)
            type_ids = type_ids.to(device)
            y_tokens = y_tokens.to(device)
            pred_mask = pred_mask.to(device)

            val_mask_counts.extend(pred_mask.sum(dim=1).cpu().tolist())
            # Count actually revealed tokens (not $missing_*$ in ground truth)
            missing_set = torch.tensor(vocab.missing_mask, device=y_tokens.device)
            is_revealed = ~torch.isin(y_tokens, missing_set)
            val_revealed_counts.extend(is_revealed.sum(dim=1).cpu().tolist())

            # one-shot eval
            logits = prediction_model.forward(x_tokens, type_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                y_tokens.view(-1),
                reduction="none",
            )
            loss = (loss * pred_mask.view(-1)).sum() / max(pred_mask.sum().item(), 1)
            oneshot_accumulator.add_batch(
                logits, y_tokens, pred_mask, type_ids, x_tokens, loss=loss
            )
            # Get one-shot predictions using the decoder
            oneshot_preds = prediction_model.oneshot_forward(
                x_tokens, type_ids, pred_mask
            )

            # One-shot semantic metrics (set-based comparison)
            oneshot_semantic_accumulator.add_batch(
                oneshot_preds.cpu(), y_tokens.cpu(), x_tokens.cpu(), t2s
            )

            # iterative eval
            iterative_preds = None
            iter_stats_for_examples = None
            if include_iterative:
                # Track tokens for visualization on first batch only
                track = batch_idx == 0 and num_examples > 0
                iterative_preds, stats = prediction_model.iterative_forward(
                    x_tokens, type_ids, pred_mask, track_tokens=track
                )
                if track:
                    iter_stats_for_examples = stats
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
                # Iterative semantic metrics (set-based comparison)
                iterative_semantic_accumulator.add_batch(
                    iterative_preds.cpu(), y_tokens.cpu(), x_tokens.cpu(), t2s
                )

            # save some predictions for fancy wandb example viz
            if batch_idx == 0 and num_examples > 0:
                for i in range(min(num_examples, x_tokens.shape[0])):
                    # Extract per-sample tokens from each iteration
                    tokens_per_iter = None
                    if iter_stats_for_examples is not None:
                        tokens_per_iter = [
                            t[i] for t in iter_stats_for_examples.tokens_per_iter
                        ]
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
                            "tokens_per_iter": tokens_per_iter,
                        }
                    )

    # summarize metrics
    oneshot_metrics = oneshot_accumulator.compute_metrics()
    oneshot_semantic_metrics = oneshot_semantic_accumulator.compute_metrics()
    iterative_metrics = (
        iterative_accumulator.compute_metrics() if iterative_accumulator else None
    )
    iterative_semantic_metrics = (
        iterative_semantic_accumulator.compute_metrics()
        if iterative_semantic_accumulator
        else None
    )
    iter_stats = (
        iter_stats_accumulator.compute_results() if iter_stats_accumulator else None
    )

    return EvalResults(
        oneshot_metrics=oneshot_metrics,
        oneshot_semantic_metrics=oneshot_semantic_metrics,
        iterative_metrics=iterative_metrics,
        iterative_semantic_metrics=iterative_semantic_metrics,
        examples=examples if num_examples > 0 else None,
        iter_stats=iter_stats,
        mask_counts=val_mask_counts,
        revealed_counts=val_revealed_counts,
    )


def log_example_predictions(
    examples: list,
    vocab: Vocabulary,
    step: int,
    include_iterative: bool = True,
    table_name: str = "val_examples",
):
    """
    Log example predictions to wandb with colored HTML output.
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

    wandb.log({table_name: table}, step=step)


def log_iterative_decoding_process(
    examples: list,
    vocab: Vocabulary,
    step: int,
    table_name: str = "iterative_decoding_process",
):
    """
    Log the iterative decoding process to wandb, showing tokens at each iteration.
    """
    if not examples:
        return

    # Find max iterations across examples
    max_iters = max(len(ex.get("tokens_per_iter", [])) for ex in examples)
    if max_iters == 0:
        return

    # Columns: iter_0 (input), iter_1, ..., iter_N, ground_truth
    columns = [f"iter_{i}" for i in range(max_iters)] + ["ground_truth"]
    table = wandb.Table(columns=columns)

    for ex in examples:
        tokens_per_iter = ex.get("tokens_per_iter", [])
        if not tokens_per_iter:
            continue

        mask = ex["mask"]
        true_seq = vocab.ints_to_pokeset_seq(ex["ground_truth"].tolist())

        row_data = []
        for iter_idx, tokens in enumerate(tokens_per_iter):
            seq = vocab.ints_to_pokeset_seq(tokens.tolist())

            # For iteration 0 (input), green = masked
            # For later iterations, blue = correct, red = wrong, green = still masked
            parts = []
            for pos, (tok_str, is_masked, true_str) in enumerate(
                zip(seq, mask, true_seq)
            ):
                tok_escaped = html.escape(tok_str)

                if iter_idx == 0:
                    # Input: just show masked in green
                    if is_masked:
                        parts.append(
                            f'<span style="color: green; font-weight: bold">{tok_escaped}</span>'
                        )
                    else:
                        parts.append(tok_escaped)
                else:
                    # Later iterations: check if token was originally masked
                    if is_masked:
                        # Check if it's been filled (no longer a $missing$ token)
                        if "$" not in tok_str:  # filled in
                            color = "blue" if tok_str == true_str else "red"
                            parts.append(
                                f'<span style="color: {color}; font-weight: bold">{tok_escaped}</span>'
                            )
                        else:
                            # Still masked
                            parts.append(
                                f'<span style="color: green; font-weight: bold">{tok_escaped}</span>'
                            )
                    else:
                        parts.append(tok_escaped)

            row_data.append(wandb.Html(" ".join(parts)))

        # Pad with empty if fewer iterations
        while len(row_data) < max_iters:
            row_data.append(wandb.Html(""))

        # Add ground truth
        true_parts = []
        for t, m in zip(true_seq, mask):
            t_escaped = html.escape(t)
            if m:
                true_parts.append(
                    f'<span style="color: purple; font-weight: bold">{t_escaped}</span>'
                )
            else:
                true_parts.append(t_escaped)
        row_data.append(wandb.Html(" ".join(true_parts)))

        table.add_data(*row_data)

    wandb.log({table_name: table}, step=step)


def train(config, use_wandb: bool = True):
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    vocab = Vocabulary()

    # maskers (create training examples)
    if config.toy_names_only:
        # debug toy: only predict pokemon names
        train_masker = NamesOnlyMasker(mask_all=False)  # random 1-6 for context
        val_masker_standard = NamesOnlyMasker(mask_all=True)
        val_masker_low = NamesOnlyMasker(mask_all=True)
        print("Using NamesOnlyMasker (toy mode)")
    elif config.curriculum_mask:
        # curriculum: masking rate anneals from 0.25 to max
        train_masker = CurriculumMasker(
            warmup_steps=config.curriculum_mask_warmup_steps,
            attrs_prob=config.mask_attrs_prob,
        )
        val_masker_standard = TeamMasker(
            attrs_prob_range=(
                config.val_hard_mask_attrs_prob,
                config.val_hard_mask_attrs_prob,
            ),
        )
        val_masker_low = TeamMasker(
            attrs_prob_range=(
                config.val_easy_mask_attrs_prob,
                config.val_easy_mask_attrs_prob,
            ),
        )
        print(
            f"Using CurriculumMasker over {config.curriculum_mask_warmup_steps} steps"
        )
    else:
        # variable: random masking rate each sample
        train_masker = TeamMasker(
            attrs_prob_range=(0.1, config.mask_attrs_prob),
        )
        val_masker_standard = TeamMasker(
            attrs_prob_range=(
                config.val_hard_mask_attrs_prob,
                config.val_hard_mask_attrs_prob,
            ),
        )
        val_masker_low = TeamMasker(
            attrs_prob_range=(
                config.val_easy_mask_attrs_prob,
                config.val_easy_mask_attrs_prob,
            ),
        )
        print(f"Using variable TeamMasker")

    # datasets
    if config.gen_weights is not None:
        print(
            f"Training on specific generations: {sorted(config.gen_weights.keys())} (uniform sampling)"
        )
    if config.curriculum_dset:
        # curriculum: start with low percentile (few samples), anneal up to 100%
        train_dset = ScoredTeamPredictionDataset(
            data_dir=config.train_data_dir,
            masker=train_masker,
            gen_weights=config.gen_weights,
            percentile=config.curriculum_dset_start_pct,
            split="train",
            validation_ratio=config.val_ratio,
            seed=config.seed,
            verbose=True,
        )
        train_dset.enable_curriculum(config.curriculum_dset_start_pct)
        print(
            f"Curriculum dataset: top {config.curriculum_dset_start_pct}% -> {config.curriculum_dset_end_pct}% over {config.curriculum_dset_warmup_steps} steps"
        )
    else:
        train_dset = TeamPredictionDataset(
            data_dir=config.train_data_dir,
            masker=train_masker,
            gen_weights=config.gen_weights,
            split="train",
            validation_ratio=config.val_ratio,
            seed=config.seed,
            verbose=True,
        )

    # Val: all teams, standard masking
    val_dset = TeamPredictionDataset(
        data_dir=config.train_data_dir,
        masker=val_masker_standard,
        gen_weights=config.gen_weights,
        split="val",
        validation_ratio=config.val_ratio,
        seed=config.seed,
        verbose=True,
    )

    # Val clean: top percentile teams, low masking (easy)
    val_clean_dset = ScoredTeamPredictionDataset(
        data_dir=config.train_data_dir,
        masker=val_masker_low,
        gen_weights=config.gen_weights,
        percentile=config.val_clean_percentile,
        split="val",
        validation_ratio=config.val_ratio,
        seed=config.seed,
        verbose=True,
    )

    # Val clean hard: top percentile teams, standard masking (hard)
    val_clean_hard_dset = ScoredTeamPredictionDataset(
        data_dir=config.train_data_dir,
        masker=val_masker_standard,
        gen_weights=config.gen_weights,
        percentile=config.val_clean_percentile,
        split="val",
        validation_ratio=config.val_ratio,
        seed=config.seed,
        verbose=True,
    )

    if config.debug_overfit:
        print(f"DEBUG OVERFIT MODE: Using {config.batch_size} samples")
        from torch.utils.data import Subset

        indices = list(range(min(config.batch_size, len(train_dset))))
        train_dset = Subset(train_dset, indices)
        val_dset = Subset(train_dset, indices)
        val_clean_indices = list(range(min(config.batch_size, len(val_clean_dset))))
        val_clean_dset = Subset(val_clean_dset, val_clean_indices)
        val_clean_hard_dset = Subset(val_clean_hard_dset, val_clean_indices)

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
    val_clean_loader = DataLoader(
        val_clean_dset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent,
    )
    val_clean_hard_loader = DataLoader(
        val_clean_hard_dset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prediction_model = create_model(
        model_type=config.model_type,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_ff,
        dropout=config.dropout,
        num_iterations=config.eval_num_iterations,
        iterative_deterministic=True,  # For fair comparison with one-shot
        oneshot_deterministic=True,  # Argmax for evaluation
        device=device,
    )

    # optimizer
    optimizer = torch.optim.AdamW(
        prediction_model.parameters(),
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

    if config.from_ckpt:
        # start from ckpt
        ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
        print(f"Loading checkpoint from {ckpt_path}")
        extra_state = prediction_model.load_checkpoint(ckpt_path, optimizer)
        global_step = extra_state.get("step", 0)
        best_val_accuracy = extra_state.get("val_accuracy", 0.0)
        # Fast-forward scheduler to match checkpoint (suppress expected warning)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*lr_scheduler.step.*optimizer.step.*"
            )
            for _ in range(global_step):
                scheduler.step()
        print(
            f"Resumed from step {global_step}, best_val_accuracy={best_val_accuracy:.4f}"
        )
        print("\nRunning initial evaluation on loaded checkpoint...")
        val_results = evaluate(
            prediction_model,
            val_loader,
            max_steps=config.max_eval_steps,
            include_iterative=config.eval_with_iterative,
            num_examples=0,
        )
        val_oneshot = val_results.oneshot_metrics
        val_iter = val_results.iterative_metrics
        print(
            f"Checkpoint eval - one-shot acc: {val_oneshot['token_accuracy']:.4f}, loss: {val_oneshot['loss']:.4f}"
        )
        if val_iter:
            print(f"Checkpoint eval - iterative acc: {val_iter['token_accuracy']:.4f}")

        # Log demo tables to wandb
        if use_wandb:
            log_demo_oneshot(prediction_model, vocab, step=global_step)
            if config.eval_with_iterative:
                log_demo_decoding(prediction_model, vocab, step=global_step)

    running_loss = 0.0
    running_metrics = {}
    steps_since_eval = 0
    train_mask_counts = []
    train_revealed_counts = []
    train_iter = iter(train_loader)
    pbar = tqdm.tqdm(total=config.max_steps, initial=global_step, desc="Training")
    t2s = Team2Seq()
    train_semantic_accumulator = SemanticMetricsAccumulator(vocab)

    print(
        f"Model parameters: {sum(p.numel() for p in prediction_model.parameters()):,}"
    )

    while global_step < config.max_steps:
        try:
            x_tokens, type_ids, y_tokens, pred_mask = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x_tokens, type_ids, y_tokens, pred_mask = next(train_iter)

        train_masker.set_step(global_step)

        # update curriculum dataset percentile if enabled
        if config.curriculum_dset:
            progress = min(1.0, global_step / config.curriculum_dset_warmup_steps)
            new_percentile = config.curriculum_dset_start_pct + progress * (
                config.curriculum_dset_end_pct - config.curriculum_dset_start_pct
            )
            train_dset.set_curriculum_percentile(new_percentile)

        # training
        prediction_model.train()
        x_tokens = x_tokens.to(device)
        type_ids = type_ids.to(device)
        y_tokens = y_tokens.to(device)
        pred_mask = pred_mask.to(device)

        logits = prediction_model.forward(x_tokens, type_ids)
        loss, metrics = compute_loss_and_metrics(
            logits, y_tokens, pred_mask, type_ids, vocab
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            prediction_model.parameters(), config.max_grad_norm
        )
        optimizer.step()
        scheduler.step()

        train_mask_counts.extend(pred_mask.sum(dim=1).cpu().tolist())
        missing_set = torch.tensor(vocab.missing_mask, device=y_tokens.device)
        is_revealed = ~torch.isin(y_tokens, missing_set)
        train_revealed_counts.extend(is_revealed.sum(dim=1).cpu().tolist())
        running_loss += loss.item()
        for k, v in metrics.items():
            running_metrics[k] = running_metrics.get(k, 0.0) + v

        # accumulate semantic metrics every N steps (decoding is expensive)
        semantic_accumulate_every = max(1, config.semantic_train_every_steps // 10)
        if global_step % semantic_accumulate_every == 0:
            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1)
                filt = vocab.filter_probs(probs, type_ids)
                train_preds = x_tokens.clone()
                train_preds[pred_mask] = filt.argmax(dim=-1)[pred_mask]
                train_semantic_accumulator.add_batch(
                    train_preds.cpu(), y_tokens.cpu(), x_tokens.cpu(), t2s
                )

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

            log_dict = {
                "global_step": global_step,
                "train/loss": avg_loss,
                "train/learning_rate": scheduler.get_last_lr()[0],
                **{f"train/{k}": v for k, v in avg_metrics.items()},
            }
            if config.curriculum_dset:
                log_dict["train/curriculum_percentile"] = train_dset.percentile
            wandb.log(log_dict, step=global_step)

            if train_mask_counts:
                wandb.log(
                    {
                        "train/num_blanks": wandb.Histogram(train_mask_counts),
                        "train/num_revealed": wandb.Histogram(train_revealed_counts),
                    },
                    step=global_step,
                )
                train_mask_counts = []
                train_revealed_counts = []

            if global_step % config.eval_every_steps != 0:
                running_loss = 0.0
                running_metrics = {}
                steps_since_eval = 0

        if use_wandb and global_step % config.semantic_train_every_steps == 0:
            train_semantic_metrics = train_semantic_accumulator.compute_metrics()
            wandb.log(
                {f"train/semantic/{k}": v for k, v in train_semantic_metrics.items()},
                step=global_step,
            )
            train_semantic_accumulator = SemanticMetricsAccumulator(vocab)

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
                prediction_model,
                val_loader,
                max_steps=config.max_eval_steps,
                include_iterative=config.eval_with_iterative,
                num_examples=config.num_examples if use_wandb else 0,
                desc="val",
            )

            val_clean_results = evaluate(
                prediction_model,
                val_clean_loader,
                max_steps=config.max_eval_steps,
                include_iterative=config.eval_with_iterative,
                num_examples=config.num_examples if use_wandb else 0,
                desc="val_clean",
            )

            val_clean_hard_results = evaluate(
                prediction_model,
                val_clean_hard_loader,
                max_steps=config.max_eval_steps,
                include_iterative=config.eval_with_iterative,
                num_examples=config.num_examples if use_wandb else 0,
                desc="val_clean_hard",
            )

            # Position-based metrics
            val_oneshot = val_results.oneshot_metrics
            val_iter = val_results.iterative_metrics
            val_clean_oneshot = val_clean_results.oneshot_metrics
            val_clean_iter = val_clean_results.iterative_metrics
            val_clean_hard_oneshot = val_clean_hard_results.oneshot_metrics
            val_clean_hard_iter = val_clean_hard_results.iterative_metrics
            # Semantic metrics (set-based)
            val_oneshot_sem = val_results.oneshot_semantic_metrics
            val_iter_sem = val_results.iterative_semantic_metrics
            val_clean_oneshot_sem = val_clean_results.oneshot_semantic_metrics
            val_clean_iter_sem = val_clean_results.iterative_semantic_metrics
            val_clean_hard_oneshot_sem = val_clean_hard_results.oneshot_semantic_metrics
            val_clean_hard_iter_sem = val_clean_hard_results.iterative_semantic_metrics

            print(f"\nStep {global_step}:")
            print(
                f"  Train Loss: {train_loss:.4f} | Acc: {train_metrics['token_accuracy']:.3f}"
            )
            val_acc_str = f"{val_oneshot['token_accuracy']:.3f}"
            if val_iter:
                val_acc_str += f" (iter: {val_iter['token_accuracy']:.3f})"
            print(f"  Val Loss:   {val_oneshot['loss']:.4f} | Acc: {val_acc_str}")

            val_clean_acc_str = f"{val_clean_oneshot['token_accuracy']:.3f}"
            if val_clean_iter:
                val_clean_acc_str += f" (iter: {val_clean_iter['token_accuracy']:.3f})"
            print(
                f"  Val Clean (low mask):  {val_clean_oneshot['loss']:.4f} | Acc: {val_clean_acc_str}"
            )

            val_clean_hard_acc_str = f"{val_clean_hard_oneshot['token_accuracy']:.3f}"
            if val_clean_hard_iter:
                val_clean_hard_acc_str += (
                    f" (iter: {val_clean_hard_iter['token_accuracy']:.3f})"
                )
            print(
                f"  Val Clean (std mask):  {val_clean_hard_oneshot['loss']:.4f} | Acc: {val_clean_hard_acc_str}"
            )

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

            print("\n  Per-Attribute Validation Accuracy (position-based):")
            for k, v in sorted(val_oneshot.items()):
                if (
                    k.endswith("_accuracy")
                    and k != "token_accuracy"
                    and not k.startswith("gen")
                ):
                    print(f"    {k}: {v:.3f}")

            # Print semantic metrics (set-based comparison)
            print("\n  Semantic Metrics (set-based):")
            for attr in ["pokemon", "move", "ability", "item", "tera"]:
                key = f"{attr}_accuracy"
                if key in val_oneshot_sem:
                    oneshot_val = val_oneshot_sem[key]
                    total = val_oneshot_sem.get(f"{attr}_total", 0)
                    iter_str = ""
                    if val_iter_sem and key in val_iter_sem:
                        iter_str = f" (iter: {val_iter_sem[key]:.3f})"
                    print(f"    {attr}: {oneshot_val:.3f}{iter_str} (n={int(total)})")

            # Print iterative decoder diagnostics
            if val_results.iter_stats and config.eval_with_iterative:
                stats = val_results.iter_stats
                print("\n  Iterative Decoder Diagnostics:")
                names_committed = stats.get("names_committed_per_iter", [])
                moves_committed = stats.get("moves_committed_per_iter", [])
                names_reset = stats.get("names_reset_per_iter", [])
                moves_reset = stats.get("moves_reset_per_iter", [])
                if names_committed or moves_committed:
                    print("    Per-iteration commits (names/moves):")
                    for i in range(len(names_committed)):
                        nc = names_committed[i] if i < len(names_committed) else 0
                        mc = moves_committed[i] if i < len(moves_committed) else 0
                        print(f"      iter {i}: {nc} names, {mc} moves")
                if names_reset or moves_reset:
                    total_names_reset = sum(names_reset) if names_reset else 0
                    total_moves_reset = sum(moves_reset) if moves_reset else 0
                    print(
                        f"    Uniqueness resets: {total_names_reset} names, {total_moves_reset} moves"
                    )
                    if total_names_reset > 0 or total_moves_reset > 0:
                        print("    Per-iteration resets:")
                        for i in range(max(len(names_reset), len(moves_reset))):
                            nr = names_reset[i] if i < len(names_reset) else 0
                            mr = moves_reset[i] if i < len(moves_reset) else 0
                            if nr > 0 or mr > 0:
                                print(f"      iter {i}: {nr} names, {mr} moves")

            if use_wandb:
                log_dict = {"global_step": global_step}

                # Position-based metrics: {dset}/one_shot/position/
                log_dict.update(
                    {f"val/one_shot/position/{k}": v for k, v in val_oneshot.items()}
                )
                log_dict.update(
                    {
                        f"val_clean/one_shot/position/{k}": v
                        for k, v in val_clean_oneshot.items()
                    }
                )
                log_dict.update(
                    {
                        f"val_clean_hard/one_shot/position/{k}": v
                        for k, v in val_clean_hard_oneshot.items()
                    }
                )

                # Semantic metrics for one-shot: {dset}/one_shot/semantic/
                log_dict.update(
                    {
                        f"val/one_shot/semantic/{k}": v
                        for k, v in val_oneshot_sem.items()
                    }
                )
                log_dict.update(
                    {
                        f"val_clean/one_shot/semantic/{k}": v
                        for k, v in val_clean_oneshot_sem.items()
                    }
                )
                log_dict.update(
                    {
                        f"val_clean_hard/one_shot/semantic/{k}": v
                        for k, v in val_clean_hard_oneshot_sem.items()
                    }
                )

                # Position-based metrics for iterative: {dset}/iterative/position/
                if val_iter:
                    log_dict.update(
                        {f"val/iterative/position/{k}": v for k, v in val_iter.items()}
                    )
                if val_clean_iter:
                    log_dict.update(
                        {
                            f"val_clean/iterative/position/{k}": v
                            for k, v in val_clean_iter.items()
                        }
                    )
                if val_clean_hard_iter:
                    log_dict.update(
                        {
                            f"val_clean_hard/iterative/position/{k}": v
                            for k, v in val_clean_hard_iter.items()
                        }
                    )

                # Semantic metrics for iterative: {dset}/iterative/semantic/
                if val_iter_sem:
                    log_dict.update(
                        {
                            f"val/iterative/semantic/{k}": v
                            for k, v in val_iter_sem.items()
                        }
                    )
                if val_clean_iter_sem:
                    log_dict.update(
                        {
                            f"val_clean/iterative/semantic/{k}": v
                            for k, v in val_clean_iter_sem.items()
                        }
                    )
                if val_clean_hard_iter_sem:
                    log_dict.update(
                        {
                            f"val_clean_hard/iterative/semantic/{k}": v
                            for k, v in val_clean_hard_iter_sem.items()
                        }
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
                        table_name="val_examples",
                    )
                    if config.eval_with_iterative:
                        log_iterative_decoding_process(
                            examples=val_results.examples,
                            vocab=vocab,
                            step=global_step,
                            table_name="val_iterative_process",
                        )
                if val_clean_results.examples:
                    log_example_predictions(
                        examples=val_clean_results.examples,
                        vocab=vocab,
                        step=global_step,
                        include_iterative=config.eval_with_iterative,
                        table_name="val_clean_examples",
                    )
                    if config.eval_with_iterative:
                        log_iterative_decoding_process(
                            examples=val_clean_results.examples,
                            vocab=vocab,
                            step=global_step,
                            table_name="val_clean_iterative_process",
                        )
                if val_clean_hard_results.examples:
                    log_example_predictions(
                        examples=val_clean_hard_results.examples,
                        vocab=vocab,
                        step=global_step,
                        include_iterative=config.eval_with_iterative,
                        table_name="val_clean_hard_examples",
                    )
                    if config.eval_with_iterative:
                        log_iterative_decoding_process(
                            examples=val_clean_hard_results.examples,
                            vocab=vocab,
                            step=global_step,
                            table_name="val_clean_hard_iterative_process",
                        )

                # Demo predictions on fixed examples per generation
                log_demo_oneshot(prediction_model, vocab, step=global_step)
                if config.eval_with_iterative:
                    log_demo_decoding(prediction_model, vocab, step=global_step)

                if val_results.iter_stats:
                    hist_dict = {}
                    stats = val_results.iter_stats

                    # Overall confidences
                    for i, conf in enumerate(stats["confidences"]):
                        if len(conf) > 0:
                            finite_conf = conf[torch.isfinite(conf)]
                            if len(finite_conf) > 0:
                                hist_dict[f"val/iterative/iter_{i}_confidences"] = (
                                    wandb.Histogram(finite_conf.numpy(), num_bins=50)
                                )

                    # Name confidences (diagnostic)
                    for i, conf in enumerate(stats.get("name_confidences", [])):
                        if len(conf) > 0:
                            finite_conf = conf[torch.isfinite(conf)]
                            if len(finite_conf) > 0:
                                hist_dict[
                                    f"val/iterative/diag/iter_{i}_name_confidences"
                                ] = wandb.Histogram(finite_conf.numpy(), num_bins=50)

                    # Move confidences (diagnostic)
                    for i, conf in enumerate(stats.get("move_confidences", [])):
                        if len(conf) > 0:
                            finite_conf = conf[torch.isfinite(conf)]
                            if len(finite_conf) > 0:
                                hist_dict[
                                    f"val/iterative/diag/iter_{i}_move_confidences"
                                ] = wandb.Histogram(finite_conf.numpy(), num_bins=50)

                    # Committed counts
                    committed = stats.get("committed_per_iter", [])
                    if committed:
                        for i, count in enumerate(committed):
                            hist_dict[f"val/iterative/iter_{i}_committed"] = count

                    # Names committed per iter
                    names_committed = stats.get("names_committed_per_iter", [])
                    if names_committed:
                        for i, count in enumerate(names_committed):
                            hist_dict[
                                f"val/iterative/diag/iter_{i}_names_committed"
                            ] = count

                    # Moves committed per iter
                    moves_committed = stats.get("moves_committed_per_iter", [])
                    if moves_committed:
                        for i, count in enumerate(moves_committed):
                            hist_dict[
                                f"val/iterative/diag/iter_{i}_moves_committed"
                            ] = count

                    # Names reset by uniqueness constraints per iter
                    names_reset = stats.get("names_reset_per_iter", [])
                    if names_reset:
                        for i, count in enumerate(names_reset):
                            hist_dict[f"val/iterative/diag/iter_{i}_names_reset"] = (
                                count
                            )
                        # Total names reset across all iterations
                        hist_dict["val/iterative/diag/total_names_reset"] = sum(
                            names_reset
                        )

                    # Moves reset by uniqueness constraints per iter
                    moves_reset = stats.get("moves_reset_per_iter", [])
                    if moves_reset:
                        for i, count in enumerate(moves_reset):
                            hist_dict[f"val/iterative/diag/iter_{i}_moves_reset"] = (
                                count
                            )
                        # Total moves reset across all iterations
                        hist_dict["val/iterative/diag/total_moves_reset"] = sum(
                            moves_reset
                        )

                    if hist_dict:
                        wandb.log(hist_dict, step=global_step)

                if val_results.mask_counts:
                    wandb.log(
                        {
                            "val/num_blanks": wandb.Histogram(val_results.mask_counts),
                            "val/num_revealed": wandb.Histogram(
                                val_results.revealed_counts
                            ),
                        },
                        step=global_step,
                    )
                if val_clean_results.mask_counts:
                    wandb.log(
                        {
                            "val_clean/num_blanks": wandb.Histogram(
                                val_clean_results.mask_counts
                            ),
                            "val_clean/num_revealed": wandb.Histogram(
                                val_clean_results.revealed_counts
                            ),
                        },
                        step=global_step,
                    )
                if val_clean_hard_results.mask_counts:
                    wandb.log(
                        {
                            "val_clean_hard/num_blanks": wandb.Histogram(
                                val_clean_hard_results.mask_counts
                            ),
                            "val_clean_hard/num_revealed": wandb.Histogram(
                                val_clean_hard_results.revealed_counts
                            ),
                        },
                        step=global_step,
                    )

            # checkpointing
            if not config.debug_overfit:
                # Use val_clean_hard semantic move accuracy as early stopping metric
                # (prefer iterative if available, fallback to one-shot)
                if (
                    val_clean_hard_iter_sem
                    and "move_accuracy" in val_clean_hard_iter_sem
                ):
                    val_score = val_clean_hard_iter_sem["move_accuracy"]
                else:
                    val_score = val_clean_hard_oneshot_sem.get("move_accuracy", 0.0)

                if val_score > best_val_accuracy:
                    # early stopping
                    best_val_accuracy = val_score
                    patience_count = 0

                    best_model_path = os.path.join(ckpt_dir, "best_model.pt")
                    prediction_model.save_checkpoint(
                        best_model_path,
                        optimizer=optimizer,
                        extra_state={
                            "step": global_step,
                            "val_semantic_move_accuracy": val_score,
                            "val_loss": val_oneshot["loss"],
                        },
                    )

                    print(f"\nNew best model! Semantic Move Acc: {val_score:.3f}")
                else:
                    patience_count += 1
                    if patience_count >= config.patience:
                        print(f"\nEarly stopping at step {global_step}")
                        break

    pbar.close()

    print("\nTraining complete! Saving final model...")
    final_model_path = os.path.join(ckpt_dir, "final_model.pt")
    prediction_model.save_checkpoint(
        final_model_path,
        optimizer=optimizer,
        extra_state={"step": global_step},
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
    parser.add_argument("--curriculum-mask", action="store_true")
    parser.add_argument("--curriculum-dset", action="store_true")
    parser.add_argument(
        "--gens",
        type=int,
        nargs="+",
        default=None,
        help="Limit training to specific generations (e.g., --gens 1 9). "
        "Samples uniformly across specified generations.",
    )
    parser.add_argument(
        "--from-ckpt",
        action="store_true",
        help="Resume training from latest checkpoint (uses --checkpoint-dir and --name)",
    )

    args = parser.parse_args()

    if args.gens is not None:
        gen_weights = {g: 1.0 for g in args.gens}  # uniform across specified gens
    else:
        gen_weights = None  # natural distribution

    sweep_defaults = {
        # dataset
        "train_data_dir": download_revealed_teams(),
        "val_ratio": 0.1,
        "batch_size": 128,
        "num_workers": 4,
        "seed": 42,
        "gen_weights": gen_weights,
        # architecture
        "model_type": "LocalGlobalTeamTransformer",  # or "LocalGlobalTeamTransformer"
        "d_model": 400,
        "nhead": 16,
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
        "semantic_train_every_steps": 5_000,
        "eval_every_steps": 5000,
        "max_eval_steps": 10,
        "patience": 500,
        "num_examples": 4,  # for wandb viz
        "from_ckpt": args.from_ckpt,
        # masking params
        "mask_attrs_prob": 1.0,  # training max (curriculum starts at 0.25)
        "val_easy_mask_attrs_prob": 0.2,
        "val_hard_mask_attrs_prob": 0.5,
        "toy_names_only": False,
        "curriculum_mask": args.curriculum_mask,
        "curriculum_mask_warmup_steps": 100_000,
        "eval_with_iterative": True,
        "eval_num_iterations": 8,
        "debug_overfit": False,
        "val_clean_percentile": 15.0,
        # curriculum dataset
        "curriculum_dset": args.curriculum_dset,
        "curriculum_dset_start_pct": 10.0,
        "curriculum_dset_end_pct": 100.0,
        "curriculum_dset_warmup_steps": 75_000,
    }

    if args.debug_overfit:
        sweep_defaults.update(
            {
                "debug_overfit": True,
                "log_train_every_steps": 1,
                "semantic_train_every_steps": 10,
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
        wandb.define_metric("val_clean/*", step_metric="global_step")
        wandb.define_metric("val_clean_hard/*", step_metric="global_step")
    else:
        from argparse import Namespace

        cfg = Namespace(**sweep_defaults)
        cfg.checkpoint_dir = args.checkpoint_dir
        cfg.run_name = args.name or "local_run"

    train(cfg, use_wandb)
