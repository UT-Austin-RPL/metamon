#!/usr/bin/env python3
"""Demo of curriculum dataset with dynamic percentile threshold."""

from metamon.backend.team_prediction.dataset import ScoredTeamPredictionDataset
from metamon.backend.team_prediction.curriculum import TeamMasker
from metamon.backend.team_prediction.vocabulary import Vocabulary
from metamon.data.download import download_revealed_teams


def main():
    data_dir = download_revealed_teams()
    vocab = Vocabulary()

    masker = TeamMasker(
        pokemon_prob_range=(0.1, 0.1),
        attrs_prob_range=(0.3, 0.3),
    )

    # Create dataset with low percentile (only top 10% most complete teams)
    dset = ScoredTeamPredictionDataset(
        data_dir=data_dir,
        masker=masker,
        gen_weights=None,
        percentile=10.0,  # Top 10%
        split="train",
        validation_ratio=0.1,
        seed=42,
        verbose=True,
    )
    dset.enable_curriculum(initial_percentile=10.0)

    print("\n" + "=" * 60)
    print("TOP 10% - Only most complete teams")
    print("=" * 60)

    for i in range(3):
        x_tokens, type_ids, y_tokens, pred_mask = dset[i]
        x_seq = vocab.ints_to_pokeset_seq(x_tokens.tolist())

        # Count revealed tokens
        missing_tokens = set(vocab.missing_mask)
        num_revealed = sum(1 for t in y_tokens.tolist() if t not in missing_tokens)

        print(f"\n--- Sample {i+1} (top 10%) ---")
        print(f"Revealed tokens: {num_revealed}/49")
        print("Input (masked):")
        for j, (x, m) in enumerate(zip(x_seq, pred_mask.tolist())):
            marker = " [MASK]" if m else ""
            print(f"  {j:2d}: {x}{marker}")

    # Raise to 100%
    print("\n" + "=" * 60)
    print("TOP 100% - All teams including incomplete")
    print("=" * 60)

    dset.set_curriculum_percentile(100.0)
    print(f"New percentile: {dset.percentile}%")

    for i in range(3):
        x_tokens, type_ids, y_tokens, pred_mask = dset[i]
        x_seq = vocab.ints_to_pokeset_seq(x_tokens.tolist())

        # Count revealed tokens
        num_revealed = sum(1 for t in y_tokens.tolist() if t not in missing_tokens)

        print(f"\n--- Sample {i+1} (top 100%) ---")
        print(f"Revealed tokens: {num_revealed}/49")
        print("Input (masked):")
        for j, (x, m) in enumerate(zip(x_seq, pred_mask.tolist())):
            marker = " [MASK]" if m else ""
            print(f"  {j:2d}: {x}{marker}")

    # Show percentile interpolation
    print("\n" + "=" * 60)
    print("PERCENTILE INTERPOLATION DEMO")
    print("=" * 60)

    start_pct = 10.0
    end_pct = 100.0
    warmup_steps = 20_000

    for step in [0, 5_000, 10_000, 15_000, 20_000, 30_000]:
        progress = min(1.0, step / warmup_steps)
        percentile = start_pct + progress * (end_pct - start_pct)
        print(f"Step {step:>6d}: progress={progress:.2f}, percentile={percentile:.1f}%")


if __name__ == "__main__":
    main()
