#!/usr/bin/env python
"""
Analyze and plot move popularity trends for a Pokemon over time.

Usage:
    python tools/analyze_moveset_trends.py --pokemon snorlax --format gen1ou --top_k 9
    python tools/analyze_moveset_trends.py --pokemon exeggutor --format gen1ou --top_k 6
    python tools/analyze_moveset_trends.py --pokemon tauros --format gen9ou --top_k 8
"""

import argparse
import datetime
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from metamon.backend.team_prediction.usage_stats import (
    get_usage_stats,
    DEFAULT_USAGE_RANK,
)


def get_monthly_dates(start_year: int, end_year: int) -> list[datetime.date]:
    """Generate first-of-month dates from start_year to end_year (inclusive)."""
    dates = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            dates.append(datetime.date(year, month, 1))
    return dates


def main(args):
    format_name = args.format
    pokemon_name = args.pokemon
    top_k = args.top_k
    start_year = args.start_year
    end_year = args.end_year
    rank = args.rank
    output_file = (
        args.output or f"{pokemon_name}_{format_name}_rank{rank}_move_trends.png"
    )

    print(
        f"Analyzing {pokemon_name} in {format_name} ({start_year}-{end_year}, rank={rank})"
    )

    # Step 1: Load all-time stats to find top K moves
    print("Loading all-time stats to determine top moves...")
    all_time_stats = get_usage_stats(
        format_name,
        start_date=datetime.date(start_year, 1, 1),
        end_date=datetime.date(end_year, 12, 1),
        rank=rank,
    )

    try:
        pokemon_data = all_time_stats[pokemon_name]
    except KeyError:
        print(f"Error: Pokemon '{pokemon_name}' not found in {format_name} usage stats")
        return

    all_moves = pokemon_data.get("moves", {})
    # Filter out "Other" and "Nothing"
    filtered_moves = {
        k: v for k, v in all_moves.items() if k not in ("Other", "Nothing")
    }
    # Sort by usage and take top K
    sorted_moves = sorted(filtered_moves.items(), key=lambda x: x[1], reverse=True)
    top_moves = [move for move, _ in sorted_moves[:top_k]]

    print(f"Top {top_k} moves (all-time): {top_moves}")

    # Step 2: Track these moves month by month
    print("Loading monthly stats...")
    monthly_dates = get_monthly_dates(start_year, end_year)
    move_trends = defaultdict(list)
    valid_dates = []

    for date in monthly_dates:
        try:
            monthly_stats = get_usage_stats(
                format_name,
                start_date=date,
                end_date=date,
                rank=rank,
            )
            pokemon_monthly = monthly_stats[pokemon_name]
            moves = pokemon_monthly.get("moves", {})
            valid_dates.append(date)
            for move in top_moves:
                # Use 0 if move not present in this month's data
                move_trends[move].append(moves.get(move, 0.0))
        except (KeyError, FileNotFoundError) as e:
            # Skip months where data is missing
            print(f"  Skipping {date}: {e}")
            continue

    if not valid_dates:
        print("Error: No valid monthly data found")
        return

    print(f"Collected data for {len(valid_dates)} months")

    # Step 3: Plot
    fig, ax = plt.subplots(figsize=(14, 7))

    # White background with grid
    ax.set_facecolor("white")
    ax.grid(True, linestyle="-", alpha=0.3, color="gray")

    # Generate colors from a pastel colormap
    cmap = cm.get_cmap("tab20", len(top_moves))

    # Plot each move
    for i, move in enumerate(top_moves):
        color = cmap(i)
        ax.plot(
            valid_dates,
            move_trends[move],
            label=move,
            color=color,
            linewidth=3.5,
            marker="o",
            markersize=8,
            alpha=0.9,
        )

    title = f"{pokemon_name.upper()} Move Trends in {format_name.upper()} ({start_year}-{end_year}, rank={rank})"
    ax.set_title(title, fontsize=18, fontweight="bold", family="monospace", pad=15)

    ax.set_xlabel("Date", fontsize=14, fontweight="bold")
    ax.set_ylabel("Usage Rate", fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", labelsize=12)
    ax.set_ylim(0, 1.05)

    # Legend outside plot
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        facecolor="white",
        edgecolor="lightgray",
        fontsize=12,
    )

    # Format x-axis dates
    fig.autofmt_xdate(rotation=45)

    plt.tight_layout()

    # Save plot
    plt.savefig(output_file, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved plot to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze and plot move popularity trends for a Pokemon over time"
    )
    parser.add_argument(
        "--pokemon",
        type=str,
        required=True,
        help="Pokemon name to analyze (e.g., snorlax, exeggutor)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="gen1ou",
        help="Format to analyze (e.g., gen1ou, gen9ou)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=9,
        help="Number of top moves to track (default: 9)",
    )
    parser.add_argument(
        "--start_year",
        type=int,
        default=2015,
        help="Start year for analysis (default: 2015)",
    )
    parser.add_argument(
        "--end_year",
        type=int,
        default=2025,
        help="End year for analysis (default: 2025)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=DEFAULT_USAGE_RANK,
        help=f"Usage stats rank/baseline (default: {DEFAULT_USAGE_RANK})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output filename for the plot (default: {pokemon}_{format}_rank{rank}_move_trends.png)",
    )
    args = parser.parse_args()
    main(args)
