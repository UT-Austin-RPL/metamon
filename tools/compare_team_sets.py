#!/usr/bin/env python3
"""
Compare N team sets for a given battle format.

Produces:
  1. A summary table (all sets as columns, at a configurable presence threshold).
  2. Threshold-sweep plots — how many species / moves / unique sets survive
     as a function of minimum team-presence threshold.
  3. Rank-frequency (log-log) plots for species and moves.
  4. Pairwise Jaccard heatmap across all sets for species, moves, and sets.
  5. Per-set "exclusives" — what is unique to each set vs. all others.

Team-file loading is parallelised across workers; each file is parsed
independently so large sets scale linearly.

Usage
-----
    python tools/compare_team_sets.py \\
        --format gen1ou \\
        --sets competitive elite_sets_filled modern_replays_v2 \\
        --workers 8 \\
        --save tools/team_set_comparison.png
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from tqdm import tqdm

import metamon
from metamon.env.wrappers import get_metamon_teams
from metamon.backend.team_prediction.team import PokemonSet
from metamon.backend.team_prediction.team import TeamSet as BackendTeamSet

# ─────────────────────────────────────────────────────────────────── #
#  Constants
# ─────────────────────────────────────────────────────────────────── #

_SKIP_MOVES = {PokemonSet.MISSING_MOVE, PokemonSet.NO_MOVE, ""}
_SKIP_ITEMS = {PokemonSet.MISSING_ITEM, PokemonSet.NO_ITEM, ""}
_SKIP_ABILITIES = {PokemonSet.MISSING_ABILITY, PokemonSet.NO_ABILITY, ""}

THRESHOLDS: List[float] = sorted(
    set(
        [0.1, 0.2, 0.5]
        + list(np.arange(1.0, 21.0, 1.0))
        + list(np.arange(25.0, 55.0, 5.0))
    )
)
TABLE_THRESHOLD = 1.0  # % presence used for summary table

# Perceptually distinct palette — cycles if more sets than colours
_PALETTE = [
    "#4878a8",  # steel blue
    "#e07040",  # burnt orange
    "#4a9a6a",  # muted green
    "#7b62a8",  # dusty violet
    "#c43c3c",  # crimson
    "#2ca089",  # teal-green
    "#d4a853",  # gold
    "#b55a82",  # muted rose
    "#8b6e4e",  # warm brown
    "#5b5ea6",  # indigo
]

MARKERS = ["o", "s", "^", "v", "D", "P", "X", "h", "*", "p"]


def _color(i: int) -> str:
    return _PALETTE[i % len(_PALETTE)]


def _marker(i: int) -> str:
    return MARKERS[i % len(MARKERS)]


# ─────────────────────────────────────────────────────────────────── #
#  Parallel loading
# ─────────────────────────────────────────────────────────────────── #


def _parse_file(args: Tuple[str, str]) -> Optional[dict]:
    """Worker: parse one team file; return per-file counters or None on error."""
    path, battle_format = args
    try:
        team = BackendTeamSet.from_showdown_file(path, battle_format)
    except Exception:
        return None

    species_ctr: Counter = Counter()
    move_ctr: Counter = Counter()
    item_ctr: Counter = Counter()
    ability_ctr: Counter = Counter()
    set_ctr: Counter = Counter()

    valid = [p for p in team.pokemon if p.name != PokemonSet.MISSING_NAME]
    for p in valid:
        species_ctr[p.name] += 1
        for m in p.moves:
            if m not in _SKIP_MOVES:
                move_ctr[m] += 1
        if p.item not in _SKIP_ITEMS:
            item_ctr[p.item] += 1
        if p.ability not in _SKIP_ABILITIES:
            ability_ctr[p.ability] += 1
        set_ctr[p.set_key] += 1

    team_key = frozenset(p.set_key for p in valid)
    return {
        "species": species_ctr,
        "moves": move_ctr,
        "items": item_ctr,
        "abilities": ability_ctr,
        "sets": set_ctr,
        "team_key": team_key,
    }


def load_set(set_name: str, battle_format: str, workers: int) -> dict:
    """Load all team files for one set in parallel; return aggregated stats."""
    wrapper = get_metamon_teams(battle_format, set_name)
    file_args = [(p, battle_format) for p in wrapper.team_files]

    species_ctr: Counter = Counter()
    move_ctr: Counter = Counter()
    item_ctr: Counter = Counter()
    ability_ctr: Counter = Counter()
    set_ctr: Counter = Counter()
    team_ctr: Counter = Counter()
    n_teams = 0
    n_errors = 0

    chunksize = max(1, len(file_args) // (workers * 4))
    with mp.Pool(workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_parse_file, file_args, chunksize=chunksize),
            total=len(file_args),
            desc=f"  {set_name}",
            leave=False,
        ):
            if result is None:
                n_errors += 1
                continue
            n_teams += 1
            species_ctr.update(result["species"])
            move_ctr.update(result["moves"])
            item_ctr.update(result["items"])
            ability_ctr.update(result["abilities"])
            set_ctr.update(result["sets"])
            team_ctr[result["team_key"]] += 1

    if n_errors:
        print(f"  [{set_name}] {n_errors} file(s) skipped due to parse errors")

    return {
        "name": set_name,
        "n_teams": n_teams,
        "species": species_ctr,
        "moves": move_ctr,
        "items": item_ctr,
        "abilities": ability_ctr,
        "sets": set_ctr,
        "teams": team_ctr,
    }


# ─────────────────────────────────────────────────────────────────── #
#  Derived statistics
# ─────────────────────────────────────────────────────────────────── #


def _above_threshold(counter: Counter, n_teams: int, pct: float) -> int:
    """Count entries whose team-presence rate ≥ pct %."""
    min_count = pct / 100.0 * n_teams
    return sum(1 for v in counter.values() if v >= min_count)


def _shannon_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((v / total) * math.log2(v / total) for v in counter.values() if v > 0)


def _jaccard(keys_a, keys_b) -> float:
    a, b = set(keys_a), set(keys_b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _threshold_curve(counter: Counter, n_teams: int) -> Tuple[List[float], List[int]]:
    return THRESHOLDS, [_above_threshold(counter, n_teams, t) for t in THRESHOLDS]


def _rank_freq(counter: Counter) -> Tuple[np.ndarray, np.ndarray]:
    counts = sorted(counter.values(), reverse=True)
    return np.arange(1, len(counts) + 1), np.array(counts, dtype=float)


# ─────────────────────────────────────────────────────────────────── #
#  Console output
# ─────────────────────────────────────────────────────────────────── #


def print_table(stats_list: List[dict]) -> None:
    T = TABLE_THRESHOLD
    names = [s["name"] for s in stats_list]

    def row(label, fn):
        return [label] + [fn(s) for s in stats_list]

    rows = [
        row("Teams", lambda s: f"{s['n_teams']:,}"),
        row("Unique species", lambda s: str(len(s["species"]))),
        row(
            f"Species ≥{T}% presence",
            lambda s: str(_above_threshold(s["species"], s["n_teams"], T)),
        ),
        row("Unique moves", lambda s: str(len(s["moves"]))),
        row(
            f"Moves ≥{T}% presence",
            lambda s: str(_above_threshold(s["moves"], s["n_teams"], T)),
        ),
        row("Unique items", lambda s: str(len(s["items"]))),
        row("Unique abilities", lambda s: str(len(s["abilities"]))),
        row("Unique sets {sp,mv,it,ab}", lambda s: f"{len(s['sets']):,}"),
        row(
            f"Sets ≥{T}% presence",
            lambda s: str(_above_threshold(s["sets"], s["n_teams"], T)),
        ),
        row("Unique full teams", lambda s: f"{len(s['teams']):,}"),
        row(
            "Species entropy (bits)", lambda s: f"{_shannon_entropy(s['species']):.2f}"
        ),
        row("Set entropy (bits)", lambda s: f"{_shannon_entropy(s['sets']):.2f}"),
    ]

    col_w = [max(len(r[i]) for r in rows) for i in range(len(names) + 1)]
    col_w[0] = max(col_w[0], 28)
    for i, name in enumerate(names):
        col_w[i + 1] = max(col_w[i + 1], len(name))

    header_parts = [f"{'Metric':<{col_w[0]}}"] + [
        f"{n:>{col_w[i+1]}}" for i, n in enumerate(names)
    ]
    header = "  ".join(header_parts)
    sep = "─" * len(header)

    print()
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        parts = [f"{r[0]:<{col_w[0]}}"] + [
            f"{r[i+1]:>{col_w[i+1]}}" for i in range(len(names))
        ]
        print("  ".join(parts))
    print(sep)
    print()


def print_pairwise_jaccard(stats_list: List[dict]) -> None:
    """Print pairwise Jaccard similarity tables for species and sets."""
    names = [s["name"] for s in stats_list]
    n = len(names)
    name_w = max(len(nm) for nm in names)

    for dim_key, dim_label in [("species", "Species"), ("sets", "Sets")]:
        print(f"Jaccard ({dim_label})")
        header = " " * (name_w + 2) + "  ".join(f"{nm:>{name_w}}" for nm in names)
        print(header)
        print("─" * len(header))
        for i, si in enumerate(stats_list):
            row_vals = []
            for j, sj in enumerate(stats_list):
                j_val = _jaccard(si[dim_key], sj[dim_key])
                row_vals.append(f"{j_val:.2f}".rjust(name_w))
            print(f"{names[i]:<{name_w}}  " + "  ".join(row_vals))
        print()


def _fmt_set_key(k) -> str:
    if isinstance(k, tuple):
        species, moves, item, ability = k
        move_str = ", ".join(sorted(moves)) if moves else "—"
        item_str = f" @ {item}" if item not in _SKIP_ITEMS else ""
        return f"{species}{item_str}  [{move_str}]"
    return str(k)


def print_exclusives(stats_list: List[dict], top_n: int = 12) -> None:
    """For each set, print top species / sets that appear nowhere else."""
    all_keys = {
        dim: [set(s[dim]) for s in stats_list] for dim in ("species", "moves", "sets")
    }

    for dim, label in [("species", "Species"), ("moves", "Moves"), ("sets", "Sets")]:
        print(f"\n{'─'*60}")
        print(f"  {label} unique to each set (not present in any other)")
        print(f"{'─'*60}")
        for i, s in enumerate(stats_list):
            others = set().union(
                *[all_keys[dim][j] for j in range(len(stats_list)) if j != i]
            )
            exclusive = set(s[dim]) - others
            if not exclusive:
                print(f"  {s['name']}: (none)")
                continue
            top = sorted(exclusive, key=lambda k: s[dim][k], reverse=True)[:top_n]
            print(f"  {s['name']} ({len(exclusive)} exclusive {label.lower()}):")
            for k in top:
                print(f"    {_fmt_set_key(k):<68}  {s[dim][k]:>5}×")


# ─────────────────────────────────────────────────────────────────── #
#  Plotting
# ─────────────────────────────────────────────────────────────────── #


def plot(
    stats_list: List[dict], battle_format: str, save_path: Optional[str] = None
) -> None:
    n = len(stats_list)
    names = [s["name"] for s in stats_list]

    # Layout: 3 columns
    #   row 0: threshold sweeps  (species | moves | sets)
    #   row 1: rank-freq log-log (species | moves) + species Jaccard heatmap
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)
    axes_sweep = [fig.add_subplot(gs[0, c]) for c in range(3)]
    axes_rf = [fig.add_subplot(gs[1, c]) for c in range(2)]
    ax_heat = fig.add_subplot(gs[1, 2])

    sweep_cfg = [
        ("species", "Unique Species"),
        ("moves", "Unique Moves"),
        ("sets", "Unique Sets\n{species, moves, item, ability}"),
    ]
    rf_cfg = [
        ("species", "Species occurrence count"),
        ("moves", "Move occurrence count"),
    ]

    # ── threshold sweeps ─────────────────────────────────────── #
    for ax, (key, ylabel) in zip(axes_sweep, sweep_cfg):
        for i, s in enumerate(stats_list):
            xs, ys = _threshold_curve(s[key], s["n_teams"])
            ax.plot(
                xs,
                ys,
                color=_color(i),
                marker=_marker(i),
                ms=4,
                lw=2,
                label=s["name"],
            )
        ax.set_xlabel("Min. team-presence threshold (%)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:g}"))
        ax.grid(True, alpha=0.22, linewidth=0.5)
        ax.legend(fontsize=8, framealpha=0.92)

    # ── rank-frequency ────────────────────────────────────────── #
    for ax, (key, ylabel) in zip(axes_rf, rf_cfg):
        for i, s in enumerate(stats_list):
            r, f = _rank_freq(s[key])
            ax.plot(r, f, color=_color(i), lw=1.8, label=s["name"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Rank", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.22, linewidth=0.5, which="both")
        ax.legend(fontsize=8, framealpha=0.92)

    # ── pairwise Jaccard heatmap (species) ───────────────────── #
    J = np.array(
        [
            [_jaccard(si["species"], sj["species"]) for sj in stats_list]
            for si in stats_list
        ]
    )
    im = ax_heat.imshow(J, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax_heat.set_xticks(range(n))
    ax_heat.set_yticks(range(n))
    ax_heat.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax_heat.set_yticklabels(names, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax_heat.text(
                j,
                i,
                f"{J[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="black" if J[i, j] < 0.6 else "white",
            )
    fig.colorbar(im, ax=ax_heat, shrink=0.82, label="Jaccard (species)")
    ax_heat.set_title("Pairwise Species Overlap", fontsize=10)

    title_sets = "  ·  ".join(names)
    fig.suptitle(
        f"Team-Set Comparison — {battle_format.upper()}\n{title_sets}",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"\nSaved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────── #
#  CLI
# ─────────────────────────────────────────────────────────────────── #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare N Metamon team sets for a given battle format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--format", default="gen1ou", help="Battle format (e.g. gen1ou)"
    )
    parser.add_argument(
        "--sets",
        nargs="+",
        default=["competitive", "elite_sets_filled"],
        metavar="SET",
        help="Two or more team-set names to compare",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, mp.cpu_count() // 2),
        help="Parallel workers for file parsing (default: half of CPU count)",
    )
    _here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "--save",
        nargs="?",
        const=os.path.join(_here, "team_set_comparison.png"),
        default=None,
        metavar="PATH",
        help="Save figure to PATH (default: <script dir>/team_set_comparison.png)",
    )
    parser.add_argument(
        "--no_exclusives",
        action="store_true",
        help="Skip the per-set exclusives printout",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=12,
        help="Max exclusive entries to print per set (default: 12)",
    )
    args = parser.parse_args()

    fmt = args.format
    print(f"\nFormat: {fmt}   Workers: {args.workers}")
    print(f"Sets   : {', '.join(args.sets)}\n")

    stats_list = []
    for set_name in args.sets:
        print(f"Loading '{set_name}' …")
        stats = load_set(set_name, fmt, args.workers)
        stats_list.append(stats)
        print(f"  → {stats['n_teams']:,} teams loaded")

    print("\n" + "═" * 60)
    print_table(stats_list)
    print_pairwise_jaccard(stats_list)

    if not args.no_exclusives:
        print_exclusives(stats_list, top_n=args.top_n)

    print("\nGenerating plots …")
    plot(stats_list, fmt, save_path=args.save)


if __name__ == "__main__":
    main()
