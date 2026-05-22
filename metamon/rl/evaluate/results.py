"""
Results tracking, crash recovery, and win matrix output for auto-evaluation.

Uses an append-only JSONL file so partial runs can be resumed.
"""

import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Set


@dataclass
class MatchupResult:
    """Result of a single head-to-head matchup."""

    matchup_id: str
    policy_a_name: str
    policy_b_name: str
    policy_a_wins: int
    policy_b_wins: int
    total_battles: int
    timestamp: str

    @property
    def policy_a_win_rate(self) -> Optional[float]:
        if self.total_battles < 1:
            return None
        return self.policy_a_wins / self.total_battles


class ResultsTracker:
    """Track matchup results with crash recovery via append-only JSONL.

    Args:
        output_dir: Directory for results files.
    """

    RESULTS_FILE = "matchup_results.jsonl"
    WIN_MATRIX_FILE = "win_matrix.csv"

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results_path = os.path.join(output_dir, self.RESULTS_FILE)
        self._completed: Dict[str, MatchupResult] = {}
        self._load_existing()

    def _load_existing(self):
        """Load previously completed matchups for crash recovery."""
        if not os.path.exists(self.results_path):
            return
        with open(self.results_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    result = MatchupResult(**data)
                    self._completed[result.matchup_id] = result
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"Warning: skipping malformed result line: {e}")

    def is_completed(self, matchup_id: str) -> bool:
        return matchup_id in self._completed

    @property
    def completed_ids(self) -> Set[str]:
        return set(self._completed.keys())

    def record_result(self, result: MatchupResult):
        """Append a result to the JSONL file and in-memory cache."""
        self._completed[result.matchup_id] = result
        with open(self.results_path, "a") as f:
            f.write(json.dumps(asdict(result)) + "\n")

    def record_from_results_dir(
        self,
        matchup_id: str,
        policy_a_name: str,
        policy_b_name: str,
        results_dir: str,
        challenger_username: str,
    ) -> Optional[MatchupResult]:
        """Read battle CSVs written by PokeEnvWrapper and record the result.

        ``PokeEnvWrapper`` writes per-player CSV files inside ``results_dir``
        with columns: Player Username, Team File, Opponent Username, Result,
        Turn Count, Battle ID (first row is a header).

        We count WIN/LOSS only from the challenger's rows (matched by exact
        username) to avoid double-counting since both sides write CSVs.
        """
        if not os.path.exists(results_dir):
            print(f"Warning: results dir not found: {results_dir}")
            return None

        a_wins = 0
        b_wins = 0
        total = 0

        for csv_file in os.listdir(results_dir):
            if not csv_file.endswith(".csv"):
                continue
            path = os.path.join(results_dir, csv_file)
            try:
                with open(path, "r") as f:
                    reader = csv.reader(f)
                    next(reader, None)  # skip header
                    for row in reader:
                        if len(row) < 4:
                            continue
                        username = row[0].strip()
                        if username != challenger_username:
                            continue
                        result_str = row[3].strip()
                        total += 1
                        if result_str == "WIN":
                            a_wins += 1
                        elif result_str == "LOSS":
                            b_wins += 1
            except Exception as e:
                print(f"Warning: failed to parse {path}: {e}")

        if total == 0:
            print(f"Warning: no battle results found in {results_dir}")
            return None

        result = MatchupResult(
            matchup_id=matchup_id,
            policy_a_name=policy_a_name,
            policy_b_name=policy_b_name,
            policy_a_wins=a_wins,
            policy_b_wins=b_wins,
            total_battles=total,
            timestamp=datetime.now().isoformat(),
        )
        self.record_result(result)
        return result

    def get_all_results(self) -> List[MatchupResult]:
        return list(self._completed.values())

    def build_win_matrix(self) -> Dict[str, Dict[str, Optional[float]]]:
        """Build a win-rate matrix from all completed matchups.

        Returns:
            Nested dict: matrix[row_policy][col_policy] = row's win rate against col.
            None for unplayed matchups, diagonal is empty.
        """
        # Collect all unique policy names
        names = set()
        for r in self._completed.values():
            names.add(r.policy_a_name)
            names.add(r.policy_b_name)
        names = sorted(names)

        matrix = {a: {b: None for b in names} for a in names}
        for r in self._completed.values():
            a, b = r.policy_a_name, r.policy_b_name
            wr = r.policy_a_win_rate
            if wr is not None:
                matrix[a][b] = wr
                matrix[b][a] = 1.0 - wr

        return matrix

    def print_win_matrix(self):
        """Print a formatted win matrix to the terminal using rich."""
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text

        matrix = self.build_win_matrix()
        if not matrix:
            print("No results to display.")
            return

        console = Console()
        names = sorted(matrix.keys())

        # Build a numbered legend so column headers stay compact
        idx_map = {name: i for i, name in enumerate(names, 1)}

        # Legend table
        legend = Table(
            title="Policy Legend",
            title_style="bold",
            show_header=True,
            header_style="bold",
            box=None,
            pad_edge=False,
            padding=(0, 1),
        )
        legend.add_column("#", style="bold yellow", justify="right", width=3)
        legend.add_column("Policy", style="bold white")
        for name in names:
            legend.add_row(str(idx_map[name]), name)

        console.print()
        console.print(legend)
        console.print()

        # Win matrix table
        table = Table(
            title="Win Matrix (row win rate vs column)",
            title_style="bold",
            show_header=True,
            header_style="bold cyan",
            show_lines=True,
            pad_edge=True,
        )
        table.add_column("", style="bold yellow", justify="right")
        for name in names:
            table.add_column(str(idx_map[name]), justify="center", width=7)

        for row in names:
            cells = [f"[bold yellow]{idx_map[row]}[/bold yellow]"]
            for col in names:
                if row == col:
                    cells.append("[dim]—[/dim]")
                elif matrix[row][col] is None:
                    cells.append("[dim italic]?[/dim italic]")
                else:
                    wr = matrix[row][col]
                    # Color by win rate
                    if wr >= 0.6:
                        style = "bold green"
                    elif wr >= 0.4:
                        style = "yellow"
                    else:
                        style = "red"
                    cells.append(f"[{style}]{wr:.1%}[/{style}]")
            table.add_row(*cells)

        console.print(table)
        console.print()

    def write_win_matrix_csv(self):
        """Write the win matrix to a CSV file."""
        matrix = self.build_win_matrix()
        if not matrix:
            return

        names = sorted(matrix.keys())
        path = os.path.join(self.output_dir, self.WIN_MATRIX_FILE)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([""] + names)
            for row in names:
                cells = []
                for col in names:
                    if row == col:
                        cells.append("")
                    elif matrix[row][col] is None:
                        cells.append("")
                    else:
                        cells.append(f"{matrix[row][col]:.4f}")
                writer.writerow([row] + cells)
        print(f"Win matrix written to {path}")
