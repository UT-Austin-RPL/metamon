"""
Dry-run visualization for auto-evaluation configs.

Parses a config file and prints a detailed, human-readable preview
of all matchups/agents that will be launched — without actually running anything.

Supports all modes: h2h (matrix), sweep (table), ladder_self_play (agent list).
"""

from typing import Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from metamon.rl.evaluate.common import (
    MatchupSpec,
    PolicySpec,
    _format_value_for_display,
)


def _fmt(value) -> str:
    """Format a raw config value (list / dict / scalar) as a compact string."""
    if value is None:
        return "-"
    return _format_value_for_display(value)


console = Console()


def _policy_detail(p: PolicySpec) -> Text:
    """Rich-formatted one-line detail string for a policy."""
    t = Text()
    t.append("model=", style="dim")
    t.append(p.model_name, style="cyan")
    t.append("  ckpt=", style="dim")
    t.append(str(p.checkpoint), style="yellow" if p.checkpoint is not None else "dim")
    t.append("  temp=", style="dim")
    t.append(str(p.temperature), style="green" if p.temperature != 1.0 else "dim")
    t.append("  teams=", style="dim")
    t.append(p.team_set, style="magenta")
    t.append("  backend=", style="dim")
    t.append(p.battle_backend, style="dim")
    return t


def preview_matchups(
    matchups: List[MatchupSpec],
    mode: str = "h2h",
    template_values: Optional[dict] = None,
):
    """Print a detailed preview of all matchups.

    Args:
        matchups: List of matchups to preview.
        mode: "h2h" shows a matrix, "sweep" shows a table.
        template_values: Resolved ``${var}`` values (shown in header if present).
    """
    if not matchups:
        console.print("[dim]No matchups to preview.[/dim]")
        return

    # Collect unique policies (ordered by first appearance)
    policies = {}
    for m in matchups:
        if m.policy_a.short_label not in policies:
            policies[m.policy_a.short_label] = m.policy_a
        if m.policy_b.short_label not in policies:
            policies[m.policy_b.short_label] = m.policy_b

    battle_format = matchups[0].battle_format
    n_battles = matchups[0].n_battles
    title = "HEAD-TO-HEAD" if mode == "h2h" else "SWEEP"

    # Header panel
    header = Text()
    header.append(f"{title} EVALUATION PREVIEW\n", style="bold white")
    header.append(f"Format: ", style="dim")
    header.append(f"{battle_format}", style="cyan bold")
    header.append(f"  │  Battles per matchup: ", style="dim")
    header.append(f"{n_battles}", style="cyan bold")
    header.append(f"  │  Total matchups: ", style="dim")
    header.append(f"{len(matchups)}", style="cyan bold")
    header.append(f"  │  Total battles: ", style="dim")
    header.append(f"{len(matchups) * n_battles}", style="cyan bold")
    if template_values:
        header.append(f"\n")
        header.append(f"Template: ", style="dim")
        for k, v in template_values.items():
            header.append(f"${{{k}}}", style="yellow")
            header.append(f"={v}", style="bold white")
            header.append(f"  ", style="dim")
    console.print()
    console.print(Panel(header, border_style="blue"))

    # Policy legend table
    legend = Table(
        title="Policies",
        title_style="bold",
        show_header=True,
        header_style="bold",
        pad_edge=False,
        box=None,
        padding=(0, 1),
    )
    legend.add_column("#", style="bold yellow", justify="right", width=3)
    legend.add_column("Label", style="bold white")
    legend.add_column("Details", no_wrap=False)

    sorted_labels = sorted(policies.keys())
    label_to_idx = {}
    for i, label in enumerate(sorted_labels, 1):
        label_to_idx[label] = i
        legend.add_row(str(i), label, _policy_detail(policies[label]))

    console.print(legend)
    console.print()

    if mode == "h2h":
        _preview_matrix(matchups, sorted_labels, label_to_idx)
    else:
        _preview_sweep_table(matchups)

    console.print()


def _preview_matrix(
    matchups: List[MatchupSpec], sorted_labels: List[str], label_to_idx: dict
):
    """Print the h2h matchup matrix using numbered indices."""
    # Build lookup: which matchups exist
    matchup_set = set()
    for m in matchups:
        matchup_set.add((m.policy_a.short_label, m.policy_b.short_label))
        matchup_set.add((m.policy_b.short_label, m.policy_a.short_label))

    # Build the Rich table with numbered columns
    table = Table(
        title="Matchup Matrix",
        title_style="bold",
        show_header=True,
        header_style="bold cyan",
        show_lines=True,
        pad_edge=True,
    )
    table.add_column("", style="bold yellow", justify="right")  # row label
    for label in sorted_labels:
        idx = label_to_idx[label]
        table.add_column(str(idx), justify="center", width=3)

    for row_label in sorted_labels:
        row_idx = label_to_idx[row_label]
        cells = [f"[bold yellow]{row_idx}[/bold yellow]"]
        for col_label in sorted_labels:
            if row_label == col_label:
                cells.append("[dim]·[/dim]")
            elif (row_label, col_label) in matchup_set:
                cells.append("[bold green]✓[/bold green]")
            else:
                cells.append("")
        table.add_row(*cells)

    console.print(table)


def _preview_sweep_table(matchups: List[MatchupSpec]):
    """Print sweep matchups as a formatted table."""
    # In sweep mode, policy_b is always the fixed opponent
    opponent = matchups[0].policy_b

    opp_text = Text()
    opp_text.append("Fixed opponent: ", style="dim")
    opp_text.append(opponent.short_label, style="bold white")
    opp_text.append("  (")
    opp_text.append_text(_policy_detail(opponent))
    opp_text.append(")")
    console.print(opp_text)
    console.print()

    table = Table(
        title="Sweep Points",
        title_style="bold",
        show_header=True,
        header_style="bold",
        show_lines=False,
        pad_edge=True,
    )
    table.add_column("#", style="bold yellow", justify="right", width=4)
    table.add_column("Policy", style="bold white")
    table.add_column("Checkpoint", justify="right", style="yellow")
    table.add_column("Temp", justify="right", style="green")
    table.add_column("Team Set", style="magenta")

    for i, m in enumerate(matchups, 1):
        p = m.policy_a
        ckpt_str = str(p.checkpoint) if p.checkpoint is not None else "default"
        table.add_row(
            str(i),
            p.short_label,
            ckpt_str,
            f"{p.temperature:.2f}",
            p.team_set,
        )

    console.print(table)


def preview_ladder_agents(agents: List[dict]):
    """Print self-play ladder agents as a formatted table.

    Args:
        agents: List of dicts with at least 'username' and 'model_name' keys.
    """
    if not agents:
        console.print("[dim]No agents to preview.[/dim]")
        return

    header = Text("LADDER SELF-PLAY PREVIEW\n", style="bold white")
    header.append("All agents play each other via random matchmaking.", style="dim")
    console.print()
    console.print(Panel(header, border_style="blue"))

    table = Table(
        title=f"Agents ({len(agents)})",
        title_style="bold",
        show_header=True,
        header_style="bold",
        show_lines=False,
        pad_edge=True,
    )
    table.add_column("#", style="bold yellow", justify="right", width=4)
    table.add_column("Username", style="bold white")
    table.add_column("Model", style="cyan")
    table.add_column("Checkpoint", justify="right", style="yellow")
    table.add_column("Temp", justify="right", style="green")
    table.add_column("Team Set", style="magenta")

    for i, a in enumerate(agents, 1):
        table.add_row(
            str(i),
            a.get("username", "?"),
            a.get("model_name", "?"),
            _fmt(a.get("checkpoint")),
            _fmt(a.get("temperature")),
            _fmt(a.get("team_set")),
        )

    console.print(table)
    console.print()
