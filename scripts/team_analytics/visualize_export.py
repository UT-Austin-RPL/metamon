#!/usr/bin/env python3
"""
Team Export Visualization Tool

Quick visualization of exported teams from team analytics.

Usage:
    python scripts/team_analytics/visualize_export.py teams_export.zip
"""

import sys
import zipfile
import json
from pathlib import Path
from typing import List, Dict, Any
import argparse


def visualize_team_file(team_path: Path) -> str:
    """
    Visualize a single team file (.{format}_team).

    Args:
        team_path: Path to team file

    Returns:
        Formatted string representation
    """
    with open(team_path, 'r') as f:
        content = f.read()

    lines = []
    lines.append("=" * 80)
    lines.append(f"Team: {team_path.name}")
    lines.append("=" * 80)
    lines.append("")
    lines.append(content)
    lines.append("")

    return "\n".join(lines)


def visualize_json_team(team_data: Dict[str, Any]) -> str:
    """
    Visualize a team from JSON format.

    Args:
        team_data: Team data dictionary

    Returns:
        Formatted string representation
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"Team: {team_data.get('team_hash', 'Unknown')}")
    lines.append("=" * 80)
    lines.append("")

    # Stats
    stats = team_data.get('stats', {})
    lines.append("Performance Stats:")
    lines.append(f"  Win Rate:     {stats.get('win_rate', 0):.1%}")
    lines.append(f"  Total Battles: {stats.get('total_battles', 0)}")
    lines.append(f"  Wins:         {stats.get('wins', 0)}")
    lines.append(f"  Avg Turns:    {stats.get('avg_turns', 0):.1f}")
    lines.append("")

    # Team composition
    species = team_data.get('species', [])
    lines.append(f"Team Composition: {', '.join(species)}")
    lines.append("")

    # Team data
    team_pokemon = team_data.get('team_data', [])
    if team_pokemon:
        lines.append("Pokemon Details:")
        for pokemon in team_pokemon:
            if isinstance(pokemon, dict):
                poke_name = pokemon.get('species') or pokemon.get('name', 'Unknown')
                lines.append(f"  - {poke_name}")

                # Ability
                if 'ability' in pokemon and pokemon['ability']:
                    lines.append(f"      Ability: {pokemon['ability']}")

                # Moves
                moves = pokemon.get('moves', [])
                if moves:
                    lines.append("      Moves:")
                    for move in moves:
                        if isinstance(move, dict):
                            move_name = move.get('id') or move.get('name', '')
                        else:
                            move_name = str(move)
                        if move_name:
                            lines.append(f"        - {move_name}")
    lines.append("")

    return "\n".join(lines)


def visualize_export(zip_path: str, max_teams: int = 10, show_lead_report: bool = True):
    """
    Visualize exported teams from ZIP archive.

    Args:
        zip_path: Path to ZIP export file
        max_teams: Maximum number of teams to display
        show_lead_report: Show lead variations report if available
    """
    print(f"\n{'='*80}")
    print("TEAM EXPORT VISUALIZATION")
    print(f"{'='*80}\n")
    print(f"Reading from: {zip_path}\n")

    with zipfile.ZipFile(zip_path, 'r') as zf:
        file_list = zf.namelist()

        # Check format
        team_files = [f for f in file_list if f.endswith('_team')]
        json_files = [f for f in file_list if f.endswith('.json') and 'team_' in f]

        if team_files:
            # Showdown format
            print(f"Found {len(team_files)} teams in Showdown format\n")
            print(f"Displaying first {min(max_teams, len(team_files))} teams:\n")

            for idx, team_file in enumerate(team_files[:max_teams]):
                with zf.open(team_file) as f:
                    content = f.read().decode('utf-8')

                print("=" * 80)
                print(f"Team {idx + 1}: {team_file}")
                print("=" * 80)
                print()
                print(content)
                print()

            # Show lead variations report if available
            if show_lead_report and 'lead_variations_report.txt' in file_list:
                print("\n")
                print("=" * 80)
                print("LEAD VARIATIONS REPORT")
                print("=" * 80)
                print()
                with zf.open('lead_variations_report.txt') as f:
                    print(f.read().decode('utf-8'))

        elif json_files:
            # JSON format
            print(f"Found {len(json_files)} teams in JSON format\n")
            print(f"Displaying first {min(max_teams, len(json_files))} teams:\n")

            for idx, json_file in enumerate(json_files[:max_teams]):
                with zf.open(json_file) as f:
                    team_data = json.load(f)

                print(visualize_json_team(team_data))

        else:
            print("No team files found in archive.")

        # Show summary CSV if available
        if 'teams_summary.csv' in file_list:
            print("\n")
            print("=" * 80)
            print("SUMMARY CSV PREVIEW (first 20 rows)")
            print("=" * 80)
            print()
            with zf.open('teams_summary.csv') as f:
                lines = f.read().decode('utf-8').split('\n')[:21]  # Header + 20 rows
                for line in lines:
                    print(line)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize exported teams from team analytics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize all teams (up to default 10)
  python scripts/team_analytics/visualize_export.py teams_export.zip

  # Show more teams
  python scripts/team_analytics/visualize_export.py teams_export.zip --max-teams 20

  # Hide lead report
  python scripts/team_analytics/visualize_export.py teams_export.zip --no-lead-report
        """
    )

    parser.add_argument(
        'zip_file',
        type=str,
        help='Path to exported teams ZIP file'
    )

    parser.add_argument(
        '--max-teams',
        type=int,
        default=10,
        help='Maximum number of teams to display (default: 10)'
    )

    parser.add_argument(
        '--no-lead-report',
        action='store_true',
        help='Hide lead variations report'
    )

    args = parser.parse_args()

    if not Path(args.zip_file).exists():
        print(f"Error: File not found: {args.zip_file}")
        sys.exit(1)

    visualize_export(
        args.zip_file,
        max_teams=args.max_teams,
        show_lead_report=not args.no_lead_report
    )


if __name__ == '__main__':
    main()
