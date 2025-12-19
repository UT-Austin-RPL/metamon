"""
Team Export Utilities

Export filtered teams to various formats (JSON, CSV, ZIP).
"""

import json
import zipfile
from pathlib import Path
from typing import List, Optional, Any, Dict
import pandas as pd
import numpy as np


class TeamExporter:
    """Export team data to various formats."""

    @staticmethod
    def _convert_to_serializable(obj: Any) -> Any:
        """
        Convert numpy types to JSON-serializable Python types.

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable version of object
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: TeamExporter._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [TeamExporter._convert_to_serializable(item) for item in obj]
        else:
            return obj

    @staticmethod
    def export_teams_to_json(
        teams_df: pd.DataFrame,
        output_dir: str,
        filename_prefix: str = "team"
    ) -> List[str]:
        """
        Export teams to individual JSON files.

        Args:
            teams_df: DataFrame with columns including 'team_hash', 'team_species', 'player_team_json'
            output_dir: Directory to save JSON files
            filename_prefix: Prefix for output filenames

        Returns:
            List of created file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        created_files = []

        for idx, row in teams_df.iterrows():
            team_hash = row['team_hash']
            team_species = row['team_species']
            team_json = row.get('player_team_json', '[]')

            # Parse team JSON
            try:
                team_data = json.loads(team_json)
            except:
                team_data = []

            # Convert species list (may be numpy array)
            species_list = TeamExporter._convert_to_serializable(team_species)

            # Create filename
            species_str = "_".join(species_list[:3])  # First 3 species
            filename = f"{filename_prefix}_{team_hash}_{species_str}.json"
            filepath = output_path / filename

            # Build export data
            export_data = {
                'team_hash': team_hash,
                'species': species_list,
                'team_data': team_data,
                'stats': {
                    'win_rate': float(row.get('win_rate', 0)),
                    'total_battles': int(row.get('total_battles', 0)),
                    'wins': int(row.get('wins', 0)),
                    'avg_turns': float(row.get('avg_turns', 0)),
                }
            }

            # Convert entire structure to ensure all numpy types are handled
            export_data = TeamExporter._convert_to_serializable(export_data)

            # Write JSON
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)

            created_files.append(str(filepath))

        return created_files

    @staticmethod
    def export_teams_to_zip(
        teams_df: pd.DataFrame,
        output_file: str,
        include_csv: bool = True
    ) -> str:
        """
        Export teams to a ZIP archive.

        Args:
            teams_df: DataFrame with team data
            output_file: Path to output ZIP file
            include_csv: Include a summary CSV in the archive

        Returns:
            Path to created ZIP file
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add individual team JSON files
            for idx, row in teams_df.iterrows():
                team_hash = row['team_hash']
                team_species = row['team_species']
                team_json = row.get('player_team_json', '[]')

                # Parse team JSON
                try:
                    team_data = json.loads(team_json)
                except:
                    team_data = []

                # Convert species list (may be numpy array)
                species_list = TeamExporter._convert_to_serializable(team_species)

                # Create filename
                species_str = "_".join(species_list[:3])
                filename = f"teams/team_{team_hash}_{species_str}.json"

                # Create team JSON
                team_export = {
                    'team_hash': team_hash,
                    'species': species_list,
                    'team_data': team_data,
                    'stats': {
                        'win_rate': float(row.get('win_rate', 0)),
                        'total_battles': int(row.get('total_battles', 0)),
                        'wins': int(row.get('wins', 0)),
                        'avg_turns': float(row.get('avg_turns', 0)),
                    }
                }

                # Convert entire structure to ensure all numpy types are handled
                team_export = TeamExporter._convert_to_serializable(team_export)

                zf.writestr(filename, json.dumps(team_export, indent=2))

            # Add summary CSV
            if include_csv:
                csv_data = teams_df.to_csv(index=False)
                zf.writestr('teams_summary.csv', csv_data)

        return str(output_path)

    @staticmethod
    def export_to_csv(
        df: pd.DataFrame,
        output_file: str,
        columns: Optional[List[str]] = None
    ) -> str:
        """
        Export DataFrame to CSV.

        Args:
            df: DataFrame to export
            output_file: Path to output CSV file
            columns: Optional list of columns to export (all if None)

        Returns:
            Path to created CSV file
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if columns:
            df = df[columns]

        df.to_csv(output_file, index=False)

        return str(output_path)

    @staticmethod
    def export_teams_to_showdown_format(
        teams_df: pd.DataFrame,
        output_dir: str,
        battle_format: str = "gen1ou",
        include_stats: bool = True
    ) -> List[str]:
        """
        Export teams to Pokemon Showdown format (.{format}_team files).

        Args:
            teams_df: DataFrame with team data
            output_dir: Directory to save team files
            battle_format: Battle format (e.g., "gen1ou")
            include_stats: Include performance stats as comments

        Returns:
            List of created file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        created_files = []
        filename_counts = {}  # Track duplicate filenames

        for idx, row in teams_df.iterrows():
            team_hash = row['team_hash']
            team_species = row['team_species']
            team_json = row.get('player_team_json', '[]')

            # Parse team JSON
            try:
                team_data = json.loads(team_json)
            except:
                team_data = []

            # Convert species list (may be numpy array)
            species_list = TeamExporter._convert_to_serializable(team_species)

            # Create base filename
            species_str = "_".join(species_list[:3])
            base_filename = f"team_{team_hash}_{species_str}"

            # Handle duplicates by adding a counter
            if base_filename in filename_counts:
                filename_counts[base_filename] += 1
                filename = f"{base_filename}_v{filename_counts[base_filename]}.{battle_format}_team"
            else:
                filename_counts[base_filename] = 1
                filename = f"{base_filename}.{battle_format}_team"

            filepath = output_path / filename

            # Convert to Showdown format
            showdown_text = TeamExporter._convert_to_showdown_format(
                team_data=team_data,
                species_list=species_list,
                team_hash=team_hash,
                row=row,
                include_stats=include_stats
            )

            # Write file
            with open(filepath, 'w') as f:
                f.write(showdown_text)

            created_files.append(str(filepath))

        return created_files

    @staticmethod
    def _convert_to_showdown_format(
        team_data: List[Dict],
        species_list: List[str],
        team_hash: str,
        row: pd.Series,
        include_stats: bool = True
    ) -> str:
        """
        Convert team data to Showdown format string.

        Args:
            team_data: List of Pokemon dicts from parsed replay
            species_list: List of species names
            team_hash: Team hash for identification
            row: DataFrame row with stats
            include_stats: Include performance stats as comments

        Returns:
            Showdown format team string
        """
        lines = []

        # Note: We DON'T add comment headers as they can confuse poke-env's parser
        # Team metadata is available in the lead_variations_report.txt and teams_summary.csv

        # Track which species we've already added (case-insensitive)
        added_species = set()
        pokemon_count = 0

        # Convert each Pokemon from team_data
        for pokemon in team_data:
            if pokemon_count >= 6:
                break

            species = pokemon.get('species') or pokemon.get('name', 'Unknown')
            species_lower = species.lower()

            # Skip if already added or unknown
            if species_lower in added_species or species == 'Unknown':
                continue

            added_species.add(species_lower)
            pokemon_count += 1

            # Pokemon name
            lines.append(species)

            # Ability (if available and not "noability")
            # Skip abilities for Gen 1 (they don't exist in Gen 1)
            ability = pokemon.get('ability', '')
            if ability and ability.lower() not in ['noability', 'none', '']:
                lines.append(f"Ability: {ability}")

            # Moves (if available)
            moves = pokemon.get('moves', [])
            if isinstance(moves, list):
                for move in moves:
                    if isinstance(move, dict):
                        move_name = move.get('id') or move.get('name', '')
                    else:
                        move_name = str(move)
                    if move_name:
                        lines.append(f"- {move_name}")

            # Empty line between Pokemon
            lines.append("")

        # If team_data is incomplete, fill in missing Pokemon from species_list
        # (up to 6 total Pokemon)
        for species in species_list:
            if pokemon_count >= 6:
                break

            species_lower = species.lower()
            if species_lower not in added_species and species != "Unknown":
                added_species.add(species_lower)
                pokemon_count += 1
                lines.append(species)
                lines.append("")

        return "\n".join(lines)

    @staticmethod
    def format_team_list(teams_df: pd.DataFrame) -> str:
        """
        Format teams as readable text list.

        Args:
            teams_df: DataFrame with team data

        Returns:
            Formatted string representation
        """
        lines = []
        lines.append("=" * 80)
        lines.append("TEAM PERFORMANCE SUMMARY")
        lines.append("=" * 80)
        lines.append("")

        for idx, row in teams_df.iterrows():
            team_species = row['team_species']
            win_rate = row.get('win_rate', 0)
            total_battles = row.get('total_battles', 0)
            wins = row.get('wins', 0)
            avg_turns = row.get('avg_turns', 0)

            lines.append(f"Team {idx + 1}: {', '.join(team_species)}")
            lines.append(f"  Win Rate: {win_rate:.1%} ({wins}/{total_battles} battles)")
            lines.append(f"  Avg Turns: {avg_turns:.1f}")
            lines.append("")

        return "\n".join(lines)
