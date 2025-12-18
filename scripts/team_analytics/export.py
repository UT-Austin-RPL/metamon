"""
Team Export Utilities

Export filtered teams to various formats (JSON, CSV, ZIP).
"""

import json
import zipfile
from pathlib import Path
from typing import List, Optional
import pandas as pd


class TeamExporter:
    """Export team data to various formats."""

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

            # Create filename
            species_str = "_".join(team_species[:3])  # First 3 species
            filename = f"{filename_prefix}_{team_hash}_{species_str}.json"
            filepath = output_path / filename

            # Write JSON
            with open(filepath, 'w') as f:
                json.dump({
                    'team_hash': team_hash,
                    'species': team_species,
                    'team_data': team_data,
                    'stats': {
                        'win_rate': float(row.get('win_rate', 0)),
                        'total_battles': int(row.get('total_battles', 0)),
                        'wins': int(row.get('wins', 0)),
                        'avg_turns': float(row.get('avg_turns', 0)),
                    }
                }, f, indent=2)

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

                # Create filename
                species_str = "_".join(team_species[:3])
                filename = f"teams/team_{team_hash}_{species_str}.json"

                # Create team JSON
                team_export = {
                    'team_hash': team_hash,
                    'species': team_species,
                    'team_data': team_data,
                    'stats': {
                        'win_rate': float(row.get('win_rate', 0)),
                        'total_battles': int(row.get('total_battles', 0)),
                        'wins': int(row.get('wins', 0)),
                        'avg_turns': float(row.get('avg_turns', 0)),
                    }
                }

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
