"""
Trajectory Parser

Extracts battle metadata and team information from .json.lz4 trajectory files.
"""

import json
import lz4.frame
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


@dataclass
class BattleRecord:
    """Structured battle record extracted from trajectory."""

    # Battle identifiers
    battle_id: str
    filename: str

    # Players
    player_name: str
    opponent_name: str
    result: str  # 'WIN' or 'LOSS'

    # Battle stats
    num_turns: int
    date: str  # ISO format
    rating: int

    # Team composition
    player_lead: str
    opponent_lead: str
    player_team_species: List[str]  # Sorted list
    opponent_team_species: List[str]  # Sorted list
    player_team_hash: str
    opponent_team_hash: str

    # Full team data (JSON strings for export)
    player_team_json: str
    opponent_team_json: str


class TrajectoryParser:
    """Parse trajectory files and extract battle records."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def parse_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Parse metadata from filename.

        Format: {battle_id}_{rating}_{player}_vs_{opponent}_{date}_{result}.json.lz4
        """
        try:
            name_without_ext = filename.replace('.json.lz4', '').replace('.json', '')
            parts = name_without_ext.split('_')

            if len(parts) < 7:
                return None

            battle_id = parts[0]
            rating_str = parts[1]
            player_name = parts[2]
            # parts[3] is 'vs'
            opponent_name = parts[4]
            date_str = parts[5]
            result = parts[6]

            # Parse rating
            try:
                rating = int(rating_str)
            except ValueError:
                rating = 1000  # Unrated

            # Parse date
            try:
                date = datetime.strptime(date_str, "%m-%d-%Y")
            except ValueError:
                try:
                    date = datetime.strptime(date_str, "%m-%d-%Y-%H:%M:%S")
                except ValueError:
                    date = datetime(2000, 1, 1)  # Default fallback

            return {
                'battle_id': battle_id,
                'rating': rating,
                'player_name': player_name,
                'opponent_name': opponent_name,
                'date': date.isoformat(),
                'result': result,
            }
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not parse filename {filename}: {e}")
            return None

    def extract_team_from_state(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract team composition from first state.

        Team = active pokemon + available switches
        """
        team = []

        # Get active pokemon
        active = state.get('player_active_pokemon')
        if active and isinstance(active, dict):
            team.append(active)

        # Get bench pokemon
        switches = state.get('available_switches', [])
        if isinstance(switches, list):
            team.extend(switches)

        return team

    def extract_opponent_team_from_state(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract opponent team from first state."""
        team = []

        # Get opponent active
        opp_active = state.get('opponent_active_pokemon')
        if opp_active and isinstance(opp_active, dict):
            team.append(opp_active)

        # Note: Opponent's bench may not be fully visible at state 0
        # We'll extract what we can

        return team

    def normalize_species(self, species: str) -> str:
        """Normalize Pokemon species name."""
        if not species:
            return "Unknown"

        # Remove forms, genders, etc. for consistency
        species = species.strip()

        # Remove common suffixes
        for suffix in ['-Mega', '-Alola', '-Galar']:
            if species.endswith(suffix):
                species = species[:-len(suffix)]

        return species

    def get_species_from_pokemon(self, pokemon: Dict[str, Any]) -> str:
        """Extract species name from pokemon dict."""
        species = pokemon.get('species') or pokemon.get('name') or "Unknown"
        return self.normalize_species(species)

    def compute_team_hash(self, species_list: List[str]) -> str:
        """Generate deterministic hash for team composition."""
        if not species_list:
            return "empty"

        # Sort to make hash order-independent
        sorted_species = sorted(species_list)
        team_str = ','.join(sorted_species)
        return hashlib.md5(team_str.encode()).hexdigest()[:16]

    def parse_trajectory_file(self, filepath: Path) -> Optional[BattleRecord]:
        """
        Parse a single trajectory file.

        Returns BattleRecord or None if parsing fails.
        """
        try:
            # Load compressed JSON
            with lz4.frame.open(filepath, 'rb') as f:
                data = json.loads(f.read().decode('utf-8'))

            # Parse filename metadata
            filename = filepath.name
            file_meta = self.parse_filename(filename)
            if not file_meta:
                return None

            # Extract states
            states = data.get('states', [])
            if not states:
                return None

            # Get num turns
            num_turns = len(states)

            # Extract teams from first state
            first_state = states[0]
            player_team = self.extract_team_from_state(first_state)
            opponent_team = self.extract_opponent_team_from_state(first_state)

            # For opponent team, scan all states to build complete team
            # (opponent pokemon are revealed gradually)
            opponent_species_set = set()
            for state in states:
                opp_active = state.get('opponent_active_pokemon')
                if opp_active and isinstance(opp_active, dict):
                    species = self.get_species_from_pokemon(opp_active)
                    if species != "Unknown":
                        opponent_species_set.add(species)

            # Get player team species
            player_species = [self.get_species_from_pokemon(p) for p in player_team]
            player_species = [s for s in player_species if s != "Unknown"]

            # Get opponent team species (from scan)
            opponent_species = sorted(list(opponent_species_set))

            # Get leads
            player_lead = player_species[0] if player_species else "Unknown"
            opponent_lead = opponent_species[0] if opponent_species else "Unknown"

            # Compute team hashes
            player_team_hash = self.compute_team_hash(player_species)
            opponent_team_hash = self.compute_team_hash(opponent_species)

            # Store full team JSON for export
            player_team_json = json.dumps(player_team)
            opponent_team_json = json.dumps(list(opponent_species_set))  # Best we can do

            return BattleRecord(
                battle_id=file_meta['battle_id'],
                filename=filename,
                player_name=file_meta['player_name'],
                opponent_name=file_meta['opponent_name'],
                result=file_meta['result'],
                num_turns=num_turns,
                date=file_meta['date'],
                rating=file_meta['rating'],
                player_lead=player_lead,
                opponent_lead=opponent_lead,
                player_team_species=sorted(player_species),
                opponent_team_species=opponent_species,
                player_team_hash=player_team_hash,
                opponent_team_hash=opponent_team_hash,
                player_team_json=player_team_json,
                opponent_team_json=opponent_team_json,
            )

        except Exception as e:
            if self.verbose:
                print(f"Error parsing {filepath}: {e}")
            return None

    def parse_directory(
        self,
        data_dir: str,
        max_workers: int = 8,
        limit: Optional[int] = None
    ) -> List[BattleRecord]:
        """
        Parse all trajectory files in directory (and subdirectories).

        Args:
            data_dir: Root directory containing .json.lz4 files
            max_workers: Number of parallel workers
            limit: Optional limit on number of files to parse (for testing)

        Returns:
            List of BattleRecord objects
        """
        data_path = Path(data_dir)

        # Find all .json.lz4 files recursively
        trajectory_files = list(data_path.rglob('*.json.lz4'))

        if not trajectory_files:
            print(f"No .json.lz4 files found in {data_dir}")
            return []

        if limit:
            trajectory_files = trajectory_files[:limit]

        print(f"Found {len(trajectory_files)} trajectory files")
        print(f"Parsing with {max_workers} workers...")

        records = []

        # Parse in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.parse_trajectory_file, f): f
                for f in trajectory_files
            }

            # Collect results with progress bar
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Parsing trajectories",
                disable=not self.verbose
            ):
                try:
                    record = future.result()
                    if record:
                        records.append(record)
                except Exception as e:
                    if self.verbose:
                        filepath = futures[future]
                        print(f"Failed to parse {filepath}: {e}")

        print(f"Successfully parsed {len(records)} battles")
        return records
