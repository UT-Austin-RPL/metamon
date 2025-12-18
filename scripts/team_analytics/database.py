"""
DuckDB Database Interface

Handles schema creation, bulk loading, and indexing for team analytics.
"""

import duckdb
import pandas as pd
from pathlib import Path
from typing import List, Optional
from dataclasses import asdict
from .parser import BattleRecord


class TeamAnalyticsDB:
    """DuckDB database for team analytics."""

    def __init__(self, db_path: str = ":memory:", verbose: bool = True):
        """
        Initialize database connection.

        Args:
            db_path: Path to database file, or ":memory:" for in-memory DB
            verbose: Print progress messages
        """
        self.db_path = db_path
        self.verbose = verbose
        self.conn = duckdb.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        """Create database schema."""
        if self.verbose:
            print("Initializing database schema...")

        # Create battles table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS battles (
                -- Battle identifiers
                battle_id VARCHAR,
                filename VARCHAR,

                -- Players
                player_name VARCHAR,
                opponent_name VARCHAR,
                result VARCHAR,  -- 'WIN' or 'LOSS'

                -- Battle stats
                num_turns INTEGER,
                date DATE,
                rating INTEGER,

                -- Team composition
                player_lead VARCHAR,
                opponent_lead VARCHAR,
                player_team_species VARCHAR[],  -- Array of species
                opponent_team_species VARCHAR[],
                player_team_hash VARCHAR,
                opponent_team_hash VARCHAR,

                -- Full team data (JSON strings)
                player_team_json VARCHAR,
                opponent_team_json VARCHAR
            )
        """)

        # Create indexes for common queries
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_player_lead
            ON battles(player_lead)
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_result
            ON battles(result)
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_player_team_hash
            ON battles(player_team_hash)
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_opponent_team_hash
            ON battles(opponent_team_hash)
        """)

        if self.verbose:
            print("Schema initialized")

    def load_records(self, records: List[BattleRecord], clear_existing: bool = True):
        """
        Bulk load battle records into database.

        Args:
            records: List of BattleRecord objects
            clear_existing: If True, clear existing data first
        """
        if not records:
            print("No records to load")
            return

        if clear_existing:
            if self.verbose:
                print("Clearing existing data...")
            self.conn.execute("DELETE FROM battles")

        if self.verbose:
            print(f"Loading {len(records)} records into database...")

        # Convert records to pandas DataFrame for DuckDB
        records_dict = [asdict(r) for r in records]
        df = pd.DataFrame(records_dict)

        # Bulk insert using DuckDB's efficient insert
        self.conn.execute("""
            INSERT INTO battles
            SELECT * FROM df
        """)

        if self.verbose:
            count = self.conn.execute("SELECT COUNT(*) FROM battles").fetchone()[0]
            print(f"Database loaded: {count} battles")

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get raw DuckDB connection for custom queries."""
        return self.conn

    def close(self):
        """Close database connection."""
        self.conn.close()

    def get_all_species(self) -> List[str]:
        """Get sorted list of all unique species in database."""
        result = self.conn.execute("""
            SELECT DISTINCT unnest(player_team_species) as species
            FROM battles
            UNION
            SELECT DISTINCT unnest(opponent_team_species) as species
            FROM battles
            ORDER BY species
        """).fetchall()

        return [row[0] for row in result if row[0] != "Unknown"]

    def get_database_stats(self) -> dict:
        """Get overall database statistics."""
        stats = {}

        # Total battles
        stats['total_battles'] = self.conn.execute(
            "SELECT COUNT(*) FROM battles"
        ).fetchone()[0]

        # Unique teams
        stats['unique_player_teams'] = self.conn.execute(
            "SELECT COUNT(DISTINCT player_team_hash) FROM battles"
        ).fetchone()[0]

        # Date range
        date_range = self.conn.execute(
            "SELECT MIN(date), MAX(date) FROM battles"
        ).fetchone()
        stats['date_range'] = {
            'min': str(date_range[0]) if date_range[0] else None,
            'max': str(date_range[1]) if date_range[1] else None,
        }

        # Most common species
        top_species = self.conn.execute("""
            SELECT species, COUNT(*) as count
            FROM (
                SELECT unnest(player_team_species) as species
                FROM battles
            )
            WHERE species != 'Unknown'
            GROUP BY species
            ORDER BY count DESC
            LIMIT 10
        """).fetchall()

        stats['top_species'] = [
            {'species': row[0], 'count': row[1]}
            for row in top_species
        ]

        # Most common leads
        top_leads = self.conn.execute("""
            SELECT player_lead, COUNT(*) as count
            FROM battles
            WHERE player_lead != 'Unknown'
            GROUP BY player_lead
            ORDER BY count DESC
            LIMIT 10
        """).fetchall()

        stats['top_leads'] = [
            {'lead': row[0], 'count': row[1]}
            for row in top_leads
        ]

        return stats

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.close()
