#!/usr/bin/env python3
"""
Team Analytics CLI

Command-line interface for analyzing Pokemon battle trajectories.

Usage:
    # Parse and launch interactive dashboard
    python scripts/team_analytics_cli.py \
        --data_dir ~/metamon/trajectories/super_dataset_loop3 \
        --launch

    # Parse only (save database)
    python scripts/team_analytics_cli.py \
        --data_dir ~/metamon/trajectories/super_dataset_loop3 \
        --db_path ~/team_analytics.duckdb

    # Load existing database and launch
    python scripts/team_analytics_cli.py \
        --db_path ~/team_analytics.duckdb \
        --launch
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add metamon to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.team_analytics.parser import TrajectoryParser
from scripts.team_analytics.database import TeamAnalyticsDB
from scripts.team_analytics.gradio_app import TeamAnalyticsApp


def parse_and_load(
    data_dir: str,
    db_path: str = ":memory:",
    max_workers: int = 8,
    limit: Optional[int] = None,
    verbose: bool = True
) -> TeamAnalyticsDB:
    """
    Parse trajectories and load into database.

    Args:
        data_dir: Directory containing .json.lz4 trajectory files
        db_path: Path to DuckDB database file (or :memory:)
        max_workers: Number of parallel workers for parsing
        limit: Optional limit on files to parse (for testing)
        verbose: Print progress messages

    Returns:
        TeamAnalyticsDB instance
    """
    print(f"\n{'='*80}")
    print("TEAM ANALYTICS - PARSING AND LOADING")
    print(f"{'='*80}\n")

    # Step 1: Parse trajectories
    print("Step 1: Parsing trajectory files...")
    parser = TrajectoryParser(verbose=verbose)
    records = parser.parse_directory(
        data_dir=data_dir,
        max_workers=max_workers,
        limit=limit
    )

    if not records:
        print("ERROR: No records parsed. Check data directory.")
        sys.exit(1)

    print(f"\n✓ Parsed {len(records)} battle records\n")

    # Step 2: Load into database
    print("Step 2: Loading into DuckDB...")
    db = TeamAnalyticsDB(db_path=db_path, verbose=verbose)
    db.load_records(records, clear_existing=True)

    print(f"\n✓ Database ready at: {db_path}\n")

    # Step 3: Show statistics
    print("Step 3: Database Statistics")
    print("-" * 80)
    stats = db.get_database_stats()
    print(f"Total Battles:     {stats['total_battles']:,}")
    print(f"Unique Teams:      {stats['unique_player_teams']:,}")
    print(f"Date Range:        {stats['date_range']['min']} to {stats['date_range']['max']}")
    print(f"\nTop 5 Species:")
    for species_data in stats['top_species'][:5]:
        print(f"  - {species_data['species']:<20} {species_data['count']:>6,} appearances")
    print(f"\nTop 5 Leads:")
    for lead_data in stats['top_leads'][:5]:
        print(f"  - {lead_data['lead']:<20} {lead_data['count']:>6,} games")
    print("-" * 80)

    return db


def load_existing_database(db_path: str, verbose: bool = True) -> TeamAnalyticsDB:
    """
    Load existing DuckDB database.

    Args:
        db_path: Path to existing database file
        verbose: Print progress messages

    Returns:
        TeamAnalyticsDB instance
    """
    if not Path(db_path).exists():
        print(f"ERROR: Database file not found: {db_path}")
        sys.exit(1)

    print(f"Loading existing database: {db_path}")
    db = TeamAnalyticsDB(db_path=db_path, verbose=verbose)

    # Show stats
    stats = db.get_database_stats()
    print(f"Loaded {stats['total_battles']:,} battles from database")

    return db


def main():
    parser = argparse.ArgumentParser(
        description="Team Analytics Tool for Metamon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse new data and launch dashboard
  python scripts/team_analytics_cli.py --data_dir ~/trajectories --launch

  # Parse and save database (no GUI)
  python scripts/team_analytics_cli.py --data_dir ~/trajectories --db_path ~/teams.duckdb

  # Load existing database and launch dashboard
  python scripts/team_analytics_cli.py --db_path ~/teams.duckdb --launch

  # Quick test on 1000 battles
  python scripts/team_analytics_cli.py --data_dir ~/trajectories --limit 1000 --launch
        """
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        help='Directory containing .json.lz4 trajectory files'
    )

    parser.add_argument(
        '--db_path',
        type=str,
        default=':memory:',
        help='Path to DuckDB database file (default: in-memory)'
    )

    parser.add_argument(
        '--max_workers',
        type=int,
        default=8,
        help='Number of parallel workers for parsing (default: 8)'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of files to parse (for testing)'
    )

    parser.add_argument(
        '--launch',
        action='store_true',
        help='Launch Gradio web interface after loading'
    )

    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Host for Gradio server (default: 127.0.0.1)'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Port for Gradio server (default: 7860)'
    )

    parser.add_argument(
        '--share',
        action='store_true',
        help='Create public Gradio share link'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )

    args = parser.parse_args()

    verbose = not args.quiet

    # Validate arguments
    if not args.data_dir and args.db_path == ':memory:':
        parser.error("Must specify either --data_dir (to parse) or --db_path (to load existing)")

    # Load or create database
    if args.data_dir:
        # Parse new data
        db = parse_and_load(
            data_dir=args.data_dir,
            db_path=args.db_path,
            max_workers=args.max_workers,
            limit=args.limit,
            verbose=verbose
        )
    else:
        # Load existing database
        db = load_existing_database(args.db_path, verbose=verbose)

    # Launch Gradio app if requested
    if args.launch:
        print(f"\n{'='*80}")
        print("LAUNCHING GRADIO DASHBOARD")
        print(f"{'='*80}\n")

        app = TeamAnalyticsApp(db)
        app.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share
        )
    else:
        print("\nDatabase ready. Use --launch to start the web interface.")
        db.close()


if __name__ == '__main__':
    main()
