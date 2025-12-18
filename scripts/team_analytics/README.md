# Team Analytics Tool

Analyze Pokemon battle trajectories to identify team performance patterns, archetypes, and optimal team compositions.

## Features

- **Team Performance Analysis** - Win rates, sample sizes, and average turn counts for each unique team
- **Archetype Queries** - Filter by species presence/absence (e.g., "teams with Zapdos but without Rhydon")
- **Lead Analysis** - Performance statistics grouped by lead Pokemon
- **Species Usage Stats** - Usage rates and win rates for individual Pokemon
- **Mirror Match Filtering** - Toggle to exclude mirror matches from calculations
- **Team Export** - Export filtered teams to ZIP archives with JSON files and CSV summaries
- **Interactive Gradio UI** - Web-based dashboard for easy exploration

## Installation

Dependencies:
```bash
source .venv/bin/activate
uv pip install duckdb gradio pandas
```

## Quick Start

### 1. Parse and Launch Dashboard

```bash
python scripts/team_analytics_cli.py \
    --data_dir ~/metamon/trajectories/super_dataset_loop3 \
    --launch
```

This will:
1. Parse all `.json.lz4` trajectory files in the directory
2. Load them into an in-memory DuckDB database
3. Launch the Gradio web interface (default: http://127.0.0.1:7860)

### 2. Parse and Save Database

```bash
python scripts/team_analytics_cli.py \
    --data_dir ~/metamon/trajectories/super_dataset_loop3 \
    --db_path ~/team_analytics.duckdb
```

This saves the database to disk for faster loading next time.

### 3. Load Existing Database

```bash
python scripts/team_analytics_cli.py \
    --db_path ~/team_analytics.duckdb \
    --launch
```

Load a previously saved database and launch the dashboard.

### 4. Quick Test (Limited Battles)

```bash
python scripts/team_analytics_cli.py \
    --data_dir ~/metamon/trajectories/super_dataset_loop3 \
    --limit 1000 \
    --launch
```

Parse only first 1000 battles for testing.

## Command-Line Arguments

```
--data_dir PATH         Directory containing .json.lz4 trajectory files
--db_path PATH          Path to DuckDB database file (default: :memory:)
--max_workers N         Number of parallel workers for parsing (default: 8)
--limit N               Limit number of files to parse (for testing)
--launch                Launch Gradio web interface after loading
--host IP               Host for Gradio server (default: 127.0.0.1)
--port PORT             Port for Gradio server (default: 7860)
--share                 Create public Gradio share link
--quiet                 Suppress progress messages
```

## Gradio Dashboard

The dashboard has 6 tabs:

### 1. Overview
- Database statistics (total battles, unique teams, date range)
- Top species and leads by usage

### 2. Team Performance
- Win rates for each unique team composition
- Configurable filters: min battles, max teams to show
- Mirror match toggle

### 3. Archetype Analysis
- Define archetypes by species presence/absence
- Example: "Teams with Zapdos vs teams with Rhydon"
- Player and opponent filters
- Shows: win rate, total battles, avg turns

### 4. Lead Analysis
- Win rates grouped by lead Pokemon
- Configurable min sample size
- Mirror match toggle

### 5. Species Usage
- Appearances, win rates, avg turns for each species
- Mirror match toggle

### 6. Export Teams
- Filter teams by archetype, win rate, sample size
- Export to ZIP (JSON files + CSV summary)
- Downloadable through browser

## Architecture

```
trajectories/*.json.lz4
  → [Parser] Parallel processing, extract metadata + teams
  → [Database] DuckDB with columnar storage, fast aggregations
  → [Analytics] Query engine with mirror match filtering
  → [Gradio UI] Interactive web dashboard
```

### Key Components

- **`parser.py`** - Parallel `.json.lz4` parsing, team extraction
- **`database.py`** - DuckDB schema, bulk loading, indexing
- **`analytics.py`** - Query functions (win rates, matchups, archetypes)
- **`export.py`** - Team export to JSON/CSV/ZIP
- **`gradio_app.py`** - Web interface
- **`team_analytics_cli.py`** - CLI entry point

## Performance

Tested on 187k battles (2.2GB compressed):

| Operation | Time |
|-----------|------|
| Parsing (8 workers) | ~1-2 min |
| Database load | ~5-10 sec |
| Simple queries (win rate by team) | < 1 sec |
| Complex queries (archetype filters) | < 5 sec |

Million-battle scale:
- Parsing: ~10-15 min
- Queries: Sub-second to few seconds (DuckDB columnar efficiency)

## Example Queries

### Teams with Zapdos but without Rhydon

**Archetype Analysis tab:**
- Team MUST have: Zapdos
- Team must NOT have: Rhydon
- Exclude mirrors: Yes

### Top Performing Tauros Leads

**Lead Analysis tab:**
- Minimum battles: 20
- Sort by win rate descending
- Filter results for "Tauros"

### Export Best Stall Teams (>60% WR)

**Export tab:**
- Team MUST have: Chansey, Snorlax
- Min win rate: 60%
- Min battles: 10
- Click "Export Teams"

## Mirror Match Filtering

**Why it matters:**

Without filtering, Rhydon archetype vs Rhydon archetype shows 50% win rate (one side always loses). This inflates sample sizes and obscures true matchup dynamics.

**How it works:**

All queries include `--exclude_mirrors` toggle. When enabled:
```sql
WHERE player_team_hash != opponent_team_hash
```

This ensures you only see non-mirror matchups where team compositions differ.

## Data Structure

Trajectory files contain:
- `states[]` - Battle state at each turn
- `actions[]` - Actions taken

Parser extracts:
- **Metadata** - From filename (battle ID, rating, date, result)
- **Player team** - Active + bench from `states[0].available_switches`
- **Opponent team** - Scanned from all states (revealed gradually)
- **Leads** - First active Pokemon
- **Team hash** - MD5 of sorted species list

## Troubleshooting

### No files found
```bash
# Check directory structure
ls ~/metamon/trajectories/super_dataset_loop3/gen1ou/*.json.lz4 | head
```

Trajectory files should be in format-specific subdirectories.

### Parsing errors
```bash
# Run with verbose output
python scripts/team_analytics_cli.py --data_dir ... --limit 100
```

Check for corrupted `.json.lz4` files or unexpected formats.

### Slow queries
```bash
# Save database to disk for faster reloading
python scripts/team_analytics_cli.py --data_dir ... --db_path ~/teams.duckdb

# Next time, just load the database
python scripts/team_analytics_cli.py --db_path ~/teams.duckdb --launch
```

### Out of memory
```bash
# Reduce max_workers
python scripts/team_analytics_cli.py --data_dir ... --max_workers 4

# Or use disk-based database instead of :memory:
python scripts/team_analytics_cli.py --data_dir ... --db_path ~/teams.duckdb
```

## Future Enhancements

Potential features:
- [ ] Matchup matrix heatmap visualization
- [ ] Lead matchup matrices (Tauros vs Starmie, etc.)
- [ ] Incremental updates (append new battles without full reparse)
- [ ] Player-specific performance tracking
- [ ] Turn-by-turn analysis (not just final results)
- [ ] Team clustering (find similar team archetypes)
- [ ] Export to Showdown team format

## Related Files

- [CLAUDE.md](../../CLAUDE.md) - Project documentation
- [scripts/self_play_tournament.py](../self_play_tournament.py) - Round-robin tournament evaluation
- [scripts/calculate_elo.py](../calculate_elo.py) - ELO rating calculation
- [scripts/filter_selfplay_data.py](../filter_selfplay_data.py) - Data quality filtering
