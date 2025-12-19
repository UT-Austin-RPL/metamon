# Team Export Guide

This guide explains how to export and use teams from the Team Analytics dashboard.

## Features

### 1. Export Formats

Two export formats are now available:

#### Showdown Format (Recommended)
- Exports teams as `.{format}_team` files (e.g., `.gen1ou_team`)
- **Compatible with metamon's evaluation system** - can be used directly with `--team_set` parameter
- Includes performance stats as comments in each file
- Includes a `lead_variations_report.txt` showing which leads were used with each team
- Includes a `teams_summary.csv` with all team stats

#### JSON Format
- Exports teams as individual JSON files
- Includes full team data and performance stats
- Useful for custom processing or analysis

### 2. Lead Variation Analytics

The export now includes lead variation data, showing:
- Which leads were used with each team
- How often each lead was used
- Win rate for each lead variation

This helps you understand:
- Which Pokemon works best as a lead for each team
- How lead choice affects win rate
- Lead usage patterns in your dataset

### 3. Visualization Tool

A new visualization script (`visualize_export.py`) lets you quickly preview exported teams.

## Usage

### Exporting Teams from Dashboard

1. Launch the dashboard:
   ```bash
   python scripts/team_analytics_cli.py --data_dir ~/trajectories --launch
   ```

2. Navigate to the "Export Teams" tab

3. Set your filters:
   - **Species filters**: Require or exclude specific Pokemon
   - **Exclude mirrors**: Remove mirror matches from stats
   - **Min win rate**: Only export teams above this win rate
   - **Min battles**: Only export teams with enough data
   - **Max teams**: Limit the number of teams exported

4. Choose export format:
   - **Showdown Format** (recommended for metamon evaluation)
   - **JSON Format** (for custom analysis)

5. For Showdown format, specify the battle format (e.g., `gen1ou`, `gen2ou`)

6. Click "Export Teams" and download the ZIP file

### Visualizing Exported Teams

```bash
# View first 10 teams
python scripts/team_analytics/visualize_export.py teams_export.zip

# View more teams
python scripts/team_analytics/visualize_export.py teams_export.zip --max-teams 20

# Hide lead variations report
python scripts/team_analytics/visualize_export.py teams_export.zip --no-lead-report
```

### Using Exported Teams in Metamon

#### Option 1: Use directly in evaluation

If you exported in Showdown format:

```bash
# 1. Extract the ZIP file
unzip teams_export.zip -d ~/my_teams

# 2. Move teams to metamon cache
mkdir -p $METAMON_CACHE_DIR/teams/my_exported_teams
mv ~/my_teams/*.gen1ou_team $METAMON_CACHE_DIR/teams/my_exported_teams/

# 3. Use in evaluation
python -m metamon.rl.evaluate \
    --agent_ckpt_name SyntheticRLV2 \
    --battle_format gen1ou \
    --team_set my_exported_teams \
    --opponent_team_set modern_replays_v2 \
    --num_battles 100
```

#### Option 2: Compare with existing teams

```bash
# Evaluate your exported teams against smogon_pass2
python -m metamon.rl.evaluate \
    --agent_ckpt_name SyntheticRLV2 \
    --battle_format gen1ou \
    --team_set my_exported_teams \
    --opponent_team_set smogon_pass2 \
    --num_battles 100
```

## Export Contents

### Showdown Format Export

The ZIP contains:
- `*.gen1ou_team` - Individual team files in Showdown format
- `lead_variations_report.txt` - Lead usage and performance analysis
- `teams_summary.csv` - Summary statistics for all teams

Each `.gen1ou_team` file contains:
```
# Team: abc123def456
# Win Rate: 68.5%
# Battles: 50
# Avg Turns: 42.3

Alakazam
Ability: Synchronize
- psychic
- recover
- seismictoss
- thunderwave

Tauros
...
```

### Lead Variations Report

Shows lead choices and their performance:
```
================================================================================
LEAD VARIATIONS REPORT
================================================================================

This report shows which leads were used with each team and their performance.

--------------------------------------------------------------------------------
Team: abc123def456
Species: Alakazam, Chansey, Exeggutor, Snorlax, Starmie, Tauros

Lead Variations:
  - Alakazam        Used:  15x  Win Rate:  73.3%  (11 wins)
  - Exeggutor       Used:   8x  Win Rate:  62.5%  (5 wins)
  - Tauros          Used:   5x  Win Rate:  80.0%  (4 wins)
```

## Tips

### Getting High-Quality Teams

1. **Filter by win rate**: Use min win rate of 55-60% to get strong teams
2. **Require minimum battles**: Use 10+ battles to ensure statistical significance
3. **Exclude mirrors**: Always exclude mirror matches to avoid 50% bias
4. **Check lead variations**: Teams with diverse successful leads are more robust

### Understanding Lead Variations

- **High usage, high win rate**: This is the "standard" lead for this team
- **Low usage, high win rate**: Potential surprise lead that works well
- **High usage, low win rate**: Common but potentially suboptimal lead choice
- **Diverse leads with similar win rates**: Flexible team that works with multiple leads

### Combining with Existing Teams

```bash
# Merge with existing team set
cp $METAMON_CACHE_DIR/teams/smogon_pass2/gen1ou/*.gen1ou_team \
   $METAMON_CACHE_DIR/teams/my_exported_teams/

# Now evaluate with combined set
python -m metamon.rl.evaluate \
    --team_set my_exported_teams \
    ...
```

## Troubleshooting

### "No teams match the specified filters"

- Lower the minimum win rate
- Lower the minimum battles requirement
- Remove species filters or make them less restrictive

### Teams missing moves/abilities

This is normal for parsed replay data. The parser extracts what's visible in the battle trajectory. Missing information will appear as:
```
Snorlax

```

You can manually add moves based on:
- Standard smogon movesets for the format
- The lead variations report (shows which leads work)
- Common competitive movesets

### Export fails with JSON serialization error

This should be fixed now. If you still see this error:
1. Ensure you're using the updated code
2. Check that numpy and pandas are properly installed
3. Try the JSON format instead of Showdown format

## Examples

### Export top 50 teams with 60%+ win rate

Filters:
- Min win rate: 60%
- Min battles: 10
- Max teams: 50
- Format: Showdown Format
- Battle format: gen1ou

### Export teams with Snorlax and Chansey

Filters:
- Team must have: Snorlax, Chansey
- Min win rate: 50%
- Min battles: 5
- Max teams: 100

### Export teams that DON'T use Alakazam

Filters:
- Team must NOT have: Alakazam
- Min win rate: 55%
- Min battles: 10
- Max teams: 50

## Advanced: Programmatic Export

You can also export teams programmatically:

```python
from scripts.team_analytics.database import TeamAnalyticsDB
from scripts.team_analytics.analytics import AnalyticsEngine
from scripts.team_analytics.export import TeamExporter

# Load database
db = TeamAnalyticsDB("~/team_analytics.duckdb")
analytics = AnalyticsEngine(db)
exporter = TeamExporter()

# Get teams
teams_df = analytics.get_teams_by_filter(
    min_win_rate=0.60,
    min_battles=10,
    limit=50
)

# Export to Showdown format
files = exporter.export_teams_to_showdown_format(
    teams_df,
    output_dir="~/my_teams",
    battle_format="gen1ou",
    include_stats=True
)

print(f"Exported {len(files)} teams")
```

## See Also

- [Team Analytics README](README.md) - Main analytics documentation
- [Team Analytics CLI](../team_analytics_cli.py) - Command-line interface
- [Metamon Evaluation Guide](../../metamon/rl/README.md) - How to evaluate agents
