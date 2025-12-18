# Format Filtering Troubleshooting

> **Category**: troubleshooting
> **Last Updated**: 2025-12-10
> **Author**: Claude

## Objective

Fix data loading issues where training loads wrong format data (Gen3/Gen4 instead of Gen1 OU) or fails silently, causing training on incorrect data distribution.

## When to Use This Skill

- [ ] When training behaves unexpectedly (wrong teams, moves not in Gen1)
- [ ] When user asks about format filtering or `--formats gen1ou`
- [ ] When troubleshooting data loading errors
- [ ] Before starting any Gen1 OU training run

## The Problem

**Critical requirement**: Data must be in format-specific subdirectories AND `--formats gen1ou` flag must be used.

**Without proper filtering**:
- Training loads all formats (Gen1-9) from parsed-replays
- Model trains on wrong generation data
- No error thrown - fails silently
- Performance degrades

## Solution: Always Use Format Filtering

### Training Command

```bash
# ALWAYS include --formats gen1ou
python -m metamon.rl.finetune_from_hf \
    --custom_replay_dir ~/metamon/trajectories/gen1_loop/gen0_filtered \
    --parsed_replay_dir ~/metamon_cache/parsed-replays \
    --formats gen1ou \  # ← CRITICAL
    ...
```

### Directory Structure Required

```
custom_replay_dir/
└── gen1ou/
    ├── battle_001.json.lz4
    ├── battle_002.json.lz4
    └── ...

parsed_replay_dir/
└── gen1ou/
    ├── replay_001.json.lz4
    ├── replay_002.json.lz4
    └── ...
```

**If data is NOT in gen1ou/ subdirectory**:
```bash
# Create subdirectory and move files
mkdir -p ~/metamon/trajectories/gen1_loop/gen0_filtered/gen1ou
mv ~/metamon/trajectories/gen1_loop/gen0_filtered/*.json.lz4 \
   ~/metamon/trajectories/gen1_loop/gen0_filtered/gen1ou/
```

## Common Errors

### Error: Training loads wrong format data

**Symptoms**: Training runs but model behaves oddly, unexpected moves/abilities

**Cause**: Forgot `--formats gen1ou` flag

**Solution**:
```bash
# Add to every training command
--formats gen1ou
```

### Error: No data loaded / FileNotFoundError

**Symptoms**: Training fails immediately, "No trajectories found"

**Cause**: Data not in format subdirectory

**Solution**:
```bash
# Check directory structure
ls ~/metamon/trajectories/gen1_loop/gen0_filtered/gen1ou/

# Should see .json.lz4 files
# If not, create subdirectory and move files
```

### Error: Mixed format data in training

**Symptoms**: Dataset stats show multiple formats, wrong team compositions

**Cause**: Multiple format subdirectories exist, flag not used

**Solution**:
```bash
# Remove other format dirs if only want Gen1
cd ~/metamon/trajectories/gen1_loop/gen0_filtered
rm -rf gen3ou gen4ou gen9ou  # Keep only gen1ou/

# Always use --formats flag
--formats gen1ou
```

## Validation Checklist

Before starting training, verify:

```bash
# 1. Check data directory structure
ls ~/metamon/trajectories/your_dir/gen1ou/*.json.lz4 | head -5

# 2. Count battles
ls ~/metamon/trajectories/your_dir/gen1ou/*.json.lz4 | wc -l

# 3. Verify command includes --formats gen1ou
grep "formats" your_training_command.sh
```

## Related Skills

- [`selfplay-loop-workflow`](./../training/selfplay-loop-workflow.md) - Self-play training workflow

## References

- [CLAUDE.md](../../CLAUDE.md) - "Critical: Format Filtering" section
- [GEN1OU_SELFPLAY_GUIDE.md](../../GEN1OU_SELFPLAY_GUIDE.md) - Format filtering warnings
