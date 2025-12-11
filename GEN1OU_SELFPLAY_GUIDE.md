# Gen1 OU Vanilla Self-Play with Dynamic Damping

## Quick Start

This guide shows how to run vanilla self-play experiments for Gen1 OU using **existing production-ready scripts** with the newly added **dynamic damping** functionality.

## Prerequisites

```bash
cd /home/eddie/repos/metamon
source .venv/bin/activate
export METAMON_CACHE_DIR=/home/eddie/metamon_cache

# Ensure local Pokémon Showdown server is running
cd server/pokemon-showdown && node pokemon-showdown start --no-security
```

## Complete Workflow: Baseline vs Dynamic Damping

### Baseline Experiment (No Dynamic Damping)

#### Generation 0: Collect Initial Data
```bash
# Collect 50k self-play battles from SynRL-V2
python scripts/generate_selfplay_data.py \
    --model SyntheticRLV2 \
    --num_battles 50000 \
    --battle_format gen1ou \
    --team_set modern_replays_v2 \
    --output_dir ~/gen1_baseline/gen0 \
    --parallel_instances 4

# Filter for quality (removes bad battles, balances win/loss)
python scripts/filter_selfplay_data.py \
    --input_dir ~/gen1_baseline/gen0/trajectories \
    --output_dir ~/gen1_baseline/gen0_filtered \
    --max_invalid_rate 0.05 \
    --balance_outcomes
```

#### Generation 1: Train on Self-Play Data
```bash
python -m metamon.rl.finetune_from_hf \
    --finetune_from_model SyntheticRLV2 \
    --run_name Baseline_Gen1 \
    --custom_replay_dir ~/gen1_baseline/gen0_filtered \
    --formats gen1ou \
    --train_gin_config vanilla_selfplay_baseline.gin \
    --epochs 20 \
    --save_dir ~/gen1_checkpoints/baseline_gen1 \
    --battle_format gen1ou \
    --team_set modern_replays_v2 \
    --obs_space TeamPreviewObservationSpace \
    --reward_function DefaultShapedReward \
    --action_space DefaultActionSpace \
    --log
```

#### Evaluate Generation 1
```bash
python scripts/self_play_tournament.py \
    --models Baseline_Gen1 SyntheticRLV2 \
    --battle_format gen1ou \
    --team_set modern_replays_v2 \
    --num_battles 100 \
    --output_dir ~/gen1_baseline/gen1_eval
```

#### Generation 2+: Iterate
```bash
# Collect data from Gen 1
python scripts/generate_selfplay_data.py \
    --model Baseline_Gen1 \
    --checkpoint_path ~/gen1_checkpoints/baseline_gen1 \
    --num_battles 50000 \
    --battle_format gen1ou \
    --output_dir ~/gen1_baseline/gen1

# Filter, train, evaluate (repeat workflow)
```

---

### Dynamic Damping Experiment

#### Generation 0: Same Data Collection
```bash
# Use same data collection as baseline
python scripts/generate_selfplay_data.py \
    --model SyntheticRLV2 \
    --num_battles 50000 \
    --battle_format gen1ou \
    --team_set modern_replays_v2 \
    --output_dir ~/gen1_damped/gen0 \
    --parallel_instances 4

python scripts/filter_selfplay_data.py \
    --input_dir ~/gen1_damped/gen0/trajectories \
    --output_dir ~/gen1_damped/gen0_filtered \
    --max_invalid_rate 0.05 \
    --balance_outcomes
```

#### Generation 1: Train with Dynamic Damping
```bash
python -m metamon.rl.finetune_from_hf \
    --finetune_from_model SyntheticRLV2 \
    --run_name Damped_Gen1 \
    --custom_replay_dir ~/gen1_damped/gen0_filtered \
    --formats gen1ou \
    --train_gin_config vanilla_selfplay_damped.gin \  # ← Enables dynamic damping!
    --epochs 20 \
    --save_dir ~/gen1_checkpoints/damped_gen1 \
    --battle_format gen1ou \
    --team_set modern_replays_v2 \
    --obs_space TeamPreviewObservationSpace \
    --reward_function DefaultShapedReward \
    --action_space DefaultActionSpace \
    --log
```

#### Evaluate Generation 1
```bash
python scripts/self_play_tournament.py \
    --models Damped_Gen1 SyntheticRLV2 Baseline_Gen1 \
    --battle_format gen1ou \
    --num_battles 100 \
    --output_dir ~/gen1_damped/gen1_eval
```

---

## What Dynamic Damping Does

When using `vanilla_selfplay_damped.gin`, the training automatically:

1. **Creates reference policy snapshot** at start of training
2. **Adds KL regularization** to keep policy close to reference: `loss += kl_coef * KL(π_new || π_ref)`
3. **Applies power-law schedules** for smooth coefficient decay:
   - KL coefficient: `kl_t = kl_0 * (1 + t/T)^(-α)`
   - Entropy coefficient: `ent_t = ent_0 * (1 + t/T)^(-β)`
4. **Adaptive LR control**: Monitors KL divergence per update:
   - If KL too high → shrink LR, increase kl_coef
   - If KL too low → grow LR, decrease kl_coef

**Monitored Metrics** (logged to WandB/TensorBoard):
- `KL Divergence` - Should hover near target (0.01)
- `Policy Entropy` - Should decay smoothly (not cliff-like)
- `Damping/KL Coefficient` - Current KL weight
- `Damping/Entropy Coefficient` - Current entropy bonus
- `Learning Rate` - Adaptively adjusted

---

## Comparing Results

### Calculate ELO Ratings
```bash
python scripts/calculate_elo.py \
    --tournament_results ~/gen1_baseline/gen1_eval/results.json \
    --output ~/gen1_baseline/elo_ratings.json
```

### Key Comparisons
1. **Win rate vs SynRL-V2**: Does Gen1 beat the base model?
2. **Baseline vs Damped**: Which approach improves more?
3. **Entropy curves**: Does damping maintain exploration better?
4. **Training stability**: Fewer NaN losses, smoother learning curves?

---

## Tuning Dynamic Damping

If experiments show issues, adjust parameters in `vanilla_selfplay_damped.gin`:

### Entropy Drops Too Fast
```gin
# Increase entropy coefficient or slow decay
MetamonAMAGOExperiment.ent_coef_init = 0.02  # was 0.01
MetamonAMAGOExperiment.ent_power_alpha = 0.5  # was 0.7 (slower decay)
```

### KL Too High (Updates Too Conservative)
```gin
# Decrease KL coefficient or increase target
MetamonAMAGOExperiment.kl_coef_init = 0.03  # was 0.05
MetamonAMAGOExperiment.target_kl_per_step = 0.02  # was 0.01
```

### Training Too Slow
```gin
# Allow larger updates
MetamonAMAGOExperiment.target_kl_per_step = 0.02  # was 0.01
MetamonAMAGOExperiment.lr_grow_factor = 1.2  # was 1.1 (faster LR growth)
```

See `DYNAMIC_DAMPING_IMPLEMENTATION.md` for complete tuning guide.

---

## Using Existing Gen1 OU Dataset

You already have 1,104 Gen1 OU replays ready to use:

```bash
# Location
ls ~/nash_phase0/trajectories/gen1ou/  # 1,104 .json.lz4 files

# Use directly for training
python -m metamon.rl.finetune_from_hf \
    --finetune_from_model SyntheticRLV2 \
    --run_name From_Existing_Data \
    --custom_replay_dir ~/nash_phase0/trajectories \
    --formats gen1ou \
    --train_gin_config vanilla_selfplay_damped.gin \
    --epochs 20
```

---

## Directory Structure

Recommended organization:

```
~/gen1_experiments/
├── baseline/
│   ├── gen0/
│   │   ├── trajectories/gen1ou/*.json.lz4
│   │   └── team_results/
│   ├── gen0_filtered/
│   │   └── gen1ou/*.json.lz4
│   ├── gen1/
│   │   └── trajectories/gen1ou/*.json.lz4
│   └── checkpoints/
│       ├── baseline_gen0/ (SynRL-V2)
│       ├── baseline_gen1/
│       └── baseline_gen2/
└── damped/
    ├── gen0/
    ├── gen0_filtered/
    ├── gen1/
    └── checkpoints/
        ├── damped_gen0/ (SynRL-V2)
        ├── damped_gen1/
        └── damped_gen2/
```

---

## Expected Timeline

Per generation (rough estimates):
- **Data collection**: 1-2 hours (50k battles, 4 parallel instances)
- **Data filtering**: 5-10 minutes
- **Training**: 2-4 hours (20 epochs on GPU)
- **Evaluation**: 30-60 minutes (100 battles per matchup)

**Total for 5 generations**: ~20-40 hours

---

## Troubleshooting

### "No module named 'metamon'"
```bash
cd /home/eddie/repos/metamon
source .venv/bin/activate
```

### "Set METAMON_CACHE_DIR environment variable"
```bash
export METAMON_CACHE_DIR=/home/eddie/metamon_cache
```

### Connection refused (ladder errors)
```bash
# Start Pokémon Showdown server
cd /home/eddie/repos/metamon/server/pokemon-showdown
node pokemon-showdown start --no-security
```

### Training loading wrong formats
**Always use `--formats gen1ou`** in training commands!

---

## Alternative: Automated Workflow

If you prefer automation over manual control, use the orchestrator script:

```bash
python -m metamon.rl.train_vanilla_selfplay \
    --run_name Gen1_Damped_Automated \
    --init_checkpoint SyntheticRLV2 \
    --num_iterations 5 \
    --epochs_per_iteration 20 \
    --episodes_per_iteration 50000 \
    --save_dir ~/gen1_automated \
    --train_gin_config vanilla_selfplay_damped.gin \
    --battle_format gen1ou \
    --team_set modern_replays_v2
```

**Note**: This script currently uses a simplified data collection approach. For production experiments, the manual workflow above is recommended.

---

## Advanced: Aggressive Sleep Strategy Training

**New Training Mode**: Offline finetuning with custom reward function to prioritize sleep moves

### Setup

This approach uses a large offline dataset (25k+ battles) with a specialized reward function that incentivizes putting opponents to sleep, combined with BC-heavy training for stability.

**Key Features**:
- **Reward**: `AggressiveShapedRewardSleep` (+200/0 win/loss, +1 sleep bonus)
- **Observation**: `ExpandedObservationSpace` (adds PP tracking, sleep/freeze flags, tera types)
- **Training**: Pure offline with BC-heavy (75%) + small DPG (25%)
- **Damping**: Conservative with looser KL target (0.015) for offline adaptation

### Training Command

```bash
python -u -m metamon.rl.finetune_from_hf \
    --run_name "aggressive-sleep-loop3v1" \
    --finetune_from_model DampedBinarySuperV1_Epoch4 \
    --custom_replay_dir ~/metamon/trajectories/super_dataset_loop3/ \
    --custom_replay_sample_weight 1.0 \
    --formats gen1ou \
    --train_gin_config selfplay_damped_aggressive.gin \
    --reward_function AggressiveShapedRewardSleep \
    --epochs 5 \
    --save_dir ~/metamon/models/gen1_aggressive_sleep_loop3 \
    --eval_gens 1 \
    --log
```

### Configuration Details

The `selfplay_damped_aggressive.gin` config is specifically tuned for offline BC training:

**Reward Scaling**:
```gin
agent.Agent.reward_multiplier = 0.05  # Handles +200/0 scale
```
Old effective: 100 × 0.1 = 10
New effective: 200 × 0.05 = 10 (matched!)

**Data Mix** (inverted from online training):
```gin
agent.Agent.online_coeff = 0.25   # Small DPG signal
agent.Agent.offline_coeff = 0.75  # Heavy BC for stability
```

**Dynamic Damping** (slightly looser for offline):
```gin
MetamonAMAGOExperiment.target_kl_per_step = 0.015  # was 0.01
MetamonAMAGOExperiment.kl_tolerance = 1.5          # was 1.25
MetamonAMAGOExperiment.kl_coef_init = 0.20         # was 0.30
```

### What This Achieves

1. **Stable offline learning**: BC-heavy prevents catastrophic forgetting
2. **Sleep specialization**: +1 reward bonus shapes policy toward sleep moves
3. **Aggressive play**: +200/0 win/loss encourages taking risks
4. **Observation improvements**: ExpandedObservationSpace adds PP and status tracking

### Monitoring

Key metrics to watch during training:

| Metric | Target Range | Red Flag |
|--------|-------------|----------|
| KL Divergence | 0.01 - 0.0225 | > 0.025 |
| Policy Entropy | > 1.0 | < 0.5 |
| BC Loss | Decreasing | Flat/increasing |
| Sleep Move Usage | Increasing | Flat |
| Win Rate | Gradual improvement | Drops |

### When to Use This Approach

✅ **Use when**:
- You have a large offline dataset (10k+ battles)
- You want to specialize for specific strategies (e.g., sleep, stall)
- You need stability (no on-policy rollouts available)
- Fine-tuning from a strong pretrained model

❌ **Don't use when**:
- Starting from scratch (use online training first)
- Dataset is small (< 5k battles)
- Need general-purpose policy (use DefaultShapedReward)

---

## Summary

✅ **Use existing production scripts**: `generate_selfplay_data.py`, `filter_selfplay_data.py`, `self_play_tournament.py`

✅ **Enable dynamic damping**: Use `vanilla_selfplay_damped.gin` training config

✅ **Always specify format**: `--formats gen1ou` in all training commands

✅ **Monitor metrics**: KL, entropy, LR to tune hyperparameters

✅ **Compare approaches**: Run both baseline and damped experiments

✅ **Advanced**: Use `selfplay_damped_aggressive.gin` for specialized offline training with custom rewards

Good luck with your Gen1 OU self-play experiments! 🚀
