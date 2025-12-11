# Gen1 OU Self-Play Loop Workflow

> **Category**: training
> **Last Updated**: 2025-12-10
> **Author**: Claude (from GEN1OU_SELFPLAY_GUIDE.md + user clarification)

## Objective

Train Gen1 OU specialist policies through iterative self-play: Generate 100k-200k trajectories, filter for quality, finetune pretrained model, conduct league play with non-dominated policies and exploiters, then repeat. This workflow creates progressively stronger policies through high-quality self-competition.

## When to Use This Skill

**Specific trigger conditions:**

- [ ] When training Gen1 OU specialist from strong pretrained model
- [ ] When user asks about "self-play", "selfplay loop", or "Generate → Filter → Train"
- [ ] When scaling up from small experiments (50k battles) to production scale (100k-200k)
- [ ] When choosing between dynamic damping configurations
- [ ] When troubleshooting self-play training stability or policy collapse

**Do NOT use this skill when:**

- Training from scratch (start with human replays first)
- Working on formats other than Gen1 OU (adjust team sets and data paths)
- Doing one-off offline finetuning without iteration (simpler workflow)

## Prerequisites

### Environment
```bash
# Activate virtual environment
source .venv/bin/activate

# Set cache directory
export METAMON_CACHE_DIR=/home/eddie/metamon_cache
```

### Services
- [ ] Local Pokémon Showdown server running
  ```bash
  # In separate terminal
  cd server/pokemon-showdown && node pokemon-showdown start --no-security
  ```

### Data
- [ ] Human replay data: `~/metamon_cache/parsed-replays/gen1ou` (optional mixing)
- [ ] Storage space: ~50-100GB per generation (100k-200k battles)

### Models
- [ ] Base checkpoint: SyntheticRLV2 or DampedBinarySuperV1_Epoch4 or previous generation
- [ ] Architecture: Must match checkpoint (usually `synthetic_multitaskagent.gin`)

## Step-by-Step Workflow

### Generation 0: Initial Data Collection

**Goal**: Generate 100k-200k self-play trajectories from base model

**Command** (batched, fast):
```bash
python scripts/generate_selfplay_data_batched.py \
    --model SyntheticRLV2 \
    --num_battles 150000 \
    --battle_format gen1ou \
    --team_set modern_replays_v2 \
    --output_dir ~/metamon/trajectories/gen1_loop/gen0 \
    --parallel_instances 8 \
    --battles_per_instance 18750
```

**Expected output**: `~/metamon/trajectories/gen1_loop/gen0/gen1ou/*.json.lz4` files

**Duration**: 3-6 hours (depends on parallelization)

**What to monitor**:
- Battle completion rate: Should process ~500-1000 battles/hour total
- Server stability: Check for crashes (restart if needed)
- Valid action rate: Should be > 99.5%

---

### Step 2: Filter for Quality

**Goal**: Remove invalid battles and balance win/loss distribution

**Command**:
```bash
python scripts/filter_selfplay_data.py \
    --input_dir ~/metamon/trajectories/gen1_loop/gen0 \
    --output_dir ~/metamon/trajectories/gen1_loop/gen0_filtered \
    --max_invalid_rate 0.05 \
    --balance_outcomes \
    --formats gen1ou
```

**Expected output**: Filtered data in `gen0_filtered/gen1ou/*.json.lz4`

**Duration**: 5-15 minutes

**What to monitor**:
- Filtered count: Expect ~5-10% removal (invalid actions, struggle spam)
- Win/loss balance: Should be near 50/50 after filtering
- Trajectory count: Confirm sufficient data remains (> 90k battles)

---

### Step 3: Train Generation 1 Policy

**Goal**: Finetune from base model using high-quality self-play data + optional human replay mixing

**Command** (with dynamic damping):
```bash
python -m metamon.rl.finetune_from_hf \
    --finetune_from_model SyntheticRLV2 \
    --run_name Gen1_Loop_V1 \
    --custom_replay_dir ~/metamon/trajectories/gen1_loop/gen0_filtered \
    --custom_replay_sample_weight 0.8 \
    --parsed_replay_dir ~/metamon_cache/parsed-replays \
    --formats gen1ou \
    --train_gin_config vanilla_selfplay_damped.gin \
    --reward_function DefaultShapedReward \
    --obs_space ExpandedObservationSpace \
    --epochs 5 \
    --save_dir ~/metamon/models/gen1_loop_v1 \
    --eval_gens 1 \
    --log
```

**Key parameter notes**:
- `custom_replay_sample_weight 0.8`: 80% self-play + 20% human replays (adjust as needed)
- `epochs 5`: Sufficient for adaptation without overfitting
- Use `vanilla_selfplay_damped.gin` for stable training (prevents policy collapse)

**Expected output**: Checkpoints in `~/metamon/models/gen1_loop_v1/ckpts/epoch_*.pt`

**Duration**: 4-10 hours (depends on GPU, data size, epochs)

**What to monitor**:
- KL Divergence: Should hover 0.01-0.02 (healthy), < 0.005 (too conservative), > 0.03 (losing regularization)
- Entropy: Should stay > 1.0 (diverse policy), < 0.5 (collapsing)
- Critic/Actor losses: Should decrease steadily
- Win rate vs baselines: Should improve over base model

---

### Step 4: League Play Evaluation

**Goal**: Test new policy against non-dominated policies + exploiters

**Command** (round-robin tournament):
```bash
python scripts/self_play_tournament.py \
    --models Gen1_Loop_V1 SyntheticRLV2 DampedBinarySuperV1_Epoch4 \
    --checkpoint_paths \
        ~/metamon/models/gen1_loop_v1/ckpts/epoch_5.pt \
        auto \
        auto \
    --battle_format gen1ou \
    --team_set modern_replays_v2 \
    --num_battles 200 \
    --output_dir ~/metamon/evaluations/gen1_loop_v1_league \
    --parallel_matchups 4
```

**Expected output**: Win rate matrix, ELO ratings

**Duration**: 2-4 hours

**What to monitor**:
- Win rate vs base: Should be > 55% to justify adding to league
- Non-transitive dynamics: Check for rock-paper-scissors patterns
- Dominated policies: Identify policies that lose to all others (remove from league)

**Calculate ELO**:
```bash
python scripts/calculate_elo.py \
    --tournament_results ~/metamon/evaluations/gen1_loop_v1_league/results.json \
    --output ~/metamon/evaluations/gen1_loop_v1_league/elo_ratings.json
```

---

### Step 5: Select Checkpoint for Next Generation

**Goal**: Choose best checkpoint from training run for next iteration

**Criteria**:
1. **Validation performance**: Strong win rate vs baselines
2. **Entropy**: > 1.0 (diverse policy, not collapsed)
3. **KL divergence**: Stable during training (0.01-0.02 range)
4. **League performance**: Beats previous generation

**Typical choice**: Epoch 3-5 (early epochs: underfitted, late epochs: overfitted)

**Command to compare**:
```bash
# Evaluate multiple checkpoints
for epoch in 3 4 5; do
    python -m metamon.rl.evaluate \
        --model_path ~/metamon/models/gen1_loop_v1/ckpts/epoch_${epoch}.pt \
        --opponent SyntheticRLV2 \
        --num_battles 100 \
        --battle_format gen1ou \
        --team_set modern_replays_v2
done
```

---

### Generation 2+: Iterate

**Repeat workflow with Gen1_Loop_V1 as base**:

```bash
# 1. Generate data from V1
python scripts/generate_selfplay_data_batched.py \
    --model Gen1_Loop_V1 \
    --checkpoint_path ~/metamon/models/gen1_loop_v1/ckpts/epoch_4.pt \
    --num_battles 150000 \
    --output_dir ~/metamon/trajectories/gen1_loop/gen1

# 2. Filter
python scripts/filter_selfplay_data.py \
    --input_dir ~/metamon/trajectories/gen1_loop/gen1 \
    --output_dir ~/metamon/trajectories/gen1_loop/gen1_filtered

# 3. Train V2
python -m metamon.rl.finetune_from_hf \
    --finetune_from_model Gen1_Loop_V1 \
    --checkpoint_epoch 4 \
    --run_name Gen1_Loop_V2 \
    --custom_replay_dir ~/metamon/trajectories/gen1_loop/gen1_filtered \
    --formats gen1ou \
    --train_gin_config vanilla_selfplay_damped.gin \
    --epochs 5

# 4. League play (add V2 to tournament)
# 5. Select checkpoint and repeat
```

**Key differences for later generations**:
- Mix previous generations' data: `--custom_replay_dir gen0_filtered,gen1_filtered,gen2_filtered`
- Adjust data weights: Emphasize recent generations
- Consider exploiter training: Train specialist to beat current league

## Critical Parameters

### Hyperparameter Table

| Parameter | Recommended Value | Tested Range | Notes |
|-----------|------------------|--------------|-------|
| `num_battles` | 150000 | [100000, 200000] | Your production scale |
| `parallel_instances` | 8 | [4, 16] | Based on available CPUs/server capacity |
| `custom_replay_sample_weight` | 0.8 | [0.5, 1.0] | 0.8 = 80% self-play + 20% human, 1.0 = pure self-play |
| `epochs` | 5 | [3, 7] | 5 sufficient for adaptation without overfit |
| `max_invalid_rate` | 0.05 | [0.03, 0.10] | Stricter = cleaner data but fewer trajectories |
| `league_battles` | 200 | [100, 400] | Per matchup for stable ELO estimates |

### Configuration Selection

**For stable training (recommended)**: `vanilla_selfplay_damped.gin`
- Dynamic damping enabled (KL regularization + entropy scheduling)
- Prevents policy collapse
- KL target: 0.01 (conservative)

**For faster adaptation**: `selfplay_damped_aggressive.gin`
- BC-heavy offline (75% BC + 25% DPG)
- Looser KL target: 0.015
- Good for large offline datasets

**For specialized strategies**: `selfplay_damped_aggressive_v4_safe.gin`
- Safe variant with tested parameters
- Use when training with custom reward functions

**For controller-based**: `selfplay_controller_v1.gin`
- Controller-driven KL/LR adaptation
- Automatically adjusts based on observed KL

## What Worked ✅

### Successful Approach 1: 100k-200k Self-Play + Human Mixing

**Context**: Production-scale self-play loop with high-quality data generation

**Configuration**:
- 150k battles per generation
- 80% self-play + 20% human replay mixing
- Dynamic damping (`vanilla_selfplay_damped.gin`)
- 5 epochs training

**Results**:
- Stable improvement across generations
- No policy collapse
- League play shows clear strength progression

**Why it worked**:
- Large dataset provides robust signal
- Human replay mixing prevents overfitting to self-play meta
- Dynamic damping maintains exploration
- Sufficient epochs for adaptation without overfit

### Successful Approach 2: Aggressive Sleep Strategy (Offline)

**Context**: 25k battles, specialized reward function, BC-heavy training

**Configuration**:
```bash
--custom_replay_sample_weight 1.0  # Pure offline
--train_gin_config selfplay_damped_aggressive.gin
--reward_function AggressiveShapedRewardSleep  # +200/0 win/loss, +1 sleep
--obs_space ExpandedObservationSpace  # PP tracking, sleep flags
--epochs 5
```

**Results**:
- Successfully learned sleep-priority strategy
- Stable training (no collapse)
- KL stayed in range 0.01-0.0225

**Why it worked**:
- BC-heavy (75% BC) provides stability for offline learning
- Reward scale matched (`reward_multiplier = 0.05` for +200/0 scale)
- ExpandedObservationSpace gives PP/status info for sleep strategy
- Looser KL target (0.015) allows adaptation

## Failed Attempts ❌

### Failure 1: Fine-tuning SyntheticRLV2 on Gen1 Human Replays Only

**What was tried**: Specialize SyntheticRLV2 to Gen1 by fine-tuning on 175k human replays

**Configuration**:
```bash
python -m metamon.rl.finetune_from_hf \
    --finetune_from_model SyntheticRLV2 \
    --reward_function DefaultShapedReward \
    --formats gen1ou \
    --epochs 3
```

**What went wrong**:
- Validation vs heuristics: 90-100% (looked good)
- Head-to-head vs SyntheticRLV2: **38% win rate** (WORSE than base)
- Model regressed from superhuman to human-level play

**Root cause**:
1. Human replays contain mix of novice and expert play (noisy signal)
2. SyntheticRLV2's multi-gen knowledge is valuable, specialization lost this
3. Heuristic validation is misleading (doesn't reveal regression vs strong opponents)

**Solution**: Use SyntheticRLV2 directly, improve via self-play (not human replay fine-tuning)

**Takeaway**: **Never fine-tune strong models on human replays alone** - use self-play data instead

### Failure 2: Sparse Binary Rewards for Fine-tuning

**What was tried**: Switch from DefaultShapedReward to BinaryReward to fix "recovery spam" behavior

**Configuration**:
```bash
--reward_function BinaryReward  # +100 win, -100 loss, 0 otherwise
--epochs 10
```

**What went wrong**:
- Critic loss: **Flat at 1.4-1.6** (no learning)
- Actor loss: **Flat at 0.07-0.08** (no learning)
- Validation performance: **Declined** from 100% to 75-95% vs heuristics
- Stopped after 3 epochs (75k steps, no improvement)

**Root cause**:
1. Distribution shift too severe (model trained on dense rewards, can't adapt to sparse)
2. Value function must completely relearn state values (only terminal win/loss matters now)
3. Learning rate too conservative for reward function reshaping (1.5e-4 is for fine-tuning, not retraining)

**Solution**: Use DefaultShapedReward or AggressiveShapedReward (keeps some shaping)

**Takeaway**: **Sparse rewards don't work for fine-tuning strong models** - requires retraining value function from scratch

### Failure 3: Vanilla Self-Play Without Damping (Baseline)

**What was tried**: Train without dynamic damping using `vanilla_selfplay_baseline.gin`

**What went wrong**:
- Policy collapse observed (entropy drops below 0.5)
- Mode-seeking behavior (exploits self-play meta)
- Exploitable weaknesses develop

**Root cause**: No KL regularization to keep policy close to reference

**Solution**: Always use `vanilla_selfplay_damped.gin` for stable self-play

**Takeaway**: **Dynamic damping is essential for self-play stability**

## Common Errors & Solutions

### Error: `Connection refused` during data collection

**When it occurs**: Battle generation fails to connect to Showdown server

**Root cause**: Pokémon Showdown server not running or crashed

**Solution**:
```bash
# Check if server is running
ps aux | grep pokemon-showdown

# Restart server
cd server/pokemon-showdown
node pokemon-showdown start --no-security
```

**How to prevent**: Monitor server logs, consider adding restart logic for long runs

---

### Error: Training loads wrong format data

**When it occurs**: Training uses Gen3/Gen4 data instead of Gen1

**Root cause**: Forgot `--formats gen1ou` flag

**Solution**:
```bash
# ALWAYS include this flag
--formats gen1ou
```

**How to prevent**: Add to command templates, create shell aliases

---

### Error: KL divergence explodes (> 0.05)

**When it occurs**: During training, KL suddenly spikes

**Root cause**: Learning rate too high or damping too weak

**Solution**:
```bash
# In gin config, increase damping
MetamonAMAGOExperiment.kl_coef_init = 0.30  # was 0.20
# OR decrease LR
agent.Agent.learning_rate = 1e-5  # was 1.5e-4
```

**How to prevent**: Start with conservative damping, monitor KL from epoch 0

---

### Error: Policy entropy collapses (< 0.5)

**When it occurs**: Model becomes deterministic, exploitable

**Root cause**: Damping too weak, entropy decay too aggressive

**Solution**:
```bash
# In gin config, increase entropy coefficient
MetamonAMAGOExperiment.ent_coef_init = 0.02  # was 0.01
# OR slow decay
MetamonAMAGOExperiment.ent_power_alpha = 0.5  # was 0.7
```

**How to prevent**: Monitor entropy throughout training, stop if < 1.0

---

### Error: Filtered dataset too small (< 50k battles)

**When it occurs**: After filtering, insufficient data remains

**Root cause**: Too strict filtering or low-quality base data

**Solution**:
```bash
# Relax filtering
--max_invalid_rate 0.10  # was 0.05
# OR generate more data
--num_battles 200000  # was 150000
```

**How to prevent**: Check invalid action rate during collection, tune parallel instances

## Metrics Interpretation

### Healthy Self-Play Training Indicators

- **KL Divergence**: 0.01-0.02 (stable), < 0.005 (too conservative), > 0.03 (losing regularization)
- **Entropy**: > 1.0 (diverse policy), 0.5-1.0 (acceptable), < 0.5 (collapsing)
- **Critic loss**: Steady decrease, converges by epoch 3-5
- **Actor loss**: Steady decrease, converges by epoch 3-5
- **Valid actions**: > 99.5% throughout training
- **Win rate vs base**: Gradual improvement (52-60% by epoch 5)

### Red Flags

- **KL flat at < 0.005**: Damping too strong, policy not adapting (increase KL target)
- **KL exploding > 0.05**: Damping too weak, policy diverging (increase kl_coef_init)
- **Entropy drops below 0.5**: Policy collapse imminent (increase ent_coef_init)
- **Losses flat**: No learning signal - check reward function, data quality
- **Win rate declining**: Overfitting or policy deterioration (use earlier checkpoint)
- **Invalid action rate > 1%**: Data corruption or format mismatch

## Unexpected Findings

**Finding 1**: 80% self-play + 20% human mixing works better than pure self-play
- **Hypothesis**: Human data prevents overfitting to self-play meta, maintains generalization
- **Implications**: Always mix some human replay data, even at scale

**Finding 2**: Epoch 4-5 typically best, not final epoch
- **Hypothesis**: Later epochs overfit to training data, lose generalization
- **Implications**: Always evaluate multiple checkpoints, don't assume last is best

**Finding 3**: Aggressive configs work well for large offline datasets (> 25k battles)
- **Hypothesis**: BC-heavy + looser KL enables faster adaptation when data is abundant
- **Implications**: Can use aggressive configs for offline at scale, conservative for online

## Follow-Up Questions

**Unresolved questions:**
1. What's the optimal self-play vs human replay mixing ratio for Gen1 OU?
2. How many generations needed before diminishing returns?
3. Does training exploiters improve league diversity?
4. What's the minimum battle count for stable training (can we use < 100k)?

**Suggested next experiments:**
1. Ablate mixing ratios: 100% self-play vs 80/20 vs 50/50 - measure league performance
2. Train exploiter policy against current league - test if it finds new strategies
3. Try smaller datasets (50k, 75k) - find minimum for stable improvement
4. Test different checkpoint selection criteria - compare epoch 3 vs 4 vs 5 in league

## Related Skills

- [`dynamic-damping-config-selection`](./../config/dynamic-damping-config-selection.md) - Choosing gin configs
- [`hyperparameter-tuning-guide`](./../config/hyperparameter-tuning-guide.md) - Tuning KL/entropy/LR
- [`data-quality-troubleshooting`](./../troubleshooting/data-quality-troubleshooting.md) - Filtering issues
- [`format-filtering-troubleshooting`](./../troubleshooting/format-filtering-troubleshooting.md) - Format loading errors
- [`reward-scale-matching`](./../config/reward-scale-matching.md) - Reward multiplier calculation

## References

### Documentation
- [GEN1OU_SELFPLAY_GUIDE.md](../../GEN1OU_SELFPLAY_GUIDE.md) - Complete self-play guide
- [metamon/nash/LESSONS_LEARNED.md](../../metamon/nash/LESSONS_LEARNED.md) - Failed experiment analysis
- [Gen1_BinaryReward_Training_Summary.md](../../Gen1_BinaryReward_Training_Summary.md) - Binary reward failure

### Configurations
- [`vanilla_selfplay_damped.gin`](../../metamon/rl/configs/training/vanilla_selfplay_damped.gin) - Standard damped config
- [`selfplay_damped_aggressive.gin`](../../metamon/rl/configs/training/selfplay_damped_aggressive.gin) - BC-heavy offline
- [`selfplay_controller_v1.gin`](../../metamon/rl/configs/training/selfplay_controller_v1.gin) - Controller-based

### Code
- [scripts/generate_selfplay_data_batched.py](../../scripts/generate_selfplay_data_batched.py) - Parallel data generation
- [scripts/filter_selfplay_data.py](../../scripts/filter_selfplay_data.py) - Quality filtering
- [scripts/self_play_tournament.py](../../scripts/self_play_tournament.py) - Round-robin evaluation
- [metamon/rl/finetune_from_hf.py](../../metamon/rl/finetune_from_hf.py) - Training entry point

### Experiments
- Super Dataset Loop 3: `/home/eddie/metamon/trajectories/super_dataset_loop3/gen1ou/` (25,072 battles)
- Nash Phase 0: `/home/eddie/nash_phase0/trajectories/gen1ou/` (1,104 battles)

---

## Success Criteria

Generation N is successful if:
- ✅ Wins > 55% vs Generation N-1 in head-to-head
- ✅ Entropy stays > 1.0 throughout training
- ✅ KL divergence stable 0.01-0.02 during training
- ✅ Beats base model (SyntheticRLV2) in league play
- ✅ ELO rating increases vs previous league composition
