# EMA Ablation Experiment: Validating Policy Averaging on Gen1 OU

> **Category**: evaluation
> **Last Updated**: 2025-12-22
> **Author**: Claude via experiment design
> **Status**: Ready to Execute

## Objective

Empirically validate the effectiveness of EMA (Exponential Moving Average) policy averaging on Gen1 OU by comparing:
1. **EMA vs No-EMA**: Does EMA improve win rates and stability?
2. **Decay rate sensitivity**: What is the optimal decay rate (0.99 vs 0.999 vs 0.9999)?
3. **Current vs EMA checkpoints**: Does the EMA policy outperform the current policy at the same epoch?

**Research Questions**:
- Q1: Does EMA provide a measurable win rate improvement over non-EMA training?
- Q2: Does EMA reduce evaluation variance (more stable win rates across runs)?
- Q3: What decay rate maximizes Gen1 OU performance?
- Q4: Does EMA prevent strategic drift better than non-EMA approaches?

## Prerequisites

### Environment
```bash
# Activate virtual environment
source .venv/bin/activate

# Set cache directory
export METAMON_CACHE_DIR=/home/eddie/metamon_cache
```

### Data
- **Dataset**: `~/metamon/trajectories/super_dataset_loop6/gen1ou/`
- **Format filtering**: `--formats gen1ou` (critical!)
- **Base model**: `SleepLoop5Controller_Epoch2` (current best)
- **Observation space**: Uses default `DefaultObservationSpace` (ExpandedObservationSpace requires model migration, planned for future)

### Available Resources
- **Existing baselines**: Multiple training runs without EMA (different gin configs)
- **Evaluation infrastructure**: `evaluate.py`, `self_play_tournament.py`
- **Team set**: `modern_replays_v2` (Gen1 OU)

---

## Experimental Design

### **Phase 1: Binary Ablation (EMA On/Off)**

**Goal**: Establish if EMA provides any benefit at all

**Approach**: Train two models with identical configs except EMA flag

#### **Run 1A: No-EMA Baseline (Control)**

```bash
python -u -m metamon.rl.finetune_from_hf \
    --run_name "ema_ablation_control" \
    --finetune_from_model SleepLoop5Controller_Epoch2 \
    --custom_replay_dir ~/metamon/trajectories/super_dataset_loop6/ \
    --custom_replay_sample_weight 1.0 \
    --formats gen1ou \
    --train_gin_config selfplay_controller_v1.gin \
    --reward_function AggressiveShapedRewardSleep \
    --epochs 3 \
    --save_dir ~/metamon/models/ema_ablation/control_no_ema \
    --eval_gens 1 \
    --log
```

**Config**: `selfplay_controller_v1.gin` (existing, no EMA)
- KL target: 0.008
- Dynamic damping: enabled
- DPG: 5%
- **EMA: disabled** (control group)

**Expected output**:
- Checkpoints: `control_no_ema/ckpts/policy_weights/policy_epoch_{0,1,2}.pt`
- No EMA directory (EMA disabled)

---

#### **Run 1B: EMA Enabled (Treatment)**

**Create config**: `metamon/rl/configs/training/selfplay_controller_v1_ema.gin`

```gin
# Controller v1 + EMA
include 'metamon/rl/configs/training/selfplay_controller_v1.gin'

# Add EMA
MetamonAMAGOExperiment.use_ema = True
MetamonAMAGOExperiment.ema_decay = 0.999           # Standard decay
MetamonAMAGOExperiment.ema_update_interval = 1     # Every step
MetamonAMAGOExperiment.ema_warmup_steps = 0        # Start immediately
MetamonAMAGOExperiment.ema_eval_only = True        # Eval uses EMA, training uses current
```

```bash
python -u -m metamon.rl.finetune_from_hf \
    --run_name "ema_ablation_treatment" \
    --finetune_from_model SleepLoop5Controller_Epoch2 \
    --custom_replay_dir ~/metamon/trajectories/super_dataset_loop6/ \
    --custom_replay_sample_weight 1.0 \
    --formats gen1ou \
    --train_gin_config selfplay_controller_v1_ema.gin \
    --reward_function AggressiveShapedRewardSleep \
    --epochs 3 \
    --save_dir ~/metamon/models/ema_ablation/treatment_ema_0999 \
    --eval_gens 1 \
    --log
```

**Expected output**:
- Checkpoints: `treatment_ema_0999/ckpts/policy_weights/policy_epoch_{0,1,2}.pt` (current)
- EMA checkpoints: `treatment_ema_0999/ckpts/ema_weights/policy_epoch_{0,1,2}.pt` (EMA)
- Logs should show: `[EMA] Initialized with decay=0.999`

---

### **Phase 2: Decay Rate Sweep**

**Goal**: Find optimal decay rate for Gen1 OU

**Approach**: Train with 3 different decay rates: 0.99 (fast), 0.999 (standard), 0.9999 (slow)

#### **Run 2A: Fast Decay (0.99)**

**Create config**: `metamon/rl/configs/training/selfplay_controller_v1_ema_fast.gin`

```gin
include 'metamon/rl/configs/training/selfplay_controller_v1.gin'

MetamonAMAGOExperiment.use_ema = True
MetamonAMAGOExperiment.ema_decay = 0.99            # Fast decay (~100 step window)
MetamonAMAGOExperiment.ema_update_interval = 1
MetamonAMAGOExperiment.ema_warmup_steps = 0
MetamonAMAGOExperiment.ema_eval_only = True
```

```bash
python -u -m metamon.rl.finetune_from_hf \
    --run_name "ema_ablation_decay_099" \
    --finetune_from_model SleepLoop5Controller_Epoch2 \
    --custom_replay_dir ~/metamon/trajectories/super_dataset_loop6/ \
    --custom_replay_sample_weight 1.0 \
    --formats gen1ou \
    --train_gin_config selfplay_controller_v1_ema_fast.gin \
    --reward_function AggressiveShapedRewardSleep \
    --epochs 3 \
    --save_dir ~/metamon/models/ema_ablation/decay_099 \
    --eval_gens 1 \
    --log
```

---

#### **Run 2B: Standard Decay (0.999)** - Already done in Phase 1, Run 1B

---

#### **Run 2C: Slow Decay (0.9999)**

**Create config**: `metamon/rl/configs/training/selfplay_controller_v1_ema_slow.gin`

```gin
include 'metamon/rl/configs/training/selfplay_controller_v1.gin'

MetamonAMAGOExperiment.use_ema = True
MetamonAMAGOExperiment.ema_decay = 0.9999          # Slow decay (~10000 step window)
MetamonAMAGOExperiment.ema_update_interval = 1
MetamonAMAGOExperiment.ema_warmup_steps = 0
MetamonAMAGOExperiment.ema_eval_only = True
```

```bash
python -u -m metamon.rl.finetune_from_hf \
    --run_name "ema_ablation_decay_09999" \
    --finetune_from_model SleepLoop5Controller_Epoch2 \
    --custom_replay_dir ~/metamon/trajectories/super_dataset_loop6/ \
    --custom_replay_sample_weight 1.0 \
    --formats gen1ou \
    --train_gin_config selfplay_controller_v1_ema_slow.gin \
    --reward_function AggressiveShapedRewardSleep \
    --epochs 3 \
    --save_dir ~/metamon/models/ema_ablation/decay_09999 \
    --eval_gens 1 \
    --log
```

---

### **Phase 3: Current vs EMA Policy Comparison**

**Goal**: Within the EMA training runs, compare current policy vs EMA policy at same epoch

**Approach**: For Run 1B (decay=0.999), evaluate both checkpoint types

This requires **no additional training** - just evaluation of existing checkpoints from Run 1B.

---

## Summary of Training Runs

| Run ID | EMA Enabled? | Decay | Config | Save Dir |
|--------|-------------|-------|--------|----------|
| **1A** | ❌ No | N/A | `selfplay_controller_v1.gin` | `control_no_ema` |
| **1B** | ✅ Yes | 0.999 | `selfplay_controller_v1_ema.gin` | `treatment_ema_0999` |
| **2A** | ✅ Yes | 0.99 | `selfplay_controller_v1_ema_fast.gin` | `decay_099` |
| **2C** | ✅ Yes | 0.9999 | `selfplay_controller_v1_ema_slow.gin` | `decay_09999` |

**Total training runs**: 4
**Estimated time per run**: 4-6 hours (3 epochs on super_dataset_loop6)
**Total training time**: ~16-24 hours

---

## Evaluation Protocol

### **Evaluation 1: Baseline Performance (vs SyntheticRLV2)**

**Goal**: Measure absolute performance of all checkpoints

**Method**: Evaluate all epoch 2 checkpoints against a strong fixed baseline

```bash
# Define base directory
BASE_DIR=~/metamon/models/ema_ablation

# Evaluate Control (No-EMA) - Epoch 2
python -m metamon.rl.evaluate \
    --model_name SleepLoop5Controller_Epoch2 \
    --checkpoint_path ${BASE_DIR}/control_no_ema/ckpts/policy_weights/policy_epoch_2.pt \
    --opponent SyntheticRLV2 \
    --num_battles 500 \
    --battle_format gen1ou \
    --team_set modern_replays_v2 \
    --output_dir ~/evaluations/ema_ablation/control_vs_synthetic

# Evaluate EMA 0.999 - Current Policy - Epoch 2
python -m metamon.rl.evaluate \
    --model_name SleepLoop5Controller_Epoch2 \
    --checkpoint_path ${BASE_DIR}/treatment_ema_0999/ckpts/policy_weights/policy_epoch_2.pt \
    --opponent SyntheticRLV2 \
    --num_battles 500 \
    --battle_format gen1ou \
    --team_set modern_replays_v2 \
    --output_dir ~/evaluations/ema_ablation/ema0999_current_vs_synthetic

# Evaluate EMA 0.999 - EMA Policy - Epoch 2
python -m metamon.rl.evaluate \
    --model_name SleepLoop5Controller_Epoch2 \
    --checkpoint_path ${BASE_DIR}/treatment_ema_0999/ckpts/ema_weights/policy_epoch_2.pt \
    --opponent SyntheticRLV2 \
    --num_battles 500 \
    --battle_format gen1ou \
    --team_set modern_replays_v2 \
    --output_dir ~/evaluations/ema_ablation/ema0999_ema_vs_synthetic

# Evaluate Decay 0.99 - EMA Policy - Epoch 2
python -m metamon.rl.evaluate \
    --model_name SleepLoop5Controller_Epoch2 \
    --checkpoint_path ${BASE_DIR}/decay_099/ckpts/ema_weights/policy_epoch_2.pt \
    --opponent SyntheticRLV2 \
    --num_battles 500 \
    --battle_format gen1ou \
    --team_set modern_replays_v2 \
    --output_dir ~/evaluations/ema_ablation/decay099_vs_synthetic

# Evaluate Decay 0.9999 - EMA Policy - Epoch 2
python -m metamon.rl.evaluate \
    --model_name SleepLoop5Controller_Epoch2 \
    --checkpoint_path ${BASE_DIR}/decay_09999/ckpts/ema_weights/policy_epoch_2.pt \
    --opponent SyntheticRLV2 \
    --num_battles 500 \
    --battle_format gen1ou \
    --team_set modern_replays_v2 \
    --output_dir ~/evaluations/ema_ablation/decay09999_vs_synthetic
```

**Expected output**: Win rates vs SyntheticRLV2 for all variants

**Duration**: ~10-15 hours total (5 evaluations × 500 battles each)

---

### **Evaluation 2: Head-to-Head Round Robin**

**Goal**: Direct comparison between all variants

**Method**: Round-robin tournament between all 5 checkpoints

**Create checkpoint list**: `ema_ablation_checkpoints.txt`
```
control_no_ema,~/metamon/models/ema_ablation/control_no_ema/ckpts/policy_weights/policy_epoch_2.pt
ema_0999_current,~/metamon/models/ema_ablation/treatment_ema_0999/ckpts/policy_weights/policy_epoch_2.pt
ema_0999_ema,~/metamon/models/ema_ablation/treatment_ema_0999/ckpts/ema_weights/policy_epoch_2.pt
ema_099_ema,~/metamon/models/ema_ablation/decay_099/ckpts/ema_weights/policy_epoch_2.pt
ema_09999_ema,~/metamon/models/ema_ablation/decay_09999/ckpts/ema_weights/policy_epoch_2.pt
```

```bash
python scripts/self_play_tournament.py \
    --checkpoint_list ema_ablation_checkpoints.txt \
    --base_model SleepLoop5Controller_Epoch2 \
    --num_battles 200 \
    --battle_format gen1ou \
    --team_set modern_replays_v2 \
    --output_dir ~/evaluations/ema_ablation/round_robin
```

**Expected output**:
- Win rate matrix (5×5 grid)
- ELO ratings for all variants
- Statistical significance tests

**Duration**: ~6-8 hours (5 choose 2 = 10 matchups × 200 battles × 2 sides)

---

### **Evaluation 3: Variance Analysis (Repeated Runs)**

**Goal**: Measure stability (does EMA reduce evaluation variance?)

**Method**: Re-evaluate control vs best EMA variant 5 times (100 battles each)

```bash
# Identify best EMA variant from Evaluation 1 (assume it's ema_0999_ema for now)
BEST_EMA_PATH=~/metamon/models/ema_ablation/treatment_ema_0999/ckpts/ema_weights/policy_epoch_2.pt
CONTROL_PATH=~/metamon/models/ema_ablation/control_no_ema/ckpts/policy_weights/policy_epoch_2.pt

# Run 5 independent evaluations
for run in {1..5}; do
    echo "Variance test run ${run}/5..."

    # Control vs SyntheticRLV2
    python -m metamon.rl.evaluate \
        --model_name SleepLoop5Controller_Epoch2 \
        --checkpoint_path ${CONTROL_PATH} \
        --opponent SyntheticRLV2 \
        --num_battles 100 \
        --battle_format gen1ou \
        --team_set modern_replays_v2 \
        --output_dir ~/evaluations/ema_ablation/variance/control_run${run}

    # Best EMA vs SyntheticRLV2
    python -m metamon.rl.evaluate \
        --model_name SleepLoop5Controller_Epoch2 \
        --checkpoint_path ${BEST_EMA_PATH} \
        --opponent SyntheticRLV2 \
        --num_battles 100 \
        --battle_format gen1ou \
        --team_set modern_replays_v2 \
        --output_dir ~/evaluations/ema_ablation/variance/ema_run${run}
done
```

**Expected output**: 5 win rate measurements for each variant

**Analysis**: Compare standard deviations
- Hypothesis: EMA should have **lower standard deviation** (more stable)

**Duration**: ~2-3 hours (5 runs × 2 variants × 100 battles)

---

### **Evaluation 4: Strategic Drift Analysis (KL Divergence)**

**Goal**: Measure how much each policy drifts from the frozen baseline (SleepLoop5Controller_Epoch2)

**Method**: Compute KL(trained_policy || frozen_baseline) using offline data

This requires custom evaluation script (optional, advanced analysis).

**Pseudocode**:
```python
# Load frozen baseline
baseline_policy = load_model("SleepLoop5Controller_Epoch2")

# Load trained checkpoint
trained_policy = load_model("control_no_ema/epoch_2.pt")

# Sample 1000 states from super_dataset_loop6
states = sample_states(dataset, n=1000)

# Compute KL divergence
kl_div = 0
for state in states:
    p_baseline = baseline_policy.action_probs(state)
    p_trained = trained_policy.action_probs(state)
    kl_div += KL(p_trained || p_baseline)

kl_div /= len(states)
```

**Hypothesis**: EMA policies should have **lower KL divergence** (less drift) than control

**Duration**: ~1 hour (offline computation)

---

## Analysis Plan

### **Metric 1: Absolute Win Rate vs Baseline**

**Data**: From Evaluation 1 (vs SyntheticRLV2)

**Comparison**:
| Variant | Win Rate (%) | 95% CI | Δ vs Control |
|---------|-------------|--------|--------------|
| Control (No-EMA) | X% | ±Y% | - |
| EMA 0.999 (Current) | X% | ±Y% | +Z% |
| EMA 0.999 (EMA) | X% | ±Y% | +Z% |
| EMA 0.99 (EMA) | X% | ±Y% | +Z% |
| EMA 0.9999 (EMA) | X% | ±Y% | +Z% |

**Success criteria**:
- ✅ EMA improves win rate by ≥ 2% (statistically significant)
- ⚠️ EMA improves win rate by 0.5-2% (marginal, needs more data)
- ❌ EMA ≤ 0.5% improvement (no meaningful benefit)

---

### **Metric 2: Head-to-Head Win Rates**

**Data**: From Evaluation 2 (round-robin)

**Win Rate Matrix** (example):
|  | Control | EMA-0.999-Curr | EMA-0.999-EMA | EMA-0.99 | EMA-0.9999 |
|--|---------|----------------|---------------|----------|------------|
| **Control** | 50% | ? | ? | ? | ? |
| **EMA-0.999-Curr** | ? | 50% | ? | ? | ? |
| **EMA-0.999-EMA** | ? | ? | 50% | ? | ? |
| **EMA-0.99** | ? | ? | ? | 50% | ? |
| **EMA-0.9999** | ? | ? | ? | ? | 50% |

**Key comparisons**:
1. **EMA-0.999-EMA vs Control**: Does EMA beat no-EMA?
2. **EMA-0.999-EMA vs EMA-0.999-Curr**: Does EMA policy beat current policy?
3. **Best decay rate**: Which of (0.99, 0.999, 0.9999) wins most matchups?

**Success criteria**:
- ✅ EMA-0.999-EMA wins ≥ 55% vs Control (clear advantage)
- ⚠️ EMA-0.999-EMA wins 52-55% vs Control (small advantage)
- ❌ EMA-0.999-EMA wins < 52% vs Control (no advantage)

---

### **Metric 3: Stability (Variance)**

**Data**: From Evaluation 3 (5 repeated runs)

**Analysis**:
```python
import numpy as np

# Win rates from 5 runs (example)
control_win_rates = [48%, 52%, 50%, 47%, 53%]  # 5 runs
ema_win_rates = [51%, 50%, 52%, 51%, 50%]      # 5 runs

control_std = np.std(control_win_rates)  # Higher variance?
ema_std = np.std(ema_win_rates)          # Lower variance?

print(f"Control std: {control_std:.2f}%")
print(f"EMA std: {ema_std:.2f}%")
print(f"Variance reduction: {(control_std - ema_std) / control_std * 100:.1f}%")
```

**Success criteria**:
- ✅ EMA reduces variance by ≥ 30% (clear stability improvement)
- ⚠️ EMA reduces variance by 10-30% (modest improvement)
- ❌ EMA reduces variance by < 10% (no meaningful stability gain)

---

### **Metric 4: Strategic Drift (KL Divergence)**

**Data**: From Evaluation 4 (optional)

**Comparison**:
| Variant | KL(policy || baseline) | Δ vs Control |
|---------|------------------------|--------------|
| Control (No-EMA) | X nats | - |
| EMA 0.999 (EMA) | Y nats | -Z nats |

**Hypothesis**: EMA should have **lower KL** (less drift from frozen baseline)

**Success criteria**:
- ✅ EMA reduces KL by ≥ 20% (clear drift prevention)
- ⚠️ EMA reduces KL by 5-20% (modest drift prevention)
- ❌ EMA reduces KL by < 5% (no meaningful drift prevention)

---

## Expected Results

### **Hypothesis 1: EMA Provides Modest Win Rate Improvement**

**Prediction**: EMA-0.999-EMA > Control by **1-3 percentage points**

**Reasoning**:
- EMA smooths training noise → more stable policy
- Hidden-information game → averaging may help preserve mixed strategies
- But: First-epoch collapse experiment showed no effect, so benefit may be small

**If confirmed**: EMA is worth using (free improvement, no downsides)

**If rejected**: EMA is neutral (use for stability, not performance)

---

### **Hypothesis 2: EMA Policy Outperforms Current Policy**

**Prediction**: EMA-0.999-EMA > EMA-0.999-Curr by **0.5-2 percentage points**

**Reasoning**:
- EMA averages out training noise
- Current policy may overfit slightly to recent batches
- EMA provides "ensemble-like" regularization

**If confirmed**: Always use EMA checkpoints for evaluation/deployment

**If rejected**: Current and EMA are equivalent (use either)

---

### **Hypothesis 3: Optimal Decay is 0.999 (Standard)**

**Prediction**: 0.999 > 0.99 and 0.999 ≈ 0.9999

**Reasoning**:
- 0.99 too fast (forgets too quickly, may not smooth enough)
- 0.9999 too slow (remembers weak early policies too long)
- 0.999 is Goldilocks zone (proven in TD3/SAC literature)

**If confirmed**: Stick with default 0.999

**If rejected**: Tune decay rate for Gen1 OU specifically

---

### **Hypothesis 4: EMA Reduces Evaluation Variance**

**Prediction**: EMA variance < Control variance by **20-40%**

**Reasoning**:
- EMA averages stochastic gradient noise
- Provides smoother policy (less sensitive to random seeds)
- Well-established benefit in deep RL (target networks)

**If confirmed**: Strong justification for using EMA (stability matters for benchmarking)

**If rejected**: EMA doesn't help with variance (surprising, would need investigation)

---

## Execution Timeline

### **Week 1: Training**

**Day 1-2**:
- Create gin config files (3 new configs)
- Launch training runs (4 parallel jobs if GPU available)
- Monitor for crashes/errors

**Day 3-4**:
- Training continues (3 epochs × ~4-6 hours = 12-18 hours per run)
- Checkpoint verification (ensure EMA checkpoints are being saved)

**Day 5**:
- Training completes
- Verify all checkpoints exist
- Quick sanity check (load checkpoints, verify they run)

---

### **Week 2: Evaluation**

**Day 1-2**:
- Run Evaluation 1 (vs SyntheticRLV2) - 5 variants × 500 battles
- Preliminary analysis of win rates

**Day 3**:
- Run Evaluation 2 (round-robin) - 10 matchups × 200 battles
- Generate win rate matrix and ELO rankings

**Day 4**:
- Run Evaluation 3 (variance analysis) - 5 runs × 2 variants × 100 battles
- Compute standard deviations

**Day 5**:
- (Optional) Run Evaluation 4 (KL divergence analysis)
- Compile all results into final report

---

### **Week 3: Analysis & Documentation**

**Day 1-2**:
- Statistical analysis (confidence intervals, significance tests)
- Generate plots (win rate comparisons, variance plots)

**Day 3-4**:
- Write up results
- Update skill documentation
- Create retrospective

**Day 5**:
- Team review (if applicable)
- Commit findings to repository
- Update CLAUDE.md with EMA recommendations

---

## Resources Required

### **Compute**

**Training**:
- 4 training runs × 3 epochs × ~6 hours = **72 GPU-hours**
- Can parallelize (4 GPUs × 18 hours or 2 GPUs × 36 hours)

**Evaluation**:
- Evaluation 1: 5 variants × 500 battles × ~1 min/battle = **~40 hours** (can parallelize)
- Evaluation 2: 10 matchups × 200 battles × ~1 min/battle = **~30 hours** (can parallelize)
- Evaluation 3: 10 runs × 100 battles × ~1 min/battle = **~15 hours**

**Total**: ~157 hours of compute (can parallelize heavily across evaluation jobs)

---

### **Storage**

**Checkpoints**:
- 4 runs × 3 epochs × 2 checkpoint types (current + EMA for 3 runs) = **~20 checkpoints**
- ~2GB per checkpoint = **~40GB total**

**Battle logs**:
- Evaluation 1: 2,500 battles × ~100KB = ~250MB
- Evaluation 2: 2,000 battles × ~100KB = ~200MB
- Evaluation 3: 1,000 battles × ~100KB = ~100MB
- **Total: ~550MB**

**Grand total**: ~40-45GB (manageable)

---

## Success Criteria

### **Minimum Viable Success**

- ✅ All 4 training runs complete without crashes
- ✅ EMA checkpoints exist for all 3 EMA runs
- ✅ Evaluation 1 (vs baseline) completes for all variants
- ✅ Statistical significance computed (p-values, confidence intervals)
- ✅ Clear answer to Q1: "Does EMA help?" (yes/no/unclear)

---

### **Strong Success**

- ✅ All evaluations (1-3) complete successfully
- ✅ EMA provides ≥ 2% win rate improvement (statistically significant)
- ✅ EMA reduces evaluation variance by ≥ 30%
- ✅ Clear optimal decay rate identified (0.99, 0.999, or 0.9999)
- ✅ Recommendations documented in skill file
- ✅ Results committed to repository

---

### **Research-Grade Success**

- ✅ All evaluations (1-4) complete successfully
- ✅ EMA benefits quantified across multiple metrics (win rate, variance, drift)
- ✅ Mechanism of improvement understood (drift prevention vs noise smoothing vs regularization)
- ✅ Decay rate tuning curves published
- ✅ Results reproducible (all commands documented)
- ✅ Findings generalizable (tested on multiple base models or datasets)

---

## Potential Issues & Solutions

### Issue 1: Training Runs Crash Due to Config Errors

**Symptom**: Gin config parsing error or attribute error during initialization

**Root cause**: Typo in new gin config files

**Solution**:
```bash
# Test config before full training
python -c "
import gin
gin.parse_config_file('metamon/rl/configs/training/selfplay_controller_v1_ema.gin')
print('Config parsed successfully')
"
```

**Prevention**: Create configs incrementally, test after each change

---

### Issue 2: EMA Checkpoints Not Being Saved

**Symptom**: `ema_weights/` directory doesn't exist after training

**Root cause**: `use_ema` not set correctly, or checkpoint interval issue

**Solution**:
```bash
# Check training logs for EMA initialization message
grep "EMA" ~/metamon/models/ema_ablation/treatment_ema_0999/logs/*.log

# Should see:
# [EMA] Initialized with decay=0.999, warmup_steps=0
# [EMA] Saved checkpoint to .../ema_weights/policy_epoch_0.pt
```

**Prevention**: Run 1-epoch test first to verify EMA is active

---

### Issue 3: Evaluation Takes Too Long

**Symptom**: 500 battles × 5 variants = 2,500 battles taking > 24 hours

**Solution**:
1. **Parallelize**: Run multiple evaluations simultaneously on different GPUs/CPUs
2. **Reduce battles**: Use 300 battles instead of 500 (still statistically valid)
3. **Batched evaluation**: Use `generate_selfplay_data_batched.py` infrastructure if available

**Config change**:
```bash
# Reduce to 300 battles per evaluation
--num_battles 300  # Down from 500
```

---

### Issue 4: No Significant Difference Between Variants

**Symptom**: All variants have ~50% win rate vs each other (within noise)

**Root cause**: EMA effect is too small to measure with current sample size

**Solutions**:
1. **Increase battles**: 500 → 1000 per matchup (reduce noise)
2. **Use tighter baseline**: Compare to weaker opponent to amplify differences
3. **Look at variance instead**: Even if mean is same, variance might differ

**Analysis adjustment**:
```bash
# If no difference in mean, compute variance explicitly
python -c "
import numpy as np
runs = [48, 52, 50, 49, 51]  # Example: control win rates
print(f'Mean: {np.mean(runs):.1f}%, Std: {np.std(runs):.2f}%')
"
```

---

### Issue 5: Decay Rates All Perform Similarly

**Symptom**: 0.99, 0.999, 0.9999 have < 1% win rate difference

**Root cause**: Gen1 OU may not be sensitive to decay rate (3 epochs not enough differentiation)

**Solutions**:
1. **Train longer**: 3 epochs → 5 epochs (give EMA more time to diverge)
2. **Test extremes**: Try 0.9 (very fast) and 0.99999 (very slow)
3. **Accept result**: Document that decay rate doesn't matter much for Gen1 OU

**Interpretation**: If decay doesn't matter, **default 0.999 is safe choice** (no tuning needed)

---

## Follow-Up Experiments

If initial results are promising, consider:

### **Experiment 1: EMA + PSRO Integration**

**Goal**: Test if EMA checkpoints improve PSRO population quality

**Method**:
- Run PSRO iteration with EMA checkpoints vs non-EMA checkpoints
- Measure exploitability reduction

---

### **Experiment 2: EMA with Different Base Configs**

**Goal**: Validate that EMA benefits generalize across configs

**Method**:
- Repeat Phase 1 (EMA on/off) with `sleep_selfplay_v2_aggressive.gin`
- Compare to controller_v1 results

---

### **Experiment 3: Adaptive Decay Schedules**

**Goal**: Test if time-varying decay improves performance

**Method**:
- Decay schedule: Start at 0.99 (epoch 0), gradually increase to 0.9999 (epoch 3)
- Compare to fixed 0.999

---

### **Experiment 4: EMA + Human Replay Mixing**

**Goal**: Test if EMA benefits change when training data includes human replays

**Method**:
- Train with 80% loop6 + 20% human replays
- Compare EMA on/off in mixed-data setting

---

## Related Skills

- [`policy-averaging-ema-overview`](./policy-averaging-ema-overview.md) - Background on EMA implementation and theory
- [`epistemic-uncertainty-negative-result`](./epistemic-uncertainty-negative-result.md) - EMA + epistemic weighting (failed to fix first-epoch collapse)
- [`large-dataset-overfitting-200k`](./large-dataset-overfitting-200k.md) - Overfitting patterns (EMA may help with stability but not overfitting)
- [`manual-evaluation-workflow`](./manual-evaluation-workflow.md) - Evaluation best practices

---

## Quick Start Commands

### **Training (All Runs)**

```bash
# Set up
source .venv/bin/activate
export METAMON_CACHE_DIR=/home/eddie/metamon_cache

# Create configs (one-time)
cat > metamon/rl/configs/training/selfplay_controller_v1_ema.gin << 'EOF'
include 'metamon/rl/configs/training/selfplay_controller_v1.gin'
MetamonAMAGOExperiment.use_ema = True
MetamonAMAGOExperiment.ema_decay = 0.999
MetamonAMAGOExperiment.ema_update_interval = 1
MetamonAMAGOExperiment.ema_warmup_steps = 0
MetamonAMAGOExperiment.ema_eval_only = True
EOF

cat > metamon/rl/configs/training/selfplay_controller_v1_ema_fast.gin << 'EOF'
include 'metamon/rl/configs/training/selfplay_controller_v1.gin'
MetamonAMAGOExperiment.use_ema = True
MetamonAMAGOExperiment.ema_decay = 0.99
MetamonAMAGOExperiment.ema_update_interval = 1
MetamonAMAGOExperiment.ema_warmup_steps = 0
MetamonAMAGOExperiment.ema_eval_only = True
EOF

cat > metamon/rl/configs/training/selfplay_controller_v1_ema_slow.gin << 'EOF'
include 'metamon/rl/configs/training/selfplay_controller_v1.gin'
MetamonAMAGOExperiment.use_ema = True
MetamonAMAGOExperiment.ema_decay = 0.9999
MetamonAMAGOExperiment.ema_update_interval = 1
MetamonAMAGOExperiment.ema_warmup_steps = 0
MetamonAMAGOExperiment.ema_eval_only = True
EOF

# Run 1A: Control (No-EMA)
python -u -m metamon.rl.finetune_from_hf \
    --run_name "ema_ablation_control" \
    --finetune_from_model SleepLoop5Controller_Epoch2 \
    --custom_replay_dir ~/metamon/trajectories/super_dataset_loop6/ \
    --custom_replay_sample_weight 1.0 \
    --formats gen1ou \
    --train_gin_config selfplay_controller_v1.gin \
    --reward_function AggressiveShapedRewardSleep \
    --epochs 3 \
    --save_dir ~/metamon/models/ema_ablation/control_no_ema \
    --eval_gens 1 \
    --log

# Run 1B: EMA 0.999
python -u -m metamon.rl.finetune_from_hf \
    --run_name "ema_ablation_treatment" \
    --finetune_from_model SleepLoop5Controller_Epoch2 \
    --custom_replay_dir ~/metamon/trajectories/super_dataset_loop6/ \
    --custom_replay_sample_weight 1.0 \
    --formats gen1ou \
    --train_gin_config selfplay_controller_v1_ema.gin \
    --reward_function AggressiveShapedRewardSleep \
    --epochs 3 \
    --save_dir ~/metamon/models/ema_ablation/treatment_ema_0999 \
    --eval_gens 1 \
    --log

# Run 2A: EMA 0.99 (Fast)
python -u -m metamon.rl.finetune_from_hf \
    --run_name "ema_ablation_decay_099" \
    --finetune_from_model SleepLoop5Controller_Epoch2 \
    --custom_replay_dir ~/metamon/trajectories/super_dataset_loop6/ \
    --custom_replay_sample_weight 1.0 \
    --formats gen1ou \
    --train_gin_config selfplay_controller_v1_ema_fast.gin \
    --reward_function AggressiveShapedRewardSleep \
    --epochs 3 \
    --save_dir ~/metamon/models/ema_ablation/decay_099 \
    --eval_gens 1 \
    --log

# Run 2C: EMA 0.9999 (Slow)
python -u -m metamon.rl.finetune_from_hf \
    --run_name "ema_ablation_decay_09999" \
    --finetune_from_model SleepLoop5Controller_Epoch2 \
    --custom_replay_dir ~/metamon/trajectories/super_dataset_loop6/ \
    --custom_replay_sample_weight 1.0 \
    --formats gen1ou \
    --train_gin_config selfplay_controller_v1_ema_slow.gin \
    --reward_function AggressiveShapedRewardSleep \
    --epochs 3 \
    --save_dir ~/metamon/models/ema_ablation/decay_09999 \
    --eval_gens 1 \
    --log
```

---

### **Evaluation (Phase 1: Baseline Performance)**

```bash
BASE_DIR=~/metamon/models/ema_ablation

# Control vs SyntheticRLV2
python -m metamon.rl.evaluate \
    --model_name SleepLoop5Controller_Epoch2 \
    --checkpoint_path ${BASE_DIR}/control_no_ema/ckpts/policy_weights/policy_epoch_2.pt \
    --opponent SyntheticRLV2 \
    --num_battles 500 \
    --battle_format gen1ou \
    --team_set modern_replays_v2 \
    --output_dir ~/evaluations/ema_ablation/control_vs_synthetic

# EMA 0.999 (EMA policy) vs SyntheticRLV2
python -m metamon.rl.evaluate \
    --model_name SleepLoop5Controller_Epoch2 \
    --checkpoint_path ${BASE_DIR}/treatment_ema_0999/ckpts/ema_weights/policy_epoch_2.pt \
    --opponent SyntheticRLV2 \
    --num_battles 500 \
    --battle_format gen1ou \
    --team_set modern_replays_v2 \
    --output_dir ~/evaluations/ema_ablation/ema0999_ema_vs_synthetic

# (Repeat for other variants...)
```

---

## Status

**Status**: Ready to Execute
**Estimated Duration**: 2-3 weeks (training + evaluation + analysis)
**Compute Cost**: ~157 GPU/CPU-hours (can parallelize)
**Storage Cost**: ~45GB
**Confidence**: High (well-designed ablation with clear metrics)
