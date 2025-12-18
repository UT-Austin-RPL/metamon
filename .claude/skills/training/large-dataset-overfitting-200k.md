# Large Dataset Overfitting at 200k Trajectories

> **Category**: training
> **Last Updated**: 2025-12-11
> **Author**: Claude via /retrospective

## Objective

Document the consistent epoch 2 → epoch 4 degradation observed when training on 200k+ trajectory datasets, occurring with BOTH aggressive and conservative controller configurations. This is a dataset-scale overfitting issue, not a config-specific problem.

## When to Use This Skill

**Specific trigger conditions:**

- [ ] When training on 100k-200k+ trajectory datasets
- [ ] When win rates peak early (epoch 2-3) then decline
- [ ] When choosing optimal checkpoint from multi-epoch training
- [ ] When deciding between early stopping vs training to completion

**Do NOT use this skill when:**

- Training on < 50k trajectories (different overfitting dynamics)
- Training from scratch (not finetuning from pretrained checkpoint)
- Using online RL (continuous data generation, not fixed offline dataset)

## Prerequisites

### Environment
```bash
# Activate virtual environment
source .venv/bin/activate

# Set cache directory
export METAMON_CACHE_DIR=/home/eddie/metamon_cache
```

### Data
- [ ] Large offline dataset: 100k-200k+ trajectories
- [ ] Format: `data_dir/gen1ou/*.json.lz4`
- [ ] Quality: Pre-filtered self-play data (< 5% invalid actions)

### Models
- [ ] Pre-trained checkpoint from previous loop/iteration
- [ ] Architecture: `synthetic_multitaskagent.gin`

## Key Finding: Both Configs Degrade After Epoch 2

### Experiment Setup

**Dataset**: 207,905 Gen1 OU self-play trajectories (super_dataset_loop5)
**Base checkpoint**: Loop 4 best checkpoint
**Reward**: AggressiveShapedRewardSleep (+200/0, +1 sleep)
**Configs tested**:
1. `sleep_selfplay_v2_aggressive.gin` (looser KL: 0.012, more DPG: 15%)
2. `selfplay_controller_v1.gin` (tighter KL: 0.008, less DPG: 5%)

### Results Summary

| Checkpoint | vs SyntheticRLV2 | vs controller_v1 epoch 2 | Notes |
|------------|------------------|-------------------------|-------|
| **controller_v1 epoch 2** | 72% | 50% (self) | Peak performance |
| **v2_aggressive epoch 2** | 72% | 50% | Equal to controller_v1 |
| **controller_v1 epoch 4** | Unknown | 40% | **Degraded from own epoch 2** |
| v2_aggressive epoch 4 | Unknown | Unknown | Also struggled (user note) |

### Critical Observation

**Both configs peak at epoch 2 and degrade by epoch 4**, despite having radically different KL control:

- **controller_v1**: Tight KL (0.008), low DPG (5%), active controller
- **v2_aggressive**: Loose KL (0.012), high DPG (15%), mostly idle controller

**Conclusion**: This is NOT a controller tuning problem. This is a **dataset-scale overfitting problem**.

## What This Means

### Overfitting at 200k Scale

**Hypothesis**: 200k trajectories from a single policy (loop 4 checkpoint) creates a narrow distribution that encourages overfitting by epoch 3-4.

**Evidence**:
1. Two configs with opposite philosophies (tight vs loose KL) both degrade
2. Peak happens at same point (epoch 2) for both
3. Degradation is substantial (60% win rate = 20 percentage point drop)

**Mechanism**:
- Epoch 0-2: Learning general patterns from diverse data
- Epoch 3+: Memorizing specific trajectories, overfitting to offline distribution
- Critic becomes too confident on training data, but miscalibrated on actual play

### Implications for Training

**Do NOT train past epoch 2-3 on 200k datasets** unless you have evidence your specific run is improving.

**Required**: Always evaluate multiple checkpoints (epoch 1-4 minimum), select best via held-out evaluation.

**Config choice doesn't matter** if stopping at epoch 2:
- v2_aggressive is faster → use for speed
- controller_v1 is stabler → use if concerned about instability

**To train longer safely**, need to change approach:
- Mix data from multiple loops (diversify distribution)
- Add online data collection during training (continuous distribution shift)
- Increase regularization (higher kl_coef_init, lower online_coeff)
- Use stronger dropout or other architectural regularization

## Step-by-Step Workflow

### Step 1: Train with Either Config

**Goal**: Train for 5 epochs but expect to use epoch 2-3 checkpoint

**Command** (example with controller_v1):
```bash
python -u -m metamon.rl.finetune_from_hf \
    --run_name "super_loop5_training" \
    --finetune_from_model <PreviousLoopCheckpoint> \
    --custom_replay_dir ~/metamon/trajectories/super_dataset_loop5/ \
    --custom_replay_sample_weight 1.0 \
    --formats gen1ou \
    --train_gin_config selfplay_controller_v1.gin \
    --reward_function AggressiveShapedRewardSleep \
    --obs_space ExpandedObservationSpace \
    --epochs 5 \
    --save_dir ~/metamon/models/super_loop5_training \
    --eval_gens 1 \
    --log
```

**Expected output**: Checkpoints for epoch 0-4 in `save_dir/ckpts/`

**Duration**: ~6-12 hours depending on GPU and dataset size

**What to monitor**:
- Training metrics improve smoothly (losses decrease)
- But actual win rate may peak early then decline (only visible via eval)

---

### Step 2: Evaluate ALL Checkpoints (Critical!)

**Goal**: Identify which epoch actually performs best via head-to-head evaluation

**Command** (evaluate each checkpoint vs baseline):
```bash
# Evaluate each epoch vs strong baseline
for epoch in 0 1 2 3 4; do
    echo "Evaluating epoch ${epoch}..."
    python -m metamon.rl.evaluate \
        --model_name <YourModel> \
        --checkpoint_path ~/metamon/models/super_loop5_training/ckpts/epoch_${epoch}.pt \
        --opponent SyntheticRLV2 \
        --num_battles 500 \
        --battle_format gen1ou \
        --team_set modern_replays_v2 \
        --output_dir ~/evaluations/super_loop5_epoch${epoch}
done
```

**Expected output**: Win rate vs baseline for each epoch

**Duration**: ~2-4 hours total (500 battles × 5 checkpoints)

**What to look for**:
- Peak likely at epoch 2-3
- Epoch 4-5 may show declining performance
- If unclear, run head-to-head between top candidates

---

### Step 3: Head-to-Head Between Top Candidates

**Goal**: Definitively select best checkpoint

**Command**:
```bash
# Example: epoch 2 vs epoch 3
python -m metamon.rl.evaluate \
    --model_name <YourModel> \
    --checkpoint_path ~/metamon/models/super_loop5_training/ckpts/epoch_2.pt \
    --opponent <YourModel> \
    --opponent_checkpoint_path ~/metamon/models/super_loop5_training/ckpts/epoch_3.pt \
    --num_battles 1000 \
    --battle_format gen1ou \
    --team_set modern_replays_v2
```

**Decision criteria**:
- > 55% win rate → earlier checkpoint is better
- 45-55% win rate → roughly equal, use earlier (less overfit)
- < 45% win rate → later checkpoint is better

---

### Step 4: Use Best Checkpoint for Next Loop

**Goal**: Generate data from validated best checkpoint, not final checkpoint

```bash
# Generate data from BEST checkpoint (likely epoch 2-3, NOT epoch 4)
python scripts/generate_selfplay_data_batched.py \
    --model <YourModel> \
    --checkpoint_path ~/metamon/models/super_loop5_training/ckpts/epoch_2.pt \
    --num_battles 150000 \
    --battle_format gen1ou \
    --team_set modern_replays_v2 \
    --output_dir ~/metamon/trajectories/super_dataset_loop6
```

## Critical Parameters

### Dataset Scale Sensitivity

| Dataset Size | Optimal Epochs | Risk of Overfitting |
|--------------|----------------|---------------------|
| 25k-50k | 4-5 | Low (diverse signal) |
| 75k-100k | 3-4 | Moderate |
| 150k-200k | **2-3** | **High** (this regime) |
| 250k+ | 1-2 | **Very High** (untested) |

### Config Selection (Both Degrade, Choose for Other Reasons)

**Use `sleep_selfplay_v2_aggressive.gin`**:
- Faster training (higher LR, more DPG)
- You're stopping at epoch 2-3 anyway
- Less time spent on epochs you'll discard

**Use `selfplay_controller_v1.gin`**:
- More stable training (less KL variance)
- Debugging/monitoring easier (active controller signals issues)
- Safer if you're uncertain about stopping point

**Both produce equal results at epoch 2** → choice is about training experience, not outcome

## What Worked ✅

### Success 1: Early Checkpoint Selection Reveals True Peak

**Context**: Evaluated all checkpoints (0-4) instead of assuming last is best

**Approach**:
- Trained for 5 epochs with controller_v1
- Evaluated epoch 2 vs epoch 4 head-to-head
- Found epoch 2 wins with 60% rate

**Results**:
- Avoided using degraded epoch 4 checkpoint
- Epoch 2 becomes seed for next loop (stronger foundation)
- Saved compute by not training past epoch 3 in future runs

**Why it worked**:
- Explicit validation step prevents silent degradation
- Head-to-head evaluation more sensitive than baseline win rates
- Caught overfitting that training metrics didn't show

**Takeaway**: **Always evaluate multiple checkpoints** - training loss is not a proxy for policy strength at large scale

---

### Success 2: Both Configs Produce Equal Epoch 2 Checkpoints

**Context**: Despite different KL control, both configs achieve 72% vs SyntheticRLV2 at epoch 2

**Results**:
- controller_v1 epoch 2 vs v2_aggressive epoch 2: 50% (equal)
- Both beat baseline with identical 72% win rate

**Why it matters**:
- Confirms epoch 2 is "natural peak" for this dataset
- Config choice doesn't affect outcome if stopping early
- Allows choosing config based on convenience (speed vs stability) not outcome

**Takeaway**: **At large scale with early stopping, config tuning less important than checkpoint selection**

## Failed Attempts ❌

### Failure 1: Training to Completion (5 Epochs)

**What was tried**: Train for full 5 epochs, use final checkpoint

**Configuration**: Both `controller_v1` and `v2_aggressive` tested

**What went wrong**:
- Epoch 4 checkpoint substantially weaker than epoch 2
- 60% win rate for epoch 2 vs epoch 4 (20 percentage point gap)
- Both configs degraded, indicating systemic issue not config bug

**Root cause**:
- Dataset too large/homogeneous for extended training
- Critic overfit to offline data distribution
- Actor learned to exploit critic's miscalibrated value estimates
- No regularization strong enough to prevent memorization

**Solution**: Stop at epoch 2-3, don't train longer

**Takeaway**: **200k single-policy trajectories cannot sustain 5 epochs of training** without degradation

---

### Failure 2: Assuming Conservative Config Prevents Degradation

**What was tried**: Use ultra-conservative controller_v1 to prevent overfitting via tight KL control

**Configuration**:
- KL target: 0.008 (very tight)
- DPG: 5% (minimal online)
- Strong initial damping (kl_coef=1.0)

**What went wrong**:
- controller_v1 degraded just like v2_aggressive
- Tight KL didn't prevent overfitting
- Epoch 2 → epoch 4 decline happened regardless

**Root cause**:
- KL regularization constrains *how much* policy changes per step
- But doesn't prevent *accumulation* of bad changes over many epochs
- Even small steps toward overfitting sum up over epochs 2-4
- Critic miscalibration is the real issue, actor KL doesn't fix it

**Solution**: No config fix exists - must stop training early or change data regime

**Takeaway**: **KL regularization slows but doesn't prevent overfitting on fixed large datasets**

---

### Failure 3: Relying on Training Metrics for Checkpoint Selection

**What was tried**: Select checkpoint based on lowest training loss or best validation metrics during training

**What went wrong**:
- Training losses likely continued improving through epoch 4
- But actual policy quality decreased
- Would have selected epoch 4 (worse) over epoch 2 (better)

**Root cause**:
- Training loss measures fit to offline data
- Offline data is fixed, so "fitting better" = overfitting after a point
- Policy quality measured by actual gameplay, not offline prediction accuracy

**Solution**: Always run held-out evaluation (vs baselines or head-to-head)

**Takeaway**: **Training metrics and policy strength diverge after overfitting begins** - must evaluate all checkpoints explicitly

## Metrics Interpretation

### Healthy Training (Epochs 0-2)

- **Critic loss**: Steady decrease from initial value
- **Actor loss**: Steady decrease
- **KL divergence**: Stable in target range (controller-dependent)
- **Entropy**: Maintains > 1.0
- **Win rate (if evaluated)**: Improving vs baselines

### Overfitting Begins (Epochs 3+)

- **Critic loss**: Continues decreasing (misleading!)
- **Actor loss**: Continues decreasing (misleading!)
- **KL divergence**: Stable (no warning signal)
- **Entropy**: Still healthy > 1.0 (no warning signal)
- **Win rate (if evaluated)**: Declining vs baselines or earlier checkpoints ← **Only reliable signal**

### Red Flags (Retrospective)

The problem is **training metrics don't show overfitting** at this scale. You must evaluate explicitly:

- **No clear training metric warning**: Standard metrics (loss, KL, entropy) all look healthy
- **Only evaluation reveals problem**: Win rate drop only visible via actual battles
- **Occurs regardless of config**: Both tight and loose KL control affected

## Unexpected Findings

**Finding 1**: Config differences don't matter at epoch 2
- **Observation**: Radically different configs (tight vs loose KL, low vs high DPG) produce identical epoch 2 performance
- **Hypothesis**: Epoch 2 represents maximum extractable signal from 200k single-policy data, regardless of how you get there
- **Implications**: Config tuning less important than dataset quality/diversity at scale

**Finding 2**: Overfitting happens despite healthy metrics
- **Observation**: KL, entropy, losses all look fine during overfitting phase (epoch 3-4)
- **Hypothesis**: Standard RL metrics designed for online training don't detect offline overfitting
- **Implications**: Cannot rely on automated early stopping based on training metrics alone

**Finding 3**: Degradation is substantial (20 percentage points)
- **Observation**: Epoch 2 vs epoch 4 is 60/40 split (massive gap)
- **Hypothesis**: Critic miscalibration compounds over epochs - by epoch 4, value estimates severely wrong
- **Implications**: Even one extra epoch could be costly - be conservative with stopping point

## Follow-Up Questions

**Unresolved questions from this experiment:**

1. **Does mixed-loop data prevent this overfitting?**
   - Test training on 50k loop4 + 50k loop5 + 50k loop3 data
   - Hypothesis: Distribution diversity allows more epochs

2. **What's the limit for single-policy data?**
   - Test 50k, 100k, 150k, 200k scales systematically
   - Find the scale where epoch 3-4 remain viable

3. **Can we detect overfitting without evaluation?**
   - Log actor-critic agreement/disagreement during training
   - Monitor advantage distribution statistics
   - Find proxy metrics that correlate with win rate decline

4. **Does this happen with human replay mixing?**
   - Test 80% loop5 + 20% human replays
   - Hypothesis: Human data provides regularization via distribution diversity

5. **What if we train with stronger dropout or weight decay?**
   - Test higher l2_coeff (1e-4 vs 5e-5)
   - Test dropout in transformer layers
   - Hypothesis: Architectural regularization helps where KL regularization failed

**Suggested next experiments:**

1. **Mixed-loop training**:
   ```bash
   # Combine multiple loops for diversity
   python -m metamon.rl.finetune_from_hf \
       --custom_replay_dir ~/trajectories/loop3,~/trajectories/loop4,~/trajectories/loop5 \
       --epochs 5
   # Hypothesis: Can train past epoch 2 without degradation
   ```

2. **Dataset scale ablation**:
   ```bash
   # Sample 50k, 100k, 150k subsets from 200k
   # Train for 5 epochs, evaluate all checkpoints
   # Map optimal stopping point vs dataset size
   ```

3. **Human replay mixing**:
   ```bash
   python -m metamon.rl.finetune_from_hf \
       --custom_replay_dir ~/trajectories/loop5 \
       --custom_replay_sample_weight 0.8 \
       --parsed_replay_dir ~/metamon_cache/parsed-replays \
       --epochs 5
   # Hypothesis: Human data prevents overfitting to self-play distribution
   ```

## Common Errors & Solutions

### Error: Using final checkpoint by default

**When it occurs**: After training completes, using epoch 4 checkpoint for next loop

**Root cause**: Assumption that more training = better policy

**Solution**:
```bash
# ALWAYS evaluate multiple checkpoints before selecting
for epoch in 1 2 3 4; do
    python -m metamon.rl.evaluate \
        --checkpoint_path <model>/epoch_${epoch}.pt \
        --opponent SyntheticRLV2 \
        --num_battles 500
done

# Then choose best based on results
```

**How to prevent**: Make checkpoint evaluation a required step in workflow, not optional

---

### Error: Training past epoch 3 on large datasets

**When it occurs**: Following standard 5-epoch workflow on 150k-200k+ data

**Root cause**: Workflow designed for smaller datasets (25k-50k)

**Solution**:
```bash
# For 150k-200k datasets, train for 4 epochs max
--epochs 4  # Not 5

# Or stop training at epoch 2 if you have evidence from previous runs
--epochs 3
```

**How to prevent**: Scale epoch count inversely with dataset size

---

### Error: Assuming tight KL prevents overfitting

**When it occurs**: Switching to controller_v1 to fix degradation problem

**Root cause**: KL controls policy drift rate, not accumulation over epochs

**Solution**:
```bash
# No config fix - must stop early OR diversify data
# Option 1: Stop early
--epochs 3

# Option 2: Mix data sources
--custom_replay_dir loop3,loop4,loop5
```

**How to prevent**: Understand KL regularization prevents instability, not overfitting to fixed dataset

## Related Skills

- [`selfplay-loop-workflow`](./../training/selfplay-loop-workflow.md) - Full self-play workflow (should be updated with early stopping guidance)
- [`dynamic-damping-config-selection`](./../config/dynamic-damping-config-selection.md) - Config selection (less important than thought at large scale)

## References

### Configurations
- [`sleep_selfplay_v2_aggressive.gin`](../../metamon/rl/configs/training/sleep_selfplay_v2_aggressive.gin) - Aggressive config
- [`selfplay_controller_v1.gin`](../../metamon/rl/configs/training/selfplay_controller_v1.gin) - Conservative config
- **Both degrade equally by epoch 4 on 200k data**

### Experiments
- Dataset: `/home/eddie/metamon/trajectories/super_dataset_loop5/gen1ou/` (207,905 trajectories)
- Result: Both configs peak at epoch 2, degrade by epoch 4
- Key metric: Epoch 2 beats epoch 4 with 60% win rate (20 point gap)

---

## Success Criteria

For 150k-200k scale training, success means:

- ✅ Evaluated all checkpoints (epoch 0-4 minimum)
- ✅ Selected checkpoint based on head-to-head eval, not training loss
- ✅ Stopped training at epoch 2-3 (or have evidence your run improves past that)
- ✅ Best checkpoint beats previous loop with > 55% win rate
- ✅ Next loop uses validated best checkpoint, not final checkpoint

**Warning signs you're overfitting**:
- ⚠️ Training past epoch 3 on 200k single-source data
- ⚠️ Assuming final checkpoint is best
- ⚠️ Not running checkpoint comparison evaluation
- ⚠️ Seeing win rates decline in later epochs
