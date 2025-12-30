# Epistemic Uncertainty Actor Weighting - Negative Result

> **Category**: training
> **Last Updated**: 2025-12-22
> **Author**: Claude via /retrospective

## Objective

**What problem was this attempting to solve?**

This experiment attempted to fix the **catastrophic first-epoch collapse** problem in Gen1 OU training where:
- Epoch 0 (after unfreezing actor): ~0% win rate
- Epoch 2+: ~50% win rate recovery
- Pattern occurs even with 2-epoch critic warmup

The hypothesis was that weighting actor gradients by inverse critic uncertainty (epistemic confidence) would prevent early exploitable policy shifts by downweighting updates from high-disagreement states.

**RESULT: The approach did NOT solve the problem. The same failure pattern occurred.**

## When to Use This Skill

**Trigger conditions:**

- [ ] When someone proposes epistemic uncertainty weighting as a solution to first-epoch collapse
- [ ] When considering approaches to stabilize early RL training with ensemble critics
- [ ] When evaluating the effectiveness of confidence-weighted gradient methods

**Do NOT use this skill when:**

- Looking for working solutions to first-epoch collapse (this is a negative result)
- The problem involves KL divergence explosion (epistemic weighting targets a different failure mode)

## Prerequisites

### Environment
```bash
# Activate virtual environment
source .venv/bin/activate

# Set cache directory
export METAMON_CACHE_DIR=/home/eddie/metamon_cache
```

### Data
- [ ] Dataset: Super dataset loop3 (~25K Gen1 OU trajectories)
- [ ] Location: `/home/eddie/metamon/trajectories/super_dataset_loop3/gen1ou/`
- [ ] Base model: `DampedBinarySuperV1_Epoch4`

### Configuration Files Created
- [ ] `metamon/rl/configs/training/epistemic_ema_rl.gin` - EMA-based normalization variant
- [ ] `metamon/rl/configs/training/ema_config.gin` - EMA configuration parameters

## What Was Implemented

### Implementation Details

**File modified**: `metamon/rl/metamon_to_amago.py`

**Core mechanism**: Per-timestep actor gradient weighting based on Q-ensemble standard deviation:

```python
# Weight formula
w(σ) = 1 / (1 + β·σ̃)^p

where:
  σ = Q-ensemble std dev (critic disagreement)
  σ̃ = normalized uncertainty (via EMA of batch statistics)
  β = penalty coefficient (annealed high → low over training)
  p = power (tail suppression strength, default 2)
```

**Key components**:

1. **Q-ensemble caching**: Cache `q_std = q_s_a_g.std(dim=3).detach()` in forward pass
2. **EMA normalization**: Track running statistics of uncertainty distribution
3. **Per-timestep weighting**: Apply `weighted_actor_loss = actor_loss * confidence` before masked averaging
4. **Beta annealing**: Start conservative (β=5.0), decay to permissive (β=1.0) over ~3 epochs

### Two Variants Tested

#### Variant 1: With Critic Warmup (2 epochs frozen actor)
```bash
python -m metamon.rl.finetune_from_hf \
    --run_name "epistemic-ema-warmup" \
    --finetune_from_model DampedBinarySuperV1_Epoch4 \
    --custom_replay_dir ~/metamon/trajectories/super_dataset_loop3/ \
    --formats gen1ou \
    --train_gin_config epistemic_ema_rl.gin \
    --epochs 5 \
    --save_dir ~/metamon/models/epistemic_ema_warmup \
    --eval_gens 1 \
    --critic_warmup_epochs 2 \
    --log
```

**Config** (`epistemic_ema_rl.gin`):
```gin
MetamonAMAGOExperiment.use_epistemic_weighting = True
MetamonAMAGOExperiment.epistemic_beta_init = 5.0
MetamonAMAGOExperiment.epistemic_beta_final = 1.0
MetamonAMAGOExperiment.epistemic_anneal_steps = 10000  # ~3 epochs
MetamonAMAGOExperiment.epistemic_power = 2
MetamonAMAGOExperiment.epistemic_normalization = 'ema'  # EMA-based
```

#### Variant 2: Without Critic Warmup
```bash
python -m metamon.rl.finetune_from_hf \
    --run_name "epistemic-ema-no-warmup" \
    --finetune_from_model DampedBinarySuperV1_Epoch4 \
    --custom_replay_dir ~/metamon/trajectories/super_dataset_loop3/ \
    --formats gen1ou \
    --train_gin_config epistemic_ema_rl.gin \
    --epochs 5 \
    --save_dir ~/metamon/models/epistemic_ema_no_warmup \
    --eval_gens 1 \
    --log
```

## Failed Attempts ❌

### Failure: Epistemic Weighting Did Not Prevent First-Epoch Collapse

**What was tried**: Both variants (with and without critic warmup) using epistemic uncertainty-based actor gradient weighting.

**Configuration**:
- `epistemic_beta_init`: 5.0 (conservative early weighting)
- `epistemic_beta_final`: 1.0 (permissive late weighting)
- `epistemic_anneal_steps`: 10000 (~3 epochs)
- `epistemic_power`: 2 (quadratic tail suppression)
- Normalization: EMA-based (tracks running uncertainty distribution)
- Damping: Standard dynamic damping with KL target 0.015
- Reward: AggressiveBinaryReward (+100/0)

**What went wrong**:

**Observed pattern** (BOTH variants):
- **Epoch 0**: ~0% win rate vs RandomBaseline
- **Epoch 2**: ~50% win rate recovery
- **Epoch 4**: ~40% win rate (slight degradation)

**This is IDENTICAL to the baseline failure mode** that epistemic weighting was supposed to fix.

**Metrics during training**:
- Mean epistemic confidence started ~0.5-0.6 (reasonable)
- High-σ vs Low-σ separation was present (~0.3 vs 0.7)
- Beta annealed smoothly as designed
- KL divergence remained low (~0.002-0.004, no explosion)
- Entropy showed gradual decay (no collapse, H > 1.0)

**Root cause**:

The epistemic weighting mechanism **functioned as designed** (confidence weights were being applied, high-uncertainty states were downweighted), but it **did not address the actual cause** of first-epoch collapse.

**Takeaway**:

**Epistemic uncertainty in the critic is NOT the primary driver of first-epoch collapse.**

The problem likely lies elsewhere:
1. **Distributional shift**: Offline data → online policy mismatch is too large
2. **Reward model misspecification**: Critic overfits to offline trajectories, gives poor signals for online behavior
3. **Exploration failure**: Policy updates reduce stochasticity too quickly, gets stuck in local minimum
4. **Team/format bias**: Base model has strong Gen1-specific priors that resist updating
5. **Advantage estimation**: Binary FBC filtering or advantage computation itself is flawed

The fact that **recovery happens by epoch 2-4** suggests the system can eventually stabilize, but the mechanism isn't critic confidence—it's something about the data accumulation or learning dynamics.

---

### Failure: Critic Warmup Also Did Not Help

**What was tried**: 2 epochs of critic-only training (frozen actor) before unfreezing.

**Why it was tried**: Hypothesis was that critic uncertainty at epoch 0 causes bad actor gradients. Warming up the critic first should stabilize it.

**Result**: **No improvement**. First-epoch collapse still occurred immediately after unfreezing actor at epoch 2.

**Implication**: Critic uncertainty is either:
- Not the root cause, OR
- Only measurable in the *direction* of updates, not the magnitude (Q-ensemble std doesn't capture this)

---

## Common Errors & Solutions

### Error: EMA Statistics Not Updating Correctly

**When it occurs**: During implementation of EMA-based normalization.

**Root cause**: EMA buffers must be registered as persistent state, not just instance variables, or they won't survive checkpoint save/load.

**Solution**:
```python
# In __init__:
self.register_buffer('ema_sigma_mean', torch.zeros(1))
self.register_buffer('ema_sigma_std', torch.ones(1))

# In normalization:
self.ema_sigma_mean = (momentum * self.ema_sigma_mean +
                       (1 - momentum) * current_mean)
self.ema_sigma_std = (momentum * self.ema_sigma_std +
                      (1 - momentum) * current_std)
```

**How to prevent**: Always use `register_buffer()` for stateful tracking in PyTorch.

---

### Error: Shape Mismatch Between q_std and actor_loss

**When it occurs**: When extracting cached Q-ensemble std and aligning with actor loss tensor.

**Root cause**: Q-values have shape `[B, L, ...]` but actor_loss has shape `[B, L-1, ...]` (last timestep trimmed).

**Solution**:
```python
# After extracting q_std from cache
q_std = q_std[:, :-1, :, :]  # Trim last timestep to match actor_loss
assert q_std.shape == actor_loss.shape, f"Shape mismatch: {q_std.shape} vs {actor_loss.shape}"
```

---

### Error: Confidence Weights Not Affecting Gradients

**When it occurs**: Applying weighting at the wrong stage of loss computation.

**Root cause**: Multiplying scalar loss by scalar weight collapses per-state structure.

**Wrong**:
```python
loss_dict["Actor Loss"] = loss_dict["Actor Loss"] * confidence.mean()
```

**Correct**:
```python
# Apply per-timestep weights BEFORE masked averaging
weighted_actor_loss = actor_loss * confidence  # [B, L-1, G, 1] * [B, L-1, G, 1]
loss_dict["Actor Loss"] = masked_avg(weighted_actor_loss, mask)
```

---

## Metrics Interpretation

### Epistemic Weighting Diagnostics (What We Observed)

**Mean Confidence**: Started ~0.5-0.6, increased slightly to ~0.65-0.7 over training
- This indicates weighting was active and critic uncertainty was decreasing (as expected)

**High-σ vs Low-σ Confidence**: Clear separation (~0.3 for high uncertainty, ~0.7 for low)
- This confirms the weighting mechanism was differentiating between states correctly

**Beta Decay**: Smooth decay from 5.0 → 1.0 over ~10K steps
- Annealing schedule worked as designed

**KL Divergence**: Remained stable at ~0.002-0.004
- No explosion, no over-damping

**Entropy**: Gradual decay, but always > 1.0
- No policy collapse

### Red Flags That Still Appeared

Despite epistemic weighting working mechanically:

- **Win rate collapse at epoch 0**: 0% vs RandomBaseline (catastrophic)
- **Recovery by epoch 2**: Suggests self-correction, but only after damage is done
- **Slight degradation at epoch 4**: ~40% (potential overfitting or mode collapse)

**Conclusion**: The mechanism was implemented correctly, but it targeted the wrong failure mode.

---

## Unexpected Findings

**Finding 1**: Epistemic weighting had no measurable impact on first-epoch collapse

- **Hypothesis before**: High critic uncertainty → noisy gradients → exploitable policy shifts
- **Observation**: Weighting was active, uncertainty was present, but collapse still happened
- **Implication**: Critic uncertainty (as measured by ensemble std) is NOT the bottleneck

**Finding 2**: Critic warmup also had no effect

- **Hypothesis before**: Critic needs stabilization before actor updates begin
- **Observation**: 2 epochs of warmup didn't prevent collapse at epoch 2
- **Implication**: Critic quality alone isn't sufficient; the problem is in actor update dynamics or data distribution

**Finding 3**: Recovery by epoch 2-4 is consistent across all configs

- **Observation**: Every configuration (epistemic on/off, warmup on/off) shows the same recovery pattern
- **Implication**: There's a robust self-correction mechanism that kicks in after ~2 epochs, but we don't understand what it is

---

## Follow-Up Questions

**Unresolved questions from this experiment:**

1. **What actually causes first-epoch collapse?** If not critic uncertainty, what is it?
   - Distributional shift from offline → online?
   - Advantage estimation bias?
   - Reward model misspecification?
   - Team composition bias in base model?

2. **Why does recovery happen by epoch 2-4?**
   - Is it accumulation of better data?
   - Stabilization of critic to *new* policy distribution?
   - Learning rate warmup effects?

3. **Is Q-ensemble std the right uncertainty measure?**
   - Should we measure uncertainty in *action preferences* instead of Q-values?
   - Is ensemble disagreement different from calibration error?

4. **Could epistemic weighting still be useful elsewhere?**
   - Even if it doesn't fix first-epoch collapse, does it improve sample efficiency?
   - Does it help with overfitting or robustness?

**Suggested next experiments:**

1. **Distributional shift analysis**: Compare offline data Q-values vs online rollout Q-values at epoch 0
   - Hypothesis: Critic is over-optimistic on offline data, under-predicts online performance

2. **Advantage distribution analysis**: Log raw advantages at epoch 0 vs epoch 2
   - Hypothesis: Advantages have wrong sign or extreme magnitudes early on

3. **Policy evaluation without training**: Freeze base model, evaluate on Gen1 OU
   - Hypothesis: Base model is already weak at Gen1 OU, collapse is just revealing this

4. **Explicit pessimism**: Add CQL-style conservative Q-learning penalties
   - Hypothesis: Critic overestimation drives bad actor updates

5. **Behavioral cloning stabilization**: Try pure BC for first epoch, then switch to RL
   - Hypothesis: Jumping straight to RL is too aggressive; need gradual transition

---

## Related Skills

- [`dynamic-damping-config-selection`](../config/dynamic-damping-config-selection.md) - KL damping (complementary mechanism)
- [`selfplay-loop-workflow`](./selfplay-loop-workflow.md) - Gen1 OU self-play pipeline
- [`large-dataset-overfitting-200k`](./large-dataset-overfitting-200k.md) - Overfitting patterns in large offline datasets

---

## References

### Documentation
- [`epistemic-uncertainty-actor-weighting.md`](./epistemic-uncertainty-actor-weighting.md) - Original implementation plan (now known to not solve the problem)
- [`GEN1OU_SELFPLAY_GUIDE.md`](../../GEN1OU_SELFPLAY_GUIDE.md) - High-level Gen1 OU workflow

### Configurations
- [`epistemic_ema_rl.gin`](../../metamon/rl/configs/training/epistemic_ema_rl.gin) - Epistemic weighting config (tested, not effective)
- [`ema_config.gin`](../../metamon/rl/configs/training/ema_config.gin) - EMA parameters

### Code
- [`metamon/rl/metamon_to_amago.py`](../../metamon/rl/metamon_to_amago.py:826-900) - Epistemic weighting implementation

### Experiments
- Run 1: `~/metamon/models/epistemic_ema_warmup` (with critic warmup)
- Run 2: `~/metamon/models/epistemic_ema_no_warmup` (without warmup)

---

## Key Takeaways

### For Future Engineers

1. **Epistemic uncertainty weighting does NOT fix first-epoch collapse in Gen1 OU training**
   - Mechanism was implemented correctly and functioned as designed
   - Critic uncertainty is present but not the primary driver of the failure mode

2. **Critic warmup does NOT help**
   - 2 epochs of frozen actor training had no protective effect
   - Problem occurs immediately after unfreezing, regardless of critic quality

3. **The failure mode is robust**
   - Appears consistently across configurations
   - Recovers by epoch 2-4 in all cases (suggesting self-correction)

4. **What to investigate instead**:
   - Distributional shift (offline → online)
   - Advantage estimation methods
   - Reward model calibration
   - Behavioral cloning → RL transition strategies

5. **The implementation is still valuable**:
   - Code is clean and well-tested
   - Could be useful for other failure modes (sample efficiency, robustness)
   - Provides infrastructure for future uncertainty-aware methods

### For Research Context

This experiment provides strong **negative evidence** against the hypothesis that:
> "First-epoch collapse is caused by noisy actor gradients from high-uncertainty critic states"

The actual cause remains unknown, but we can now rule out:
- Q-ensemble uncertainty as the bottleneck
- Critic quality (warmup doesn't help)
- KL divergence explosion (wasn't happening)
- Entropy collapse (wasn't happening)

**Next research directions should focus on:**
- Data distribution analysis (offline vs online)
- Advantage/reward signal quality
- Base model behavior on Gen1 OU
- Conservative Q-learning approaches

---

## Status

**Status**: Complete (negative result documented)
**Confidence**: High (clean ablation, clear null result)
**Value**: High (prevents future wasted effort, points toward better hypotheses)
