# Epistemic Uncertainty-Aware Actor Updates

**Category**: Training Infrastructure
**Status**: Designed (Not Yet Implemented)
**Priority**: Critical - Addresses catastrophic first-epoch failure mode
**Complexity**: Medium (90 lines of code, 1 file modified)

---

## Problem Statement

### The Failure Mode

Across all Gen1 OU training runs, the model exhibits **catastrophic first-epoch collapse**:
- **Before training**: 50% win rate (frozen actor, random baseline)
- **After epoch 0**: 0% win rate (actor unfrozen, catastrophic failure)
- **After epoch 2+**: ~50% win rate (gradual recovery)

**Even with critic-only warmup** (2 epochs of frozen actor), the failure persists immediately upon unfrozen.

### Root Cause Analysis (From External Colleague)

Analysis of training metrics across runs revealed:

1. **KL divergence is not exploding** (~0.002-0.004, conservative by PPO standards)
2. **Entropy is not collapsing** (slow decay, no sudden cliff)
3. **Policy is stochastic** (activation probs show healthy variance)
4. **BUT: Q-ensemble uncertainty remains high at epoch-0** (~0.12-0.14 std dev)

**Key insight**: In imperfect-information games like Gen1 OU, **small KL can still cause catastrophic exploitability if it shifts action frequencies in critical information sets**.

The problem is **directional damage, not step-size damage**:
- Actor is being updated using advantages from a **critic that internally disagrees** (high ensemble std)
- Binary FBC filter lets ~30% of actions through regardless of critic uncertainty
- Actor confidently follows noisy gradients → exploitable policy

### Why Existing Damping Fails

**Ataraxos-style KL damping** (implemented in `metamon/rl/dynamic_damping.py`):
- ✅ Limits **magnitude** of policy updates (KL constraint)
- ✅ Prevents policy collapse (reverse-KL regularization)
- ❌ Does **NOT** prevent **directionally wrong** updates from uncertain critic states

**Quote from analysis**: *"Direction matters! Ataraxos-style damping is incomplete without epistemic awareness."*

---

## Solution: Epistemic-Aware Actor Weighting

### Core Idea

Weight actor gradients by **inverse critic uncertainty** at the per-timestep level:

```
L_actor = E[ w(σ) · A(s,a) · log π(a|s) ]

where: w(σ) = 1 / (1 + β·σ̃)^p
```

**Effect**:
- High-uncertainty states (critic ensemble disagrees) → low weight → small gradient
- Low-uncertainty states (critic ensemble agrees) → high weight → full gradient
- Critic stabilizes over training → uncertainty drops → weights increase

### Why This Works

1. **Preserves signal where critic is confident** (low σ → w ≈ 1.0)
2. **Suppresses noise where critic disagrees** (high σ → w ≈ 0.2-0.5)
3. **Automatically anneals** as critic stabilizes (σ↓ → w↑ over training)
4. **Complementary to KL damping** (direction safety + magnitude safety)

---

## Implementation Plan

### Overview

**Files modified**: `metamon/rl/metamon_to_amago.py` only
**Lines of code**: ~90 lines total
**Complexity**: Medium (requires understanding AMAGO loss flow)

### Critical Background: AMAGO Loss Flow

Understanding the loss computation pipeline is **essential** for correct implementation:

```
1. Agent.forward(batch)
   → Returns: (critic_loss, actor_loss)
   → Shapes: [B, L-1, num_critics, G, 1], [B, L-1, G, 1]
   → These are PER-TIMESTEP tensors, NOT scalars

2. Experiment.compute_loss(batch)
   → Calls Agent.forward()
   → Applies masking (for padding)
   → Does masked_avg() → produces SCALARS
   → Returns: {"Actor Loss": scalar, "Critic Loss": scalar}

3. Experiment.train_step(batch)
   → Calls compute_loss()
   → Computes: loss = l["Actor Loss"] + 10.0 * l["Critic Loss"]
   → Calls: backward(loss)
```

**Key insight**: We must apply per-timestep weighting **BEFORE** `masked_avg()` in step 2, otherwise we collapse all structure and the weighting becomes useless.

### Actor Loss Components

From `amago/agent.py:569-611`:

```python
actor_loss = torch.zeros((B, L - 1, G, 1))

if online_coeff > 0:
    # DPG term: -Q(s, a ~ π)
    actor_loss += online_coeff * -Q_values

if offline_coeff > 0:
    # BC term: -f(A(s,a)) * log π(a|s)
    actor_loss += offline_coeff * -(filter * log_prob)
```

**Both terms are negative** (minimization objectives). Multiplying by `w ∈ (0,1)` reduces magnitude → smaller updates. This is correct.

**Current metamon configs**: `offline_coeff=0.75`, `online_coeff=0.25`

---

## Step-by-Step Implementation

### Step 1: Cache Q-Ensemble in Forward Pass

**File**: `metamon/rl/metamon_to_amago.py`
**Location**: In `MetamonMultiTaskAgent.forward()` method, around line 635

**Add before final return**:

```python
# Cache Q-ensemble for epistemic weighting
# IMPORTANT: Cache BEFORE .mean(dim=3) collapses critic dimension
# q_s_a_g shape: [1, B, L, C, G, 1] where C = num_critics
if not hasattr(self, 'cached_epistemic'):
    self.cached_epistemic = {}

# Detach immediately to save memory
q_std = q_s_a_g.std(dim=3).detach()  # [1, B, L, G, 1]
self.cached_epistemic['q_std'] = q_std
```

**Why detach**: We don't want gradients flowing through the uncertainty calculation back to the critic. Uncertainty is a stop-gradient "meta-signal".

**Why cache std, not ensemble**: Full ensemble is `[1, B, L, C, G, 1]` which is memory-expensive. We only need the std dev across critics.

---

### Step 2: Override compute_loss() to Apply Per-Timestep Weighting

**File**: `metamon/rl/metamon_to_amago.py`
**Location**: Replace the existing `MetamonAMAGOExperiment.compute_loss()` method (lines ~826-869)

**New implementation**:

```python
def compute_loss(self, batch: Batch, log_step: bool) -> dict:
    """
    Compute actor and critic losses with optional epistemic weighting.

    Overrides parent to apply per-timestep confidence weights BEFORE
    masked averaging (critical for epistemic weighting to affect gradients).
    """

    # Call Agent.forward() to get per-timestep losses
    critic_loss, actor_loss = self.policy_aclr(batch, log_step=log_step)
    # critic_loss: [B, L-1, num_critics, G, 1]
    # actor_loss: [B, L-1, G, 1]

    # Apply epistemic weighting to actor_loss if enabled
    if self.use_epistemic_weighting:
        actor_loss = self._apply_epistemic_weighting(
            actor_loss, batch, log_step
        )

    # Apply masking (copied from parent Experiment.compute_loss)
    # This removes padding timesteps before averaging
    mask = batch["mask"][:, 1:]  # [B, L-1]
    actor_state_mask = mask.unsqueeze(-1).unsqueeze(-1)  # [B, L-1, 1, 1]
    critic_state_mask = actor_state_mask.unsqueeze(2)  # [B, L-1, 1, 1, 1]

    # Compute scalar losses via masked averaging
    from amago import utils
    masked_actor_loss = utils.masked_avg(actor_loss, actor_state_mask)
    masked_critic_loss = utils.masked_avg(critic_loss, critic_state_mask)

    loss_dict = {
        "Critic Loss": masked_critic_loss,
        "Actor Loss": masked_actor_loss,
    }

    # Add KL regularization if dynamic damping enabled
    # (Independent mechanism - both can be active simultaneously)
    if self.dd_state is not None and self.dd_config.enabled:
        kl_loss, kl_metrics = self._compute_kl_loss(batch, log_step)
        loss_dict["Actor Loss"] = loss_dict["Actor Loss"] + kl_loss
        loss_dict.update(kl_metrics)

    # Clean up cache to prevent stale data
    if hasattr(self.policy, 'cached_epistemic'):
        self.policy.cached_epistemic.clear()

    return loss_dict
```

**Why this structure**:
1. Call `Agent.forward()` directly to get per-timestep tensors
2. Apply weighting **before** `masked_avg()`
3. Do masking ourselves (copied from parent)
4. Return scalar losses for backward pass
5. KL damping adds to the scalar (independent of epistemic weighting)

---

### Step 3: Implement _apply_epistemic_weighting()

**Add new method to MetamonAMAGOExperiment**:

```python
def _apply_epistemic_weighting(
    self,
    actor_loss: torch.Tensor,  # [B, L-1, G, 1]
    batch: Batch,
    log_step: bool
) -> torch.Tensor:
    """
    Apply per-timestep confidence weighting based on critic uncertainty.

    Weights actor gradients by inverse uncertainty: w = 1/(1 + β·σ̃)^p
    where σ̃ is normalized critic ensemble std dev.
    """

    # Extract Q-ensemble std from cache
    q_std = self.policy.cached_epistemic['q_std']  # [1, B, L, G, 1]

    # Shape alignment: trim to match actor_loss [B, L-1, G, 1]
    q_std = q_std.squeeze(0)[:, :-1, :, :]  # [B, L-1, G, 1]

    # Defensive shape check
    assert q_std.shape == actor_loss.shape, \
        f"Shape mismatch: q_std {q_std.shape} vs actor_loss {actor_loss.shape}"

    # Get mask for valid (non-padding) timesteps
    mask = batch["mask"][:, 1:]  # [B, L-1]
    state_mask = mask.unsqueeze(-1).unsqueeze(-1)  # [B, L-1, 1, 1]

    # Normalize uncertainty (using only valid timesteps)
    sigma_norm = self._normalize_uncertainty(q_std, state_mask)  # [B, L-1, G, 1]

    # Compute confidence weights: w = 1 / (1 + β·σ̃)^p
    beta = self._get_current_beta()
    confidence = 1.0 / (1.0 + beta * sigma_norm).pow(self.epistemic_power)

    # Ensure stop-gradient (no backprop through uncertainty)
    confidence = confidence.detach()

    # Apply per-timestep weighting
    weighted_actor_loss = actor_loss * confidence

    # Log metrics
    if log_step:
        self._log_epistemic_metrics(sigma_norm, confidence, state_mask)

    return weighted_actor_loss
```

**Critical details**:
- `q_std[:, :-1, :, :]` trims last timestep (actor loss is L-1, Q-values are L)
- Masking prevents padding from biasing normalization stats
- `confidence.detach()` ensures no gradients flow to critic through weighting
- Power `p` (default 2) controls tail suppression strength

---

### Step 4: Implement Uncertainty Normalization

**Add helper method**:

```python
def _normalize_uncertainty(
    self,
    q_std: torch.Tensor,      # [B, L-1, G, 1]
    mask: torch.Tensor        # [B, L-1, 1, 1]
) -> torch.Tensor:
    """
    Normalize uncertainty to prevent scale drift across training loops.

    Uses per-gamma median normalization for stability.
    """

    # Apply mask to exclude padding
    masked_std = q_std * mask  # Zero out padding

    # Compute median per-gamma (more stable than global median)
    # Shape: [G, 1]
    valid_stds = masked_std[mask.squeeze(-1).squeeze(-1) > 0]  # Flatten valid

    if valid_stds.numel() == 0:
        # Fallback: all padding (shouldn't happen)
        return torch.ones_like(q_std)

    # Simple version: global median (can upgrade to per-gamma later)
    median = valid_stds.median()

    # Ratio normalization: σ̃ = σ / median(σ)
    sigma_norm = q_std / (median + 1e-8)

    # Clamp to prevent extreme outliers
    sigma_norm = sigma_norm.clamp(0, 10)

    return sigma_norm
```

**Why median normalization**:
- **Robust to outliers** (unlike mean)
- **Scale-invariant** (handles reward scale changes)
- **Stable across loops** (median shifts slowly as critic improves)

**Upgrade path**: For better stability, use CDF-based normalization with EMA histogram. See "Advanced Normalization" section below.

---

### Step 5: Implement Beta Annealing Schedule

**Add helper method**:

```python
def _get_current_beta(self) -> float:
    """
    Anneal beta from high (conservative) to low (permissive) over training.

    Schedule: β(t) = β_final + (β_init - β_final) * (1 - progress)^α

    This yields:
    - step 0: β = β_init (high penalty, ~5-10)
    - end: β = β_final (low penalty, ~0.5-1.0)
    """

    if not hasattr(self, 'epistemic_step'):
        self.epistemic_step = 0

    # Compute training progress [0, 1]
    progress = min(1.0, self.epistemic_step / self.epistemic_anneal_steps)

    # Power-law decay: high → low
    beta = self.epistemic_beta_final + \
           (self.epistemic_beta_init - self.epistemic_beta_final) * \
           (1.0 - progress) ** self.epistemic_anneal_power

    self.epistemic_step += 1
    return beta
```

**Why anneal high → low**:
- **Early training**: Critic is uncertain → need strong penalty (β=5-10)
- **Late training**: Critic is confident → can trust more (β=0.5-1.0)
- Power-law decay (α=0.5) gives smooth transition

**CRITICAL BUG TO AVOID**: Do NOT use `beta = beta_init * progress^α`. This starts at 0 and increases, which is backwards.

---

### Step 6: Implement Logging

**Add helper method**:

```python
def _log_epistemic_metrics(
    self,
    sigma_norm: torch.Tensor,   # [B, L-1, G, 1]
    confidence: torch.Tensor,   # [B, L-1, G, 1]
    mask: torch.Tensor          # [B, L-1, 1, 1]
) -> None:
    """Log epistemic weighting diagnostics."""

    with torch.no_grad():
        # Only consider valid (non-padding) timesteps
        valid_mask = mask.squeeze(-1).squeeze(-1) > 0
        sigma_valid = sigma_norm[valid_mask]
        conf_valid = confidence[valid_mask]

        # Basic stats
        self.logger.log({
            "Epistemic/Mean Uncertainty": sigma_valid.mean().item(),
            "Epistemic/Mean Confidence": conf_valid.mean().item(),
            "Epistemic/Beta": self._get_current_beta(),
        })

        # High vs low uncertainty impact
        median_sigma = sigma_valid.median()
        high_unc_mask = sigma_valid > median_sigma
        low_unc_mask = ~high_unc_mask

        self.logger.log({
            "Epistemic/Confidence (High σ)": conf_valid[high_unc_mask].mean().item(),
            "Epistemic/Confidence (Low σ)": conf_valid[low_unc_mask].mean().item(),
        })

        # Effective learning mass (what fraction of gradients we're allowing)
        effective_mass = conf_valid.mean().item()
        self.logger.log({"Epistemic/Effective Mass": effective_mass})
```

**What to monitor**:
- **Mean Confidence**: Should start ~0.3-0.5, increase to ~0.8-0.9 as critic stabilizes
- **High σ vs Low σ**: Should show clear separation (e.g., 0.3 vs 0.8)
- **Effective Mass**: Should be > 0.5 (otherwise learning is too suppressed)
- **Beta**: Should decay smoothly from init to final over ~3 epochs

---

### Step 7: Add Configuration Parameters

**Add to MetamonAMAGOExperiment.__init__()**:

```python
# Epistemic weighting configuration
self.use_epistemic_weighting = False  # gin-configurable flag
self.epistemic_beta_init = 5.0        # high initial penalty
self.epistemic_beta_final = 1.0       # low final penalty
self.epistemic_anneal_steps = 10000   # ~3 epochs of training steps
self.epistemic_anneal_power = 0.5     # sqrt decay
self.epistemic_power = 2              # w = 1/(1 + β·σ)^p (tail suppression)
```

**Tuning guide**:
- `beta_init`: Start at 5-10 (conservative). Lower if learning is too slow.
- `beta_final`: End at 0.5-1.0 (permissive). Raise if instability returns.
- `anneal_steps`: ~3 epochs worth of gradient steps (e.g., 10k steps for 500 replays/epoch)
- `power`: Use 2 for stronger tail suppression, 1 for gentler

---

### Step 8: Create Gin Config

**New file**: `metamon/rl/configs/training/epistemic_aware_rl.gin`

```gin
# Epistemic uncertainty-aware actor updates
# Base config: inherits from selfplay_damped_aggressive.gin

include 'metamon/rl/configs/training/selfplay_damped_aggressive.gin'

# Enable epistemic weighting
MetamonAMAGOExperiment.use_epistemic_weighting = True

# Beta annealing schedule
MetamonAMAGOExperiment.epistemic_beta_init = 5.0      # conservative start
MetamonAMAGOExperiment.epistemic_beta_final = 1.0     # permissive end
MetamonAMAGOExperiment.epistemic_anneal_steps = 10000 # ~3 epochs
MetamonAMAGOExperiment.epistemic_anneal_power = 0.5   # sqrt decay

# Weighting function parameters
MetamonAMAGOExperiment.epistemic_power = 2  # w = 1/(1 + β·σ)^2

# Keep existing KL damping (complementary mechanism)
# No changes to dynamic_damping parameters
```

---

## Testing & Validation

### Pre-Flight Check (5 Minutes)

Before running full training, add debug prints to `_apply_epistemic_weighting()`:

```python
# At the end of _apply_epistemic_weighting():
if log_step and self.epistemic_step < 5:  # Only first few steps
    print(f"\n=== Epistemic Weighting Debug ===")
    print(f"actor_loss shape: {actor_loss.shape}")
    print(f"q_std shape: {q_std.shape}")
    print(f"confidence range: [{confidence.min():.3f}, {confidence.max():.3f}]")
    print(f"confidence mean: {confidence.mean():.3f}")

    # Check high vs low uncertainty separation
    valid = state_mask.squeeze(-1).squeeze(-1) > 0
    sigma_valid = sigma_norm[valid]
    conf_valid = confidence[valid]
    high_mask = sigma_valid > sigma_valid.median()

    print(f"High-σ confidence: {conf_valid[high_mask].mean():.3f}")
    print(f"Low-σ confidence: {conf_valid[~high_mask].mean():.3f}")
    print(f"Beta: {self._get_current_beta():.3f}")
    print("=" * 40)
```

**Expected output**:
```
=== Epistemic Weighting Debug ===
actor_loss shape: torch.Size([32, 63, 3, 1])
q_std shape: torch.Size([32, 63, 3, 1])
confidence range: [0.167, 0.952]
confidence mean: 0.523
High-σ confidence: 0.312
Low-σ confidence: 0.734
Beta: 5.000
========================================
```

**Red flags**:
- Shapes don't match → indexing error
- Confidence all ~1.0 → beta too low or normalization broken
- Confidence all ~0.1 → beta too high or σ not normalized
- High-σ == Low-σ → no separation, normalization not working

---

### Ablation Experiments

**Setup**: Gen1 OU, 2 epochs, eval after each epoch

**Baseline (current failure)**:
```bash
python -m metamon.rl.finetune_from_hf \
    --run_name "baseline-no-epistemic" \
    --finetune_from_model DampedBinarySuperV1_Epoch4 \
    --custom_replay_dir ~/metamon/trajectories/super_dataset_loop3/ \
    --formats gen1ou \
    --train_gin_config selfplay_damped_aggressive.gin \
    --epochs 2 \
    --save_dir ~/metamon/models/ablation_baseline \
    --eval_gens 1 \
    --log
```

**Expected**: Epoch-0 win rate ~0%, Epoch-1 ~50%

**Treatment (epistemic weighting)**:
```bash
python -m metamon.rl.finetune_from_hf \
    --run_name "epistemic-weighted" \
    --finetune_from_model DampedBinarySuperV1_Epoch4 \
    --custom_replay_dir ~/metamon/trajectories/super_dataset_loop3/ \
    --formats gen1ou \
    --train_gin_config epistemic_aware_rl.gin \
    --epochs 2 \
    --save_dir ~/metamon/models/ablation_epistemic \
    --eval_gens 1 \
    --log
```

**Expected**: Epoch-0 win rate ~40-50% (matches frozen baseline), Epoch-1 ~50-55%

---

### Success Metrics

**Primary metric**: Epoch-0 win rate vs RandomBaseline
- ❌ Failure: < 30% (catastrophic collapse)
- ⚠️ Partial: 30-40% (some collapse)
- ✅ Success: > 40% (maintains baseline performance)

**Secondary metrics**:
- Policy entropy: Should NOT collapse (H > 1.0 throughout)
- KL divergence: Should remain < 0.02 (not exploding)
- Q-ensemble std: Should decrease over training (critic stabilizing)
- Confidence weights: High-σ should be ~0.3-0.5, Low-σ should be ~0.7-0.9

**Diagnostic metric** (most direct test):
- **Gradient mass from top-σ decile**: Should drop sharply with epistemic weighting
  - Baseline: ~15-20% of gradient from top 10% uncertainty states
  - Treatment: < 5% of gradient from top 10% uncertainty states

To compute: Log `(confidence * |actor_loss|)[top_decile_sigma].sum() / (confidence * |actor_loss|).sum()`

---

## Common Pitfalls & Solutions

### Pitfall 1: Confidence.mean() Multiplication

**WRONG**:
```python
loss_dict["Actor Loss"] = loss_dict["Actor Loss"] * confidence.mean()
```

**Why it fails**: Collapses all per-state structure into a single scalar. Doesn't prevent "high-σ states dominate gradient" failure.

**Correct**: Apply per-timestep weights BEFORE masked_avg (see Step 2).

---

### Pitfall 2: Backwards Beta Annealing

**WRONG**:
```python
beta = beta_init * (progress ** anneal_power)
```

**Why it fails**: Starts at 0 and increases. We want high → low.

**Correct**:
```python
beta = beta_final + (beta_init - beta_final) * (1 - progress) ** anneal_power
```

---

### Pitfall 3: Computing Normalization Stats on Padded Data

**WRONG**:
```python
median = q_std.median()  # Includes padding
```

**Why it fails**: Padding is typically zero, which biases median downward → over-penalization.

**Correct**: Apply mask before computing stats (see Step 4).

---

### Pitfall 4: Not Detaching Confidence Weights

**WRONG**:
```python
confidence = 1.0 / (1.0 + beta * sigma_norm)  # No detach
weighted_loss = actor_loss * confidence
```

**Why it fails**: Gradients flow back through uncertainty → critic updates to minimize weighting penalty (unintended optimization target).

**Correct**: Always `confidence.detach()` before multiplication.

---

### Pitfall 5: Caching Full Q-Ensemble

**WRONG**:
```python
self.cached_epistemic['q_ensemble'] = q_s_a_g  # [1, B, L, C, G, 1]
```

**Why it fails**: Memory bloat. For B=32, L=64, C=4, G=3, this is 24,576 floats per batch.

**Correct**: Cache only the std (see Step 1).

---

## Advanced: CDF Normalization (Future Upgrade)

The median-ratio normalization is simple but has limitations:
- Doesn't distinguish "all uncertainty is high" from "relative uncertainty within batch"
- Can drift if batch composition changes (team distribution, episode lengths)

**Better approach**: CDF-based normalization with EMA reference distribution.

### Implementation

```python
def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # EMA histogram for uncertainty distribution
    self.ema_sigma_hist = None
    self.ema_momentum = 0.99

def _normalize_uncertainty_cdf(self, q_std, mask):
    """Normalize using CDF relative to EMA distribution."""

    valid_stds = q_std[mask.squeeze(-1).squeeze(-1) > 0]

    # Initialize or update EMA histogram
    if self.ema_sigma_hist is None:
        self.ema_sigma_hist = valid_stds.detach().cpu().numpy()
    else:
        new_stds = valid_stds.detach().cpu().numpy()
        # Keep last N samples (e.g., 10k timesteps)
        combined = np.concatenate([
            self.ema_sigma_hist * self.ema_momentum,
            new_stds
        ])[-10000:]
        self.ema_sigma_hist = combined

    # Compute percentile rank (CDF)
    from scipy.stats import rankdata
    ranks = rankdata(valid_stds.cpu().numpy(), method='average')
    sigma_cdf = torch.tensor(ranks / len(ranks), device=q_std.device)

    # Reshape back to original
    sigma_norm = torch.zeros_like(q_std)
    sigma_norm[mask.squeeze(-1).squeeze(-1) > 0] = sigma_cdf

    return sigma_norm  # Now in [0, 1]
```

**Benefits**:
- σ̃ ∈ [0, 1] (uniform scale across loops)
- Automatically tracks "critic is getting more confident" (σ̃ distribution shifts left)
- Beta tuning is stable ("top 10% uncertainty gets weight 0.3" is interpretable)

**Drawback**: Requires scipy and careful EMA management. Use median-ratio for v0, upgrade later.

---

## Interaction with Existing Systems

### Dynamic Damping (KL Regularization)

**Status**: Keeps running independently (complementary)

**Why both**:
- **KL damping**: Limits step size (prevents large policy shifts)
- **Epistemic weighting**: Improves direction (prevents wrong-direction shifts)

**No parameter changes needed**: Existing KL targets (0.01-0.015) and damping coefficients remain optimal.

### Binary FBC Filter

**Current behavior**: Filters to positive advantages only (threshold=0.0)

**With epistemic weighting**: Filter still runs, but now filtered advantages are further weighted by confidence.

**Future upgrade**: Consider filtering on **effective advantage** `A_eff = w(σ) · A` instead of raw advantage. This aligns "train on big signals" heuristic with epistemic safety.

### Reward Functions

**Compatible with all reward functions**: AggressiveShapedRewardSleep, BinaryReward, DefaultShapedReward, etc.

**No reward_multiplier changes needed**: Epistemic weighting operates on gradients, not rewards.

---

## Hyperparameter Tuning Guide

### If Learning is Too Slow (Effective Mass < 0.5)

**Symptom**: Mean confidence very low (~0.3), training barely updates policy

**Solutions**:
1. Lower `beta_init` (5.0 → 3.0)
2. Lower `epistemic_power` (2 → 1)
3. Use faster annealing (`anneal_power = 1.0` for linear decay)

### If Instability Returns (Epoch-0 collapse still happens)

**Symptom**: Win rate still drops to 0% at epoch-0

**Solutions**:
1. Raise `beta_init` (5.0 → 8.0 or 10.0)
2. Raise `epistemic_power` (2 → 3 for stronger tail suppression)
3. Slower annealing (`anneal_steps = 20000` for 6 epochs)
4. Check that normalization is working (inspect High-σ vs Low-σ separation)

### If Weighting Seems to Do Nothing

**Symptom**: High-σ confidence ≈ Low-σ confidence (no separation)

**Possible causes**:
1. Normalization broken (all σ̃ ≈ 1.0)
2. Beta too low (increase `beta_init`)
3. Padding contaminating stats (check masking in Step 4)
4. Q-ensemble not actually cached (check Step 1)

**Diagnostic**: Add logging for raw `q_std.mean()` and `sigma_norm.mean()`. If they're not changing over training, something is wrong.

---

## Future Work & Extensions

### 1. Advantage-Level Weighting (Instead of Loss-Level)

**Current**: Weight entire `actor_loss = offline_coeff*BC + online_coeff*DPG`

**Better**: Weight the advantage term specifically before it enters BC/DPG

**Requires**: Modifying `amago/agent.py` to pass confidence through advantage computation. More invasive but cleaner semantically.

### 2. Separate BC vs DPG Weighting

**Idea**: Only weight the BC term (which uses offline data, higher risk), leave DPG unweighted (online signal, lower risk)

**Implementation**: Cache BC and DPG components separately, apply weights before summing.

### 3. Epistemic-Aware FBC Filtering

**Idea**: Filter based on `A_eff = w(σ) · A` instead of raw `A`

**Effect**: Aligns filtering with epistemic safety (don't train on high-magnitude but high-uncertainty advantages)

### 4. Adaptive Beta (Controller-Driven)

**Idea**: Adjust beta based on observed win rate or critic loss slope (similar to dynamic damping's adaptive LR)

**Example**: If win rate drops, increase beta (more conservative). If stable, decrease beta (learn faster).

### 5. Multi-Head Uncertainty

**Current**: Treat all gammas (G dimension) equally

**Better**: Per-gamma uncertainty normalization and weighting (short-horizon vs long-horizon have different uncertainty profiles)

---

## References & Context

### Related Files
- `metamon/rl/dynamic_damping.py` - KL-based damping (complementary mechanism)
- `metamon/rl/metamon_to_amago.py` - Training experiment wrapper (where to implement)
- `amago/agent.py` - Core RL agent (actor/critic, loss computation)
- `amago/experiment.py` - Training loop (where backward happens)

### Related Skills
- `dynamic-damping-config-selection.md` - Choosing KL damping configs
- `selfplay-loop-workflow.md` - Gen1 OU self-play pipeline

### External Analysis
This approach was informed by detailed analysis from an external colleague who identified:
- High Q-ensemble std dev at epoch-0 (~0.12-0.14)
- Correlation between critic uncertainty and catastrophic updates
- Binary FBC filter lets 30%+ actions through regardless of uncertainty
- Small KL is not protective in imperfect-information games

**Key quote**: *"You need to make early actor updates conditional on critic trustworthiness."*

---

## Questions / Uncertainties

1. **Optimal beta schedule**: Is power-law (sqrt) best, or should we try exponential decay?
2. **Per-gamma normalization**: Worth the complexity, or is global median sufficient?
3. **Weighting both BC and DPG**: Or should we only weight BC (offline) term?
4. **CDF vs median normalization**: Does CDF meaningfully improve stability in practice?
5. **Interaction with reward shaping annealing**: DefaultShapedReward anneals shaping terms. Does this interact with epistemic weighting?

---

## Implementation Checklist

For a new engineer taking over:

- [ ] **Step 1**: Add Q-ensemble std caching in `MetamonMultiTaskAgent.forward()`
  - [ ] Cache `q_std = q_s_a_g.std(dim=3).detach()`
  - [ ] Store in `self.cached_epistemic` (not `cached_kl_data`)
  - [ ] Add shape assertion: `assert q_s_a_g.size(3) >= 2`

- [ ] **Step 2**: Override `MetamonAMAGOExperiment.compute_loss()`
  - [ ] Call `Agent.forward()` directly for per-timestep tensors
  - [ ] Apply epistemic weighting before `masked_avg()`
  - [ ] Copy masking logic from parent
  - [ ] Clear cache after use

- [ ] **Step 3**: Implement `_apply_epistemic_weighting()`
  - [ ] Extract `q_std` from cache
  - [ ] Trim to match actor_loss shape `[:, :-1, :, :]`
  - [ ] Shape assertion
  - [ ] Normalize uncertainty with masking
  - [ ] Compute confidence weights
  - [ ] Detach confidence
  - [ ] Multiply and return

- [ ] **Step 4**: Implement `_normalize_uncertainty()`
  - [ ] Apply mask before stats
  - [ ] Compute median (global or per-gamma)
  - [ ] Ratio normalization
  - [ ] Clamp to [0, 10]

- [ ] **Step 5**: Implement `_get_current_beta()`
  - [ ] Initialize step counter
  - [ ] Compute progress
  - [ ] **CORRECT formula**: `beta_final + (beta_init - beta_final) * (1 - progress)^α`
  - [ ] Increment step counter

- [ ] **Step 6**: Implement `_log_epistemic_metrics()`
  - [ ] Log mean uncertainty/confidence
  - [ ] Log high-σ vs low-σ split
  - [ ] Log effective mass
  - [ ] Log current beta

- [ ] **Step 7**: Add config parameters to `__init__()`
  - [ ] `use_epistemic_weighting = False`
  - [ ] `epistemic_beta_init = 5.0`
  - [ ] `epistemic_beta_final = 1.0`
  - [ ] `epistemic_anneal_steps = 10000`
  - [ ] `epistemic_anneal_power = 0.5`
  - [ ] `epistemic_power = 2`

- [ ] **Step 8**: Create gin config `epistemic_aware_rl.gin`
  - [ ] Include base config
  - [ ] Set epistemic parameters
  - [ ] Document expected behavior

- [ ] **Testing**: Run 5-minute validation
  - [ ] Add debug prints
  - [ ] Check shape alignment
  - [ ] Check confidence range [0.1, 1.0]
  - [ ] Check high-σ vs low-σ separation

- [ ] **Ablation**: Run 2-epoch experiment
  - [ ] Baseline (no epistemic)
  - [ ] Treatment (epistemic enabled)
  - [ ] Compare epoch-0 win rates

- [ ] **Commit**: Create skill and commit implementation
  - [ ] Document actual results
  - [ ] Update hyperparameters if tuning needed
  - [ ] Cross-reference in `selfplay-loop-workflow.md`

---

## Expected Timeline

- Implementation: 2-3 hours (90 lines of code, careful testing)
- Validation: 5 minutes (debug prints, shape checks)
- Ablation: 2-4 hours (2 epochs each, 2 runs)
- **Total**: 4-7 hours from start to validated results

---

## Success Criteria (Final)

**Minimum viable success**:
- Epoch-0 win rate > 40% (vs current 0%)
- No entropy collapse (H > 1.0)
- Clear confidence separation (high-σ < 0.5, low-σ > 0.7)

**Strong success**:
- Epoch-0 win rate ≈ 50% (matches frozen baseline exactly)
- Epoch-1 win rate > baseline (faster learning due to better gradients)
- Top-σ decile contributes < 5% of gradient mass

**Research validation**:
- Ablation shows clear causality (epistemic ON/OFF difference)
- Metrics align with hypothesis (σ↓ → confidence↑ → learning↑ over training)
- Generalizes to other formats/teams (not just Gen1 OU)

---

**Status**: Ready for implementation
**Confidence**: High (solid theoretical basis, clear failure mode, careful design)
**Risk**: Low (single file, easy to toggle off, independent of other systems)
