# Skill Name (kebab-case, specific)

> **Category**: [training | config | troubleshooting | evaluation]
> **Last Updated**: YYYY-MM-DD
> **Author**: [Name or "Claude via /retrospective"]

## Objective

**What problem does this skill solve?** (1-2 sentences)

Be specific: Not "helps with training", but "prevents policy collapse in Gen1 OU self-play via dynamic damping"

## When to Use This Skill

**Specific trigger conditions:**

- [ ] When [specific scenario, e.g., "training Gen1 OU specialist from SyntheticRLV2"]
- [ ] When encountering [specific error, e.g., "vllm_skip_weight_sync errors during GRPO"]
- [ ] When [user asks about X, e.g., "asking how to choose between aggressive vs conservative damping"]
- [ ] When [metric pattern observed, e.g., "KL divergence > 0.03 with entropy dropping below 1.0"]

**Do NOT use this skill when:**

- [Scenarios where this skill doesn't apply]

## Prerequisites

**Required setup before following this skill:**

### Environment
```bash
# Activate virtual environment
source .venv/bin/activate

# Set cache directory
export METAMON_CACHE_DIR=/home/eddie/metamon_cache
```

### Services
- [ ] Local Pokémon Showdown server running (if needed)
  ```bash
  cd server/pokemon-showdown && node pokemon-showdown start --no-security
  ```

### Data
- [ ] Dataset location: `/path/to/data`
- [ ] Required format: [e.g., "gen1ou/*.json.lz4 structure"]
- [ ] Expected size: [e.g., "~25K trajectories"]

### Models
- [ ] Base checkpoint: [e.g., "SyntheticRLV2, epoch 48"]
- [ ] Architecture config: [e.g., "synthetic_multitaskagent.gin"]

## Step-by-Step Workflow

### Step 1: [Action Name]

**Goal**: [What this step achieves]

**Command**:
```bash
# Copy-paste ready command with all flags
python -m metamon.rl.finetune_from_hf \
    --run_name "experiment-name" \
    --finetune_from_model SyntheticRLV2 \
    --formats gen1ou \
    --train_gin_config config_name.gin \
    --epochs 5 \
    --save_dir ~/metamon/models/experiment-name \
    --log
```

**Expected output**: [What you should see]

**Duration**: [Typical runtime]

**What to monitor**:
- Metric 1: Expected range [0.01-0.03], red flag if > 0.05
- Metric 2: Should increase steadily, plateau indicates [problem]

---

### Step 2: [Next Action]

[Continue pattern for each step...]

## Critical Parameters

### Hyperparameter Table

| Parameter | Recommended Value | Tested Range | Notes |
|-----------|------------------|--------------|-------|
| `reward_multiplier` | 0.05 | [0.01, 0.1] | Must match reward scale (200×0.05=10) |
| `kl_target` | 0.015 | [0.01, 0.03] | Conservative: 0.01, Aggressive: 0.02 |
| `learning_rate` | 1e-5 | [5e-6, 5e-5] | Too high causes instability, too low stalls |

### Configuration Selection

**For [scenario A]**: Use `config_name_v1.gin`
- Reason: [Why this config]
- Key settings: [Distinctive parameters]

**For [scenario B]**: Use `config_name_v2.gin`
- Reason: [Why this config instead]
- Key settings: [What's different]

## What Worked ✅

### Successful Approach 1: [Description]

**Context**: [When this was tried, what model/data/goal]

**Configuration**:
```gin
# Key gin settings
reward_multiplier = 0.05
kl_target = 0.015
online_coeff = 0.25
offline_coeff = 0.75
```

**Results**:
- Metric 1: Achieved [value], target was [value]
- Metric 2: Improved from [before] to [after]
- Win rate: [percentage] vs [opponent]

**Why it worked**: [Root cause analysis]

---

### Successful Approach 2: [Description]

[Continue pattern...]

## Failed Attempts ❌

> **Most valuable section** - Document failures in detail to prevent repetition

### Failure 1: [Short description of what was tried]

**What was tried**:
```bash
# Exact command that failed
python -m metamon.rl.train ...
```

**Configuration**:
- Hyperparameter 1: [value]
- Hyperparameter 2: [value]

**What went wrong**:
- Symptom: [Observed behavior, e.g., "losses stayed flat after epoch 1"]
- Metrics: [Specific values, e.g., "KL: 0.001, Entropy: 0.3"]
- Error message (if applicable):
  ```
  Exact error message copied here
  ```

**Root cause**: [Why it failed]

**Solution**: [How it was fixed, or why approach was abandoned]

**Takeaway**: [Key lesson - what to avoid or check next time]

---

### Failure 2: [Another failed attempt]

[Continue pattern...]

## Common Errors & Solutions

### Error: `[Exact error message or pattern]`

**When it occurs**: [Trigger conditions]

**Root cause**: [Technical explanation]

**Solution**:
```bash
# Fix command or code change
```

**How to prevent**: [Validation check or prerequisite]

---

### Error: [Another common error]

[Continue pattern...]

## Metrics Interpretation

### Healthy Training Indicators

- **KL Divergence**: 0.01-0.02 (stable), < 0.005 (too conservative), > 0.03 (losing regularization)
- **Entropy**: > 1.0 (diverse policy), < 0.5 (collapsing)
- **Win Rate**: Gradual improvement (2-5% per epoch), sudden jumps suspicious
- **[Other metrics]**: [Expected patterns]

### Red Flags

- **Flat losses**: Likely reward scale mismatch or learning rate too low
- **Exploding KL**: Damping too weak or KL target too loose
- **Entropy collapse**: Policy mode-seeking, increase damping coefficient
- **[Other warning signs]**: [What they indicate]

## Unexpected Findings

**Finding 1**: [Surprising result]
- **Hypothesis**: [Why this might have happened]
- **Implications**: [What this means for future experiments]

**Finding 2**: [Another unexpected result]
- [Continue pattern...]

## Follow-Up Questions

**Unresolved questions from this experiment:**
1. Question 1 that needs investigation
2. Question 2 that needs investigation

**Suggested next experiments:**
1. Test [variation] to determine [hypothesis]
2. Try [approach] to improve [metric]

## Related Skills

- [`other-skill-name`](./../category/other-skill-name.md) - When to use that instead
- [`related-skill`](./../category/related-skill.md) - Complementary information

## References

### Documentation
- [GUIDE_NAME.md](../../GUIDE_NAME.md) - High-level workflow
- [Code file](../../metamon/path/to/code.py:123) - Implementation

### Configurations
- [`config_name.gin`](../../metamon/rl/configs/training/config_name.gin) - Training config
- [`model_config.gin`](../../metamon/rl/configs/models/model_config.gin) - Architecture

### Experiments
- Run directory: `/home/user/metamon/models/experiment-name`
- W&B link (if applicable): [URL]
- Checkpoint used: Model name, epoch number

---

## Template Usage Notes

**When creating a new skill:**

1. **Be SPECIFIC**:
   - ❌ "Helps with pruning experiments"
   - ✅ "Prunes Gen1 OU models for inference speedup on 4GB GPUs, handles embedding quantization errors"

2. **Emphasize FAILURES**:
   - Failed attempts section should be longest/most detailed
   - Include exact error messages and metric values
   - Explain root causes, not just symptoms

3. **Copy-paste READY**:
   - All commands must be executable as-is
   - Include all required flags
   - Use absolute paths or document path expectations

4. **Concrete PARAMETERS**:
   - ❌ "Use a small learning rate"
   - ✅ "Learning rate: 1e-5 (tested 5e-6 too slow, 5e-5 unstable)"

5. **Hardware CONTEXT**:
   - Specify where tested (e.g., "4x A100 80GB", "single RTX 3090")
   - Note memory requirements
   - Include approximate runtime

6. **Validation CRITERIA**:
   - Define clear success metrics with ranges
   - Specify red flags with thresholds
   - Explain how to know if it's working
