# Reward Scale Matching

> **Category**: config
> **Last Updated**: 2025-12-10
> **Author**: Claude (from LESSONS_LEARNED.md)

## Objective

Match reward_multiplier to reward function scale to prevent gradient explosions or training stalls. Wrong multiplier causes either unstable training (too high) or no learning (too low).

## When to Use This Skill

- [ ] When switching reward functions (DefaultShapedReward ↔ AggressiveShapedReward)
- [ ] When training shows gradient explosions or NaN losses
- [ ] When flat loss curves indicate no learning
- [ ] When creating custom reward functions

## Quick Reference

### Standard Reward Functions

| Reward Function | Range | Reward Multiplier | Effective Scale |
|----------------|-------|------------------|----------------|
| DefaultShapedReward | ±100 | 10.0 | 1000 |
| AggressiveShapedReward | +200/0 | 0.05 | 10 |
| AggressiveShapedRewardSleep | +200/0 | 0.05 | 10 |
| BinaryReward | ±100 | N/A | ❌ Don't use for finetuning |

### Calculation Formula

```python
reward_multiplier = target_effective_scale / (R_max - R_min)

# Examples:
# DefaultShapedReward: 100 - (-100) = 200 range
#   → 1000 / 200 = 5.0 or 2000 / 200 = 10.0 (both work)
#
# AggressiveShapedReward: 200 - 0 = 200 range
#   → 10 / 200 = 0.05
```

## Common Mistakes

### ❌ Using reward_multiplier=10.0 with AggressiveShapedReward

**Problem**: 200 × 10.0 = 2000 (way too large)
**Symptoms**: Gradient explosions, NaN losses, unstable training
**Fix**: Use 0.05 multiplier (200 × 0.05 = 10)

### ❌ Switching reward functions without updating multiplier

**Problem**: Config has reward_multiplier from previous reward function
**Symptoms**: Either explosions or flat losses depending on direction
**Fix**: Always update multiplier when changing reward function

## Failed Experiments

**Binary Reward for Finetuning**: Tried BinaryReward (±100) with standard multiplier → **flat losses, no learning**
- Root cause: Distribution shift too severe (model trained on dense rewards)
- Takeaway: **Don't use sparse rewards for finetuning strong models**

## Related Skills

- [`dynamic-damping-config-selection`](./../config/dynamic-damping-config-selection.md) - Config selection
- [`selfplay-loop-workflow`](./../training/selfplay-loop-workflow.md) - Training workflow

## References

- [metamon/nash/LESSONS_LEARNED.md](../../metamon/nash/LESSONS_LEARNED.md) - Binary reward failure
- [Gen1_BinaryReward_Training_Summary.md](../../Gen1_BinaryReward_Training_Summary.md) - Detailed analysis
