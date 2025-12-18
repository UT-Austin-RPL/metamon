# Dynamic Damping Configuration Selection

> **Category**: config
> **Last Updated**: 2025-12-10
> **Author**: Claude

## Objective

Quick guide to selecting the right dynamic damping gin config for Gen1 OU self-play training.

## When to Use This Skill

- [ ] When starting new training run and choosing `--train_gin_config`
- [ ] When troubleshooting KL divergence or policy collapse issues
- [ ] When switching reward functions (need to match reward multiplier)

## Quick Decision Guide

### Current Active Configs

| Config | Use When | Reward Multiplier | KL Target |
|--------|----------|------------------|-----------|
| `selfplay_damped_aggressive.gin` | Large offline datasets (100k-200k), +200/0 rewards | 0.05 | 0.015 |
| `selfplay_damped_aggressive_v3.gin` | Version 3 iteration (document what changed) | 0.05 | 0.015 |
| `selfplay_damped_aggressive_v4_safe.gin` | Safe variant, tested and stable | 0.05 | 0.015 |
| `selfplay_controller_v1.gin` | Maximum safety, if others collapsed | 0.05 | 0.008 |
| `sleep_selfplay_v1.gin` | Sleep strategy experiments | 0.05 | TBD |
| `sleep_selfplay_v2_aggressive.gin` | Aggressive sleep variant | 0.05 | TBD |

### Simple Rules

**For standard 100k-200k self-play**:
- Start with: `selfplay_damped_aggressive_v4_safe.gin`
- BC-heavy (75/25), stable, tested

**If policy collapses**:
- Switch to: `selfplay_controller_v1.gin`
- Ultra-tight KL (0.008), nearly pure BC (85/5)

**Reward multiplier matching**:
- AggressiveShapedReward (+200/0) → 0.05
- DefaultShapedReward (±100) → 10.0

## Key Lessons

### Failed: kl_coef_init = 0.20 (too weak)
- Policy collapsed on 200k dataset
- **Solution**: Increased to 0.30 in v2+

### Success: BC-heavy (75/25) for offline
- Stable training on large datasets
- Prevents catastrophic forgetting

### Success: Controller-driven (no decay)
- Adapts dynamically to observed KL
- No need for power-law schedules

## Monitoring

**Healthy**:
- KL: 0.01-0.0225
- Entropy: > 1.0
- Win rate: improving

**Red flags**:
- KL > 0.03 → increase kl_coef_init
- Entropy < 0.5 → switch to controller_v1
- Win rate drops after epoch 1 → reduce DPG (lower online_coeff)

## Related Skills

- [`selfplay-loop-workflow`](./../training/selfplay-loop-workflow.md) - Full training workflow
- [`reward-scale-matching`](./../config/reward-scale-matching.md) - Reward multiplier calculation

## References

- [GEN1OU_SELFPLAY_GUIDE.md](../../GEN1OU_SELFPLAY_GUIDE.md)
- Config directory: `metamon/rl/configs/training/`
