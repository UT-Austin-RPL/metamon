---
name: online-validator
description: "The CPU-sim evaluation process for metamon online RL. Use when asked about the validator's Tauros eval, the competitive teamset, val/Average Win Rate wandb metric, val_interval/val_timesteps tuning, or why validation win rate can be ~50% while per-teamset collection win rates are much lower. Covers the val opponent pool, the competitive vs smogon teamset distinction, and best practices for using validation as a generalization canary."
---

# Online RL Validator — Evaluation & Generalization Canary

The validator (`--mode validate`) evaluates the learner's rolling
`latest/policy.pt` against a fixed val opponent each epoch. It is CPU-sim-bound,
never touches the GPU, and reloads weights every epoch
(`always_load_latest=True`).

## How it works

### The validation loop

1. Each `val_interval` epochs (default 1), the validator reloads
   `latest/policy.pt` (the learner's most recent published weights).
2. It runs `val_timesteps` env steps (default 1000) across `lanes` battle lanes
   (32) against the val opponent.
3. It logs `val/Average Win Rate` and `val/Average Total Return` to wandb.

The validator runs its own wandb run (separate from the learner and collector),
so you see 3 runs in wandb for a split-layout run. The validator's run is the
one with `val/` prefixed keys.

### The val opponent and teamset

By default, the validator battles against **TaurosV0** on the **`competitive`
teamset** (28 curated teams). This is configured via:

- `--val_opponent TaurosV0` (or `--val_pool <yaml>` for a custom val pool)
- `--val_team_set competitive` (default; gen9ou runs use `gl_05_26`)

The `competitive` teamset is a small, curated set of 28 high-quality teams. It
is **not** the same as the smogon collection teamsets (`smogon_pass2` = 475
teams, `smogon_pass2_selected` = 283 teams). This distinction matters — see
"The generalization canary" below.

### Why the validator doesn't need the schedule

The validator uses its own val pool (TaurosV0) and its own teamset
(`competitive`), neither of which uses `@schedule`. So the validator does **not**
need `--train_team_schedule` and can keep running independently when the
collector's schedule changes. This is why we could relaunch the learner and
collector mid-run without restarting the validator.

### `always_load_latest=True` — the validator never trains

The validator is configured by `apply_async_mode` (`metamon/rl/online_envs.py`):
- `start_collecting_at_epoch = inf` (never collects)
- `train_timesteps_per_epoch = 0` (no env interaction for training)
- `start_learning_at_epoch = inf` (never takes grad updates)
- `train_batches_per_epoch = 0`
- `ckpt_interval = None` (never saves checkpoints)
- `always_save_latest = False` (doesn't write policy.pt)
- `always_load_latest = True` (reads the learner's policy.pt each epoch)
- `epochs = max(epochs, 1_000_000)` (runs effectively forever)

It's a pure evaluation loop — it reads weights, plays battles, logs metrics,
repeats.

## The generalization canary

### Why val WR can be ~50% while per-teamset collection WRs are ~15%

This was the key diagnostic finding of this session. The validator showed
~50% win rate vs Tauros on `competitive` teams, while the collector's
per-teamset diagnostics showed ~15% win rate on `smogon_pass2_selected` teams.
These are **not contradictory** — they measure different things:

- **Validation** (~50%): the learner vs Tauros on 28 curated `competitive`
  teams. The `competitive` set is apparently close enough to gl_05_26
  compositions for the learner to handle (it was trained on gl for 1140 epochs).
  Tauros itself may have similar familiarity limits.
- **Collection** (~15% on smogon): the learner vs various opponents when
  **handed smogon teams**. The learner was never trained on smogon compositions
  and misplays them badly. The ~50% overall collection WR is propped up by the
  ~61% WR on gl teams (which dominate the old collection mix).

**The validation WR is the generalization canary** — if it drops, the policy is
forgetting how to play standard-team Pokémon (the diverse-team grounding is
failing). If it stays ~50% while smogon WRs climb, the specialization is
working without catastrophic forgetting. That's the goal.

### The result perspective

Trajectory filenames record `WIN`/`LOSS` from `final_state.battle_won` where
`final_state` is `universal_state(self.eval_side)` — i.e., the **learner's
perspective**. So `WIN` in the buffer means the learner won, regardless of which
side (p1/p2) the learner was on. The validator's `val/Average Win Rate` is
likewise from the learner's perspective.

## Resume flow

The validator resumes like the other roles (from the latest `training_state`
to keep epoch counters aligned), but it doesn't strictly need to — it just
reads `latest/policy.pt`:

```bash
EPOCHS=3950 bash scripts/launch_mini_online_v1.sh validator --log \
  --resume_training_state
```

**The validator can keep running across learner/collector relaunches.** Since it
only reads `latest/policy.pt` (which the learner continuously publishes), it
picks up new weights each epoch without needing to restart. This session, the
validator ran continuously for 3.8+ hours while the learner and collector were
relaunched multiple times.

### To change the val opponent or teamset mid-run

Relaunch the validator with new flags:
```bash
bash scripts/launch_mini_online_v1.sh validator --log \
  --resume_training_state \
  --val_opponent Kakuna        # or --val_pool <yaml>
  --val_team_set smogon_pass2_selected  # eval on smogon teams instead
```
The learner and collector keep running.

## Best practices

### Using validation as the forgetting signal

1. **Watch `val/Average Win Rate in gen1ou_vs_TaurosV0-competitive` first.**
   This is the canary. If it drops below ~40% after a schedule shift, the policy
   is forgetting diverse-team play → raise the gl_05_26 collection floor or
   lower the learner's teamset up-sampling.
2. **The validator's `competitive` teamset is NOT a smogon proxy.** Don't infer
   smogon performance from validation. Smogon performance is measured by the
   collector's `psro/<agent>/smogon_pass2*/win_rate` diagnostics. If you want
   direct smogon validation, relaunch the validator with
   `--val_team_set smogon_pass2_selected`.
3. **Don't over-tune to the val opponent.** TaurosV0 is a fixed opponent; the
   learner could overfit to Tauros-specific exploits. The PSRO collector
   opponents provide the diverse self-play signal; validation is just a
   sanity check.

### Throughput

1. **32 lanes is sufficient for validation** (vs 128 for collection). Validation
   is less throughput-critical — you need enough battles for a stable win-rate
   estimate (1000 env steps / ~48 turns ≈ 20 battles/lane × 32 lanes ≈ 640
   battles/epoch), not maximum throughput.
2. **`val_interval=1` (every epoch)** gives the tightest signal but costs CPU.
   For long runs, `val_interval=5` or `10` is fine — the validator shares CPU
   with the collector.
3. **Set the CPU governor to `performance`** — same as the collector (single-
   threaded Node sim, throttled by `powersave`).

### When the validator's epoch counter diverges

The validator runs `epochs = max(epochs, 1_000_000)` so it effectively never
stops. Its epoch counter increments alongside the learner's (both resumed from
the same training_state epoch). If the validator is relaunched independently,
its epoch counter resets to the training_state epoch — this is fine, it just
affects the wandb x-axis, not the evaluation.

## Reference

- `metamon/rl/online_envs.py` — `apply_async_mode` (the `validate` branch),
  `make_val_env`, `resolve_val_opponent_config`
- `metamon/rl/online_rl.py` — `run_online_rl` (the `validate` dataset branch
  uses `amago.loading.DoNothingDataset()`)
- `metamon/rl/configs/opponent_pools/tauros_val.yaml` — the default val pool
- `metamon/rl/configs/opponent_pools/kakuna_val.yaml` — alternative val pool
- `metamon/env/vectorized/vector_env.py` — `_save_lane_outcome` (the WIN/LOSS
  perspective), `eval_side`
- `scripts/launch_mini_online_v1.sh` — launch script (`validator` subcommand)
- See also: the `online-training` skill for the split-layout launch recipe and
  the `online-learner` / `online-collector` skills for the other two roles.
