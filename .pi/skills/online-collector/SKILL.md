---
name: online-collector
description: "The CPU-sim self-play collection process for metamon online RL. Use when asked about the collector's team-mix schedule, @schedule opponent pool, PSRO-Lite sidecar, WeightedMixedTeamSet, per-teamset diagnostics (_ts- filename tokens), collection throughput, or why the curriculum shifts from gl_05_26 to smogon teams. Covers the teamset curriculum, schedule YAMLs, and best practices for collection mix design."
---

# Online RL Collector — Self-Play Collection & Teamset Curriculum

The collector (`--mode collect`) runs self-play rollouts into the shared FIFO
buffer. It is CPU-sim-bound (single-threaded Node Showdown sim workers), never
touches the GPU, and syncs to the learner's `latest/policy.pt` each epoch
(`always_load_latest=True`).

## How it works

### The collection pipeline

1. Each epoch, the collector runs `train_timesteps_per_epoch` env steps (500)
   across `lanes` parallel battle lanes (128) with `n_workers` Node-sim
   subprocesses (28).
2. Each lane plays a full battle; when it ends, the trajectory is saved to
   `buffer_dir/<format>/` as a `.json.lz4` file and a row is appended to the
   battle-log CSV.
3. The PSRO-Lite step runs after collection (`_psro_step` in
   `MetamonOnlineExperiment`): it scores the buffer, computes prioritized
   opponent weights, and writes `meta_weights.json` (the sidecar the learner's
   FIFO sampler reads).

### Throughput (measured on 1× RTX 5090, 32-core host)

- **~234 battles/min** (14,040/hr) at 128 lanes / 28 workers
- **~48.6 env steps per battle** (avg battle length)
- **~1,317 battles per collector epoch** (128 lanes × 500 steps / 48.6 turns)
- **~5.6 min per collector epoch**
- **Main process at 100% CPU** (GIL-bound Python orchestration) — this is the
  bottleneck, not the sim workers. Doubling workers won't double throughput
  without multi-process collection architecture.

### The team-mix schedule (epoch-driven curriculum)

The collector's player team set and the opponent pool's `@schedule` agents both
follow a shared `TeamMixSchedule` + `EpochRef` loaded from a YAML file via
`--train_team_schedule`. The schedule shifts the team mix automatically at epoch
boundaries — no restart needed.

**How it's wired** (reconstructed this session — the original integration was
lost from the working tree, never committed):

1. `make_schedule_state(path)` in `metamon/rl/online_schedule.py` loads the YAML
   into a `TeamMixSchedule` + a fresh `EpochRef(epoch=0)`.
2. `make_collect_train_env` builds the player team set via
   `get_metamon_team_set_from_schedule` (schedule-aware `WeightedMixedTeamSet`)
   and passes `opponent_team_schedule` / `opponent_epoch_ref` to the opponent
   pool so `@schedule` agents follow the same curriculum.
3. `MetamonOnlineExperiment` stores the `EpochRef` and bumps it to `self.epoch`
   at the start of each `collect_new_training_data`, so schedule-aware team
   sets lazily refresh weights on the next `yield_team()`.
4. On resume, `run_online_rl` sets `epoch_ref.epoch` to the resumed epoch so the
   schedule picks up at the right phase.

**The schedule is REQUIRED when the opponent pool uses `@schedule`** — without
it, the pool raises `ValueError: opponent pool team_set '@schedule' requires a
TeamMixSchedule and EpochRef`. The `launch_psro_v1.sh` script defaults
`TRAIN_TEAM_SCHEDULE` to the gen1ou competitive curriculum YAML.

### Why the curriculum shifts from gl_05_26 to smogon teams

**The goal:** a policy that is superhuman at piloting competitive/smogon teams
and playing against them, without falling over against diverse teams.

**The problem the data revealed:** the learner was trained for ~1140 epochs on
100%/75% gl_05_26 teams. It overfit to gl compositions and is incompetent with
smogon compositions (15% win rate when handed smogon teams, vs 61% with gl
teams). The per-teamset diagnostics (`_ts-` filename tokens, see below) exposed
this — the overall win rate (~50%) was hiding the generalization gap.

**The curriculum design:**

| phase | epochs | gl_05_26 | smogon_pass2 | smogon_pass2_selected | rationale |
|---|---|---|---|---|---|
| 0 | 0-939 | 100% | 0% | 0% | broad ladder (historical, done) |
| 1 | 940-1259 | 75% | 15% | 10% | acquaint (v1.3, done) |
| 2 | 1260-1599 | 30% | 30% | 40% | **specialize** (v1.4 current) |
| 3 | 1600-1999 | 20% | 25% | 55% | sharpen |
| 4 | 2000-3950 | 15% | 20% | 65% | maintain (permanent gl floor) |

**The permanent gl_05_26 floor (15-30%)** is the "don't forget diverse teams"
safety net. The collector always produces some gl battles so the learner keeps
seeing them. Combined with the 60% offline data (also gl-like), the learner
always has substantial diverse-team exposure.

**The v1.4 schedule** (`gen1ou_competitive_curriculum_v1.4.yaml`) compressed
the timeline aggressively — the original schedule didn't reach heavy smogon
until epoch 2800, which was far too slow given the generalization gap was
already visible at epoch 1140.

### Per-teamset diagnostics (`_ts-` filename tokens)

The collector records the concrete team set the learner drew for each battle as
a `_ts-<teamset>` token in the trajectory filename:

```
metamon-gen1ou-..._vs_TaurosV0-1-ckpt54-..._ts-gl_05_26_WIN.json.lz4
metamon-gen1ou-..._vs_Kakuna-1-ckpt24-..._ts-smogon_pass2_LOSS.json.lz4
```

The PSRO-Lite solver (`compute_prioritized_weights` in `metamon/rl/psro_lite.py`)
parses these tokens and breaks per-opponent win rates down by learner teamset,
logged to wandb as `psro/<agent>/<teamset>/win_rate` and `psro/<agent>/<teamset>/n`.

**This is diagnostic only** — the solver weights still use the overall aggregate
per-agent win rate. The per-teamset breakdown reveals where the policy is strong
vs weak across team compositions, which is how we discovered the gl vs smogon
generalization gap.

**Backward compat:** files without the `_ts-` token (pre-v1.3 collection) bucket
as `_unknown` in the diagnostics. These age out of the rolling 50k window over
~20-30 epochs.

### PSRO-Lite prioritized opponent sampling

The collector writes `meta_weights.json` (the sidecar) each PSRO update interval
(default 5 epochs). The solver scores opponents by the learner's empirical win
rate against them:

```
score = max(0, 0.5 - win_rate)   # winnable matchups score high
weight = score^(1/temp) × confidence   # confidence = n/(n+min_games)
```

Opponents the learner barely beats get boosted → the learner faces them more.
This is PFSP-style (prioritized fictitious self-play), analogous to AlphaStar's
league mechanism. See `metamon/rl/online_psro.py` and `docs/psro_lite_plan.md`.

**Quota-based diversification** (`--psro_quota_min_games`, `--psro_quota_window`)
guarantees every pool agent a minimum number of games over a rolling window so
dominated opponents don't fall to ~0 games played.

### Both sides draw independently from the same schedule

When the player draws `smogon_pass2_selected`, the opponent draws independently
from the same 30/30/40 mix — so the opponent might get gl_05_26 (easier) or
smogon (harder). The battle-log CSV records **both** team files per battle,
which is how we cross-tabulated player-teamset × opponent-teamset × result to
confirm the teamsets are being used correctly.

## Resume flow

The collector resumes from the latest `training_state` just like the learner
(to keep epoch counters aligned with the schedule):

```bash
EPOCHS=3950 bash scripts/launch_mini_online_v1.sh collector --log \
  --resume_training_state \
  --psro_weighting --psro_start_epoch 0 \
  --psro_temp 2.0 --psro_floor 0.01 \
  --psro_quota_min_games 200 --psro_quota_window 256
```

With `TRAIN_TEAM_SCHEDULE` pointing to the schedule YAML (the launch script
defaults it). On resume, the schedule prints:
```
Team mix schedule: starting at epoch 1260 (from latest training_state)
Team mix schedule: e0: gl_05_26=100%,... | e1260: gl_05_26=30%,...
```

**The collector can be relaunched independently** of the learner and validator.
If only the schedule changes (not the training state), just relaunch the
collector with the new `TRAIN_TEAM_SCHEDULE` path — the learner keeps training,
the validator keeps validating.

## Best practices

### Schedule design

1. **Always keep a permanent gl_05_26 floor (≥15%)** in the schedule. This +
   the 60% offline data = the "don't forget diverse teams" guarantee. Going to
   0% gl risks catastrophic forgetting of broad-team play.
2. **Compress the timeline if the generalization gap is visible early.** The
   original schedule waited until epoch 2800 for heavy smogon — too slow. v1.4
   shifts at epoch 1260 (the resume point). Don't be afraid to edit the YAML
   mid-run; the schedule is epoch-driven and picks up at the right phase on
   resume.
3. **The schedule YAML is the single source of truth for the collection mix.**
   Both the player team set and the `@schedule` opponent pool agents follow it.
   Don't try to set `--train_team_set` and `--train_team_schedule` at the same
   time — the schedule overrides the static set.
4. **Schedule YAMLs live in `metamon/rl/configs/team_schedules/`.** Create a
   new file (e.g. `_v1.4.yaml`) rather than editing the existing one, so you
   can revert by pointing `TRAIN_TEAM_SCHEDULE` at the old file.

### Throughput

1. **The collector main process is the bottleneck** (100% CPU, GIL-bound).
   Adding more workers beyond ~28 doesn't help — the orchestrator can't
   coordinate them faster. To scale collection, you'd need multi-process
   collection (architecture change, not a flag).
2. **128 lanes / 28 workers is the proven config** on a 32-core host. The
   validator (32 lanes) shares CPU but runs at a different cadence.
3. **Set the CPU governor to `performance` before launching** — collection is
   single-threaded Node sim, and `powersave` throttles it:
   `sudo cpupower frequency-set -g performance`

### PSRO-Lite

1. **`--psro_temp 2.0 --psro_floor 0.01`** sharpen the prioritized solver so
   near-break-even opponents still get boosted (temp=1.0 floored Kakuna to
   uniform because its win rate ~0.45 was below the floor).
2. **The sidecar is atomic** — the learner reads it with mtime-caching and
   falls back to uniform if it's momentarily absent. Don't worry about races.
3. **The per-teamset diagnostics are diagnostic only.** The solver weights use
   the overall aggregate. To make PSRO target the smogon-generalization gap
   specifically, the solver would need a `teamset_filter` option (not yet
   implemented — see the online-learner skill for the plan).

## Reference

- `metamon/rl/online_schedule.py` — `ScheduleState`, `make_schedule_state`,
  `resolve_train_team_set`, `log_schedule_start`
- `metamon/rl/online_envs.py` — `make_collect_train_env` (wires the schedule
  into the env)
- `metamon/env/wrappers.py` — `TeamMixSchedule`, `EpochRef`,
  `WeightedMixedTeamSet`, `get_metamon_team_set_from_schedule`
- `metamon/rl/configs/team_schedules/` — schedule YAMLs (v1 original + v1.4
  compressed)
- `metamon/rl/configs/opponent_pools/hl_gen1ou.yaml` — the `@schedule` pool
- `metamon/rl/psro_lite.py` — `compute_prioritized_weights`,
  `parse_trajectory_filename` (the `_ts-` token parser)
- `metamon/env/vectorized/vector_env.py` — `_save_lane_outcome` (writes the
  `_ts-` token), `_teamset_from_team_file`
- `scripts/launch_psro_v1.sh` — the PSRO split-layout launcher
- See also: the `online-training` skill for the split-layout launch recipe and
  the `online-learner` / `online-validator` skills for the other two roles.
