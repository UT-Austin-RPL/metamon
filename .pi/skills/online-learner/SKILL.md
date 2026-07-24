---
name: online-learner
description: "The GPU gradient-update process for metamon online RL. Use when asked about the learner's online/offline mixture, FIFO replay ratio, buffer freshness, teamset up-sampling, dset_max_size/online_weight tuning, resume from training_state, or why the 40% online / 60% offline split was chosen. Covers the replay-staleness tradeoff and best practices for specializing without forgetting."
---

# Online RL Learner — Gradient Updates & Replay Mixture

The learner (`--mode learn`) is the only GPU consumer in the split layout. It
drains the FIFO buffer + offline dataset mixture, takes gradient updates, and
publishes `ckpts/latest/policy.pt` each epoch for the collector and validator
to sync to.

## How it works

### The training mixture (40% online / 60% offline)

The learner trains on a `MixtureOfDatasets` built by
`build_online_mixture_dataset` (`metamon/rl/online_envs.py`):

- **40% online FIFO buffer** — self-play battles the collector writes to
  `buffer_dir/<format>/`. Sampled with replacement (off-policy replay).
- **60% offline dataset** — the static replay corpus from the metamon paper
  (5M+ human battles + 20M+ self-play battles). This is the same data the
  paper used to build strong policies via offline RL.

The 40% online weight is the CLI default (`--online_weight 0.40`) set in the
original "online training draft" commit and never changed. It is **not** from
the paper (the paper was purely offline); it was the author's choice when
extending to online RL.

### Why 40% online and not higher

The 40% was left unexamined for the entire v1 → v1.3 progression and produced
strong policies. The analysis below explains why it works and what its cost is.

**The replay ratio is fixed by three numbers:**

```
repeats per battle = learner_consumption × online_weight ÷ collection_rate
                   = 1,280,000 × online_weight ÷ 14,040
                   = 91.2 × online_weight
```

At 40% online, each battle is replayed ~36× before aging out of the buffer.
Standard RL replay literature (DQN/Atari) finds 4–10× optimal; 36× is high but
tolerable because:

1. The 60% offline data is **also** heavily replayed and works fine — replay-heavy
   training is proven in this codebase.
2. The offline data provides permanent diverse-team grounding (the "don't forget"
   floor), so high online replay doesn't cause catastrophic forgetting.
3. Hidden-info games make staleness costlier (stale data = stale belief-gathering
   behavior), but the offline mix dilutes this.

**The cost of 36× replay is slower specialization**, not memorization collapse.
The policy adapts to new team compositions at a fraction of the rate it could
with fresher data. Observed: smogon-team win rates stayed flat at ~0.17 for 50
epochs while validation vs Tauros improved — the policy wasn't broken, but it
was anchored to old-policy smogon data.

### Buffer freshness is independent of repeats

A critical insight: **buffer size does not change the repeat ratio** — it cancels
out (a smaller buffer means each battle lives shorter but gets sampled faster,
netting the same total replays). Buffer size is purely a **freshness** lever:

```
buffer turnover = buffer_size ÷ collection_rate
```

| buffer size | turnover | median data age | repeats (at 40% online) |
|---|---|---|---|
| 300,000 | 21.4h | 10.7h | 36× |
| 100,000 | 7.1h | 3.6h | 36× |
| 50,000 | 3.6h | 1.8h | 36× |

**Decision made this session:** shrink `--dset_max_size` from 300k → 50k to evict
~250k stale gl_05_26 battles (collected by the epoch-1140 policy) so the 36×
replays hit **fresh** smogon-dominant data (1.8h old) instead of **stale** gl
data (21h old). This keeps the author's 40% online weight while making the
replays productive for specialization.

### Teamset up-sampling (`--fifo_teamset_weights`)

Added this session. The learner's FIFO sampler multiplies each file's weight by
a per-teamset multiplier parsed from the `_ts-<teamset>` filename token:

```
--fifo_teamset_weights 'smogon_pass2:4.0,smogon_pass2_selected:4.0'
```

Files with gl_05_26 or no `_ts-` token get `--fifo_teamset_default_weight`
(default 1.0). This composes multiplicatively with PSRO opponent-weight
reweighting (`--psro_fifo_reweight`). At 4× up-sampling, ~90% of the learner's
online samples go to smogon teams even though smogon is only ~70% of collection.

**Why this matters:** the learner was trained for ~1140 epochs on 100%/75%
gl_05_26 teams. It knows gl compositions and is incompetent with smogon
compositions. Up-sampling smogon trajectories forces the learner to train on
the compositions it's weak at, without changing the collection mix (which is
the collector's job — see the online-collector skill).

### Epoch sync with the collector

The learner and collector run "epochs" at different speeds:

| | rate |
|---|---|
| learner | ~40 epochs/hr (1000 batches/epoch × 32 samples, ~11 it/s) |
| collector | ~11 epochs/hr (128 lanes × 500 steps, ~234 battles/min) |

The learner runs 3.6× faster. Each learner epoch, ~350 fresh battles arrive.
At 50k buffer, the freshest epoch's battles are ~0.7% of the buffer — the
learner almost always samples slightly older battles, but with a 3.6h turnover
they're at most 3.6h stale (vs 21h at 300k buffer).

## Resume flow

The learner resumes from `ckpts/training_states/<run>_epoch_<N>/` (full
accelerate state: model + optimizer + scheduler + PopArt + RNG) via
`--resume_training_state`. The newest state is found by
`latest_training_state_epoch()` in `metamon/rl/online_envs.py`.

```bash
EPOCHS=3950 bash scripts/launch_mini_online_v1.sh learner --log \
  --resume_training_state \
  --psro_weighting --psro_fifo_reweight --psro_start_epoch 0 \
  --dset_max_size 50000 \
  --fifo_teamset_weights 'smogon_pass2:4.0,smogon_pass2_selected:4.0' \
  --fifo_teamset_default_weight 1.0
```

**GOTCHA:** `EPOCHS` must exceed the resumed epoch (default 300 → learner exits
immediately if resuming from epoch ≥ 300). Always set `EPOCHS=3950` for
mini_online runs.

**GOTCHA:** `--resume_training_state` is incompatible with `--from_scratch` and
`--prev_run_dir`. The launch script auto-drops `--from_scratch`.

### To resume with a new buffer size (evict stale battles)

Just relaunch the learner with `--dset_max_size <new>`. On the first
`on_end_of_collection`, `_evict_oldest()` deletes the oldest files down to the
new cap. The collector and validator keep running — they don't need to restart.
This is how we dropped 300k → 50k mid-run without losing training state.

## Best practices

### Tuning the replay mixture

1. **Keep `--online_weight 0.40`** (the author's proven recipe) unless you have
   a specific reason to change it. The 60% offline floor prevents forgetting.
2. **Shrink `--dset_max_size` to control freshness**, not repeats. 50k gives
   3.6h turnover at 14k battles/hr collection. Going below ~20k risks the
   buffer dropping below `--dset_min_size` (default 5000) and stalling training.
3. **Use `--fifo_teamset_weights` to specialize** without changing the online
   weight. 4× is a good starting point; 6× pushes harder but raises per-teamset
   repeats (smogon at 4× + 40% online ≈ 47× per smogon battle).
4. **The repeat ratio formula** (bookmark this):
   `repeats = 91.2 × online_weight` at 14k battles/hr collection. To reduce
   repeats without lowering online weight, scale the collector (hard — the
   main process is GIL-bound at 100% CPU) or accept the ratio and optimize
   freshness via buffer size.

### What to watch in wandb

- `val/Average Win Rate in gen1ou_vs_TaurosV0-competitive` (validator run) —
  the generalization canary. Should stay ~45-55%. If it drops, the policy is
  forgetting diverse-team play → raise the gl_05_26 collection floor or lower
  the teamset up-sampling.
- `psro/<agent>/<teamset>/win_rate` (collector run) — per-teamset specialization
  signal. If smogon WRs climb from ~0.17, the up-sampling + fresh buffer is
  working. If flat after 50+ epochs, replay saturation is capping
  specialization → consider lowering online weight or raising up-sampling.
- `psro/<agent>/gl_05_26/win_rate` — should stay ~60%. If it drops, that's the
  forgetting signal.

### Don't scale the collector to fix a learner problem

The collector main process is GIL-bound at 100% CPU (1 core orchestrating 28
Node-sim workers). Adding more workers/lanes won't scale collection linearly —
it requires multi-process collection architecture changes. If the learner is
data-starved, the simpler fix is a smaller buffer (fresher data) or higher
teamset up-sampling (more focus on the data you have).

## Reference

- `metamon/rl/online_rl.py` — CLI, `create_online_experiment`, `run_online_rl`
- `metamon/rl/online_envs.py` — `build_online_mixture_dataset`,
  `StatsDropoutObservationSpace`, env factories
- `metamon/rl/metamon_to_amago.py` — `MetamonFIFODataset` (the FIFO sampler with
  opponent + teamset reweighting), `MetamonOnlineExperiment`
- `metamon/rl/online_psro.py` — PSRO-Lite CLI/config/sidecar helpers
- `scripts/launch_mini_online_v1.sh` — launch script (`learner` subcommand)
- See also: the `online-training` skill for the split-layout launch recipe and
  the `online-collector` / `online-validator` skills for the other two roles.
