---
name: online-training
description: "Launch and resume online RL training runs in this repo (metamon.rl.online_rl via scripts/launch_*.sh). Use when the user asks to start, launch, kick off, resume, restart, recover, or relaunch an online RL training run, or asks about online training layout / wandb runs / checkpoints / the launch scripts."
---

# Online RL Training — Launch & Resume

`metamon.rl.online_rl` runs online RL finetuning with three roles that can run
as one process or as separate processes:

- **learner** (`--mode learn`) — gradient updates; the only GPU consumer; writes
  `ckpts/latest/policy.pt` each epoch and `ckpts/training_states/<run>_epoch_<N>/`
  every `ckpt_interval` epochs (full accelerate state: model + optimizer +
  scheduler + PopArt + RNG).
- **collector** (`--mode collect`) — self-play rollouts into the FIFO buffer
  (`buffer_dir/<format>/`); CPU-sim-bound; syncs to the learner's
  `latest/policy.pt` each epoch (`always_load_latest=True`).
- **validator** (`--mode validate`) — evaluates `latest/policy.pt` vs the val
  opponent each epoch; CPU-sim-bound; reloads weights every epoch.

## The two layouts

### Split layout (preferred for real runs) — 3 processes, 3 wandb runs

One process per role, launched separately. Overlaps the CPU-bound collector and
validator with the GPU-bound learner so the GPU stays fed during sim-bound
phases. Each process logs to its **own** wandb run (so you see 3 runs in wandb).

```bash
# Learner first — it owns the GPU and publishes latest/policy.pt.
EPOCHS=3950 nohup bash scripts/launch_mini_online_v1.sh learner --log \
  > ~/metamon_runs/learner.log 2>&1 &
# Collector and validator sync to latest/policy.pt (which may already exist
# from a prior run — they start immediately and pick up fresh weights each
# epoch once the learner publishes a new one).
EPOCHS=3950 nohup bash scripts/launch_mini_online_v1.sh collector --log \
  > ~/metamon_runs/collector.log 2>&1 &
EPOCHS=3950 nohup bash scripts/launch_mini_online_v1.sh validator --log \
  > ~/metamon_runs/validator.log 2>&1 &
```

### Single-process `--mode both` — 1 wandb run

One process collects, learns, and validates. Most memory-efficient on a single
GPU and fine for smoke tests or quick runs, but it serializes the CPU-sim-bound
phases with GPU updates and logs only **one** wandb run. If the user expects to
see multiple wandb runs (because their original run used the split layout), use
the split layout above instead.

```bash
EPOCHS=3950 nohup bash scripts/launch_mini_online_v1.sh run --log \
  > ~/metamon_runs/run.log 2>&1 &
```

## Resume after a crash

The run is fully resumable as long as `ckpts/training_states/<run>_epoch_<N>/`
and `buffer_dir/` survived. Three things resume:

1. **Training state** — the learner reloads the newest
   `training_states/<run>_epoch_<N>/` (full accelerate state) and continues at
   that epoch. Driven by `--resume_training_state`.
2. **Policy weights** — `latest/policy.pt` (and `policy_epoch_<N>.pt`) are on
   disk; collector and validator sync to them.
3. **Data collection** — the FIFO buffer is append-only; collection just keeps
   adding trajectories. No replay/re-derive needed.

```bash
# Split-layout resume: learner resumes state; collector+validator sync to
# the existing latest/policy.pt and keep going.
EPOCHS=3950 nohup bash scripts/launch_mini_online_v1.sh learner --log --resume_training_state \
  > ~/metamon_runs/learner.log 2>&1 &
EPOCHS=3950 nohup bash scripts/launch_mini_online_v1.sh collector --log \
  > ~/metamon_runs/collector.log 2>&1 &
EPOCHS=3950 nohup bash scripts/launch_mini_online_v1.sh validator --log \
  > ~/metamon_runs/validator.log 2>&1 &
```

The `resume` subcommand does the single-process (`--mode both`) resume
equivalent with speed-tuned defaults (val every 10 epochs, 256 lanes, 28
workers, batch 32):

```bash
EPOCHS=3950 bash scripts/launch_mini_online_v1.sh resume --log
```

### GOTCHA: `EPOCHS` must exceed the resumed epoch

The launch scripts default `EPOCHS=300`. When resuming from a checkpoint at
epoch N ≥ 300, the learner exits immediately ("0 epochs remaining"). **Always
set `EPOCHS=<target>` on resume** to the run's original target (e.g. `3950` for
`mini_online_v1`, per its `SmallG1OnlineV0` registration in
`metamon/rl/pretrained.py`).

### How `--resume_training_state` interacts with other flags

- **Incompatible with `--from_scratch`** and `--prev_run_dir`. The launch script
  auto-drops `--from_scratch` whenever `--resume_training_state` (or
  `--prev_run_dir`) is present in the forwarded args.
- `--resume_epoch N` pins a specific epoch instead of the newest saved.
- Relaunch the learner with the **same accelerate / GPU config** used originally
  (single-GPU plain `python -m` here — no `accelerate launch` needed on one
  5090).

## Before launching: set the CPU governor to performance

Collection and validation are CPU-sim-bound (single-threaded Node Showdown sim
workers). The default `powersave` governor throttles them. Run once before
launch:

```bash
sudo cpupower frequency-set -g performance
```

## Where things live

For a run with `--save_dir $SAVE_DIR --run_name $RUN_NAME --buffer_dir $BUFFER_DIR`:

```
$SAVE_DIR/$RUN_NAME/
├── ckpts/
│   ├── latest/policy.pt                    # rolling; collector+validator sync here
│   ├── policy_weights/policy_epoch_<N>.pt  # every ckpt_interval epochs
│   ├── training_states/<run>_epoch_<N>/    # full accelerate state (resume source)
│   └── config.txt                          # resolved gin config
├── wandb_logs/wandb/run-*/                 # per-process wandb run dirs
└── dataset_config.yaml
$BUFFER_DIR/<format>/                        # FIFO self-play trajectories (append-only)
```

## Key env vars (read by scripts/launch_*.sh)

- `EPOCHS` — target epoch count. **Set this on resume** (default 300).
- `RUN_NAME`, `SAVE_DIR`, `BUFFER_DIR` — defaults to
  `~/metamon_runs/$RUN_NAME` and `${SAVE_DIR}/buffer`.
- `BASE_MODEL`, `DATASET_CONFIG`, `TRAIN_POOL`, `VAL_POOL`, `BATTLE_FORMAT`,
  `TRAIN_TEAM_SET`, `VAL_TEAM_SET`, `TRAIN_TEAM_MIX`, `VAL_TEAM_MIX`,
  `TRAIN_TEAM_SCHEDULE` — defaults match the registered run. The `*_MIX` vars
  are optional weighted-mix specs (`'set:weight,set:weight,...'`) that override
  the single-set vars when set. `TRAIN_TEAM_SCHEDULE` is a path to a schedule
  YAML that shifts the collection team mix automatically at epoch boundaries
  (no restart needed); the opponent pool follows the same schedule when its
  `team_set` is `"@schedule"`. See `docs/teamset_curriculum_proposal.md`.
  `launch_psro_v1.sh` accepts `RESUME=1` to resume from the latest training
  state (instead of bootstrapping from `--prev_run_dir`).
- `BATCH_PER_GPU`, `LANES_BOTH`, `N_WORKERS`, `STEPS_PER_EPOCH`,
  `TRAIN_TIMESTEPS_PER_EPOCH`, `DLOADER_WORKERS`, `VAL_INTERVAL`,
  `VAL_TIMESTEPS` — speed knobs; the `resume`/`learner`/`collector` subcommands
  raise tuned defaults where you haven't set them explicitly.
- `EVAL_DURING_TRAINING=1` — split learner periodically evals vs TaurosV0 and
  logs win rate to wandb.
- `METAMON_CACHE_DIR` — HF checkpoints + teams cache (default
  `/home/eddie/metamon_cache`).
- `METAMON_WANDB_ENTITY` / `METAMON_WANDB_PROJECT` — wandb destination (NOT the
  standard `WANDB_*` vars). `--log` on the CLI turns logging on.

## Survival across shell exit / reboot

`nohup ... &` survives the shell but **not** a machine restart. For
crash-proof survival wrap each process in `tmux` or a `systemd` service.

## Reference

- `scripts/launch_mini_online_v1.sh` — the launch script (subcommands: `run`,
  `resume`, `smoke`, `collector`, `learner`, `validator`). Header comment
  documents every env var and the split-vs-both tradeoff.
- `metamon/rl/online_rl.py` — the `--mode` logic, `--resume_training_state`,
  `_latest_training_state_epoch`, and the `__main__` resume flow.
- `metamon/rl/pretrained.py` — registered runs (`SmallG1OnlineV0` =
  `mini_online_v1`, target epoch 3950, `ckpt_interval=50`) and where
  `MINI_ONLINE_V1_SAVE_DIR` is defined.
