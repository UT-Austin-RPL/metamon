#!/usr/bin/env bash
# Launch the from-scratch gen1ou online RL run "mini_online_v1" on a SINGLE GPU.
#
# Reproduces the registered SmallG1OnlineV0 run:
#   --base_model V2AGroupedV2DataAblation --from_scratch
#   (small GroupedV2 arch, ~12-15M params, GroupedObservationSpace, no computed stats)
#
# Hardware target: 1x RTX 5090 (32GB), compute cap 12.0, torch 2.12+cu130.
#   - No `accelerate launch`: AMAGO's internal Accelerator runs single-device
#     fine under plain `python -m`. No accelerate config required.
#   - Primary mode is `--mode both`: one process, one model copy on the GPU,
#     collects self-play then takes grad updates each epoch. This is the most
#     memory-efficient layout for a single GPU (no second process contending).
#
# Verification already passed (run before writing this script):
#   - all 46 opponent-pool checkpoints exist on jakegrigsby/metamon
#   - online_selfplay.yaml resolves (gen1ou, pac-base/pac-exploratory/pac-tauros, 5% replay)
#   - V2AGroupedV2DataAblation + --from_scratch => random init, no HF weight load
#   - hl_gen1ou.yaml / tauros_val.yaml parse; gl_05_26 / competitive team sets load for gen1ou
#
# Usage:
#   bash scripts/launch_mini_online_v1.sh run         [N_LANES]  # primary: --mode both (collect+learn+val)
#   bash scripts/launch_mini_online_v1.sh resume      [N_LANES]  # resume THIS run from latest training_state (speed-tuned)
#   bash scripts/launch_mini_online_v1.sh smoke                   # 1-epoch wiring smoke test
#   bash scripts/launch_mini_online_v1.sh collector   [N_LANES]   # optional split: rollouts only
#   bash scripts/launch_mini_online_v1.sh learner                 # optional split: grad updates only
#   bash scripts/launch_mini_online_v1.sh validator   [N_LANES]   # optional split: val only
set -euo pipefail

# ----------------------------------------------------------------------------
# Configurable paths / knobs (override via env vars)
# ----------------------------------------------------------------------------
RUN_NAME="${RUN_NAME:-mini_online_v1}"
SAVE_DIR="${SAVE_DIR:-$HOME/metamon_runs}/${RUN_NAME}"
BUFFER_DIR="${BUFFER_DIR:-${SAVE_DIR}/buffer}"
BASE_MODEL="${BASE_MODEL:-V2AGroupedV2DataAblation}"
DATASET_CONFIG="${DATASET_CONFIG:-online_selfplay.yaml}"
TRAIN_POOL="${TRAIN_POOL:-hl_gen1ou.yaml}"
VAL_POOL="${VAL_POOL:-tauros_val.yaml}"

# Opponent pools are opened verbatim (load_opponent_pool does open(config_path)
# with no config-dir resolution), so bare basenames must be resolved to absolute
# paths against the repo's opponent pool dir.
OPP_POOL_DIR="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/metamon/rl/configs/opponent_pools"
resolve_pool() { case "$1" in /*) echo "$1";; *) echo "${OPP_POOL_DIR}/$1";; esac; }
TRAIN_POOL="$(resolve_pool "$TRAIN_POOL")"
VAL_POOL="$(resolve_pool "$VAL_POOL")"
BATTLE_FORMAT="${BATTLE_FORMAT:-gen1ou}"
TRAIN_TEAM_SET="${TRAIN_TEAM_SET:-gl_05_26}"
VAL_TEAM_SET="${VAL_TEAM_SET:-competitive}"
# Optional team-mix schedule (epoch-driven curriculum YAML) and static mix specs.
# TRAIN_TEAM_SCHEDULE is required when the train pool uses "@schedule" agents.
# The *_MIX vars override the single-set vars when set.
TRAIN_TEAM_SCHEDULE="${TRAIN_TEAM_SCHEDULE:-}"
TRAIN_TEAM_MIX="${TRAIN_TEAM_MIX:-}"
VAL_TEAM_MIX="${VAL_TEAM_MIX:-}"
# Learner FIFO teamset up-sampling: aggressively shift the online mix toward
# trajectories where the learner drew a specific teamset (e.g. smogon). Empty
# = disabled. Format: 'set:mult,set:mult,...'. See --fifo_teamset_weights.
FIFO_TEAMSET_WEIGHTS="${FIFO_TEAMSET_WEIGHTS:-}"
FIFO_TEAMSET_DEFAULT_WEIGHT="${FIFO_TEAMSET_DEFAULT_WEIGHT:-1.0}"

# Single-GPU knobs. Defaults are conservative for a 32GB card; tune up if headroom.
#   BATCH_PER_GPU: grad-update batch size. 14 is the proven default; the small
#     (~15M) model likely fits 24-32 on a 5090 — raise via env if OOM-free.
#   LANES_BOTH:    vectorized sim battles per env batch in `both` mode. 128 fills
#     the FIFO buffer fast; GPU inference is one batched forward pass regardless
#     of lane count, so this is CPU-sim-bound, not GPU-bound.
#   N_WORKERS:     Node.js Showdown sim processes that shard the lanes. The sim
#     is single-threaded per worker, so n_workers=1 bottlenecks collection.
#     Default 16 spreads the sim across CPU cores (you have 32 threads), leaving
#     headroom for the 10 dataloader workers + main Python process.
#   STEPS_PER_EPOCH, TRAIN_TIMESTEPS_PER_EPOCH, EPOCHS: defaults match online_rl.py.
#   VAL_INTERVAL, VAL_TIMESTEPS: validation cadence. `run` defaults (1 / 1000)
#     match online_rl.py; the `resume` subcommand bumps these to (10 / 250) to
#     cut the dominant per-epoch validation cost (see the resume case below).
# Remember which speed knobs the user set explicitly via env, so the `resume`
# subcommand can raise its tuned defaults only where NOT overridden. This must
# run BEFORE the :- defaults below consume the env vars. (${VAR+x} is the
# set-ness test and is safe under `set -u`.)
_BATCH_SET=0
[[ -n "${BATCH_PER_GPU+x}" ]] && _BATCH_SET=1
_LANES_SET=0
[[ -n "${LANES_BOTH+x}" ]] && _LANES_SET=1
_NWORKERS_SET=0
[[ -n "${N_WORKERS+x}" ]] && _NWORKERS_SET=1
_VAL_INTERVAL_SET=0
[[ -n "${VAL_INTERVAL+x}" ]] && _VAL_INTERVAL_SET=1
_VAL_TIMESTEPS_SET=0
[[ -n "${VAL_TIMESTEPS+x}" ]] && _VAL_TIMESTEPS_SET=1
# EVAL_DURING_TRAINING: when enabled, the split learner pauses every VAL_INTERVAL
# epochs to eval the in-memory policy vs TaurosV0 for VAL_TIMESTEPS env steps and
# logs win rate to wandb. Accepts 1/0, on/off, true/false (case-insensitive);
# unset or empty disables.
EVAL_DURING_TRAINING="${EVAL_DURING_TRAINING:-0}"
_eval_on=0
case "${EVAL_DURING_TRAINING}" in
  1|on|ON|true|True|TRUE) _eval_on=1 ;;
esac
BATCH_PER_GPU="${BATCH_PER_GPU:-14}"
LANES_BOTH="${LANES_BOTH:-128}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-1000}"
TRAIN_TIMESTEPS_PER_EPOCH="${TRAIN_TIMESTEPS_PER_EPOCH:-500}"
EPOCHS="${EPOCHS:-300}"
N_WORKERS="${N_WORKERS:-16}"
DLOADER_WORKERS="${DLOADER_WORKERS:-10}"
VAL_INTERVAL="${VAL_INTERVAL:-1}"
VAL_TIMESTEPS="${VAL_TIMESTEPS:-1000}"
# Extra flags after the optional [N_LANES] positional are forwarded verbatim to
# `metamon.rl.online_rl`. e.g. `... run --log` adds --log, letting your existing
# WANDB_* env vars handle auth just like your other training runs.

# metamon requires this; HF checkpoints + teams cache live here
export METAMON_CACHE_DIR="${METAMON_CACHE_DIR:-/home/eddie/metamon_cache}"
# W&B destination (override via env vars if needed). metamon's online_rl.py
# reads METAMON_WANDB_ENTITY / METAMON_WANDB_PROJECT (not the standard WANDB_*
# vars). --log on the CLI turns logging on; these control where it lands.
export METAMON_WANDB_ENTITY="${METAMON_WANDB_ENTITY:-costacosta-personal-research}"
export METAMON_WANDB_PROJECT="${METAMON_WANDB_PROJECT:-metamon}"
mkdir -p "$METAMON_CACHE_DIR" "$SAVE_DIR" "$BUFFER_DIR/${BATTLE_FORMAT}"

# Activate the repo venv if present and not already active
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
if [[ -f "${REPO_ROOT}/.venv/bin/activate" && -z "${VIRTUAL_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi
cd "$REPO_ROOT"

# Parse the subcommand BEFORE building COMMON_ARGS so we can drop --from_scratch
# for `resume` and `learner` (both incompatible with --resume_training_state /
# --prev_run_dir) and apply speed-tuned knob defaults where the user didn't set
# env overrides.
mode="${1:-}"
shift || true

# --from_scratch is incompatible with --resume_training_state and --prev_run_dir
# (online_rl.py validates this). Auto-blank it whenever either resume flag is
# present in the forwarded args, for ANY subcommand. The `resume` subcommand
# always adds --resume_training_state itself (after this scan), so blank it
# explicitly for that mode too. (Unquoted ${FROM_SCRATCH} in COMMON_ARGS then
# expands to nothing.)
FROM_SCRATCH="--from_scratch"
if [[ "$mode" == "resume" ]]; then
  FROM_SCRATCH=""
fi
for _a in "$@"; do
  case "$_a" in
    --resume_training_state|--prev_run_dir) FROM_SCRATCH=""; break ;;
  esac
done

if [[ "$mode" == "resume" ]]; then
  # Speed-tuned defaults for the single-process resumed run; honor explicit env
  # overrides. (Guarded with `||` so `set -e` doesn't trip on `(( 0 ))`.)
  (( _BATCH_SET )) || BATCH_PER_GPU=32
  (( _LANES_SET )) || LANES_BOTH=256
  (( _NWORKERS_SET )) || N_WORKERS=28
  (( _VAL_INTERVAL_SET )) || VAL_INTERVAL=10
  (( _VAL_TIMESTEPS_SET )) || VAL_TIMESTEPS=250
fi
if [[ "$mode" == "collector" ]]; then
  # Collection is CPU-sim-bound; more Node showdown workers = faster rollouts.
  (( _NWORKERS_SET )) || N_WORKERS=28
fi
if [[ "$mode" == "learner" ]]; then
  # The learner is the only GPU consumer in the split layout, so a bigger batch
  # fills the otherwise-idle GPU. lanes/workers/val are irrelevant (no envs)
  # UNLESS EVAL_DURING_TRAINING is on, in which case val envs are built and kept
  # alive for periodic Tauros evals. Default to a light cadence (every 5 epochs,
  # 500 ts) when the user enables eval without overriding the knobs.
  (( _BATCH_SET )) || BATCH_PER_GPU=32
  if (( _eval_on )); then
    (( _VAL_INTERVAL_SET )) || VAL_INTERVAL=5
    (( _VAL_TIMESTEPS_SET )) || VAL_TIMESTEPS=500
  fi
fi

# Common args shared by every role. ${FROM_SCRATCH} is blank when resuming.
COMMON_ARGS=(
  --run_name "$RUN_NAME"
  --save_dir "$SAVE_DIR"
  --base_model "$BASE_MODEL"
  ${FROM_SCRATCH}
  --buffer_dir "$BUFFER_DIR"
  --dataset_config "$DATASET_CONFIG"
  --train_pool "$TRAIN_POOL"
  --val_pool "$VAL_POOL"
  --battle_format "$BATTLE_FORMAT"
  --train_team_set "$TRAIN_TEAM_SET"
  --val_team_set "$VAL_TEAM_SET"
  --batch_size_per_gpu "$BATCH_PER_GPU"
  --steps_per_epoch "$STEPS_PER_EPOCH"
  --epochs "$EPOCHS"
  --dloader_workers "$DLOADER_WORKERS"
)
# Conditionally forward optional team-mix schedule + static mix specs (only
# when the env var is set, so unset vars don't pass empty strings to argparse).
[[ -n "$TRAIN_TEAM_SCHEDULE" ]] && COMMON_ARGS+=(--train_team_schedule "$TRAIN_TEAM_SCHEDULE")
[[ -n "$TRAIN_TEAM_MIX" ]] && COMMON_ARGS+=(--train_team_mix "$TRAIN_TEAM_MIX")
[[ -n "$VAL_TEAM_MIX" ]] && COMMON_ARGS+=(--val_team_mix "$VAL_TEAM_MIX")
[[ -n "$FIFO_TEAMSET_WEIGHTS" ]] && COMMON_ARGS+=(--fifo_teamset_weights "$FIFO_TEAMSET_WEIGHTS")
[[ "${FIFO_TEAMSET_DEFAULT_WEIGHT:-1.0}" != "1.0" ]] && COMMON_ARGS+=(--fifo_teamset_default_weight "$FIFO_TEAMSET_DEFAULT_WEIGHT")

# After the mode, an optional leading [N_LANES] integer positional is consumed by
# run/collector/validator; everything remaining is forwarded verbatim to
# metamon.rl.online_rl so you can pass --log (and any other CLI flag) straight
# through, with your existing WANDB_* env vars handling auth the same way as your
# other training runs. e.g. `... run --log` or `... run 96 --log`.
case "$mode" in
  run)
    # PRIMARY single-GPU path: one process collects self-play, takes grad
    # updates, and runs validation each epoch (val_timesteps default 1000).
    LANES="$LANES_BOTH"
    if [[ "${1:-}" =~ ^[0-9]+$ ]]; then LANES="$1"; shift; fi
    echo ">> single-GPU run: --mode both, ${LANES} lanes, batch=${BATCH_PER_GPU}, save=${SAVE_DIR}"
    exec python -m metamon.rl.online_rl \
      --mode both "${COMMON_ARGS[@]}" \
      --train_timesteps_per_epoch "$TRAIN_TIMESTEPS_PER_EPOCH" \
      --lanes "$LANES" --n_workers "$N_WORKERS" \
      --val_interval "$VAL_INTERVAL" --val_timesteps "$VAL_TIMESTEPS" "$@"
    ;;

  resume)
    # Resume the SAME run (--run_name/--save_dir) from its newest full
    # accelerate training state (ckpts/training_states/<run_name>_epoch_<N>),
    # restoring model + optimizer + scheduler + PopArt + RNG, then continue to
    # --epochs. Drops --from_scratch (incompatible with --resume_training_state).
    # Speed-tuned defaults (rare validation, more sim workers/lanes, bigger
    # batch) apply unless overridden via env. Run
    #   sudo cpupower frequency-set -g performance
    # first — collection/validation are CPU-sim-bound and the default
    # `powersave` governor throttles the single-threaded Node showdown sim.
    # Optional [N_LANES] positional overrides --lanes (default ${LANES_BOTH}).
    # To pin a specific epoch instead of the newest, forward --resume_epoch N
    # (e.g. `... resume --log --resume_epoch 10`); it passes through via "$@".
    LANES="$LANES_BOTH"
    if [[ "${1:-}" =~ ^[0-9]+$ ]]; then LANES="$1"; shift; fi
    echo ">> resume: --mode both, ${LANES} lanes, batch=${BATCH_PER_GPU}, val every ${VAL_INTERVAL} epochs (${VAL_TIMESTEPS} ts), save=${SAVE_DIR}"
    exec python -m metamon.rl.online_rl \
      --mode both "${COMMON_ARGS[@]}" \
      --train_timesteps_per_epoch "$TRAIN_TIMESTEPS_PER_EPOCH" \
      --lanes "$LANES" --n_workers "$N_WORKERS" \
      --val_interval "$VAL_INTERVAL" --val_timesteps "$VAL_TIMESTEPS" \
      --resume_training_state "$@"
    ;;

  smoke)
    # 1-epoch wiring smoke test: tiny lanes/steps to sanity-check the pipeline.
    echo ">> smoke test: single-process --mode both, minimal work"
    exec python -m metamon.rl.online_rl \
      --mode both "${COMMON_ARGS[@]}" \
      --lanes 2 --epochs 1 --train_timesteps_per_epoch 5 --steps_per_epoch 2 \
      --dset_min_size 0 --val_timesteps 10 "$@"
    ;;

  # --- Optional split roles (only if you want dedicated processes sharing the GPU) ---
  # The split layout overlaps the CPU-bound collector with the GPU-bound learner:
  # the collector fills the FIFO buffer (self-play, no grad updates) and syncs to
  # the learner's rolling latest/policy.pt each epoch; the learner drains the
  # buffer and takes grad updates continuously. This keeps the GPU fed during the
  # phases that are CPU-sim-bound in single-process --mode both. Launch in two
  # terminals (or background one):
  #   bash scripts/launch_mini_online_v1.sh collector --log
  #   bash scripts/launch_mini_online_v1.sh learner --log --resume_training_state
  # The collector must see the learner's latest/policy.pt, so start the learner
  # first (or at the same time — the collector's read_latest_policy is a no-op
  # until the first policy.pt appears).
  collector)
    # Self-play rollouts only. Default lanes=128 (matches the proven `run`
    # collect rate; 256 doubled per-timestep sim work without speeding training).
    # n_workers=28 spreads the sim across CPU cores. Override via [N_LANES] or env.
    LANES="${COLLECTOR_LANES:-128}"
    if [[ "${1:-}" =~ ^[0-9]+$ ]]; then LANES="$1"; shift; fi
    echo ">> collector: ${LANES} lanes, ${N_WORKERS} workers, buffer=${BUFFER_DIR} (no grad updates, no val)"
    exec python -m metamon.rl.online_rl \
      --mode collect "${COMMON_ARGS[@]}" \
      --lanes "$LANES" --n_workers "$N_WORKERS" "$@"
    ;;

  learner)
    # Grad updates only (drains the FIFO buffer the collector fills). The learner
    # writes latest/policy.pt each epoch so the collector syncs to fresh weights.
    # To RESUME an existing run, forward --resume_training_state (optionally
    # --resume_epoch N); --from_scratch is auto-dropped so they don't conflict:
    #   bash scripts/launch_mini_online_v1.sh learner --log --resume_training_state
    #   bash scripts/launch_mini_online_v1.sh learner --log --resume_training_state --resume_epoch 10
    # For a brand-new run (random init), omit the resume flag and --from_scratch
    # applies as usual. batch=32 by default (set above, before COMMON_ARGS) — the
    # learner is the only thing using the GPU in the split layout, so a bigger
    # batch fills it.
    echo ">> learner: grad updates only, batch=${BATCH_PER_GPU}, save=${SAVE_DIR}${FROM_SCRATCH:+ (from scratch)}"
    if (( _eval_on )); then
      echo ">>   + periodic eval vs TaurosV0 every ${VAL_INTERVAL} epochs (${VAL_TIMESTEPS} ts, ${VAL_LANES:-32} lanes / ${VAL_N_WORKERS:-8} workers), logged to wandb"
    fi
    _eval_args=()
    (( _eval_on )) && _eval_args=(--eval_during_training --val_interval "$VAL_INTERVAL" --val_timesteps "$VAL_TIMESTEPS" --lanes "${VAL_LANES:-32}" --n_workers "${VAL_N_WORKERS:-8}")
    exec python -m metamon.rl.online_rl \
      --mode learn "${COMMON_ARGS[@]}" \
      --train_timesteps_per_epoch "$TRAIN_TIMESTEPS_PER_EPOCH" "${_eval_args[@]}" "$@"
    ;;

  validator)
    LANES=32
    if [[ "${1:-}" =~ ^[0-9]+$ ]]; then LANES="$1"; shift; fi
    echo ">> validator: vs TaurosV0, reloads latest/policy.pt each epoch"
    exec python -m metamon.rl.online_rl \
      --mode validate "${COMMON_ARGS[@]}" \
      --lanes "$LANES" "$@"
    ;;

  *)
    echo "Usage: $0 {run [N_LANES] | resume [N_LANES] | smoke | collector [N_LANES] | learner | validator [N_LANES]}" >&2
    echo "  run        — PRIMARY: --mode both, single process, single GPU (default ${LANES_BOTH} lanes, ${N_WORKERS} sim workers)" >&2
    echo "  resume     — resume THIS run from its latest training_state (drops --from_scratch; speed-tuned: val every 10 epochs, 256 lanes, 28 workers, batch 32)" >&2
    echo "  smoke      — 1-epoch wiring smoke test" >&2
    echo "  collector  — split: self-play rollouts only (default 128 lanes, ${N_WORKERS} workers; syncs to learner's latest/policy.pt)" >&2
    echo "  learner    — split: grad updates only (batch 32; add --resume_training_state to resume THIS run; set EVAL_DURING_TRAINING=1 to eval vs Tauros every 5 epochs)" >&2
    echo "  validator  — split: reload latest/policy.pt each epoch vs TaurosV0 (default 32 lanes)" >&2
    exit 1
    ;;
esac
