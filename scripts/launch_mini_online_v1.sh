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

# Single-GPU knobs. Defaults are conservative for a 32GB card; tune up if headroom.
#   BATCH_PER_GPU: grad-update batch size. 14 is the proven default; the small
#     (~15M) model likely fits 24-32 on a 5090 — raise via env if OOM-free.
#   LANES_BOTH:    vectorized sim battles per env batch in `both` mode. 64 keeps
#     collection+learning balanced on one process; raise for faster buffer fill.
#   STEPS_PER_EPOCH, TRAIN_TIMESTEPS_PER_EPOCH, EPOCHS: defaults match online_rl.py.
BATCH_PER_GPU="${BATCH_PER_GPU:-14}"
LANES_BOTH="${LANES_BOTH:-64}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-1000}"
TRAIN_TIMESTEPS_PER_EPOCH="${TRAIN_TIMESTEPS_PER_EPOCH:-500}"
EPOCHS="${EPOCHS:-300}"
N_WORKERS="${N_WORKERS:-1}"
DLOADER_WORKERS="${DLOADER_WORKERS:-10}"
# W&B logging is off by default (the hardcoded ut-austin-rpl-metamon entity is
# not writable by outside users). Set WANDB_LOG=1 AND override the entity/project
# via WANDB_ENTITY / WANDB_PROJECT to enable. The online_rl.py defaults are
# ut-austin-rpl-metamon / online-metamon.
WANDB_LOG="${WANDB_LOG:-0}"
LOG_FLAG=""
if [[ "$WANDB_LOG" == "1" ]]; then LOG_FLAG="--log"; fi

# metamon requires this; HF checkpoints + teams cache live here
export METAMON_CACHE_DIR="${METAMON_CACHE_DIR:-/home/eddie/metamon_cache}"
mkdir -p "$METAMON_CACHE_DIR" "$SAVE_DIR" "$BUFFER_DIR/${BATTLE_FORMAT}"

# Activate the repo venv if present and not already active
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
if [[ -f "${REPO_ROOT}/.venv/bin/activate" && -z "${VIRTUAL_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi
cd "$REPO_ROOT"

# Common args shared by every role
COMMON_ARGS=(
  --run_name "$RUN_NAME"
  --save_dir "$SAVE_DIR"
  --base_model "$BASE_MODEL"
  --from_scratch
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

mode="${1:-}"
shift || true

case "$mode" in
  run)
    # PRIMARY single-GPU path: one process collects self-play, takes grad
    # updates, and runs validation each epoch (val_timesteps default 1000).
    LANES="${1:-$LANES_BOTH}"
    echo ">> single-GPU run: --mode both, ${LANES} lanes, batch=${BATCH_PER_GPU}, save=${SAVE_DIR}"
    exec python -m metamon.rl.online_rl \
      --mode both "${COMMON_ARGS[@]}" \
      --train_timesteps_per_epoch "$TRAIN_TIMESTEPS_PER_EPOCH" \
      --lanes "$LANES" --n_workers "$N_WORKERS" $LOG_FLAG
    ;;

  smoke)
    # 1-epoch wiring smoke test: tiny lanes/steps to sanity-check the pipeline.
    echo ">> smoke test: single-process --mode both, minimal work"
    exec python -m metamon.rl.online_rl \
      --mode both "${COMMON_ARGS[@]}" \
      --lanes 2 --epochs 1 --train_timesteps_per_epoch 5 --steps_per_epoch 2 \
      --dset_min_size 0 --val_timesteps 10 $LOG_FLAG
    ;;

  # --- Optional split roles (only if you want dedicated processes sharing the GPU) ---
  collector)
    LANES="${1:-256}"
    echo ">> collector: ${LANES} lanes, buffer=${BUFFER_DIR} (no grad updates, no val)"
    exec python -m metamon.rl.online_rl \
      --mode collect "${COMMON_ARGS[@]}" \
      --lanes "$LANES" --n_workers "$N_WORKERS"
    ;;

  learner)
    # Grad updates only. Plain `python -m` (no accelerate launch) on 1 GPU.
    echo ">> learner: grad updates only, batch=${BATCH_PER_GPU}, save=${SAVE_DIR}"
    exec python -m metamon.rl.online_rl \
      --mode learn "${COMMON_ARGS[@]}" \
      --train_timesteps_per_epoch "$TRAIN_TIMESTEPS_PER_EPOCH" $LOG_FLAG
    ;;

  validator)
    LANES="${1:-32}"
    echo ">> validator: vs TaurosV0, reloads latest/policy.pt each epoch"
    exec python -m metamon.rl.online_rl \
      --mode validate "${COMMON_ARGS[@]}" \
      --lanes "$LANES" $LOG_FLAG
    ;;

  *)
    echo "Usage: $0 {run [N_LANES] | smoke | collector [N_LANES] | learner | validator [N_LANES]}" >&2
    echo "  run        — PRIMARY: --mode both, single process, single GPU (default ${LANES_BOTH} lanes)" >&2
    echo "  smoke      — 1-epoch wiring smoke test" >&2
    echo "  collector  — split: self-play rollouts only (default 256 lanes)" >&2
    echo "  learner    — split: grad updates only (no accelerate launch; 1 GPU)" >&2
    echo "  validator  — split: reload latest/policy.pt each epoch vs TaurosV0 (default 32 lanes)" >&2
    exit 1
    ;;
esac
