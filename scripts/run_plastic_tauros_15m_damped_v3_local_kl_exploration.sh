#!/usr/bin/env bash
set -euo pipefail

export METAMON_CACHE_DIR="${METAMON_CACHE_DIR:-/home/eddie/metamon_cache}"
export METAMON_WANDB_ENTITY="${METAMON_WANDB_ENTITY:-costacosta-personal-research}"
export METAMON_WANDB_PROJECT="${METAMON_WANDB_PROJECT:-metamon}"
export WANDB_MODE="${WANDB_MODE:-online}"

SAVE_DIR="${SAVE_DIR:-/home/eddie/metamon/models/plastic_tauros_15m_damped_v3_local_kl}"
EPOCHS="${EPOCHS:-1}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-1000}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-12}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
DLOADER_WORKERS="${DLOADER_WORKERS:-10}"
CKPT_INTERVAL="${CKPT_INTERVAL:-1}"
EVAL_GENS="${EVAL_GENS-1}"
RUNS="${RUNS:-small-damped-v3-local8 small-damped-v3-local16 small-damped-v3-local32}"

run_one() {
  local run_name="$1"
  local train_gin
  local dataset_config

  case "$run_name" in
    small-damped-v3-local8)
      train_gin="plastic_tauros_15m_damped_v3_local8.gin"
      dataset_config="gen1ou_small_tauros_core.yaml"
      ;;
    small-damped-v3-local16)
      train_gin="plastic_tauros_15m_damped_v3_local16.gin"
      dataset_config="gen1ou_small_tauros_core.yaml"
      ;;
    small-damped-v3-local32)
      train_gin="plastic_tauros_15m_damped_v3_local32.gin"
      dataset_config="gen1ou_small_tauros_core.yaml"
      ;;
    *)
      echo "Unknown run: $run_name" >&2
      exit 2
      ;;
  esac

  uv run python -m metamon.rl.train \
    --run_name "$run_name" \
    --save_dir "$SAVE_DIR" \
    --model_gin_config smaller_multitaskagent_grouped_v2_arch.gin \
    --train_gin_config "$train_gin" \
    --dataset_config "$dataset_config" \
    --obs_space GroupedObservationSpace \
    --action_space MinimalActionSpace \
    --reward_function AggressiveShapedReward \
    --eval_gens $EVAL_GENS \
    --epochs "$EPOCHS" \
    --steps_per_epoch "$STEPS_PER_EPOCH" \
    --ckpt_interval "$CKPT_INTERVAL" \
    --batch_size_per_gpu "$BATCH_SIZE_PER_GPU" \
    --grad_accum "$GRAD_ACCUM" \
    --dloader_workers "$DLOADER_WORKERS" \
    --async_env_mp_context forkserver \
    --log
}

for run_name in $RUNS; do
  run_one "$run_name"
done
