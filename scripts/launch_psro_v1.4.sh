#!/usr/bin/env bash
# Launch the split-layout PSRO-Lite run "mini_online_psro_v1.4" in tmux.
#
# v1.4 is a clean break from v1.3: bootstraps policy WEIGHTS from v1.3 epoch
# 1800 (the latest clean checkpoint after the gl-specialist incident + rollback)
# via --prev_run_dir, but starts a FRESH training state (optimizer / scheduler /
# PopArt / RNG reset, epoch counter restarts at 0). Uses the maintenance team
# schedule (gen1ou_competitive_maintenance_v1.4.yaml) — the policy already
# completed the competitive curriculum in v1.3, so v1.4 starts directly at the
# 15/20/65 (gl/pass2/selected) maintenance mix.
#
# Two modes:
#   Default (first launch): boots from v1.3 epoch-1800 policy via --prev_run_dir.
#     PSRO-Lite active from epoch 0 (--psro_start_epoch 0).
#   RESUME=1: resumes THIS v1.4 run from its newest training_state via
#     --resume_training_state (full accelerate state). Use after a crash or to
#     apply a config change without losing epochs.
#
# Three processes, three tmux windows in session "mini_online_psro_v1_4":
#   learner    — grad updates; writes latest/policy.pt each epoch;
#                --psro_fifo_reweight reads the sidecar; 4x smogon up-sampling;
#                --dset_max_size 50000 (3.6h freshness at 14k battles/hr).
#   collector  — self-play rollouts into the v1.4 buffer; --psro_weighting
#                writes the meta_weights.json sidecar; maintenance schedule.
#   validator  — reloads latest/policy.pt each epoch vs TaurosV0 on competitive
#                (the generalization canary); unaffected by PSRO/schedule.
#
# The PSRO opponent pool (hl_gen1ou.yaml) has 11 agents: the 9 fixed external
# gl-era models (TaurosV0, Kakuna, Alakazam, SyntheticRLV2, V2ADataAblation,
# Superkazam, Kadabra3) + 2 smogon-side self-play specialists from v1.3
# (MiniOnlinePsroV1_3 on @schedule, MiniOnlinePsroV1_3P2 on smogon_pass2).
# NO gl-pinned specialists — see the warning in hl_gen1ou.yaml.
#
# Override via env vars:
#   EPOCHS (default 3950), SESSION, RUN_NAME, RESUME=1, PSRO_QUOTA_*,
#   FIFO_TEAMSET_WEIGHTS (default 'smogon_pass2:4.0,smogon_pass2_selected:4.0').
#   To reuse the clean v1.3 buffer instead of a fresh one, set
#   BUFFER_DIR=$HOME/metamon_runs/mini_online_v1/buffer (the v1.3 buffer was
#   purged of harmful files on 2026-07-24 and is clean at ~50k trajectories).
#
# Before launching: sudo cpupower frequency-set -g performance
#
# Attach with: tmux attach -t mini_online_psro_v1_4
set -euo pipefail

SESSION="${SESSION:-mini_online_psro_v1_4}"   # tmux session (dot-free)
RUN_NAME="${RUN_NAME:-mini_online_psro_v1.4}" # ckpt dir + wandb run name
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREV_SAVE_DIR="$HOME/metamon_runs/mini_online_psro_v1.3"
PREV_RUN_NAME="mini_online_psro_v1.3"
PREV_CKPT="1800"
EPOCHS="${EPOCHS:-3950}"
RESUME="${RESUME:-0}"

cd "$REPO_ROOT"

METAMON_CACHE_DIR="${METAMON_CACHE_DIR:-/home/eddie/metamon_cache}"
METAMON_WANDB_ENTITY="${METAMON_WANDB_ENTITY:-costacosta-personal-research}"
METAMON_WANDB_PROJECT="${METAMON_WANDB_PROJECT:-metamon}"
TRAIN_TEAM_SCHEDULE="${TRAIN_TEAM_SCHEDULE:-$REPO_ROOT/metamon/rl/configs/team_schedules/gen1ou_competitive_maintenance_v1.4.yaml}"
FIFO_TEAMSET_WEIGHTS="${FIFO_TEAMSET_WEIGHTS:-smogon_pass2:4.0,smogon_pass2_selected:4.0}"
FIFO_TEAMSET_DEFAULT_WEIGHT="${FIFO_TEAMSET_DEFAULT_WEIGHT:-1.0}"
PSRO_QUOTA_MIN_GAMES="${PSRO_QUOTA_MIN_GAMES:-200}"
PSRO_QUOTA_WINDOW="${PSRO_QUOTA_WINDOW:-256}"
export METAMON_CACHE_DIR METAMON_WANDB_ENTITY METAMON_WANDB_PROJECT

if [[ "$RESUME" == "1" ]]; then
  echo ">> RESUME mode: --resume_training_state (continues from newest v1.4 training_state)"
  RESUME_FLAGS="--resume_training_state"
  BUFFER_TRIM=""
else
  echo ">> First-launch mode: --prev_run_dir (boots from $PREV_RUN_NAME epoch $PREV_CKPT)"
  RESUME_FLAGS="--prev_run_dir $PREV_SAVE_DIR --prev_run_name $PREV_RUN_NAME --prev_checkpoint $PREV_CKPT"
  BUFFER_TRIM="--psro_buffer_trim 50000"
fi

tmux kill-session -t "$SESSION" 2>/dev/null || true

# --- Learner (GPU; grad updates; writes latest/policy.pt each epoch) ----------
tmux new-session -d -s "$SESSION" -n learner \
  "cd $REPO_ROOT && EPOCHS=$EPOCHS RUN_NAME=$RUN_NAME \
     METAMON_CACHE_DIR=$METAMON_CACHE_DIR \
     METAMON_WANDB_ENTITY=$METAMON_WANDB_ENTITY \
     METAMON_WANDB_PROJECT=$METAMON_WANDB_PROJECT \
     TRAIN_TEAM_SCHEDULE=$TRAIN_TEAM_SCHEDULE \
     FIFO_TEAMSET_WEIGHTS='$FIFO_TEAMSET_WEIGHTS' \
     FIFO_TEAMSET_DEFAULT_WEIGHT=$FIFO_TEAMSET_DEFAULT_WEIGHT \
   bash scripts/launch_mini_online_v1.sh learner --log \
     $RESUME_FLAGS \
     --psro_weighting --psro_fifo_reweight --psro_start_epoch 0 \
     --dset_max_size 50000 \
   2>&1 | tee $HOME/metamon_runs/${SESSION}_learner.log; echo '[learner exited]'; bash"

# --- Collector (CPU-sim; self-play rollouts; writes the PSRO sidecar) --------
tmux new-window -t "$SESSION" -n collector \
  "cd $REPO_ROOT && EPOCHS=$EPOCHS RUN_NAME=$RUN_NAME \
     METAMON_CACHE_DIR=$METAMON_CACHE_DIR \
     METAMON_WANDB_ENTITY=$METAMON_WANDB_ENTITY \
     METAMON_WANDB_PROJECT=$METAMON_WANDB_PROJECT \
     TRAIN_TEAM_SCHEDULE=$TRAIN_TEAM_SCHEDULE \
   bash scripts/launch_mini_online_v1.sh collector --log \
     $RESUME_FLAGS \
     --psro_weighting --psro_start_epoch 0 $BUFFER_TRIM \
     --psro_temp 2.0 --psro_floor 0.01 \
     --psro_quota_min_games '$PSRO_QUOTA_MIN_GAMES' --psro_quota_window '$PSRO_QUOTA_WINDOW' \
   2>&1 | tee $HOME/metamon_runs/${SESSION}_collector.log; echo '[collector exited]'; bash"

# --- Validator (CPU-sim; reloads latest/policy.pt each epoch vs TaurosV0) -----
tmux new-window -t "$SESSION" -n validator \
  "cd $REPO_ROOT && EPOCHS=$EPOCHS RUN_NAME=$RUN_NAME \
     METAMON_CACHE_DIR=$METAMON_CACHE_DIR \
     METAMON_WANDB_ENTITY=$METAMON_WANDB_ENTITY \
     METAMON_WANDB_PROJECT=$METAMON_WANDB_PROJECT \
   bash scripts/launch_mini_online_v1.sh validator --log \
     $RESUME_FLAGS \
   2>&1 | tee $HOME/metamon_runs/${SESSION}_validator.log; echo '[validator exited]'; bash"

echo "tmux session '$SESSION' launched with 3 windows: learner, collector, validator."
echo "  run name:    $RUN_NAME"
echo "  bootstrap:   $PREV_RUN_NAME epoch $PREV_CKPT (weights only; fresh optimizer)"
echo "  schedule:    $(basename "$TRAIN_TEAM_SCHEDULE") (maintenance: 15/20/65)"
echo "  buffer:      fresh at \$SAVE_DIR/buffer (set BUFFER_DIR to reuse v1.3's)"
echo "  attach:      tmux attach -t $SESSION"
echo "  logs:        ~/metamon_runs/${SESSION}_{learner,collector,validator}.log"
