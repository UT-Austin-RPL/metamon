#!/usr/bin/env bash
# Launch the split-layout PSRO-Lite run "mini_online_smogon_v0" in tmux.
#
# A FROM-SCRATCH (random init) gen1ou online RL run with a ~35M-param
# TaurosV0-inspired GroupedV2 architecture (V2AGroupedV2Tauros35M /
# grouped_v2_35m.gin). The online collection universe is SMOGON-ONLY:
#   smogon_pass2          — 475 competitive-breadth teams
#   smogon_pass2_selected — 283 battle-validated winners (up-weighted over time)
# gl_05_26 is deliberately EXCLUDED from online collection; the 60% offline
# floor (online_selfplay.yaml: gl-like pac-base / pac-exploratory / pac-tauros
# self-play) provides the diverse-team grounding. smogon_pass2_selected ramps
# 0% -> 75% of the collector mix over collector epochs 0-900 (see
# metamon/rl/configs/team_schedules/gen1ou_smogon_curriculum_v0.yaml).
#
# Two modes:
#   Default (first launch): --from_scratch (random init). PSRO-Lite active from
#     epoch 0 (--psro_start_epoch 0).
#   RESUME=1: resumes THIS run from its newest training_state via
#     --resume_training_state (full accelerate state: model + optimizer +
#     scheduler + PopArt + RNG). Use after a crash or to apply a config change
#     without losing epochs.
#
# Three processes, three tmux windows in session "mini_online_smogon_v0":
#   learner    — grad updates; writes latest/policy.pt each epoch;
#                --psro_fifo_reweight reads the sidecar in its FIFO sampler;
#                4x smogon_pass2_selected up-sampling; dset_max_size 50000
#                (3.6h freshness at 14k battles/hr); 35M Tauros-inspired arch.
#   collector  — self-play rollouts into the fresh smogon_v0 buffer;
#                --psro_weighting writes the meta_weights.json sidecar;
#                smogon-only schedule; psro_temp 2.0 / floor 0.01.
#   validator  — reloads latest/policy.pt each epoch vs TaurosV0 on competitive
#                (the 50%-trigger canary + generalization canary); unaffected
#                by PSRO/schedule.
#
# The PSRO opponent pool (hl_gen1ou.yaml) has 11 agents: the 9 fixed external
# gl-era models (TaurosV0, Kakuna, Alakazam, SyntheticRLV2, V2ADataAblation,
# Superkazam, Kadabra3) + 2 smogon-side self-play specialists from v1.3
# (MiniOnlinePsroV1_3 on @schedule, MiniOnlinePsroV1_3P2 on smogon_pass2).
# All @schedule agents draw smogon via the smogon-only schedule. NO gl-pinned
# specialists (see the warning in hl_gen1ou.yaml). Past selves from THIS run
# are added incrementally (cap 5) as the Tauros canary crosses 50% / hits new
# highs — add rows to hl_gen1ou.yaml + relaunch the collector (the learner reads
# the updated sidecar automatically).
#
# Override via env vars:
#   EPOCHS (default 3950 — learner ~4.1 days at 40 ep/hr; collector+validator
#     run forever, stopped when the learner exits), SESSION, RUN_NAME, RESUME=1,
#   PSRO_QUOTA_*, FIFO_TEAMSET_WEIGHTS
#   (default 'smogon_pass2_selected:4.0' — up-sample selected in the online 40%).
#
# Before launching: sudo cpupower frequency-set -g performance  (DONE for this run)
#
# Attach with: tmux attach -t mini_online_smogon_v0
set -euo pipefail

SESSION="${SESSION:-mini_online_smogon_v0}"   # tmux session (dot-free)
RUN_NAME="${RUN_NAME:-mini_online_smogon_v0}" # ckpt dir + wandb run name
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EPOCHS="${EPOCHS:-3950}"
RESUME="${RESUME:-0}"

cd "$REPO_ROOT"

METAMON_CACHE_DIR="${METAMON_CACHE_DIR:-/home/eddie/metamon_cache}"
METAMON_WANDB_ENTITY="${METAMON_WANDB_ENTITY:-costacosta-personal-research}"
METAMON_WANDB_PROJECT="${METAMON_WANDB_PROJECT:-metamon}"
TRAIN_TEAM_SCHEDULE="${TRAIN_TEAM_SCHEDULE:-$REPO_ROOT/metamon/rl/configs/team_schedules/gen1ou_smogon_curriculum_v0.yaml}"
FIFO_TEAMSET_WEIGHTS="${FIFO_TEAMSET_WEIGHTS:-smogon_pass2_selected:4.0}"
FIFO_TEAMSET_DEFAULT_WEIGHT="${FIFO_TEAMSET_DEFAULT_WEIGHT:-1.0}"
PSRO_QUOTA_MIN_GAMES="${PSRO_QUOTA_MIN_GAMES:-200}"
PSRO_QUOTA_WINDOW="${PSRO_QUOTA_WINDOW:-256}"
# The 35M GroupedV2 model's perceiver activations at batch 32 OOM a 32GB
# 5090 when collector+validator each hold an inference copy (~1.3-3 GiB).
# Learner batch 16, grad_accum 1 (effective batch 16, matching the proven
# from-scratch mini_online_v1 recipe: batch 14, grad_accum 1, LR 8e-5) fits
# in ~17 GiB and maximizes optimizer steps/hr for the 4-day window. The 35M
# model is compute-bound at ~5.3 it/s, so grad_accum 1 (~2× more updates than
# grad_accum 2) is the better use of the window. expandable_segments reduces
# fragmentation across the 3 GPU-sharing processes.
LEARNER_BATCH="${LEARNER_BATCH:-16}"
LEARNER_GRAD_ACCUM="${LEARNER_GRAD_ACCUM:-1}"
export METAMON_CACHE_DIR METAMON_WANDB_ENTITY METAMON_WANDB_PROJECT
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "$RESUME" == "1" ]]; then
  echo ">> RESUME mode: --resume_training_state (continues from newest smogon_v0 training_state)"
  RESUME_FLAGS="--resume_training_state"
  BUFFER_TRIM=""
else
  echo ">> First-launch mode: --from_scratch (random init, 35M Tauros-inspired GroupedV2)"
  RESUME_FLAGS=""
  BUFFER_TRIM="--psro_buffer_trim 50000"
fi

tmux kill-session -t "$SESSION" 2>/dev/null || true

# --- Learner (GPU; grad updates; writes latest/policy.pt each epoch) ----------
tmux new-session -d -s "$SESSION" -n learner \
  "cd $REPO_ROOT && EPOCHS=$EPOCHS RUN_NAME=$RUN_NAME \
     BASE_MODEL=V2AGroupedV2Tauros35M \
     TRAIN_TEAM_SET=smogon_pass2 VAL_TEAM_SET=competitive \
     BATCH_PER_GPU=$LEARNER_BATCH \
     METAMON_CACHE_DIR=$METAMON_CACHE_DIR \
     METAMON_WANDB_ENTITY=$METAMON_WANDB_ENTITY \
     METAMON_WANDB_PROJECT=$METAMON_WANDB_PROJECT \
     TRAIN_TEAM_SCHEDULE=$TRAIN_TEAM_SCHEDULE \
     FIFO_TEAMSET_WEIGHTS='$FIFO_TEAMSET_WEIGHTS' \
     FIFO_TEAMSET_DEFAULT_WEIGHT=$FIFO_TEAMSET_DEFAULT_WEIGHT \
     PYTORCH_CUDA_ALLOC_CONF='$PYTORCH_CUDA_ALLOC_CONF' \
   bash scripts/launch_mini_online_v1.sh learner --log \
     $RESUME_FLAGS \
     --grad_accum $LEARNER_GRAD_ACCUM \
     --psro_weighting --psro_fifo_reweight --psro_start_epoch 0 \
     --dset_max_size 50000 \
     --ckpt_interval 25 \
   2>&1 | tee $HOME/metamon_runs/${SESSION}_learner.log; echo '[learner exited]'; bash"

# --- Collector (CPU-sim; self-play rollouts; writes the PSRO sidecar) --------
tmux new-window -t "$SESSION" -n collector \
  "cd $REPO_ROOT && EPOCHS=$EPOCHS RUN_NAME=$RUN_NAME \
     BASE_MODEL=V2AGroupedV2Tauros35M \
     TRAIN_TEAM_SET=smogon_pass2 VAL_TEAM_SET=competitive \
     METAMON_CACHE_DIR=$METAMON_CACHE_DIR \
     METAMON_WANDB_ENTITY=$METAMON_WANDB_ENTITY \
     METAMON_WANDB_PROJECT=$METAMON_WANDB_PROJECT \
     TRAIN_TEAM_SCHEDULE=$TRAIN_TEAM_SCHEDULE \
     PYTORCH_CUDA_ALLOC_CONF='$PYTORCH_CUDA_ALLOC_CONF' \
   bash scripts/launch_mini_online_v1.sh collector --log \
     $RESUME_FLAGS \
     --psro_weighting --psro_start_epoch 0 $BUFFER_TRIM \
     --psro_temp 2.0 --psro_floor 0.01 \
     --psro_quota_min_games '$PSRO_QUOTA_MIN_GAMES' --psro_quota_window '$PSRO_QUOTA_WINDOW' \
   2>&1 | tee $HOME/metamon_runs/${SESSION}_collector.log; echo '[collector exited]'; bash"

# --- Validator (CPU-sim; reloads latest/policy.pt each epoch vs TaurosV0) -----
tmux new-window -t "$SESSION" -n validator \
  "cd $REPO_ROOT && EPOCHS=$EPOCHS RUN_NAME=$RUN_NAME \
     BASE_MODEL=V2AGroupedV2Tauros35M \
     TRAIN_TEAM_SET=smogon_pass2 VAL_TEAM_SET=competitive \
     METAMON_CACHE_DIR=$METAMON_CACHE_DIR \
     METAMON_WANDB_ENTITY=$METAMON_WANDB_ENTITY \
     METAMON_WANDB_PROJECT=$METAMON_WANDB_PROJECT \
     TRAIN_TEAM_SCHEDULE=$TRAIN_TEAM_SCHEDULE \
     PYTORCH_CUDA_ALLOC_CONF='$PYTORCH_CUDA_ALLOC_CONF' \
   bash scripts/launch_mini_online_v1.sh validator --log \
     $RESUME_FLAGS \
   2>&1 | tee $HOME/metamon_runs/${SESSION}_validator.log; echo '[validator exited]'; bash"

echo "tmux session '$SESSION' launched with 3 windows: learner, collector, validator."
echo "  run name:    $RUN_NAME"
echo "  arch:        V2AGroupedV2Tauros35M (grouped_v2_35m.gin, ~34.94M params, Tauros-inspired)"
echo "  start:       from scratch (random init)"
echo "  schedule:    $(basename "$TRAIN_TEAM_SCHEDULE") (smogon-only; selected ramps 0%->75%)"
echo "  buffer:      fresh at \$SAVE_DIR/buffer"
echo "  learner:     batch=$LEARNER_BATCH, grad_accum=$LEARNER_GRAD_ACCUM (effective batch $((LEARNER_BATCH*LEARNER_GRAD_ACCUM)))"
echo "  epochs:      $EPOCHS (learner ~4.1 days; collector+validator run forever)"
echo "  attach:      tmux attach -t $SESSION"
echo "  logs:        ~/metamon_runs/${SESSION}_{learner,collector,validator}.log"
