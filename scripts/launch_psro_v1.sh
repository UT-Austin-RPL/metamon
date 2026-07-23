#!/usr/bin/env bash
# Launch the split-layout PSRO-Lite continuation run "mini_online_psro_v1" in tmux.
#
# Resumes from the mini_online_v1 epoch-700 policy checkpoint (the run was
# interrupted mid-epoch 708; epoch 700 is the newest full training_state).
# PSRO-Lite is active from epoch 0 of the new run (--psro_start_epoch 0).
#
# Three processes, three tmux windows in one session "mini_online_psro_v1":
#   learner    — grad updates; resumes from policy_epoch_700.pt via --prev_run_dir;
#                --psro_fifo_reweight reads the sidecar in its FIFO sampler.
#   collector  — self-play rollouts into the SHARED mini_online_v1 buffer;
#                --psro_weighting writes the meta_weights.json sidecar each update;
#                --psro_buffer_trim 50000 trims the uniform backlog once at epoch 0;
#                --psro_temp 2.0 / --psro_floor 0.01 sharpen the prioritized solver
#                so near-break-even opponents (score < floor under temp=1.0) still
#                get boosted — Kakuna (~0.45 win rate) was being floored to uniform.
#                The pool also includes the run's own past checkpoints
#                (MiniOnlinePsroV0 in hl_gen1ou.yaml) as PSRO self-play opponents.
#                --psro_quota_min_games 200 / --psro_quota_window 256 enforce a
#                per-agent minimum-games guarantee over a rolling window so
#                dominated, ladder-strong policies never fall to ~0 games played.
#   validator  — reloads the learner's latest/policy.pt each epoch vs TaurosV0;
#                unaffected by PSRO (uses its own val pool).
#
# All three load policy_epoch_700.pt at startup (no cold-start gap), then the
# collector/validator sync to the learner's rolling latest/policy.pt each epoch.
#
# Attach with:   tmux attach -t mini_online_psro_v1
# Switch windows with Ctrl-b n / Ctrl-b p (or Ctrl-b <number>).
#
# To revert PSRO: relaunch without the --psro_* flags (clean; no persistent
# state beyond the sidecar JSON, which can be deleted).
set -euo pipefail

SESSION="mini_online_psro_v1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREV_SAVE_DIR="$HOME/metamon_runs/mini_online_v1"
PREV_RUN_NAME="mini_online_v1"
PREV_CKPT="700"
SHARED_BUFFER="$HOME/metamon_runs/mini_online_v1/buffer"
EPOCHS="${EPOCHS:-3950}"

cd "$REPO_ROOT"

# Resolve env defaults once (overridable from the caller's environment).
METAMON_CACHE_DIR="${METAMON_CACHE_DIR:-/home/eddie/metamon_cache}"
METAMON_WANDB_ENTITY="${METAMON_WANDB_ENTITY:-costacosta-personal-research}"
METAMON_WANDB_PROJECT="${METAMON_WANDB_PROJECT:-metamon}"
# Optional weighted team-set mix spec (overrides --train_team_set when set).
# Format: 'set_name:weight,set_name:weight,...'
# e.g. "gl_05_26:0.75,smogon_pass2:0.15,smogon_pass2_selected:0.10" (Phase 1)
# See docs/teamset_curriculum_proposal.md for the full phase schedule.
TRAIN_TEAM_MIX="${TRAIN_TEAM_MIX:-}"
VAL_TEAM_MIX="${VAL_TEAM_MIX:-}"
export METAMON_CACHE_DIR METAMON_WANDB_ENTITY METAMON_WANDB_PROJECT

# Tear down any prior session of the same name so this is idempotent.
tmux kill-session -t "$SESSION" 2>/dev/null || true

# --- Learner (GPU; grad updates; writes latest/policy.pt each epoch) ----------
tmux new-session -d -s "$SESSION" -n learner \
  "cd $REPO_ROOT && EPOCHS=$EPOCHS RUN_NAME=$SESSION BUFFER_DIR=$SHARED_BUFFER \
     METAMON_CACHE_DIR=$METAMON_CACHE_DIR \
     METAMON_WANDB_ENTITY=$METAMON_WANDB_ENTITY \
     METAMON_WANDB_PROJECT=$METAMON_WANDB_PROJECT \
     TRAIN_TEAM_MIX='$TRAIN_TEAM_MIX' VAL_TEAM_MIX='$VAL_TEAM_MIX' \
   bash scripts/launch_mini_online_v1.sh learner --log \
     --prev_run_dir $PREV_SAVE_DIR --prev_run_name $PREV_RUN_NAME --prev_checkpoint $PREV_CKPT \
     --psro_weighting --psro_fifo_reweight --psro_start_epoch 0 \
   2>&1 | tee $HOME/metamon_runs/psro_learner.log; echo '[learner exited]'; bash"

# --- Collector (CPU-sim; self-play rollouts; writes the PSRO sidecar) --------
tmux new-window -t "$SESSION" -n collector \
  "cd $REPO_ROOT && EPOCHS=$EPOCHS RUN_NAME=$SESSION BUFFER_DIR=$SHARED_BUFFER \
     METAMON_CACHE_DIR=$METAMON_CACHE_DIR \
     METAMON_WANDB_ENTITY=$METAMON_WANDB_ENTITY \
     METAMON_WANDB_PROJECT=$METAMON_WANDB_PROJECT \
     TRAIN_TEAM_MIX='$TRAIN_TEAM_MIX' VAL_TEAM_MIX='$VAL_TEAM_MIX' \
   bash scripts/launch_mini_online_v1.sh collector --log \
     --prev_run_dir $PREV_SAVE_DIR --prev_run_name $PREV_RUN_NAME --prev_checkpoint $PREV_CKPT \
     --psro_weighting --psro_start_epoch 0 --psro_buffer_trim 50000 \
     --psro_temp 2.0 --psro_floor 0.01 \
     --psro_quota_min_games "$PSRO_QUOTA_MIN_GAMES" --psro_quota_window "$PSRO_QUOTA_WINDOW" \
   2>&1 | tee $HOME/metamon_runs/psro_collector.log; echo '[collector exited]'; bash"

# --- Validator (CPU-sim; reloads latest/policy.pt each epoch vs TaurosV0) -----
tmux new-window -t "$SESSION" -n validator \
  "cd $REPO_ROOT && EPOCHS=$EPOCHS RUN_NAME=$SESSION BUFFER_DIR=$SHARED_BUFFER \
     METAMON_CACHE_DIR=$METAMON_CACHE_DIR \
     METAMON_WANDB_ENTITY=$METAMON_WANDB_ENTITY \
     METAMON_WANDB_PROJECT=$METAMON_WANDB_PROJECT \
     TRAIN_TEAM_MIX='$TRAIN_TEAM_MIX' VAL_TEAM_MIX='$VAL_TEAM_MIX' \
   bash scripts/launch_mini_online_v1.sh validator --log \
     --prev_run_dir $PREV_SAVE_DIR --prev_run_name $PREV_RUN_NAME --prev_checkpoint $PREV_CKPT \
   2>&1 | tee $HOME/metamon_runs/psro_validator.log; echo '[validator exited]'; bash"

echo "tmux session '$SESSION' launched with 3 windows: learner, collector, validator."
echo "  attach:    tmux attach -t $SESSION"
echo "  switch:    Ctrl-b n (next) / Ctrl-b p (prev) / Ctrl-b <number>"
echo "  logs:      ~/metamon_runs/psro_{learner,collector,validator}.log"
echo "  GPU is owned by the learner; collector+validator are CPU-sim-bound."
