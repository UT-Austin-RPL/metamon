#!/usr/bin/env bash
# Launch the split-layout PSRO-Lite continuation run "mini_online_psro_v1" in tmux.
#
# Override the tmux session / run name via env vars:
#   SESSION=mini_online_psro_v1_3 RUN_NAME=mini_online_psro_v1.3 RESUME=1 \
#     bash scripts/launch_psro_v1.sh
# SESSION is the tmux session name (MUST be dot-free — tmux parses '.' as a pane
# separator and new-window -t <dotted> fails). RUN_NAME is the ckpt dir + wandb
# run name and may contain dots, so you can label resumed runs e.g.
# mini_online_psro_v1.3 while keeping a dot-free tmux session like
# mini_online_psro_v1_3. Defaults: SESSION=mini_online_psro_v1, RUN_NAME=$SESSION.
#
# Two modes:
#   Default (first launch): boots from the mini_online_v1 epoch-700 policy via
#     --prev_run_dir. PSRO-Lite active from epoch 0 (--psro_start_epoch 0).
#   RESUME=1: resumes THIS run from its newest training_state via
#     --resume_training_state (full accelerate state: model + optimizer +
#     scheduler + PopArt + RNG). Drops --prev_run_dir and --psro_buffer_trim.
#     Use this to relaunch after a crash or to apply a team-mix schedule change
#     without losing epochs.
#
# Three processes, three tmux windows in one session "mini_online_psro_v1":
#   learner    — grad updates; writes latest/policy.pt each epoch;
#                --psro_fifo_reweight reads the sidecar in its FIFO sampler.
#   collector  — self-play rollouts into the SHARED mini_online_v1 buffer;
#                --psro_weighting writes the meta_weights.json sidecar each update;
#                --psro_temp 2.0 / --psro_floor 0.01 sharpen the prioritized solver
#                so near-break-even opponents still get boosted.
#                The pool includes the run's own past checkpoints
#                (MiniOnlinePsroV0 in hl_gen1ou.yaml) as PSRO self-play opponents.
#   validator  — reloads the learner's latest/policy.pt each epoch vs TaurosV0;
#                unaffected by PSRO (uses its own val pool).
#
# Optional env vars:
#   TRAIN_TEAM_SCHEDULE — path to a team-mix schedule YAML (epoch-driven, no
#                         restart needed). The collector's player team set and
#                         the opponent pool's "@schedule" agents both follow it.
#                         Defaults to the gen1ou competitive curriculum
#                         (metamon/rl/configs/team_schedules/gen1ou_competitive_curriculum.yaml)
#                         and is REQUIRED for this run (the opponent pool uses
#                         '@schedule'). See docs/teamset_curriculum_proposal.md.
#   TRAIN_TEAM_MIX / VAL_TEAM_MIX — static weighted mix specs (overridden by
#                         TRAIN_TEAM_SCHEDULE for the collector).
#   PSRO_QUOTA_MIN_GAMES / PSRO_QUOTA_WINDOW — per-agent minimum-games guarantee.
#
# Attach with:   tmux attach -t <SESSION>
# Switch windows with Ctrl-b n / Ctrl-b p (or Ctrl-b <number>).
#
# To revert PSRO: relaunch without the --psro_* flags (clean; no persistent
# state beyond the sidecar JSON, which can be deleted).
set -euo pipefail

SESSION="${SESSION:-mini_online_psro_v1}"   # tmux session name (must be dot-free; tmux treats '.' as a pane separator)
RUN_NAME="${RUN_NAME:-$SESSION}"            # ckpt dir + wandb run name (may contain dots, e.g. mini_online_psro_v1.3)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREV_SAVE_DIR="$HOME/metamon_runs/mini_online_v1"
PREV_RUN_NAME="mini_online_v1"
PREV_CKPT="700"
SHARED_BUFFER="$HOME/metamon_runs/mini_online_v1/buffer"
EPOCHS="${EPOCHS:-3950}"
RESUME="${RESUME:-0}"

cd "$REPO_ROOT"

# Resolve env defaults once (overridable from the caller's environment).
METAMON_CACHE_DIR="${METAMON_CACHE_DIR:-/home/eddie/metamon_cache}"
METAMON_WANDB_ENTITY="${METAMON_WANDB_ENTITY:-costacosta-personal-research}"
METAMON_WANDB_PROJECT="${METAMON_WANDB_PROJECT:-metamon}"
TRAIN_TEAM_MIX="${TRAIN_TEAM_MIX:-}"
VAL_TEAM_MIX="${VAL_TEAM_MIX:-}"
TRAIN_TEAM_SCHEDULE="${TRAIN_TEAM_SCHEDULE:-$REPO_ROOT/metamon/rl/configs/team_schedules/gen1ou_competitive_curriculum.yaml}"   # required: opponent pool uses '@schedule' (defaults to the gen1ou competitive curriculum)
PSRO_QUOTA_MIN_GAMES="${PSRO_QUOTA_MIN_GAMES:-200}"
PSRO_QUOTA_WINDOW="${PSRO_QUOTA_WINDOW:-256}"
export METAMON_CACHE_DIR METAMON_WANDB_ENTITY METAMON_WANDB_PROJECT

# Build the resume/continuation flags based on mode.
if [[ "$RESUME" == "1" ]]; then
  echo ">> RESUME mode: --resume_training_state (continues from newest training_state)"
  RESUME_FLAGS="--resume_training_state"
  BUFFER_TRIM=""
else
  echo ">> First-launch mode: --prev_run_dir (boots from $PREV_RUN_NAME epoch $PREV_CKPT)"
  RESUME_FLAGS="--prev_run_dir $PREV_SAVE_DIR --prev_run_name $PREV_RUN_NAME --prev_checkpoint $PREV_CKPT"
  BUFFER_TRIM="--psro_buffer_trim 50000"
fi

# Tear down any prior session of the same name so this is idempotent.
tmux kill-session -t "$SESSION" 2>/dev/null || true

# --- Learner (GPU; grad updates; writes latest/policy.pt each epoch) ----------
tmux new-session -d -s "$SESSION" -n learner \
  "cd $REPO_ROOT && EPOCHS=$EPOCHS RUN_NAME=$RUN_NAME BUFFER_DIR=$SHARED_BUFFER \
     METAMON_CACHE_DIR=$METAMON_CACHE_DIR \
     METAMON_WANDB_ENTITY=$METAMON_WANDB_ENTITY \
     METAMON_WANDB_PROJECT=$METAMON_WANDB_PROJECT \
     TRAIN_TEAM_MIX=$TRAIN_TEAM_MIX VAL_TEAM_MIX=$VAL_TEAM_MIX \
     TRAIN_TEAM_SCHEDULE=$TRAIN_TEAM_SCHEDULE \
   bash scripts/launch_mini_online_v1.sh learner --log \
     $RESUME_FLAGS \
     --psro_weighting --psro_fifo_reweight --psro_start_epoch 0 \
   2>&1 | tee $HOME/metamon_runs/${SESSION}_learner.log; echo '[learner exited]'; bash"

# --- Collector (CPU-sim; self-play rollouts; writes the PSRO sidecar) --------
tmux new-window -t "$SESSION" -n collector \
  "cd $REPO_ROOT && EPOCHS=$EPOCHS RUN_NAME=$RUN_NAME BUFFER_DIR=$SHARED_BUFFER \
     METAMON_CACHE_DIR=$METAMON_CACHE_DIR \
     METAMON_WANDB_ENTITY=$METAMON_WANDB_ENTITY \
     METAMON_WANDB_PROJECT=$METAMON_WANDB_PROJECT \
     TRAIN_TEAM_MIX=$TRAIN_TEAM_MIX VAL_TEAM_MIX=$VAL_TEAM_MIX \
     TRAIN_TEAM_SCHEDULE=$TRAIN_TEAM_SCHEDULE \
   bash scripts/launch_mini_online_v1.sh collector --log \
     $RESUME_FLAGS \
     --psro_weighting --psro_start_epoch 0 $BUFFER_TRIM \
     --psro_temp 2.0 --psro_floor 0.01 \
     --psro_quota_min_games '$PSRO_QUOTA_MIN_GAMES' --psro_quota_window '$PSRO_QUOTA_WINDOW' \
   2>&1 | tee $HOME/metamon_runs/${SESSION}_collector.log; echo '[collector exited]'; bash"

# --- Validator (CPU-sim; reloads latest/policy.pt each epoch vs TaurosV0) -----
tmux new-window -t "$SESSION" -n validator \
  "cd $REPO_ROOT && EPOCHS=$EPOCHS RUN_NAME=$RUN_NAME BUFFER_DIR=$SHARED_BUFFER \
     METAMON_CACHE_DIR=$METAMON_CACHE_DIR \
     METAMON_WANDB_ENTITY=$METAMON_WANDB_ENTITY \
     METAMON_WANDB_PROJECT=$METAMON_WANDB_PROJECT \
     TRAIN_TEAM_MIX=$TRAIN_TEAM_MIX VAL_TEAM_MIX=$VAL_TEAM_MIX \
     TRAIN_TEAM_SCHEDULE=$TRAIN_TEAM_SCHEDULE \
   bash scripts/launch_mini_online_v1.sh validator --log \
     $RESUME_FLAGS \
   2>&1 | tee $HOME/metamon_runs/${SESSION}_validator.log; echo '[validator exited]'; bash"

echo "tmux session '$SESSION' launched with 3 windows: learner, collector, validator."
echo "  run name:  $RUN_NAME  (wandb run + ckpt dir; tmux session is '$SESSION')"
echo "  attach:    tmux attach -t $SESSION"
echo "  switch:    Ctrl-b n (next) / Ctrl-b p (prev) / Ctrl-b <number>"
echo "  logs:      ~/metamon_runs/${SESSION}_{learner,collector,validator}.log"
echo "  GPU is owned by the learner; collector+validator are CPU-sim-bound."
