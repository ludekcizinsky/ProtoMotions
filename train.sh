#!/usr/bin/env bash
set -euo pipefail

# --- Basics (override via env when calling) ---
PYTHON_BIN="${PYTHON_BIN:-/workspace/isaaclab/_isaac_sim/python.sh}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/isaaclab/ProtoMotions/results}"

MOTION_FILE="${MOTION_FILE:-/workspace/isaaclab/ProtoMotions/data/zurihack/data/motion_states/initial_demo.pt}"
SIMULATOR="${SIMULATOR:-isaaclab}"
ROBOT="${ROBOT:-smpl}"
TERRAIN="${TERRAIN:-flat}"

# Which to run: 1, 2, 3, both (1+2), or all (1+2+3)
STAGE="${STAGE:-1}"

# --- Stage 1 (Full-body tracker) ---
TRACKER_EXPERIMENT_NAME="${TRACKER_EXPERIMENT_NAME:-initial_demo}"
TRACKER_NUM_ENVS="${TRACKER_NUM_ENVS:-512}"
TRACKER_NUM_STEPS="${TRACKER_NUM_STEPS:-32}"
TRACKER_BATCH_SIZE="$((TRACKER_NUM_ENVS * TRACKER_NUM_STEPS))"

# --- Stage 2 (MaskedMimic) ---
EXPERIMENT_NAME="${EXPERIMENT_NAME:-football}"
MM_NUM_ENVS="${MM_NUM_ENVS:-256}"
MM_NUM_STEPS="${MM_NUM_STEPS:-32}"
MM_BATCH_SIZE="$((MM_NUM_ENVS * MM_NUM_STEPS))"
EXPERT_DIR="${EXPERT_DIR:-${OUTPUT_DIR}/${TRACKER_EXPERIMENT_NAME}}"

# --- Stage 3 (Evaluation) ---
# By default, eval the Stage-2 experiment's last.ckpt with the user_control task.
EVAL_SIMULATOR="${EVAL_SIMULATOR:-isaaclab}"   # eval commonly uses isaacgym
EVAL_OPT="${EVAL_OPT:-[masked_mimic/tasks/user_control]}"
EVAL_CHECKPOINT="${EVAL_CHECKPOINT:-${OUTPUT_DIR}/${EXPERIMENT_NAME}/last.ckpt}"

# --- Optional Weights & Biases ---
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-zurihack}"
WANDB_ENTITY="${WANDB_ENTITY:-cbrander}"
WANDB_GROUP="${WANDB_GROUP:-dev}"
WANDB_TAGS="${WANDB_TAGS:-dev}"

WANDB_ARGS=()
if [[ "$USE_WANDB" -eq 1 ]]; then
  WANDB_ARGS=(+opt=[wandb] wandb.wandb_project="${WANDB_PROJECT}")
  [[ -n "${WANDB_ENTITY}" ]] && WANDB_ARGS+=(wandb.wandb_entity="${WANDB_ENTITY}")
  [[ -n "${WANDB_GROUP}"  ]] && WANDB_ARGS+=(wandb.wandb_group="${WANDB_GROUP}")
  [[ -n "${WANDB_TAGS}"   ]] && WANDB_ARGS+=(wandb.wandb_tags="${WANDB_TAGS}")
fi

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"

print_stage_1() {
  echo "=================================================================="
  echo " Stage 1: Full-body tracker"
  echo "=================================================================="
  echo "  motion_file     = ${MOTION_FILE}"
  echo "  experiment_name = ${TRACKER_EXPERIMENT_NAME}"
  echo "  simulator       = ${SIMULATOR}"
  echo "  robot           = ${ROBOT}"
  echo "  terrain         = ${TERRAIN}"
  echo "  num_envs        = ${TRACKER_NUM_ENVS}"
  echo "  num_steps       = ${TRACKER_NUM_STEPS}"
  echo "  batch_size      = ${TRACKER_BATCH_SIZE}"
  echo "  output_dir      = ${OUTPUT_DIR}"
  if [[ "$USE_WANDB" -eq 1 ]]; then
    echo "  wandb_project   = ${WANDB_PROJECT}"
    [[ -n "${WANDB_ENTITY}" ]] && echo "  wandb_entity    = ${WANDB_ENTITY}"
    [[ -n "${WANDB_GROUP}"  ]] && echo "  wandb_group     = ${WANDB_GROUP}"
    [[ -n "${WANDB_TAGS}"   ]] && echo "  wandb_tags      = ${WANDB_TAGS}"
  fi
}

print_stage_2() {
  echo "=================================================================="
  echo " Stage 2: MaskedMimic"
  echo "=================================================================="
  echo "  motion_file     = ${MOTION_FILE}"
  echo "  expert_path     = ${EXPERT_DIR}"
  echo "  experiment_name = ${EXPERIMENT_NAME}"
  echo "  simulator       = ${SIMULATOR}"
  echo "  robot           = ${ROBOT}"
  echo "  terrain         = ${TERRAIN}"
  echo "  num_envs        = ${MM_NUM_ENVS}"
  echo "  num_steps       = ${MM_NUM_STEPS}"
  echo "  batch_size      = ${MM_BATCH_SIZE}"
  echo "  output_dir      = ${OUTPUT_DIR}"
  if [[ "$USE_WANDB" -eq 1 ]]; then
    echo "  wandb_project   = ${WANDB_PROJECT}"
  fi
}

print_stage_3() {
  echo "=================================================================="
  echo " Stage 3: Evaluation"
  echo "=================================================================="
  echo "  opt task        = ${EVAL_OPT}"
  echo "  checkpoint      = ${EVAL_CHECKPOINT}"
  echo "  simulator       = ${EVAL_SIMULATOR}"
  echo "  robot           = ${ROBOT}"
}

run_stage_1() {
  print_stage_1
  HYDRA_FULL_ERROR=1 "$PYTHON_BIN" protomotions/train_agent.py \
    +exp=full_body_tracker/transformer_flat_terrain \
    +robot="$ROBOT" \
    +simulator="$SIMULATOR" \
    +terrain="$TERRAIN" \
    motion_file="$MOTION_FILE" \
    +experiment_name="$TRACKER_EXPERIMENT_NAME" \
    base_dir="$OUTPUT_DIR" \
    num_envs="$TRACKER_NUM_ENVS" \
    agent.config.num_steps="$TRACKER_NUM_STEPS" \
    agent.config.batch_size="$TRACKER_BATCH_SIZE" \
    "${WANDB_ARGS[@]}"
}

run_stage_2() {
  print_stage_2
  HYDRA_FULL_ERROR=1 "$PYTHON_BIN" protomotions/train_agent.py \
    +exp=masked_mimic/flat_terrain \
    +robot="$ROBOT" \
    +simulator="$SIMULATOR" \
    +terrain="$TERRAIN" \
    motion_file="$MOTION_FILE" \
    agent.config.expert_model_path="$EXPERT_DIR" \
    +experiment_name="$EXPERIMENT_NAME" \
    base_dir="$OUTPUT_DIR" \
    num_envs="$MM_NUM_ENVS" \
    agent.config.num_steps="$MM_NUM_STEPS" \
    agent.config.batch_size="$MM_BATCH_SIZE" \
    "${WANDB_ARGS[@]}"
}

run_stage_3() {
  print_stage_3
  HYDRA_FULL_ERROR=1 "$PYTHON_BIN" protomotions/eval_agent.py \
    +robot="$ROBOT" \
    +simulator="$EVAL_SIMULATOR" \
    +checkpoint="$EVAL_CHECKPOINT"
    #+opt="$EVAL_OPT" \
}

case "${STAGE}" in
  1)    run_stage_1 ;;
  2)    run_stage_2 ;;
  3)    run_stage_3 ;;
  both) run_stage_1; echo; run_stage_2 ;;
  all)  run_stage_1; echo; run_stage_2; echo; run_stage_3 ;;
  *)    echo "STAGE must be 1, 2, 3, both, or all"; exit 1 ;;
esac

echo "Done."

# USAGE:
#   STAGE=1 bash train.sh                       # run tracker only
#   STAGE=2 MM_NUM_ENVS=1024 bash train.sh      # run masked mimic only
#   STAGE=3 EVAL_CHECKPOINT=/path/ckpt.ckpt bash train.sh   # run evaluation only
#   STAGE=all bash train.sh                     # run 1 -> 2 -> 3