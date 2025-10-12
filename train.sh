#!/usr/bin/env bash
set -euo pipefail

# --- Basics (override via env when calling) ---
PYTHON_BIN="${PYTHON_BIN:-/workspace/isaaclab/_isaac_sim/python.sh}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/isaaclab/ProtoMotions/results}"

MOTION_FILE="${MOTION_FILE:-/workspace/isaaclab/ProtoMotions/data/zurihack/data/motion_states/walking.pt}"
SIMULATOR="${SIMULATOR:-isaaclab}"
ROBOT="${ROBOT:-h1}"
TERRAIN="${TERRAIN:-flat}"

# Which to run: 1, 2, 3, both (1+2), or all (1+2+3)
STAGE="${STAGE:-all}"

# Liftable box toggle (0/1, false/true)
ENABLE_LIFTABLE_BOX="${ENABLE_LIFTABLE_BOX:-0}"
case "${ENABLE_LIFTABLE_BOX,,}" in
  1|true|yes|on) ENABLE_LIFTABLE_BOX_OVERRIDE=true ;;
  *) ENABLE_LIFTABLE_BOX_OVERRIDE=false ;;
esac

# TODO RECORD_INITIAL_FRAMES="${RECORD_INITIAL_FRAMES:-0}"
# TODO RAW_FRAME_CAPTURE="${RAW_FRAME_CAPTURE:-0}"
# TODO case "${RAW_FRAME_CAPTURE,,}" in
# TODO   1|true|yes|on) RAW_FRAME_CAPTURE_OVERRIDE=true ;;
# TODO   *) RAW_FRAME_CAPTURE_OVERRIDE=false ;;
# TODO esac


# --- Stage 1 (Full-body tracker) ---
TRACKER_EXPERIMENT_NAME="${TRACKER_EXPERIMENT_NAME:-h1_walking_night_v2}"
TRACKER_NUM_ENVS="${TRACKER_NUM_ENVS:-512}"
TRACKER_NUM_STEPS="${TRACKER_NUM_STEPS:-32}"
TRACKER_BATCH_SIZE="$((TRACKER_NUM_ENVS * TRACKER_NUM_STEPS))"

# --- Stage 2 (MaskedMimic) ---
EXPERIMENT_NAME="${EXPERIMENT_NAME:-h1_walking_night_mimic_v2}"
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
  ENABLE_LIFTABLE_BOX="$ENABLE_LIFTABLE_BOX" HYDRA_FULL_ERROR=1 "$PYTHON_BIN" protomotions/train_agent.py \
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
    agent.config.max_epochs="2000" \
    env.config.enable_liftable_box="$ENABLE_LIFTABLE_BOX_OVERRIDE" \
    "${WANDB_ARGS[@]}"
    # TODO env.config.record_raw_frames_only="$RAW_FRAME_CAPTURE_OVERRIDE" \
    # TODO +env.config.record_initial_frames="$RECORD_INITIAL_FRAMES" \
}

run_stage_2() {
  print_stage_2
  ENABLE_LIFTABLE_BOX="$ENABLE_LIFTABLE_BOX" HYDRA_FULL_ERROR=1 "$PYTHON_BIN" protomotions/train_agent.py \
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
    agent.config.max_epochs="500" \
    env.config.enable_liftable_box="$ENABLE_LIFTABLE_BOX_OVERRIDE" \
    "${WANDB_ARGS[@]}"
}

run_stage_3() {
  print_stage_3
  ENABLE_LIFTABLE_BOX="$ENABLE_LIFTABLE_BOX" HYDRA_FULL_ERROR=1 "$PYTHON_BIN" protomotions/eval_agent.py \
    +robot="$ROBOT" \
    +simulator="$EVAL_SIMULATOR" \
    +checkpoint="$EVAL_CHECKPOINT" \
    +headless=False \
    +env.config.headless=False \
    +agent.config.max_eval_steps=1000 \
    +env.config.enable_liftable_box="$ENABLE_LIFTABLE_BOX_OVERRIDE" 
    #+opt="$EVAL_OPT" \

  echo "Encoding evaluation videos from rendered frames (if any)..."
  "$PYTHON_BIN" create_videos.py || echo "create_videos.py failed (see logs above)."

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
