#!/usr/bin/env bash

# Helper script to launch MaskedMimic training with the custom packaged motions.
# Update the configuration block below to point at your motion file and expert tracker checkpoint.

set -euo pipefail
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"


###############################################################################
# Configuration (override via environment variables if desired)
###############################################################################

PYTHON_BIN=${PYTHON_BIN:-python}
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="/scratch/izar/cizinsky/zurihack/results"

# Path to the packaged MotionLib state produced by prepare_custom_smpl.sh
MOTION_FILE=${MOTION_FILE:-/scratch/izar/cizinsky/zurihack/data/motion_states/football_high_res.pt}

# Single knob to control workload (smaller -> less GPU memory/time)
ENV_COUNT=${ENV_COUNT:-256}

# Stage 1 (full-body tracker) settings
TRACKER_EXPERIMENT_NAME=${TRACKER_EXPERIMENT_NAME:-football_tracker}
TRACKER_NUM_ENVS=${TRACKER_NUM_ENVS:-${ENV_COUNT}}
TRACKER_NUM_STEPS=${TRACKER_NUM_STEPS:-32}
TRACKER_BATCH_SIZE=${TRACKER_BATCH_SIZE:-$((TRACKER_NUM_ENVS * TRACKER_NUM_STEPS))}

# Stage 2 (MaskedMimic) settings
EXPERIMENT_NAME=${EXPERIMENT_NAME:-football_masked_mimic}
SIMULATOR=${SIMULATOR:-isaacgym}
ROBOT=${ROBOT:-smpl}
TERRAIN=${TERRAIN:-flat}
NUM_ENVS=${NUM_ENVS:-${ENV_COUNT}}
MM_NUM_STEPS=${MM_NUM_STEPS:-32}
MM_BATCH_SIZE=${MM_BATCH_SIZE:-$((NUM_ENVS * MM_NUM_STEPS))}

# Optional Weights & Biases logging
USE_WANDB=1
WANDB_PROJECT="zurihack"
WANDB_ENTITY="ludekcizinsky"
WANDB_GROUP="dev"
WANDB_TAGS="dev"

# Where to read the expert tracker checkpoints from (defaults to stage-1 output)
EXPERT_DIR=${EXPERT_DIR:-${OUTPUT_DIR}/${TRACKER_EXPERIMENT_NAME}}

# Re-run tracker even if a checkpoint exists? (set FORCE_TRACKER=1 to force)
FORCE_TRACKER=1

###############################################################################
# Launch training
###############################################################################

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

echo "=================================================================="
echo " Stage 1: Full-body tracker"
echo "=================================================================="
echo "  motion_file          = ${MOTION_FILE}"
echo "  experiment_name      = ${TRACKER_EXPERIMENT_NAME}"
echo "  simulator            = ${SIMULATOR}"
echo "  robot                = ${ROBOT}"
echo "  terrain              = ${TERRAIN}"
echo "  num_envs             = ${TRACKER_NUM_ENVS}"
echo "  num_steps            = ${TRACKER_NUM_STEPS}"
echo "  batch_size           = ${TRACKER_BATCH_SIZE}"
echo "  output_dir           = ${OUTPUT_DIR}"
if [[ "${USE_WANDB}" -eq 1 ]]; then
  echo "  wandb_project       = ${WANDB_PROJECT}"
  [[ -n "${WANDB_ENTITY}" ]] && echo "  wandb_entity        = ${WANDB_ENTITY}"
  [[ -n "${WANDB_GROUP}" ]] && echo "  wandb_group         = ${WANDB_GROUP}"
  [[ -n "${WANDB_TAGS}" ]] && echo "  wandb_tags          = ${WANDB_TAGS}"
fi

WANDB_ARGS=()
if [[ "${USE_WANDB}" -eq 1 ]]; then
  WANDB_ARGS=(+opt=[wandb] wandb.wandb_project="${WANDB_PROJECT}")
  [[ -n "${WANDB_ENTITY}" ]] && WANDB_ARGS+=(wandb.wandb_entity="${WANDB_ENTITY}")
  [[ -n "${WANDB_GROUP}" ]] && WANDB_ARGS+=(wandb.wandb_group="${WANDB_GROUP}")
  [[ -n "${WANDB_TAGS}" ]] && WANDB_ARGS+=(wandb.wandb_tags="${WANDB_TAGS}")
fi

TRACKER_LAST_CKPT="${OUTPUT_DIR}/${TRACKER_EXPERIMENT_NAME}/last.ckpt"
if [[ ${FORCE_TRACKER} -eq 1 || ! -f "${TRACKER_LAST_CKPT}" ]]; then
  "${PYTHON_BIN}" protomotions/train_agent.py \
    +exp=full_body_tracker/transformer_flat_terrain \
    +robot="${ROBOT}" \
    +simulator="${SIMULATOR}" \
    +terrain="${TERRAIN}" \
    motion_file="${MOTION_FILE}" \
    +experiment_name="${TRACKER_EXPERIMENT_NAME}" \
    base_dir="${OUTPUT_DIR}" \
    num_envs="${TRACKER_NUM_ENVS}" \
    agent.config.num_steps="${TRACKER_NUM_STEPS}" \
  agent.config.batch_size="${TRACKER_BATCH_SIZE}" \
  "${WANDB_ARGS[@]}"
else
  echo "Tracker checkpoint already exists at ${TRACKER_LAST_CKPT}; skipping (set FORCE_TRACKER=1 to re-run)."
fi

echo
echo "=================================================================="
echo " Stage 2: MaskedMimic"
echo "=================================================================="
echo "  motion_file          = ${MOTION_FILE}"
echo "  expert_model_path    = ${EXPERT_DIR}"
echo "  experiment_name      = ${EXPERIMENT_NAME}"
echo "  simulator            = ${SIMULATOR}"
echo "  robot                = ${ROBOT}"
echo "  terrain              = ${TERRAIN}"
echo "  num_envs             = ${NUM_ENVS}"
echo "  num_steps            = ${MM_NUM_STEPS}"
echo "  batch_size           = ${MM_BATCH_SIZE}"
echo "  output_dir           = ${OUTPUT_DIR}"
if [[ "${USE_WANDB}" -eq 1 ]]; then
  echo "  wandb_project       = ${WANDB_PROJECT}"
fi

"${PYTHON_BIN}" protomotions/train_agent.py \
  +exp=masked_mimic/flat_terrain \
  +robot="${ROBOT}" \
  +simulator="${SIMULATOR}" \
  +terrain="${TERRAIN}" \
  motion_file="${MOTION_FILE}" \
  agent.config.expert_model_path="${EXPERT_DIR}" \
  +experiment_name="${EXPERIMENT_NAME}" \
  base_dir="${OUTPUT_DIR}" \
  num_envs="${NUM_ENVS}" \
  agent.config.num_steps="${MM_NUM_STEPS}" \
  agent.config.batch_size="${MM_BATCH_SIZE}" \
  "${WANDB_ARGS[@]}"

echo "MaskedMimic training launched."
