#!/usr/bin/env bash

# End-to-end helper script that converts custom SMPL outputs into a packaged
# MotionLib state ready for ProtoMotions training. Update the configuration
# section before running.

set -euo pipefail

# Ensure all relative paths resolve from the repository root even if invoked elsewhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pushd "${SCRIPT_DIR}" >/dev/null
cleanup() {
  popd >/dev/null
}
trap cleanup EXIT

###############################################################################
# Configuration
###############################################################################

source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate protomotions
if type module >/dev/null 2>&1; then
  module load gcc git-lfs
fi

PYTHON_BIN=${PYTHON_BIN:-python}


SEQUENCE_NAME="pushups_smpl"
PREPROCESS_DIR="/scratch/izar/cizinsky/multiply-output/preprocessing/data/$SEQUENCE_NAME"
WORK_ROOT="/scratch/izar/cizinsky/zurihack/data/"

FPS=30
GENDER="neutral"
HUMANOID_TYPE="smpl"
ROBOT_TYPE="smpl"

# Leave empty to convert every detected track.
TRACK_IDS=(0)

# Optional override for the exported sequence identifier used in intermediate files.
# By default we strip common humanoid suffixes so convert_amass_to_isaac processes it.
SEQUENCE_EXPORT_NAME="${SEQUENCE_EXPORT_NAME:-${SEQUENCE_NAME}}"

sanitize_identifier() {
  local value="$1"
  value="${value// /_}"
  value="${value//(/}"
  value="${value//)/}"
  value="${value//[/}"
  value="${value//]/}"
  echo "$value"
}

sanitize_for_convert() {
  local value
  value="$(sanitize_identifier "$1")"
  local token
  for token in "smplx" "smplh" "smpl" "h1" "g1" "${ROBOT_TYPE}"; do
    [[ -z "${token}" ]] && continue
    value="${value//${token}/}"
  done
  # Collapse duplicate separators and trim leading/trailing ones.
  while [[ "${value}" == *"__"* ]]; do
    value="${value//__/_}"
  done
  while [[ "${value}" == *"--"* ]]; do
    value="${value//--/-}"
  done
  value="${value//-_/-}"
  value="${value//_-/_}"
  value="${value##[_-]}"
  value="${value%%[_-]}"
  if [[ -z "${value}" ]]; then
    value="sequence"
  fi
  echo "${value}"
}

###############################################################################
# Derived paths (feel free to adjust)
###############################################################################

AMASS_EXPORT_DIR="${WORK_ROOT}/amass_export"
MOTION_DESC_PATH="${WORK_ROOT}/motion_descriptors/${SEQUENCE_NAME}.yaml"
PACKAGED_OUTPUT="${WORK_ROOT}/motion_states/${SEQUENCE_NAME}.pt"
DATASET_REPO_ROOT="/scratch/izar/cizinsky/zurihack/"

# The conversion script sanitizes the sequence name; mirror the same logic here.
SANITIZED_SEQUENCE_NAME="$(sanitize_identifier "${SEQUENCE_NAME}")"
SANITIZED_EXPORT_NAME="$(sanitize_for_convert "${SEQUENCE_EXPORT_NAME}")"
CONVERTED_SUBDIR="${SANITIZED_EXPORT_NAME}-${ROBOT_TYPE}"
AMASS_SEQUENCE_DIR="${AMASS_EXPORT_DIR}/${SANITIZED_EXPORT_NAME}"

if [[ "${SANITIZED_EXPORT_NAME}" != "$(sanitize_identifier "${SEQUENCE_EXPORT_NAME}")" ]]; then
  echo "[INFO] Using sanitized export identifier '${SANITIZED_EXPORT_NAME}' for intermediate artifacts."
fi

###############################################################################
# Create working directories
###############################################################################

mkdir -p "${AMASS_EXPORT_DIR}"
mkdir -p "$(dirname "${MOTION_DESC_PATH}")"
mkdir -p "$(dirname "${PACKAGED_OUTPUT}")"

###############################################################################
# Step 1: Custom SMPL -> AMASS-style .npz
###############################################################################

echo "[1/4] Exporting AMASS-style clips..."
TRACK_ARGS=()
if ((${#TRACK_IDS[@]} > 0)); then
  for tid in "${TRACK_IDS[@]}"; do
    TRACK_ARGS+=(--track-id "${tid}")
  done
fi

"${PYTHON_BIN}" data/scripts/custom_smpl_to_amass.py \
  "${PREPROCESS_DIR}" \
  "${AMASS_EXPORT_DIR}" \
  --sequence-name "${SANITIZED_EXPORT_NAME}" \
  --fps "${FPS}" \
  --gender "${GENDER}" \
  "${TRACK_ARGS[@]}"

# The converter expects the input directory to match the sanitized export name.
if [[ ! -d "${AMASS_SEQUENCE_DIR}" ]]; then
  echo "[ERROR] Expected AMASS export directory '${AMASS_SEQUENCE_DIR}' not found." >&2
  exit 1
fi

###############################################################################
# Step 2: AMASS -> Isaac/poselib npy
###############################################################################

echo "[2/4] Converting AMASS clips to Isaac format..."
RETARGET_FLAGS=()
if [[ "${ROBOT_TYPE}" == "h1" || "${ROBOT_TYPE}" == "g1" ]]; then
  RETARGET_FLAGS+=(--force-retarget)
fi

"${PYTHON_BIN}" data/scripts/convert_amass_to_isaac.py \
  "${AMASS_EXPORT_DIR}" \
  --robot-type "${ROBOT_TYPE}" \
  --humanoid-type "${HUMANOID_TYPE}" \
  --output-dir "${AMASS_EXPORT_DIR}" \
  --force-remake \
  "${RETARGET_FLAGS[@]}"

CONVERTED_DIR="${AMASS_EXPORT_DIR}/${CONVERTED_SUBDIR}"
if [[ ! -d "${CONVERTED_DIR}" ]]; then
  echo "[ERROR] Converted motion root does not exist: ${CONVERTED_DIR}" >&2
  exit 1
fi

###############################################################################
# Step 3: Build motion descriptor YAML
###############################################################################

echo "[3/4] Creating motion descriptor..."
"${PYTHON_BIN}" data/scripts/create_motion_descriptor.py \
  "${AMASS_EXPORT_DIR}" \
  "${MOTION_DESC_PATH}" \
  --fps "${FPS}" \
  --sequence-subdir "${CONVERTED_SUBDIR}"

###############################################################################
# Step 4: Package MotionLib state
###############################################################################

echo "[4/4] Packaging MotionLib state..."
"${PYTHON_BIN}" data/scripts/package_motion_lib.py \
  "${MOTION_DESC_PATH}" \
  "${CONVERTED_DIR}" \
  "${PACKAGED_OUTPUT}" \
  --humanoid-type "${HUMANOID_TYPE}"

###############################################################################
# Step 5: Commit and push dataset repo to Hugging Face
###############################################################################

if [[ -d "${DATASET_REPO_ROOT}/.git" ]]; then
  echo "[5/5] Syncing dataset repo with Hugging Face..."
  pushd "${DATASET_REPO_ROOT}" >/dev/null
  git add .
  if git diff --cached --quiet; then
    echo "[5/5] No dataset changes to commit."
  else
    COMMIT_MSG="Update ${SEQUENCE_NAME} dataset ($(date -u +'%Y-%m-%dT%H:%M:%SZ'))"
    git commit -m "${COMMIT_MSG}"
    git push
  fi
  popd >/dev/null
else
  echo "[5/5] Skipping dataset sync; no Git repository found at ${DATASET_REPO_ROOT}"
fi

echo "[DONE] Packaged motion saved to ${PACKAGED_OUTPUT}"
