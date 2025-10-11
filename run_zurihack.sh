#!/usr/bin/env bash

PY_CMD=(/workspace/isaaclab/isaaclab.sh -p)
#EXP="zuri_hack"
ROBOT="smpl"
SIMULATOR="isaaclab"
MOTION_FILE="/workspace/isaaclab/ProtoMotions/data/zurihack/data/motion_states/football_high_res.pt"
EXPERIMENT_NAME="custom_data_zurihack"
NUM_ENVS="1024"

HYDRA_FULL_ERROR=1 "${PY_CMD[@]}" protomotions/train_agent.py \
  +exp=full_body_tracker/transformer_flat_terrain \
  +robot="$ROBOT" \
  +simulator="$SIMULATOR" \
  motion_file="$MOTION_FILE" \
  +experiment_name="$EXPERIMENT_NAME" \
  num_envs="$NUM_ENVS"