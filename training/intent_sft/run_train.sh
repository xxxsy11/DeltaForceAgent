#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)/src
export WANDB_PROJECT=${WANDB_PROJECT:-deltaforceagent-sft}
export WANDB_MODE=${WANDB_MODE:-online}
ACCELERATE_BIN=${ACCELERATE_BIN:-accelerate}
PYTHON_BIN=${PYTHON_BIN:-python}
LAUNCH_MODE=${INTENT_SFT_LAUNCH_MODE:-distributed}
TRAIN_CONFIG=${TRAIN_CONFIG:-training/intent_sft/configs/train.json}
ACCEL_CONFIG=${ACCEL_CONFIG:-training/common/configs/accelerate_2gpu_zero2.yaml}

if [[ "$LAUNCH_MODE" == "single" ]]; then
  "$PYTHON_BIN" training/intent_sft/train.py \
    --train_config "$TRAIN_CONFIG"
else
  "$ACCELERATE_BIN" launch \
    --config_file "$ACCEL_CONFIG" \
    training/intent_sft/train.py \
    --train_config "$TRAIN_CONFIG"
fi
