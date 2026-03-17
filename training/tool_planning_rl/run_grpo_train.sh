#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN=${PYTHON_BIN:-python}
TRAIN_CONFIG=${TRAIN_CONFIG:-training/tool_planning_rl/configs/grpo_train.json}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-auto}
NUM_GPUS=${NUM_GPUS:-2}

# 检测是否使用 DeepSpeed
if grep -q '"deepspeed"' "$TRAIN_CONFIG" 2>/dev/null; then
    # 使用 DeepSpeed 启动
    if command -v deepspeed &> /dev/null; then
        echo "使用 DeepSpeed 启动训练（GPUs: $NUM_GPUS）"
        deepspeed --num_gpus=$NUM_GPUS training/tool_planning_rl/train_grpo.py \
            --train_config "$TRAIN_CONFIG" \
            --resume_from_checkpoint "$RESUME_FROM_CHECKPOINT"
    else
        # 回退到 torchrun
        echo "DeepSpeed 未安装，使用 torchrun 启动（GPUs: $NUM_GPUS）"
        torchrun --nproc_per_node=$NUM_GPUS training/tool_planning_rl/train_grpo.py \
            --train_config "$TRAIN_CONFIG" \
            --resume_from_checkpoint "$RESUME_FROM_CHECKPOINT"
    fi
else
    # 不使用 DeepSpeed
    echo "不使用 DeepSpeed，单 GPU 训练"
    "$PYTHON_BIN" training/tool_planning_rl/train_grpo.py \
        --train_config "$TRAIN_CONFIG" \
        --resume_from_checkpoint "$RESUME_FROM_CHECKPOINT"
fi
