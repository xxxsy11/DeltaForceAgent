#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

BASE_MODEL=${BASE_MODEL:-models/Qwen3-8B}
ADAPTER_PATH=${ADAPTER_PATH:-outputs/intent_sft/qwen3_8b_lora}
TEST_FILE=${TEST_FILE:-data/dataset/final/intent/test.jsonl}
OUT_DIR=${OUT_DIR:-outputs/intent_sft/eval}

mkdir -p "$OUT_DIR"

python training/intent_sft/eval.py \
  --base_model_path "$BASE_MODEL" \
  --adapter_path "$ADAPTER_PATH" \
  --test_file "$TEST_FILE" \
  --output_file "$OUT_DIR/predictions.jsonl" \
  --report_file "$OUT_DIR/report.json" \
  --max_samples ${MAX_SAMPLES:-200}
