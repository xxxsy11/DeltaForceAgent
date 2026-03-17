#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

echo "[Self-Improve] 启动自进化管理器（按阈值自动判断是否训练）"
python data/scripts/self_improve/self_improve_manager.py --llm-filter-enabled
