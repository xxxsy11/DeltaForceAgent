#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

python data/scripts/self_improve/self_improve_manager.py --llm-filter-enabled
