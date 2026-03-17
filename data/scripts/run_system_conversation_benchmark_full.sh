#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  bash data/scripts/run_system_conversation_benchmark_full.sh [extra args...]

Environment variables (optional):
  BENCHMARK_FILE            default: data/benchmarks/system_conversation_benchmark_100.json
  REPORT_PREFIX             default: SYSTEM_CONVERSATION_BENCHMARK_100_TRIPLE
  PROFILES                  default: kimi,base_qwen3_8b,qwen3_8b_lora
  LOCAL_DEVICE              default: cuda
  MAX_CASES                 default: 0 (full)
  CASE_OFFSET               default: 0
  MAX_TURNS_PER_SESSION     default: 0 (full)
  MAX_ATTEMPTS              default: 1
  LOG_DIR                   default: outputs/benchmark_logs

Examples:
  bash data/scripts/run_system_conversation_benchmark_full.sh
  REPORT_PREFIX=RUN_A MAX_CASES=20 bash data/scripts/run_system_conversation_benchmark_full.sh
  bash data/scripts/run_system_conversation_benchmark_full.sh --profiles kimi --max-cases 5
EOF
  exit 0
fi

BENCHMARK_FILE="${BENCHMARK_FILE:-data/benchmarks/system_conversation_benchmark_100.json}"
REPORT_PREFIX="${REPORT_PREFIX:-SYSTEM_CONVERSATION_BENCHMARK_100_TRIPLE}"
PROFILES="${PROFILES:-kimi,base_qwen3_8b,qwen3_8b_lora}"
LOCAL_DEVICE="${LOCAL_DEVICE:-cuda}"
MAX_CASES="${MAX_CASES:-0}"
CASE_OFFSET="${CASE_OFFSET:-0}"
MAX_TURNS_PER_SESSION="${MAX_TURNS_PER_SESSION:-0}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-1}"

LOG_DIR="${LOG_DIR:-outputs/benchmark_logs}"
mkdir -p "${LOG_DIR}"

STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${REPORT_PREFIX}_${STAMP}.log"
PID_FILE="${LOG_DIR}/${REPORT_PREFIX}_${STAMP}.pid"

CMD=(
  python data/scripts/system_conversation_benchmark_suite.py
  --benchmark-file "${BENCHMARK_FILE}"
  --report-prefix "${REPORT_PREFIX}"
  --profiles "${PROFILES}"
  --local-device "${LOCAL_DEVICE}"
  --max-cases "${MAX_CASES}"
  --case-offset "${CASE_OFFSET}"
  --max-turns-per-session "${MAX_TURNS_PER_SESSION}"
  --max-attempts "${MAX_ATTEMPTS}"
  "$@"
)

nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
PID=$!
echo "${PID}" > "${PID_FILE}"

cat <<EOF
[OK] Benchmark started in background
PID: ${PID}
PID file: ${PID_FILE}
Log file: ${LOG_FILE}

Monitor:
  tail -f ${LOG_FILE}
  ps -p ${PID} -o pid,etime,cmd

Stop:
  kill ${PID}

Result files (after finish):
  docs/${REPORT_PREFIX}_RESULT.json
  docs/${REPORT_PREFIX}_REPORT.md
  docs/${REPORT_PREFIX}_REPORT_BRIEF.md
EOF
