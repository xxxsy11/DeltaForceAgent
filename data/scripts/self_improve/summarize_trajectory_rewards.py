#!/usr/bin/env python3
"""统计 self-improving 轨迹奖励分布与关键质量指标。"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "self_improve" / "raw_trajectories"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "self_improve" / "reports" / "trajectory_reward_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计轨迹奖励分布")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="轨迹目录")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="输出 JSON 路径")
    return parser.parse_args()


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _pct(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    idx = max(0, min(len(values) - 1, int(round((p / 100.0) * (len(values) - 1)))))
    return values[idx]


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    files = sorted(input_dir.glob("tool_planning_trajectory_*.jsonl"))
    if not files:
        raise SystemExit(f"未找到轨迹文件: {input_dir}")

    rewards: List[float] = []
    quality_pass = 0
    retry_total_sum = 0
    retry_budget_exhausted = 0
    sample_count = 0

    for file in files:
        for row in _iter_jsonl(file):
            sample_count += 1
            reward = float(((row.get("reward", {}) or {}).get("total", 0.0)) or 0.0)
            rewards.append(reward)
            outcome = row.get("outcome", {}) or {}
            if bool(outcome.get("quality_gate_passed", False)):
                quality_pass += 1
            retry_total_sum += int(outcome.get("retry_count_total", 0) or 0)
            if bool(outcome.get("retry_budget_exhausted", False)):
                retry_budget_exhausted += 1

    rewards_sorted = sorted(rewards)
    summary: Dict[str, Any] = {
        "files": [str(f) for f in files],
        "sample_count": sample_count,
        "reward": {
            "mean": statistics.fmean(rewards) if rewards else 0.0,
            "std": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
            "min": min(rewards) if rewards else 0.0,
            "max": max(rewards) if rewards else 0.0,
            "p50": _pct(rewards_sorted, 50),
            "p90": _pct(rewards_sorted, 90),
            "p95": _pct(rewards_sorted, 95),
        },
        "quality_gate_pass_rate": (quality_pass / sample_count) if sample_count else 0.0,
        "avg_retry_count": (retry_total_sum / sample_count) if sample_count else 0.0,
        "retry_budget_exhausted_rate": (retry_budget_exhausted / sample_count) if sample_count else 0.0,
    }

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
