#!/usr/bin/env python3
"""Self-Improve Manager：按阈值触发自进化训练（采样 -> 构建 -> 训练）。"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "self_improve" / "raw_trajectories"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "data" / "self_improve" / "reports"
DEFAULT_STATE_FILE = DEFAULT_REPORT_DIR / "self_improve_manager_state.json"
DEFAULT_BUILD_SUMMARY = PROJECT_ROOT / "data" / "dataset" / "agentic_rl" / "tool_planning" / "build_summary.json"


@dataclass
class ManagerConfig:
    input_dir: Path
    state_file: Path
    report_dir: Path
    min_valid_samples: int
    min_days_since_last_train: float
    reward_min: float
    current_window_hours: int
    baseline_window_days: int
    min_baseline_samples: int
    trigger_tool_match_drop_pp: float
    trigger_quality_drop_pp: float
    trigger_retry_budget_spike_ratio: float
    trigger_blind_retry_spike_ratio: float
    trigger_reward_p50_drop_sigma: float
    min_train_records: int
    llm_filter_enabled: bool
    train_dry_run: bool
    dry_run: bool
    force_train: bool


def parse_args() -> ManagerConfig:
    parser = argparse.ArgumentParser(description="Self-Improve Manager")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_FILE))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--min-valid-samples", type=int, default=50)
    parser.add_argument("--min-days-since-last-train", type=float, default=2.0)
    parser.add_argument("--reward-min", type=float, default=-10.0)
    parser.add_argument("--current-window-hours", type=int, default=24)
    parser.add_argument("--baseline-window-days", type=int, default=7)
    parser.add_argument("--min-baseline-samples", type=int, default=20)
    parser.add_argument("--trigger-tool-match-drop-pp", type=float, default=0.08)
    parser.add_argument("--trigger-quality-drop-pp", type=float, default=0.05)
    parser.add_argument("--trigger-retry-budget-spike-ratio", type=float, default=2.0)
    parser.add_argument("--trigger-blind-retry-spike-ratio", type=float, default=1.5)
    parser.add_argument("--trigger-reward-p50-drop-sigma", type=float, default=1.0)
    parser.add_argument("--min-train-records", type=int, default=200)
    parser.add_argument("--llm-filter-enabled", action="store_true")
    parser.add_argument("--train-dry-run", action="store_true", help="训练阶段使用 GRPO dry_run 配置")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    args = parser.parse_args()
    return ManagerConfig(
        input_dir=Path(args.input_dir).resolve(),
        state_file=Path(args.state_file).resolve(),
        report_dir=Path(args.report_dir).resolve(),
        min_valid_samples=int(args.min_valid_samples),
        min_days_since_last_train=float(args.min_days_since_last_train),
        reward_min=float(args.reward_min),
        current_window_hours=int(args.current_window_hours),
        baseline_window_days=int(args.baseline_window_days),
        min_baseline_samples=int(args.min_baseline_samples),
        trigger_tool_match_drop_pp=float(args.trigger_tool_match_drop_pp),
        trigger_quality_drop_pp=float(args.trigger_quality_drop_pp),
        trigger_retry_budget_spike_ratio=float(args.trigger_retry_budget_spike_ratio),
        trigger_blind_retry_spike_ratio=float(args.trigger_blind_retry_spike_ratio),
        trigger_reward_p50_drop_sigma=float(args.trigger_reward_p50_drop_sigma),
        min_train_records=int(args.min_train_records),
        llm_filter_enabled=bool(args.llm_filter_enabled),
        train_dry_run=bool(args.train_dry_run),
        dry_run=bool(args.dry_run),
        force_train=bool(args.force_train),
    )


def _parse_utc(ts: str) -> datetime | None:
    text = str(ts or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
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


def _load_rows(input_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for file in sorted(input_dir.glob("tool_planning_trajectory_*.jsonl")):
        rows.extend(_iter_jsonl(file))
    return rows


def _is_valid_row(row: Dict[str, Any], reward_min: float) -> bool:
    reward_total = float(((row.get("reward", {}) or {}).get("total", 0.0) or 0.0))
    if reward_total < reward_min:
        return False
    action = row.get("action", {}) or {}
    selected_tool = str(action.get("selected_tool", "") or "").strip()
    tool_calls = action.get("tool_calls", []) or []
    return bool(selected_tool or tool_calls)


def _pct(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = max(0, min(len(arr) - 1, int(round((p / 100.0) * (len(arr) - 1)))))
    return arr[idx]


def _metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    n = len(rows)
    if n <= 0:
        return {
            "sample_count": 0.0,
            "tool_match_rate": 0.0,
            "quality_pass_rate": 0.0,
            "retry_budget_exhausted_rate": 0.0,
            "blind_retry_rate": 0.0,
            "reward_mean": 0.0,
            "reward_std": 0.0,
            "reward_p50": 0.0,
        }
    tool_match_hit = 0
    quality_pass = 0
    retry_budget_exhausted = 0
    blind_retry = 0
    rewards: List[float] = []
    for row in rows:
        reward = row.get("reward", {}) or {}
        components = reward.get("components", {}) or {}
        tool_match = float(components.get("tool_match", 0.0) or 0.0)
        if tool_match > 0:
            tool_match_hit += 1
        rewards.append(float(reward.get("total", 0.0) or 0.0))

        outcome = row.get("outcome", {}) or {}
        if bool(outcome.get("quality_gate_passed", False)):
            quality_pass += 1
        if bool(outcome.get("retry_budget_exhausted", False)):
            retry_budget_exhausted += 1
        tags = outcome.get("failure_tags", []) or []
        if any(str(tag) == "blind_retry" for tag in tags):
            blind_retry += 1

    return {
        "sample_count": float(n),
        "tool_match_rate": tool_match_hit / n,
        "quality_pass_rate": quality_pass / n,
        "retry_budget_exhausted_rate": retry_budget_exhausted / n,
        "blind_retry_rate": blind_retry / n,
        "reward_mean": statistics.fmean(rewards) if rewards else 0.0,
        "reward_std": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
        "reward_p50": _pct(rewards, 50),
    }


def _read_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _write_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _days_since_last_train(state: Dict[str, Any]) -> float:
    last_train = _parse_utc(str(state.get("last_train_utc", "") or ""))
    if last_train is None:
        return math.inf
    now = datetime.now(timezone.utc)
    return (now - last_train).total_seconds() / 86400.0


def _run(cmd: List[str], cwd: Path) -> Tuple[int, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    text = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, text.strip()


def _prepare_grpo_config(train_dry_run: bool) -> Path:
    base_cfg = PROJECT_ROOT / "training" / "tool_planning_rl" / "configs" / "grpo_train.json"
    if not train_dry_run:
        return base_cfg
    raw = json.loads(base_cfg.read_text(encoding="utf-8"))
    raw["dry_run"] = True
    raw["report_to"] = []
    raw["wandb_mode"] = "disabled"
    out = PROJECT_ROOT / "data" / "self_improve" / "reports" / "grpo_train.dry_run.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> None:
    cfg = parse_args()
    now = datetime.now(timezone.utc)
    rows = _load_rows(cfg.input_dir)
    valid_rows = [row for row in rows if _is_valid_row(row, cfg.reward_min)]

    current_start = now - timedelta(hours=cfg.current_window_hours)
    baseline_start = current_start - timedelta(days=cfg.baseline_window_days)
    current_rows: List[Dict[str, Any]] = []
    baseline_rows: List[Dict[str, Any]] = []
    for row in valid_rows:
        ts = _parse_utc(str(row.get("created_at_utc", "") or ""))
        if ts is None:
            continue
        if ts >= current_start:
            current_rows.append(row)
        elif baseline_start <= ts < current_start:
            baseline_rows.append(row)

    current_m = _metrics(current_rows)
    baseline_m = _metrics(baseline_rows)
    state = _read_state(cfg.state_file)
    days_since = _days_since_last_train(state)

    triggered_rules: List[str] = []
    if baseline_m["sample_count"] >= cfg.min_baseline_samples and current_m["sample_count"] > 0:
        if (baseline_m["tool_match_rate"] - current_m["tool_match_rate"]) >= cfg.trigger_tool_match_drop_pp:
            triggered_rules.append("tool_match_drop")
        if (baseline_m["quality_pass_rate"] - current_m["quality_pass_rate"]) >= cfg.trigger_quality_drop_pp:
            triggered_rules.append("quality_pass_drop")
        if baseline_m["retry_budget_exhausted_rate"] > 0:
            if (
                current_m["retry_budget_exhausted_rate"] / baseline_m["retry_budget_exhausted_rate"]
                >= cfg.trigger_retry_budget_spike_ratio
            ):
                triggered_rules.append("retry_budget_exhausted_spike")
        if baseline_m["blind_retry_rate"] > 0:
            if (current_m["blind_retry_rate"] / baseline_m["blind_retry_rate"]) >= cfg.trigger_blind_retry_spike_ratio:
                triggered_rules.append("blind_retry_spike")
        if baseline_m["reward_std"] > 0:
            reward_drop = baseline_m["reward_p50"] - current_m["reward_p50"]
            if reward_drop >= (cfg.trigger_reward_p50_drop_sigma * baseline_m["reward_std"]):
                triggered_rules.append("reward_p50_drop")

    gate_sample = len(valid_rows) >= cfg.min_valid_samples
    gate_days = days_since >= cfg.min_days_since_last_train
    gate_degrade = len(triggered_rules) > 0
    should_train = cfg.force_train or (gate_sample and gate_days and gate_degrade)

    decision = "hold"
    actions: List[Dict[str, Any]] = []
    train_status = "skipped"
    error_text = ""

    if should_train:
        decision = "train"
        actions.append({"step": "summarize", "cmd": "python data/scripts/self_improve/summarize_trajectory_rewards.py"})
        code, out = _run([sys.executable, "data/scripts/self_improve/summarize_trajectory_rewards.py"], cwd=PROJECT_ROOT)
        actions[-1]["code"] = code
        actions[-1]["output_tail"] = out[-500:]
        if code != 0:
            train_status = "failed"
            error_text = "summarize_failed"
        else:
            build_cmd = [sys.executable, "data/scripts/self_improve/build_tool_planning_rl_dataset.py"]
            if cfg.llm_filter_enabled:
                build_cmd.append("--llm-filter-enabled")
            actions.append({"step": "build_dataset", "cmd": " ".join(build_cmd)})
            code, out = _run(build_cmd, cwd=PROJECT_ROOT)
            actions[-1]["code"] = code
            actions[-1]["output_tail"] = out[-500:]
            if code != 0:
                train_status = "failed"
                error_text = "build_dataset_failed"
            else:
                if DEFAULT_BUILD_SUMMARY.exists():
                    build_summary = json.loads(DEFAULT_BUILD_SUMMARY.read_text(encoding="utf-8"))
                    records_after = int(build_summary.get("rows_after_filter", 0) or 0)
                else:
                    records_after = 0
                if records_after < cfg.min_train_records and not cfg.force_train:
                    decision = "hold"
                    train_status = "skipped_low_records"
                    error_text = f"records_after_filter={records_after} < min_train_records={cfg.min_train_records}"
                elif cfg.dry_run:
                    train_status = "dry_run"
                else:
                    cfg_path = _prepare_grpo_config(cfg.train_dry_run)
                    train_cmd = [
                        sys.executable,
                        "training/tool_planning_rl/train_grpo.py",
                        "--train_config",
                        str(cfg_path),
                    ]
                    actions.append({"step": "train_grpo", "cmd": " ".join(train_cmd)})
                    code, out = _run(train_cmd, cwd=PROJECT_ROOT)
                    actions[-1]["code"] = code
                    actions[-1]["output_tail"] = out[-500:]
                    if code != 0:
                        train_status = "failed"
                        error_text = "train_failed"
                    else:
                        train_status = "done"

    if train_status == "done":
        new_state = dict(state)
        new_state["last_train_utc"] = now.isoformat()
        new_state["last_train_reason"] = triggered_rules
        _write_state(cfg.state_file, new_state)

    report = {
        "generated_at_utc": now.isoformat(),
        "decision": decision,
        "train_status": train_status,
        "error": error_text,
        "gates": {
            "sample_gate": gate_sample,
            "days_gate": gate_days,
            "degrade_gate": gate_degrade,
            "force_train": cfg.force_train,
        },
        "thresholds": {
            "min_valid_samples": cfg.min_valid_samples,
            "min_days_since_last_train": cfg.min_days_since_last_train,
            "min_train_records": cfg.min_train_records,
        },
        "days_since_last_train": days_since if math.isfinite(days_since) else None,
        "counts": {
            "rows_all": len(rows),
            "rows_valid": len(valid_rows),
            "rows_current_window": len(current_rows),
            "rows_baseline_window": len(baseline_rows),
        },
        "metrics": {
            "current": current_m,
            "baseline": baseline_m,
        },
        "triggered_rules": triggered_rules,
        "actions": actions,
    }

    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = cfg.report_dir / f"self_improve_manager_report_{now.strftime('%Y%m%dT%H%M%SZ')}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nreport_path={report_path}")


if __name__ == "__main__":
    main()
