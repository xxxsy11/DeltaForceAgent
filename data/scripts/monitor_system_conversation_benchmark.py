#!/usr/bin/env python3
"""Monitor conversation benchmark progress in real time."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "outputs" / "benchmark_logs"

TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
RUN_RE = re.compile(r"^\[benchmark\] running profile=(.+)$")
DONE_RE = re.compile(r"^\[benchmark\] profile=(.+)\s+elapsed_sec=")
TURN_RE = re.compile(r"^\[(.+)\]\s+turn=\d+\s+query=")


@dataclass
class RuntimeConfig:
    benchmark_file: Path
    profiles: List[str]
    max_cases: int
    case_offset: int
    max_turns_per_session: int


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monitor benchmark progress")
    p.add_argument(
        "--prefix",
        type=str,
        default="SYSTEM_CONVERSATION_BENCHMARK_100_TRIPLE",
        help="Report/log prefix used by run script",
    )
    p.add_argument(
        "--log-file",
        type=str,
        default="",
        help="Optional explicit log file path (overrides --prefix lookup)",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Refresh interval seconds",
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="Print once and exit",
    )
    return p.parse_args()


def _find_latest_log(prefix: str) -> Path:
    pattern = f"{prefix}_*.log"
    logs = sorted(LOG_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        raise FileNotFoundError(f"No log found under {LOG_DIR} with pattern {pattern}")
    return logs[0]


def _read_pid(log_path: Path) -> int | None:
    pid_path = log_path.with_suffix(".pid")
    if not pid_path.exists():
        return None
    raw = pid_path.read_text(encoding="utf-8").strip()
    if not raw.isdigit():
        return None
    return int(raw)


def _is_pid_running(pid: int | None) -> bool:
    if not pid:
        return False
    return Path(f"/proc/{pid}").exists()


def _parse_cmdline(pid: int | None) -> Dict[str, str]:
    if not pid:
        return {}
    p = Path(f"/proc/{pid}/cmdline")
    if not p.exists():
        return {}
    raw = p.read_bytes().decode("utf-8", errors="ignore")
    argv = [x for x in raw.split("\x00") if x]
    out: Dict[str, str] = {}
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--"):
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                out[tok] = argv[i + 1]
                i += 2
                continue
            out[tok] = "1"
        i += 1
    return out


def _build_runtime_cfg(cmd: Dict[str, str]) -> RuntimeConfig:
    benchmark_file = cmd.get("--benchmark-file", "data/benchmarks/system_conversation_benchmark_100.json")
    profiles_raw = cmd.get("--profiles", "kimi,base_qwen3_8b,qwen3_8b_lora")
    profiles = [x.strip() for x in profiles_raw.split(",") if x.strip()]
    max_cases = int(cmd.get("--max-cases", "0"))
    case_offset = int(cmd.get("--case-offset", "0"))
    max_turns_per_session = int(cmd.get("--max-turns-per-session", "0"))
    b = Path(benchmark_file)
    if not b.is_absolute():
        b = ROOT / b
    return RuntimeConfig(
        benchmark_file=b,
        profiles=profiles,
        max_cases=max_cases,
        case_offset=case_offset,
        max_turns_per_session=max_turns_per_session,
    )


def _load_expected_turns(cfg: RuntimeConfig) -> Tuple[int, Dict[str, int]]:
    data = json.loads(cfg.benchmark_file.read_text(encoding="utf-8"))
    cases = data.get("cases", [])
    start = max(0, cfg.case_offset)
    cases = cases[start:]
    if cfg.max_cases > 0:
        cases = cases[: cfg.max_cases]

    per_case_turns = 0
    for case in cases:
        for sess in case.get("sessions", []):
            turns = sess.get("turns", [])
            if cfg.max_turns_per_session > 0:
                per_case_turns += min(len(turns), cfg.max_turns_per_session)
            else:
                per_case_turns += len(turns)

    per_profile = {p: per_case_turns for p in cfg.profiles}
    total = per_case_turns * len(cfg.profiles)
    return total, per_profile


def _parse_log_progress(log_path: Path) -> Tuple[Dict[str, int], List[str], List[str], datetime | None]:
    if not log_path.exists():
        return {}, [], [], None

    started_profiles: List[str] = []
    done_profiles: List[str] = []
    turn_count: Dict[str, int] = {}
    first_ts: datetime | None = None

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            m = TS_RE.match(s)
            if m and first_ts is None:
                first_ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")

            m = RUN_RE.match(s)
            if m:
                started_profiles.append(m.group(1).strip())
                continue

            m = DONE_RE.match(s)
            if m:
                done_profiles.append(m.group(1).strip())
                continue

            m = TURN_RE.match(s)
            if m:
                profile = m.group(1).strip()
                turn_count[profile] = turn_count.get(profile, 0) + 1

    return turn_count, started_profiles, done_profiles, first_ts


def _fmt_secs(v: float) -> str:
    if v <= 0:
        return "--"
    v = int(v)
    h = v // 3600
    m = (v % 3600) // 60
    s = v % 60
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _print_once(log_path: Path, pid: int | None) -> None:
    cmd = _parse_cmdline(pid)
    cfg = _build_runtime_cfg(cmd)
    total_expected, per_profile_expected = _load_expected_turns(cfg)
    turns, started, done, first_ts = _parse_log_progress(log_path)

    completed = sum(turns.values())
    pct = (completed / total_expected * 100) if total_expected > 0 else 0.0

    if first_ts:
        elapsed = max(0.0, time.time() - first_ts.timestamp())
    else:
        elapsed = 0.0
    speed = (completed / elapsed) if elapsed > 0 else 0.0
    remain = max(0, total_expected - completed)
    eta = (remain / speed) if speed > 0 else 0.0

    print("=" * 80)
    print("System Conversation Benchmark Progress")
    print(f"log: {log_path}")
    print(f"pid: {pid if pid else '-'}  running: {'yes' if _is_pid_running(pid) else 'no'}")
    print(f"benchmark: {cfg.benchmark_file}")
    print(f"profiles: {', '.join(cfg.profiles)}")
    print(f"case_offset={cfg.case_offset}, max_cases={cfg.max_cases}, max_turns_per_session={cfg.max_turns_per_session}")
    print("-" * 80)
    print(f"overall: {completed}/{total_expected} ({pct:.2f}%)")
    print(f"elapsed: {_fmt_secs(elapsed)}  speed: {speed:.4f} turn/s  eta: {_fmt_secs(eta)}")
    print("-" * 80)
    for p in cfg.profiles:
        d = turns.get(p, 0)
        t = per_profile_expected.get(p, 0)
        pp = (d / t * 100) if t > 0 else 0.0
        status = "done" if p in done else ("running" if p in started else "pending")
        print(f"{p:18s} {d:4d}/{t:<4d}  {pp:6.2f}%  status={status}")
    print("=" * 80)


def main() -> None:
    args = _parse_args()
    if args.log_file:
        log_path = Path(args.log_file)
        if not log_path.is_absolute():
            log_path = ROOT / log_path
    else:
        log_path = _find_latest_log(args.prefix)

    pid = _read_pid(log_path)

    while True:
        os.system("clear")
        _print_once(log_path, pid)
        if args.once:
            break
        if not _is_pid_running(pid):
            print("\nProcess exited. Monitor stop.")
            break
        time.sleep(max(1.0, float(args.interval)))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)

