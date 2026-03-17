#!/usr/bin/env python3
"""Compare input->tool-selection latency between Kimi and local Qwen+LoRA."""

from __future__ import annotations

import asyncio
import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

load_dotenv(PROJECT_ROOT / ".env")

from agents.runner import _build_initial_state, _build_runtime  # noqa: E402
from config import DEFAULT_CONFIG, GraphRAGConfig  # noqa: E402
from memory import SessionMemoryManager  # noqa: E402


@dataclass
class Profile:
    name: str
    patch: Dict[str, Any]


def _build_cfg(patch: Dict[str, Any]) -> GraphRAGConfig:
    data = DEFAULT_CONFIG.to_dict()
    data.update(patch)
    return GraphRAGConfig.from_dict(data)


def _mean(values: List[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _run_once(runtime, query: str, run_idx: int, user_id: str, session_id: str) -> Dict[str, Any]:
    manager = SessionMemoryManager()
    patch = manager.build_state_patch(
        user_id=user_id,
        session_id=session_id,
        include_pending_in_prompt=runtime.config.memory_include_pending_in_prompt,
    )
    wall_start = time.perf_counter()
    result = asyncio.run(
        runtime.graph.ainvoke(
            _build_initial_state(
                query=query,
                user_id=user_id,
                session_id=session_id,
                memory_patch=patch,
            )
        )
    )
    wall_elapsed_ms = round((time.perf_counter() - wall_start) * 1000, 2)

    meta = result.get("orchestration_meta", {}) or {}
    return {
        "run_index": run_idx,
        "query": query,
        "selected_tool": str(result.get("selected_tool", "") or ""),
        "selected_query": str(result.get("tool_query", "") or ""),
        "first_tool_selected_latency_ms": float(meta.get("first_tool_selected_latency_ms", 0.0) or 0.0),
        "latest_tool_selected_latency_ms": float(meta.get("latest_tool_selected_latency_ms", 0.0) or 0.0),
        "intent_node_latency_ms": float(meta.get("intent_node_latency_ms", 0.0) or 0.0),
        "end_to_end_elapsed_ms": wall_elapsed_ms,
    }


def _run_profile(profile: Profile, query: str, runs: int, pause_sec: float, warmup_runs: int) -> Dict[str, Any]:
    cfg = _build_cfg(profile.patch)
    runtime = _build_runtime(cfg)
    now_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    user_id = f"latency-bench-{profile.name}-{now_tag}"
    session_prefix = f"latency-{profile.name}"

    details: List[Dict[str, Any]] = []
    try:
        for w in range(1, warmup_runs + 1):
            session_id = f"{session_prefix}-warmup-{w:03d}"
            _run_once(runtime=runtime, query=query, run_idx=0, user_id=user_id, session_id=session_id)
            if pause_sec > 0:
                time.sleep(pause_sec)

        for i in range(1, runs + 1):
            session_id = f"{session_prefix}-{i:03d}"
            record = _run_once(runtime=runtime, query=query, run_idx=i, user_id=user_id, session_id=session_id)
            details.append(record)
            if pause_sec > 0 and i < runs:
                time.sleep(pause_sec)
    finally:
        asyncio.run(runtime.close_async())

    first_latencies = [x["first_tool_selected_latency_ms"] for x in details if x["first_tool_selected_latency_ms"] > 0]
    end_to_end = [x["end_to_end_elapsed_ms"] for x in details if x["end_to_end_elapsed_ms"] > 0]
    intent_node = [x["intent_node_latency_ms"] for x in details if x["intent_node_latency_ms"] > 0]

    return {
        "profile": profile.name,
        "runs": runs,
        "query": query,
        "warmup_runs": warmup_runs,
        "summary": {
            "input_to_tool_avg_ms": round(_mean(first_latencies), 2),
            "input_to_tool_min_ms": round(min(first_latencies), 2) if first_latencies else 0.0,
            "input_to_tool_max_ms": round(max(first_latencies), 2) if first_latencies else 0.0,
            "intent_node_avg_ms": round(_mean(intent_node), 2),
            "end_to_end_avg_ms": round(_mean(end_to_end), 2),
        },
        "details": details,
    }


def _profiles(qwen_device: str) -> List[Profile]:
    return [
        Profile(
            name="kimi",
            patch={
                "agent_local_enabled": False,
                "agent_intent_model": "kimi-k2-0711-preview",
                "agent_planner_model": "kimi-k2-0711-preview",
            },
        ),
        Profile(
            name="qwen3_8b_lora",
            patch={
                "agent_local_enabled": True,
                "agent_local_device": qwen_device,
                "agent_intent_model": "models/Qwen3-8B",
                "agent_planner_model": "models/Qwen3-8B",
                "agent_intent_adapter_path": DEFAULT_CONFIG.agent_intent_adapter_path,
                "agent_tool_selection_adapter_path": DEFAULT_CONFIG.agent_tool_selection_adapter_path,
                "agent_planning_adapter_path": DEFAULT_CONFIG.agent_planning_adapter_path,
            },
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare input->tool-selection latency.")
    parser.add_argument("--query", default="非洲之心现在什么价格", help="benchmark query")
    parser.add_argument("--runs", type=int, default=5, help="runs for each profile")
    parser.add_argument("--pause-sec", type=float, default=0.3, help="sleep between runs")
    parser.add_argument("--warmup-runs", type=int, default=1, help="warmup turns (excluded from metrics)")
    parser.add_argument("--qwen-device", default=DEFAULT_CONFIG.agent_local_device, help="device for qwen profile, e.g. cuda:0 or cpu")
    parser.add_argument(
        "--profiles",
        default="kimi,qwen3_8b_lora",
        help="comma-separated profiles to run: kimi,qwen3_8b_lora",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "docs" / "TOOL_SELECTION_LATENCY_COMPARE.json"),
        help="result json path",
    )
    args = parser.parse_args()

    enabled = {x.strip() for x in str(args.profiles).split(",") if x.strip()}
    results = []
    for profile in _profiles(args.qwen_device):
        if enabled and profile.name not in enabled:
            continue
        print(f"[run] profile={profile.name} runs={args.runs}")
        results.append(
            _run_profile(
                profile=profile,
                query=args.query,
                runs=max(1, args.runs),
                pause_sec=max(0.0, args.pause_sec),
                warmup_runs=max(0, args.warmup_runs),
            )
        )

    out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "query": args.query,
        "runs_per_profile": max(1, args.runs),
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Tool Selection Latency Compare ===")
    for item in results:
        s = item["summary"]
        print(
            f"{item['profile']}: input->tool avg={s['input_to_tool_avg_ms']}ms "
            f"(min={s['input_to_tool_min_ms']} max={s['input_to_tool_max_ms']}), "
            f"intent_node_avg={s['intent_node_avg_ms']}ms, e2e_avg={s['end_to_end_avg_ms']}ms"
        )
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
