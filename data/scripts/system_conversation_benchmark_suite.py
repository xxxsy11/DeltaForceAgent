#!/usr/bin/env python3
"""Conversation-level system benchmark runner (Kimi vs Base Qwen vs Qwen+LoRA)."""

from __future__ import annotations

import asyncio
import argparse
import json
import math
import shutil
import statistics
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

load_dotenv(PROJECT_ROOT / ".env")

from agents.graph import build_multi_agent_graph
from agents.runner import _build_initial_state, _finalize_session_memory
from config import DEFAULT_CONFIG, GraphRAGConfig
from memory import PersistentMemoryStore, SessionMemoryManager
from services import RAGService
from tools import ToolRegistry

FAIL_MARKERS = (
    "查询失败",
    "工具调用失败",
    "系统错误",
    "未获得可用结果",
    "HTTP 400",
    "HTTP 500",
    "HTTP 502",
    "请至少提供两个物品名称",
)
TRANSIENT_MARKERS = (
    "timeout",
    "timed out",
    "read timed out",
    "temporary",
    "http 429",
    "http 500",
    "http 502",
    "connection",
)


@dataclass
class ModelProfile:
    name: str
    label: str
    cfg_patch: Dict[str, Any]


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _avg(values: Iterable[float]) -> float:
    arr = list(values)
    return float(statistics.mean(arr)) if arr else 0.0


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    if len(arr) == 1:
        return float(arr[0])
    pp = max(0.0, min(1.0, float(p)))
    pos = (len(arr) - 1) * pp
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(arr[lo])
    w = pos - lo
    return float(arr[lo] + (arr[hi] - arr[lo]) * w)


def _is_failure(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return True
    return any(x in raw for x in FAIL_MARKERS)


def _is_transient_failure(text: str) -> bool:
    raw = str(text or "").lower()
    return any(x in raw for x in TRANSIENT_MARKERS)


def _intent_from_tool(tool_name: str) -> str:
    mapping = {
        "rag_knowledge_search": "knowledge_query",
        "df_market_latest_price": "market_price_latest_query",
        "df_market_history_price": "market_price_history_query",
        "df_market_price_advice": "market_price_advice_query",
        "df_place_profit_rank": "place_profit_query",
        "df_multi_item_compare": "market_compare_query",
        "df_profit_stability": "profit_stability_query",
        "df_answer_composer": "answer_composer_query",
    }
    return mapping.get(str(tool_name or "").strip(), "")


def _intent_match(actual_intent: str, selected_tool: str, expected_intents: List[str]) -> bool:
    expected = [str(x).strip() for x in expected_intents if str(x).strip()]
    if not expected:
        return True
    canonical = _intent_from_tool(selected_tool)
    if canonical and canonical in expected:
        return True
    raw = str(actual_intent or "").strip()
    if raw in expected:
        return True
    return False


def _keyword_coverage(answer: str, keywords: List[str]) -> float:
    if not keywords:
        return 1.0
    raw = str(answer or "")
    hits = sum(1 for k in keywords if k and k in raw)
    return hits / max(1, len(keywords))


def _entity_hit(expected_entities: List[str], entities: List[str], tool_query: str, answer: str) -> bool:
    if not expected_entities:
        return True
    known = "\n".join([str(tool_query or ""), str(answer or ""), "\n".join(entities or [])])
    return all(e in known for e in expected_entities)


def _stage_seen(turn: Dict[str, Any], stage_prefix: str) -> bool:
    for step in turn.get("debug_steps", []) or []:
        if str(step).startswith(stage_prefix):
            return True
    return False


def _stage_skipped(turn: Dict[str, Any], stage_name: str) -> bool:
    prefix = f"{stage_name}:"
    for step in turn.get("debug_steps", []) or []:
        raw = str(step).strip()
        if raw.startswith(prefix) and "skipped" in raw:
            return True
    return False


def _stage_done(turn: Dict[str, Any], stage_name: str) -> bool:
    done_marker = f"{stage_name}: done"
    for step in turn.get("debug_steps", []) or []:
        if str(step).strip() == done_marker:
            return True
    return False


def _reset_persistent_memory(store: PersistentMemoryStore) -> Dict[str, Any]:
    info: Dict[str, Any] = {"db_reset": False, "local_reset": False}

    conn = store._connect()  # noqa: SLF001
    if conn is not None:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    TRUNCATE TABLE
                        memory_facts,
                        memory_summaries,
                        chat_turns,
                        chat_sessions
                    RESTART IDENTITY CASCADE;
                    """
                )
        info["db_reset"] = True

    for target in [PROJECT_ROOT / "data/memory/readable", PROJECT_ROOT / "data/memory/exports"]:
        if target.exists():
            for child in target.iterdir():
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
    info["local_reset"] = True
    return info


def _run_turn(
    *,
    graph,
    memory_manager: SessionMemoryManager,
    user_id: str,
    session_id: str,
    query: str,
    include_pending_in_prompt: bool,
    persistent_gate_threshold: int,
    max_attempts: int = 3,
) -> Dict[str, Any]:
    attempts = 0
    start = time.time()
    result: Dict[str, Any] = {}

    while True:
        attempts += 1
        patch = memory_manager.build_state_patch(
            user_id=user_id,
            session_id=session_id,
            include_pending_in_prompt=include_pending_in_prompt,
        )
        result = asyncio.run(
            graph.ainvoke(
                _build_initial_state(
                    query,
                    session_id=session_id,
                    user_id=user_id,
                    memory_patch=patch,
                )
            )
        )

        answer = str(result.get("final_answer", "") or "")
        tool_outputs = [str(x.get("output", "")) for x in (result.get("tool_results", []) or [])]
        failed = _is_failure(answer)
        transient = _is_transient_failure(answer) or any(_is_transient_failure(x) for x in tool_outputs)
        if (not failed) or attempts >= max_attempts or (not transient):
            break
        time.sleep(1.0 * attempts)

    elapsed = round(time.time() - start, 2)
    memory_manager.save_from_state(user_id=user_id, session_id=session_id, state=result)

    tool_results = [x for x in (result.get("tool_results", []) or []) if isinstance(x, dict)]
    gate_score = int(result.get("memory_persistent_gate_score", 0) or 0)

    return {
        "timestamp_utc": _now_utc(),
        "query": query,
        "attempts": attempts,
        "elapsed_sec": elapsed,
        "intent": result.get("intent", ""),
        "flow_type": result.get("flow_type", ""),
        "selected_tool": result.get("selected_tool", ""),
        "selected_skill": result.get("selected_skill", ""),
        "tool_query": result.get("tool_query", ""),
        "understanding_entities": result.get("understanding_entities", []) or [],
        "requires_task_planning": bool(result.get("requires_task_planning", False)),
        "task_plan": result.get("task_plan", []) or [],
        "tool_calls": result.get("tool_calls", []) or [],
        "tool_results": tool_results,
        "tool_result_count": len(tool_results),
        "tool_result_fail_count": sum(1 for x in tool_results if _is_failure(str(x.get("output", "")))),
        "memory_gate_score": gate_score,
        "memory_gate_triggered": gate_score >= persistent_gate_threshold,
        "memory_persistent_used": bool(result.get("memory_persistent_used", False)),
        "memory_recall_hits": len(result.get("memory_persistent_hits", []) or []),
        "final_answer": result.get("final_answer", ""),
        "final_answer_failed": _is_failure(result.get("final_answer", "")),
        "quality_gate_passed": bool(result.get("quality_gate_passed", False)),
        "retry_count_total": int(result.get("retry_count_total", 0) or 0),
        "retry_budget_exhausted": bool(result.get("retry_budget_exhausted", False)),
        "validator_reject": any(
            bool(x.get("retry_requested", False))
            for x in (result.get("retry_trace", []) or [])
            if isinstance(x, dict) and str(x.get("stage", "")) == "tool_output_validator"
        ),
        "reviewer_reject": any(
            bool(x.get("retry_requested", False))
            for x in (result.get("retry_trace", []) or [])
            if isinstance(x, dict) and str(x.get("stage", "")) == "answer_reviewer"
        ),
        "debug_steps": result.get("debug_steps", []) or [],
    }


def _build_profiles(device: str) -> List[ModelProfile]:
    return [
        ModelProfile(
            name="kimi",
            label="kimi",
            cfg_patch={
                "agent_local_enabled": False,
                "agent_intent_model": "kimi-k2-0711-preview",
                "agent_planner_model": "kimi-k2-0711-preview",
            },
        ),
        ModelProfile(
            name="base_qwen3_8b",
            label="qwen8b",
            cfg_patch={
                "agent_local_enabled": True,
                "agent_local_device": device,
                "agent_intent_model": "models/Qwen3-8B",
                "agent_planner_model": "models/Qwen3-8B",
                "agent_intent_adapter_path": "",
                "agent_tool_selection_adapter_path": "",
                "agent_planning_adapter_path": "",
            },
        ),
        ModelProfile(
            name="qwen3_8b_lora",
            label="qwen8b_lora",
            cfg_patch={
                "agent_local_enabled": True,
                "agent_local_device": device,
                "agent_intent_model": "models/Qwen3-8B",
                "agent_planner_model": "models/Qwen3-8B",
                "agent_intent_adapter_path": "outputs/intent_sft/qwen3_8b_lora",
                "agent_tool_selection_adapter_path": "outputs/tool_selection_sft/qwen3_8b_lora",
                "agent_planning_adapter_path": "outputs/planning_sft/qwen3_8b_lora",
            },
        ),
    ]


def _extract_executed_chain(turn: Dict[str, Any]) -> List[str]:
    chain: List[str] = []
    for item in turn.get("tool_results", []) or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("tool_name", "") or "").strip()
        if name and name not in chain:
            chain.append(name)
    if not chain:
        sel = str(turn.get("selected_tool", "") or "").strip()
        if sel:
            chain.append(sel)
    return chain


def _evaluate_turn(turn: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, Any]:
    expected_intents = [str(x) for x in expected.get("expected_intents", []) if str(x)]
    expected_entities = [str(x) for x in expected.get("expected_entities", []) if str(x)]
    expected_q_contains = [str(x) for x in expected.get("expected_tool_query_contains", []) if str(x)]
    expected_candidates = [str(x) for x in expected.get("expected_tool_candidates", []) if str(x)]
    expected_chain = [str(x) for x in expected.get("expected_tool_chain", []) if str(x)]

    selected_tool = str(turn.get("selected_tool", "") or "")
    selected_skill = str(turn.get("selected_skill", "") or "")
    tool_query = str(turn.get("tool_query", "") or "")

    tool_ok = selected_tool == str(expected.get("expected_tool", "") or "")
    candidate_ok = (selected_tool in expected_candidates) if expected_candidates else tool_ok
    skill_ok = selected_skill == str(expected.get("expected_skill", "") or "")
    intent_ok = _intent_match(str(turn.get("intent", "") or ""), selected_tool, expected_intents)
    entity_ok = _entity_hit(
        expected_entities=expected_entities,
        entities=[str(x) for x in (turn.get("understanding_entities", []) or [])],
        tool_query=tool_query,
        answer=str(turn.get("final_answer", "") or ""),
    )
    tool_query_ok = all(x in tool_query for x in expected_q_contains)
    coverage = _keyword_coverage(str(turn.get("final_answer", "") or ""), [str(x) for x in expected.get("answer_keywords", [])])

    expected_complex = str(expected.get("complexity", "simple") or "simple")
    actual_complex = str(turn.get("flow_type", "") or "simple")
    complexity_ok = actual_complex == expected_complex

    planning_expected = bool(expected.get("expect_requires_task_planning", False))
    planning_triggered = bool(turn.get("requires_task_planning", False))
    planning_done = _stage_done(turn, "task_planning") or (not _stage_skipped(turn, "task_planning") and _stage_seen(turn, "task_planning:"))

    actual_chain = _extract_executed_chain(turn)
    chain_ok = True
    if expected_chain:
        chain_ok = all(x in actual_chain for x in expected_chain)

    memory_expected = bool(expected.get("expect_memory_resolution", False))
    memory_ok = True if not memory_expected else (entity_ok or tool_query_ok)

    persistent_expected = bool(expected.get("expect_persistent_recall", False))
    persistent_triggered = bool(turn.get("memory_gate_triggered", False))
    persistent_hit = int(turn.get("memory_recall_hits", 0) or 0) > 0
    persistent_ok = True if not persistent_expected else (persistent_triggered and persistent_hit)

    return {
        "intent_ok": intent_ok,
        "tool_ok": tool_ok,
        "tool_candidate_ok": candidate_ok,
        "skill_ok": skill_ok,
        "entity_ok": entity_ok,
        "tool_query_ok": tool_query_ok,
        "keyword_coverage": round(float(coverage), 4),
        "complexity_ok": complexity_ok,
        "planning_expected": planning_expected,
        "planning_triggered": planning_triggered,
        "planning_done": planning_done,
        "chain_expected": bool(expected_chain),
        "chain_ok": chain_ok,
        "actual_chain": actual_chain,
        "memory_expected": memory_expected,
        "memory_ok": memory_ok,
        "persistent_expected": persistent_expected,
        "persistent_triggered": persistent_triggered,
        "persistent_hit": persistent_hit,
        "persistent_ok": persistent_ok,
        "success": not bool(turn.get("final_answer_failed", False)),
        "selected_tool": selected_tool,
        "selected_skill": selected_skill,
        "intent": str(turn.get("intent", "") or ""),
        "complexity_expected": expected_complex,
        "complexity_actual": actual_complex,
        "elapsed_sec": float(turn.get("elapsed_sec", 0.0) or 0.0),
        "retry_count_total": int(turn.get("retry_count_total", 0) or 0),
        "quality_gate_passed": bool(turn.get("quality_gate_passed", False)),
        "validator_reject": bool(turn.get("validator_reject", False)),
        "reviewer_reject": bool(turn.get("reviewer_reject", False)),
        "retry_budget_exhausted": bool(turn.get("retry_budget_exhausted", False)),
    }


def _safe_rate(num: int, den: int) -> float:
    return round((num / den), 4) if den > 0 else 0.0


def _aggregate_model_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    lat = [float(x["elapsed_sec"]) for x in rows]

    simple = [x for x in rows if x["complexity_expected"] == "simple"]
    complex_rows = [x for x in rows if x["complexity_expected"] == "complex"]

    kb_rows = [x for x in rows if x.get("selected_tool") == "rag_knowledge_search" or x.get("selected_tool") == ""]
    expected_kb_rows = [x for x in rows if x.get("expected_tool") == "rag_knowledge_search"]

    planning_expected_rows = [x for x in rows if x["planning_expected"]]
    chain_expected_rows = [x for x in rows if x["chain_expected"]]
    memory_expected_rows = [x for x in rows if x["memory_expected"]]
    persistent_expected_rows = [x for x in rows if x["persistent_expected"]]

    out = {
        "turn_count": n,
        "intent_accuracy": _safe_rate(sum(1 for x in rows if x["intent_ok"]), n),
        "tool_top1_accuracy": _safe_rate(sum(1 for x in rows if x["tool_ok"]), n),
        "tool_candidate_hit_rate": _safe_rate(sum(1 for x in rows if x["tool_candidate_ok"]), n),
        "skill_accuracy": _safe_rate(sum(1 for x in rows if x["skill_ok"]), n),
        "entity_resolution_accuracy": _safe_rate(sum(1 for x in rows if x["entity_ok"]), n),
        "tool_query_entity_hit_rate": _safe_rate(sum(1 for x in rows if x["tool_query_ok"]), n),
        "answer_keyword_coverage": round(_avg(x["keyword_coverage"] for x in rows), 4),
        "complexity_accuracy": _safe_rate(sum(1 for x in rows if x["complexity_ok"]), n),
        "simple_tool_accuracy": _safe_rate(sum(1 for x in simple if x["tool_ok"]), len(simple)),
        "complex_tool_accuracy": _safe_rate(sum(1 for x in complex_rows if x["tool_ok"]), len(complex_rows)),
        "planning_trigger_recall": _safe_rate(sum(1 for x in planning_expected_rows if x["planning_triggered"]), len(planning_expected_rows)),
        "planning_execution_recall": _safe_rate(sum(1 for x in planning_expected_rows if x["planning_done"]), len(planning_expected_rows)),
        "planning_chain_hit_rate": _safe_rate(sum(1 for x in chain_expected_rows if x["chain_ok"]), len(chain_expected_rows)),
        "short_memory_resolution_rate": _safe_rate(sum(1 for x in memory_expected_rows if x["memory_ok"]), len(memory_expected_rows)),
        "persistent_recall_trigger_rate_expected": _safe_rate(sum(1 for x in persistent_expected_rows if x["persistent_triggered"]), len(persistent_expected_rows)),
        "persistent_recall_hit_rate_expected": _safe_rate(sum(1 for x in persistent_expected_rows if x["persistent_hit"]), len(persistent_expected_rows)),
        "persistent_recall_success_rate_expected": _safe_rate(sum(1 for x in persistent_expected_rows if x["persistent_ok"]), len(persistent_expected_rows)),
        "kb_route_accuracy": _safe_rate(sum(1 for x in expected_kb_rows if x.get("selected_tool") == "rag_knowledge_search"), len(expected_kb_rows)),
        "kb_entity_recall_proxy": _safe_rate(sum(1 for x in expected_kb_rows if x["entity_ok"]), len(expected_kb_rows)),
        "kb_answer_keyword_coverage": round(_avg(x["keyword_coverage"] for x in expected_kb_rows), 4) if expected_kb_rows else 0.0,
        "final_success_rate": _safe_rate(sum(1 for x in rows if x["success"]), n),
        "quality_gate_pass_rate": _safe_rate(sum(1 for x in rows if x["quality_gate_passed"]), n),
        "validator_reject_rate": _safe_rate(sum(1 for x in rows if x["validator_reject"]), n),
        "reviewer_reject_rate": _safe_rate(sum(1 for x in rows if x["reviewer_reject"]), n),
        "retry_invocation_rate": _safe_rate(sum(1 for x in rows if x["retry_count_total"] > 0), n),
        "retry_budget_exhausted_rate": _safe_rate(sum(1 for x in rows if x["retry_budget_exhausted"]), n),
        "avg_latency_sec": round(_avg(lat), 2),
        "p50_latency_sec": round(_percentile(lat, 0.5), 2),
        "p95_latency_sec": round(_percentile(lat, 0.95), 2),
    }
    return out


def _run_profile(
    profile: ModelProfile,
    benchmark: Dict[str, Any],
    max_cases: int | None = None,
    case_offset: int = 0,
    max_attempts: int = 3,
    max_turns_per_session: int | None = None,
) -> Dict[str, Any]:
    cfg_dict = DEFAULT_CONFIG.to_dict()
    cfg_dict.update(profile.cfg_patch)
    cfg = GraphRAGConfig.from_dict(cfg_dict)

    rag_service = RAGService(cfg)
    registry = ToolRegistry(rag_service=rag_service, config=cfg)
    store = PersistentMemoryStore(cfg)
    graph = build_multi_agent_graph(registry, persistent_store=store)

    try:
        _reset_persistent_memory(store)

        rows: List[Dict[str, Any]] = []
        all_cases = benchmark.get("cases", [])
        start = max(0, int(case_offset))
        if isinstance(max_cases, int) and max_cases > 0:
            cases = all_cases[start : start + max_cases]
        else:
            cases = all_cases[start:]

        for case in cases:
            case_id = str(case.get("case_id", "") or "")
            user_id_base = str(case.get("user_id", "") or "user")
            user_id = f"{user_id_base}_{profile.label}"
            print(f"[{profile.name}] case={case_id} user={user_id}")

            for session in case.get("sessions", []):
                session_id = f"{str(session.get('session_id', '') or '')}_{profile.label}"
                manager = SessionMemoryManager()
                raw_turns = session.get("turns", []) or []
                turns = raw_turns[:max_turns_per_session] if isinstance(max_turns_per_session, int) and max_turns_per_session > 0 else raw_turns
                print(f"[{profile.name}]  session={session_id} turns={len(turns)}")

                for ti, expected in enumerate(turns, 1):
                    query = str(expected.get("query", "") or "")
                    print(f"[{profile.name}]   turn={ti} query={query[:40]}")
                    try:
                        turn = _run_turn(
                            graph=graph,
                            memory_manager=manager,
                            user_id=user_id,
                            session_id=session_id,
                            query=query,
                            include_pending_in_prompt=cfg.memory_include_pending_in_prompt,
                            persistent_gate_threshold=int(cfg.memory_persistent_trigger_threshold or 2),
                            max_attempts=max_attempts,
                        )
                    except Exception as e:  # noqa: BLE001
                        turn = {
                            "timestamp_utc": _now_utc(),
                            "query": query,
                            "attempts": 1,
                            "elapsed_sec": 0.0,
                            "intent": "",
                            "flow_type": "",
                            "selected_tool": "",
                            "selected_skill": "",
                            "tool_query": "",
                            "understanding_entities": [],
                            "requires_task_planning": False,
                            "task_plan": [],
                            "tool_calls": [],
                            "tool_results": [],
                            "tool_result_count": 0,
                            "tool_result_fail_count": 0,
                            "memory_gate_score": 0,
                            "memory_gate_triggered": False,
                            "memory_persistent_used": False,
                            "memory_recall_hits": 0,
                            "final_answer": f"系统错误: {type(e).__name__}: {e}",
                            "final_answer_failed": True,
                            "quality_gate_passed": False,
                            "retry_count_total": 0,
                            "retry_budget_exhausted": False,
                            "validator_reject": False,
                            "reviewer_reject": False,
                            "debug_steps": [f"benchmark_turn_error: {type(e).__name__}"],
                        }
                        print(f"[{profile.name}]   turn_error={type(e).__name__}: {e}")
                    eval_item = _evaluate_turn(turn, expected)
                    eval_item.update(
                        {
                            "case_id": case_id,
                            "user_id": user_id,
                            "session_id": session_id,
                            "turn_index": ti,
                            "query": query,
                            "expected_tool": expected.get("expected_tool", ""),
                            "expected_skill": expected.get("expected_skill", ""),
                            "expected_intents": expected.get("expected_intents", []),
                            "expected_entities": expected.get("expected_entities", []),
                            "expected_tool_chain": expected.get("expected_tool_chain", []),
                            "expect_requires_task_planning": bool(expected.get("expect_requires_task_planning", False)),
                            "expect_memory_resolution": bool(expected.get("expect_memory_resolution", False)),
                            "expect_persistent_recall": bool(expected.get("expect_persistent_recall", False)),
                        }
                    )
                    rows.append(eval_item)

                asyncio.run(
                    _finalize_session_memory(
                        user_id=user_id,
                        session_id=session_id,
                        config=cfg,
                        memory_manager=manager,
                        persistent_store=store,
                    )
                )

        metrics = _aggregate_model_metrics(rows)
        return {
            "profile": profile.name,
            "label": profile.label,
            "config_patch": profile.cfg_patch,
            "case_count": len(cases),
            "turn_count": len(rows),
            "metrics": metrics,
            "rows": rows,
        }
    finally:
        asyncio.run(registry.close_async())


def _metric_definitions() -> Dict[str, str]:
    return {
        "intent_accuracy": "意图识别准确率（预测intent命中期望intent）",
        "tool_top1_accuracy": "工具Top-1准确率（selected_tool==expected_tool）",
        "tool_candidate_hit_rate": "工具候选命中率（selected_tool在expected_tool_candidates中）",
        "skill_accuracy": "技能匹配准确率（selected_skill==expected_skill）",
        "entity_resolution_accuracy": "实体解析准确率（实体在理解/工具query/回答中被正确解析）",
        "tool_query_entity_hit_rate": "工具query实体命中率（query包含期望实体）",
        "answer_keyword_coverage": "回答关键词覆盖率（answer_keywords命中占比）",
        "complexity_accuracy": "简单/复杂任务分型准确率",
        "simple_tool_accuracy": "简单任务工具准确率",
        "complex_tool_accuracy": "复杂任务工具准确率",
        "planning_trigger_recall": "应触发规划的样例中，实际触发规划比例",
        "planning_execution_recall": "应触发规划的样例中，实际执行task_planning节点比例",
        "planning_chain_hit_rate": "期望工具链样例中，执行链覆盖期望链的比例",
        "short_memory_resolution_rate": "需要短期记忆补全的样例中，补全成功率",
        "persistent_recall_trigger_rate_expected": "需要长期记忆的样例中，门控触发率",
        "persistent_recall_hit_rate_expected": "需要长期记忆的样例中，召回命中率",
        "persistent_recall_success_rate_expected": "需要长期记忆的样例中，触发+命中综合成功率",
        "kb_route_accuracy": "知识库RAG路由准确率（知识类是否走rag_knowledge_search）",
        "kb_entity_recall_proxy": "知识库RAG实体召回代理指标",
        "kb_answer_keyword_coverage": "知识库RAG回答关键词覆盖率",
        "final_success_rate": "最终回答成功率（非失败文案）",
        "quality_gate_pass_rate": "质量审查通过率",
        "validator_reject_rate": "工具输出审查拒绝率",
        "reviewer_reject_rate": "回答审查拒绝率",
        "retry_invocation_rate": "触发重试比例",
        "retry_budget_exhausted_rate": "重试预算耗尽比例",
        "avg_latency_sec": "平均时延(秒)",
        "p50_latency_sec": "P50时延(秒)",
        "p95_latency_sec": "P95时延(秒)",
    }


def _build_sft_focus_summary(model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    focus_metrics = ["intent_accuracy", "tool_top1_accuracy", "planning_chain_hit_rate"]
    by_name: Dict[str, Dict[str, Any]] = {str(x["profile"]): x for x in model_results}
    base = by_name.get("base_qwen3_8b")
    lora = by_name.get("qwen3_8b_lora")
    kimi = by_name.get("kimi")

    rows: List[Dict[str, Any]] = []
    for m in focus_metrics:
        row: Dict[str, Any] = {"metric": m}
        if kimi:
            row["kimi"] = float(kimi["metrics"].get(m, 0.0) or 0.0)
        if base:
            row["base_qwen3_8b"] = float(base["metrics"].get(m, 0.0) or 0.0)
        if lora:
            row["qwen3_8b_lora"] = float(lora["metrics"].get(m, 0.0) or 0.0)
        if base and lora:
            row["delta_lora_vs_base"] = round(row["qwen3_8b_lora"] - row["base_qwen3_8b"], 4)
        if kimi and lora:
            row["gap_lora_vs_kimi"] = round(row["qwen3_8b_lora"] - row["kimi"], 4)
        rows.append(row)

    return {"focus_metrics": focus_metrics, "rows": rows}


def _write_report(result: Dict[str, Any], prefix: str) -> Dict[str, str]:
    docs = PROJECT_ROOT / "docs"
    json_path = docs / f"{prefix}_RESULT.json"
    md_path = docs / f"{prefix}_REPORT.md"
    brief_path = docs / f"{prefix}_REPORT_BRIEF.md"

    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    metric_defs = _metric_definitions()
    model_results = result["model_results"]
    sft_focus = _build_sft_focus_summary(model_results)

    # markdown full
    lines: List[str] = []
    lines.append("# Conversation Benchmark 三模型对比报告")
    lines.append("")
    lines.append(f"- 生成时间(UTC): `{result['meta']['generated_at_utc']}`")
    lines.append(f"- Benchmark: `{result['meta']['benchmark_file']}`")
    lines.append(f"- Case数: `{result['meta']['case_count']}`")
    lines.append("")

    lines.append("## 指标说明")
    for k, v in metric_defs.items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")

    lines.append("## SFT聚焦指标（Intent/ToolSelection/Planning）")
    lines.append("| metric | kimi | base_qwen3_8b | qwen3_8b_lora | Δ(lora-base) |")
    lines.append("|---|---|---|---|---|")
    for row in sft_focus["rows"]:
        lines.append(
            "| {metric} | {kimi} | {base} | {lora} | {delta} |".format(
                metric=row["metric"],
                kimi=row.get("kimi", ""),
                base=row.get("base_qwen3_8b", ""),
                lora=row.get("qwen3_8b_lora", ""),
                delta=row.get("delta_lora_vs_base", ""),
            )
        )
    lines.append("")

    lines.append("## 模型指标")
    for mr in model_results:
        lines.append(f"### {mr['profile']}")
        for k, v in mr["metrics"].items():
            lines.append(f"- `{k}`: `{v}`")
        lines.append("")

    # comparison table
    keys = list(metric_defs.keys())
    lines.append("## 横向对比")
    header = ["metric"] + [mr["profile"] for mr in model_results]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for k in keys:
        row = [k] + [str(mr["metrics"].get(k, "")) for mr in model_results]
        lines.append("| " + " | ".join(row) + " |")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    brief: List[str] = []
    brief.append("# Conversation Benchmark 简报")
    brief.append("")
    brief.append(f"- 时间(UTC): `{result['meta']['generated_at_utc']}`")
    brief.append(f"- Benchmark: `{result['meta']['benchmark_file']}`")
    brief.append(f"- Case数: `{result['meta']['case_count']}`")
    brief.append("")
    brief.append("## SFT聚焦指标（Intent/ToolSelection/Planning）")
    for row in sft_focus["rows"]:
        brief.append(
            "- `{metric}`: kimi=`{kimi}` / base=`{base}` / lora=`{lora}` / Δ(lora-base)=`{delta}`".format(
                metric=row["metric"],
                kimi=row.get("kimi", ""),
                base=row.get("base_qwen3_8b", ""),
                lora=row.get("qwen3_8b_lora", ""),
                delta=row.get("delta_lora_vs_base", ""),
            )
        )
    brief.append("")
    for mr in model_results:
        m = mr["metrics"]
        brief.append(f"## {mr['profile']}")
        brief.append(f"- intent_accuracy: `{m.get('intent_accuracy')}`")
        brief.append(f"- tool_top1_accuracy: `{m.get('tool_top1_accuracy')}`")
        brief.append(f"- complex_tool_accuracy: `{m.get('complex_tool_accuracy')}`")
        brief.append(f"- planning_chain_hit_rate: `{m.get('planning_chain_hit_rate')}`")
        brief.append(f"- short_memory_resolution_rate: `{m.get('short_memory_resolution_rate')}`")
        brief.append(f"- persistent_recall_success_rate_expected: `{m.get('persistent_recall_success_rate_expected')}`")
        brief.append(f"- kb_route_accuracy: `{m.get('kb_route_accuracy')}`")
        brief.append(f"- final_success_rate: `{m.get('final_success_rate')}`")
        brief.append(f"- p95_latency_sec: `{m.get('p95_latency_sec')}`")
        brief.append("")

    brief_path.write_text("\n".join(brief), encoding="utf-8")

    return {
        "json": str(json_path),
        "md": str(md_path),
        "md_brief": str(brief_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conversation-level benchmark runner")
    parser.add_argument(
        "--benchmark-file",
        type=str,
        default="data/benchmarks/system_conversation_benchmark_100.json",
        help="Benchmark json path",
    )
    parser.add_argument(
        "--report-prefix",
        type=str,
        default="SYSTEM_CONVERSATION_BENCHMARK_100",
        help="Output report prefix",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Optional cap for quick run; 0 means full",
    )
    parser.add_argument(
        "--case-offset",
        type=int,
        default=0,
        help="Start benchmark from this case index (0-based)",
    )
    parser.add_argument(
        "--local-device",
        type=str,
        default="cuda",
        help="Local Qwen device for base/lora profiles",
    )
    parser.add_argument(
        "--profiles",
        type=str,
        default="kimi,base_qwen3_8b,qwen3_8b_lora",
        help="Comma-separated profile names to run",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Max retry attempts for transient turn-level failures",
    )
    parser.add_argument(
        "--max-turns-per-session",
        type=int,
        default=0,
        help="Optional cap of turns per session; 0 means full session",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    benchmark_path = Path(args.benchmark_file)
    if not benchmark_path.is_absolute():
        benchmark_path = PROJECT_ROOT / benchmark_path
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))

    all_profiles = _build_profiles(device=str(args.local_device or "cuda"))
    enabled = {x.strip() for x in str(args.profiles or "").split(",") if x.strip()}
    profiles = [p for p in all_profiles if p.name in enabled]
    if not profiles:
        raise ValueError(f"No valid profiles selected: {sorted(enabled)}")
    max_cases = int(args.max_cases) if int(args.max_cases) > 0 else None
    case_offset = max(0, int(args.case_offset))
    max_attempts = max(1, int(args.max_attempts))
    max_turns_per_session = int(args.max_turns_per_session) if int(args.max_turns_per_session) > 0 else None

    model_results: List[Dict[str, Any]] = []
    for profile in profiles:
        print(f"[benchmark] running profile={profile.name}")
        started = time.time()
        try:
            model_results.append(
                _run_profile(
                    profile=profile,
                    benchmark=benchmark,
                    max_cases=max_cases,
                    case_offset=case_offset,
                    max_attempts=max_attempts,
                    max_turns_per_session=max_turns_per_session,
                )
            )
        except Exception as e:  # noqa: BLE001
            model_results.append(
                {
                    "profile": profile.name,
                    "label": profile.label,
                    "config_patch": profile.cfg_patch,
                    "case_count": 0,
                    "turn_count": 0,
                    "metrics": {},
                    "rows": [],
                    "fatal_error": {
                        "type": type(e).__name__,
                        "message": str(e),
                        "traceback": traceback.format_exc(limit=8),
                    },
                }
            )
            print(f"[benchmark] profile_failed={profile.name} err={type(e).__name__}: {e}")
        finally:
            print(f"[benchmark] profile={profile.name} elapsed_sec={round(time.time() - started, 2)}")

    result = {
        "meta": {
            "generated_at_utc": _now_utc(),
            "benchmark_file": str(benchmark_path),
            "case_offset": case_offset,
            "case_count": (
                max(0, len(benchmark.get("cases", [])) - case_offset)
                if max_cases is None
                else min(max(0, len(benchmark.get("cases", [])) - case_offset), max_cases)
            ),
            "profiles": [p.name for p in profiles],
            "max_cases": max_cases or 0,
            "max_turns_per_session": max_turns_per_session or 0,
        },
        "metric_definitions": _metric_definitions(),
        "sft_focus_summary": _build_sft_focus_summary(model_results),
        "model_results": model_results,
    }

    paths = _write_report(result, prefix=str(args.report_prefix).strip() or "SYSTEM_CONVERSATION_BENCHMARK_100")
    print(json.dumps({"status": "ok", "reports": paths}, ensure_ascii=False))


if __name__ == "__main__":
    main()
