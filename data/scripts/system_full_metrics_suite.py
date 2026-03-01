#!/usr/bin/env python3
"""全量系统测试 + 指标评估（功能、工具、审查重试、知识库RAG、长期记忆RAG）。"""

from __future__ import annotations

import json
import re
import shutil
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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
from config import DEFAULT_CONFIG
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
PRONOUN_MARKERS = ("它", "他", "她", "这两个", "这三个", "刚才", "上次", "之前", "那个")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        "df_multi_item_compare": "market_compare_query",
        "df_place_profit_rank": "place_profit_query",
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
    intent_text = (raw + " " + canonical).lower()
    if "history" in " ".join(expected) and ("历史" in intent_text or "history" in intent_text):
        return True
    if "latest" in " ".join(expected) and ("价格" in intent_text and "历史" not in intent_text):
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


def _find_memory_compression(result: Dict[str, Any]) -> Dict[str, Any]:
    for msg in reversed(result.get("agent_messages", []) or []):
        if msg.get("message_type") != "memory_update":
            continue
        payload = msg.get("payload")
        if isinstance(payload, dict):
            compression = payload.get("compression")
            if isinstance(compression, dict):
                return compression
    return {}


def _extract_retry_signals(trace: List[Dict[str, Any]]) -> Dict[str, bool]:
    validator_reject = False
    reviewer_reject = False
    for item in trace:
        if not isinstance(item, dict):
            continue
        stage = str(item.get("stage", "") or "")
        requested = bool(item.get("retry_requested", False))
        if stage == "tool_output_validator" and requested:
            validator_reject = True
        if stage == "answer_reviewer" and requested:
            reviewer_reject = True
    return {
        "validator_reject": validator_reject,
        "reviewer_reject": reviewer_reject,
    }


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
        result = graph.invoke(_build_initial_state(query, session_id=session_id, user_id=user_id, memory_patch=patch))

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
    retry_trace = [x for x in (result.get("retry_trace", []) or []) if isinstance(x, dict)]
    retry_signals = _extract_retry_signals(retry_trace)

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
        "tool_results": tool_results,
        "tool_result_count": len(tool_results),
        "tool_result_fail_count": sum(1 for x in tool_results if _is_failure(str(x.get("output", "")))),
        "memory_gate_score": gate_score,
        "memory_gate_triggered": gate_score >= persistent_gate_threshold,
        "memory_persistent_used": bool(result.get("memory_persistent_used", False)),
        "memory_recall_hits": len(result.get("memory_persistent_hits", []) or []),
        "memory_recall_effective": len(result.get("memory_persistent_hits", []) or []) > 0,
        "memory_compression": _find_memory_compression(result),
        "memory_stats_after": memory_manager.stats(user_id=user_id, session_id=session_id),
        "final_answer": result.get("final_answer", ""),
        "final_answer_failed": _is_failure(result.get("final_answer", "")),
        "quality_gate_passed": bool(result.get("quality_gate_passed", False)),
        "quality_score": float(result.get("quality_score", 0.0) or 0.0),
        "validation_result": result.get("validation_result", {}) or {},
        "review_result": result.get("review_result", {}) or {},
        "retry_count_total": int(result.get("retry_count_total", 0) or 0),
        "retry_count_by_stage": result.get("retry_count_by_stage", {}) or {},
        "retry_budget_exhausted": bool(result.get("retry_budget_exhausted", False)),
        "retry_trace": retry_trace,
        "validator_reject": retry_signals["validator_reject"],
        "reviewer_reject": retry_signals["reviewer_reject"],
        "block_persistent_write": bool(result.get("block_persistent_write", False)),
        "execution_attempt": int(result.get("execution_attempt", 0) or 0),
        "summary_attempt": int(result.get("summary_attempt", 0) or 0),
        "debug_steps": result.get("debug_steps", []) or [],
    }


def _run_fault_injection_retry_case(
    *,
    graph,
    registry: ToolRegistry,
    include_pending_in_prompt: bool,
    persistent_gate_threshold: int,
) -> Dict[str, Any]:
    """故障注入：首次价格查询强制返回 502，验证 execution->validator->retry_router->execution 链路。"""

    manager = SessionMemoryManager()
    session_id = f"fault-injection-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    user_id = "fault-user-0001"

    original_invoke = registry.invoke
    counter = {"injected": 0}

    def _flaky_invoke(tool_name: str, query: str) -> str:
        if (
            tool_name == "df_market_latest_price"
            and "非洲之心" in str(query)
            and counter["injected"] == 0
        ):
            counter["injected"] += 1
            return "查询失败：HTTP 502（fault_injection）"
        return original_invoke(tool_name, query)

    registry.invoke = _flaky_invoke
    try:
        turn = _run_turn(
            graph=graph,
            memory_manager=manager,
            user_id=user_id,
            session_id=session_id,
            query="非洲之心现在什么价格",
            include_pending_in_prompt=include_pending_in_prompt,
            persistent_gate_threshold=persistent_gate_threshold,
            max_attempts=1,
        )
    finally:
        registry.invoke = original_invoke

    retry_trace = [x for x in (turn.get("retry_trace", []) or []) if isinstance(x, dict)]
    retry_stages = [str(x.get("stage", "") or "") for x in retry_trace]

    turn["fault_injection_meta"] = {
        "injected_failures": counter["injected"],
        "retry_stages": retry_stages,
        "retry_count_total": int(turn.get("retry_count_total", 0) or 0),
        "validator_reject": bool(turn.get("validator_reject", False)),
        "final_answer_failed": bool(turn.get("final_answer_failed", False)),
    }
    return turn


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


def _db_counts(store: PersistentMemoryStore, user_id: str, session_id: str) -> Dict[str, Any]:
    conn = store._connect()  # noqa: SLF001
    if conn is None:
        return {"user_id": user_id, "session_id": session_id, "error": "db_connect_failed"}

    with conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chat_turns WHERE user_id=%s AND session_id=%s;", (user_id, session_id))
            chat_turns = int((cur.fetchone() or [0])[0] or 0)
            cur.execute("SELECT COUNT(*) FROM memory_summaries WHERE user_id=%s AND session_id=%s;", (user_id, session_id))
            summaries = int((cur.fetchone() or [0])[0] or 0)
            cur.execute("SELECT COUNT(*) FROM memory_facts WHERE user_id=%s AND session_id=%s;", (user_id, session_id))
            facts = int((cur.fetchone() or [0])[0] or 0)

    return {
        "user_id": user_id,
        "session_id": session_id,
        "chat_turns": chat_turns,
        "memory_summaries": summaries,
        "memory_facts": facts,
    }


def _avg(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.mean(values)) if values else 0.0


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = max(0, min(len(arr) - 1, int((len(arr) - 1) * p)))
    return float(arr[idx])


def _aggregate_core_metrics(single: List[Dict[str, Any]], multi: List[Dict[str, Any]], isolation_ok: bool) -> Dict[str, Any]:
    n = max(1, len(single))
    mt_n = max(1, len(multi))

    intent_acc = sum(1 for x in single if x.get("intent_ok")) / n
    tool_acc = sum(1 for x in single if x.get("tool_ok")) / n
    skill_acc = sum(1 for x in single if x.get("skill_ok")) / n
    entity_acc = sum(1 for x in single if x.get("entity_ok")) / n
    keyword_cov = _avg(float(x.get("answer_keyword_coverage", 0.0) or 0.0) for x in single)
    success_rate = sum(1 for x in single if not x.get("final_answer_failed")) / n

    mt_success = (
        sum(
            1
            for x in multi
            if x.get("tool_ok") and x.get("skill_ok") and x.get("tool_query_ok") and not x.get("final_answer_failed")
        )
        / mt_n
    )
    mt_keyword_cov = _avg(float(x.get("answer_keyword_coverage", 0.0) or 0.0) for x in multi)
    isolation_score = 1.0 if isolation_ok else 0.0

    overall = (
        0.16 * intent_acc
        + 0.18 * tool_acc
        + 0.08 * skill_acc
        + 0.10 * entity_acc
        + 0.10 * keyword_cov
        + 0.08 * success_rate
        + 0.08 * mt_success
        + 0.04 * mt_keyword_cov
        + 0.04 * isolation_score
    )

    return {
        "intent_accuracy": round(intent_acc, 4),
        "tool_accuracy": round(tool_acc, 4),
        "skill_accuracy": round(skill_acc, 4),
        "entity_resolution_accuracy": round(entity_acc, 4),
        "answer_keyword_coverage": round(keyword_cov, 4),
        "single_turn_success_rate": round(success_rate, 4),
        "multi_turn_success_rate": round(mt_success, 4),
        "multi_turn_keyword_coverage": round(mt_keyword_cov, 4),
        "user_isolation_success": bool(isolation_ok),
        "overall_score": round(overall, 4),
    }


def _aggregate_kb_rag_metrics(single: List[Dict[str, Any]]) -> Dict[str, Any]:
    kb = [x for x in single if str(x.get("expected_tool", "")) == "rag_knowledge_search"]
    n = max(1, len(kb))

    route_acc = sum(1 for x in kb if str(x.get("selected_tool", "")) == "rag_knowledge_search") / n
    entity_recall = sum(1 for x in kb if x.get("entity_ok")) / n
    answer_cov = _avg(float(x.get("answer_keyword_coverage", 0.0) or 0.0) for x in kb)
    success_rate = sum(1 for x in kb if not x.get("final_answer_failed")) / n
    lat = _avg(float(x.get("elapsed_sec", 0.0) or 0.0) for x in kb)

    return {
        "kb_rag_case_count": len(kb),
        "kb_rag_route_accuracy": round(route_acc, 4),
        "kb_rag_entity_recall_proxy": round(entity_recall, 4),
        "kb_rag_answer_keyword_coverage": round(answer_cov, 4),
        "kb_rag_success_rate": round(success_rate, 4),
        "kb_rag_avg_latency_sec": round(lat, 2),
    }


def _aggregate_ltm_rag_metrics(ltm_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = max(1, len(ltm_cases))
    triggered = [x for x in ltm_cases if x.get("memory_gate_triggered")]
    t_n = max(1, len(triggered))

    gate_rate = sum(1 for x in ltm_cases if x.get("memory_gate_triggered")) / n
    hit_rate = sum(1 for x in ltm_cases if int(x.get("memory_recall_hits", 0) or 0) > 0) / n
    hit_rate_when_triggered = sum(1 for x in triggered if int(x.get("memory_recall_hits", 0) or 0) > 0) / t_n
    entity_acc = sum(1 for x in ltm_cases if x.get("entity_resolved")) / n
    success_rate = sum(1 for x in ltm_cases if not x.get("final_answer_failed")) / n
    avg_hits = _avg(int(x.get("memory_recall_hits", 0) or 0) for x in ltm_cases)

    return {
        "ltm_case_count": len(ltm_cases),
        "ltm_gate_trigger_rate": round(gate_rate, 4),
        "ltm_recall_hit_rate": round(hit_rate, 4),
        "ltm_recall_hit_rate_when_triggered": round(hit_rate_when_triggered, 4),
        "ltm_entity_resolution_accuracy": round(entity_acc, 4),
        "ltm_success_rate": round(success_rate, 4),
        "ltm_avg_recall_hits": round(avg_hits, 2),
    }


def _aggregate_engineering_metrics(all_turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = max(1, len(all_turns))
    lat = [float(x.get("elapsed_sec", 0.0) or 0.0) for x in all_turns]

    validator_reject_rate = sum(1 for x in all_turns if x.get("validator_reject")) / n
    reviewer_reject_rate = sum(1 for x in all_turns if x.get("reviewer_reject")) / n
    retry_invocation_rate = sum(1 for x in all_turns if int(x.get("retry_count_total", 0) or 0) > 0) / n
    retry_budget_exhausted_rate = sum(1 for x in all_turns if x.get("retry_budget_exhausted")) / n
    quality_gate_pass_rate = sum(1 for x in all_turns if x.get("quality_gate_passed")) / n
    persistent_write_block_rate = sum(1 for x in all_turns if x.get("block_persistent_write")) / n

    tool_total = sum(int(x.get("tool_result_count", 0) or 0) for x in all_turns)
    tool_fail = sum(int(x.get("tool_result_fail_count", 0) or 0) for x in all_turns)
    tool_success_rate = (tool_total - tool_fail) / max(1, tool_total)

    compression_cases = [x for x in all_turns if isinstance(x.get("memory_compression"), dict)]
    compression_trigger_rate = (
        sum(1 for x in compression_cases if bool((x.get("memory_compression") or {}).get("triggered")))
        / max(1, len(compression_cases))
    )

    return {
        "turn_count": len(all_turns),
        "avg_latency_sec": round(_avg(lat), 2),
        "p50_latency_sec": round(_percentile(lat, 0.5), 2),
        "p95_latency_sec": round(_percentile(lat, 0.95), 2),
        "avg_attempts": round(_avg(int(x.get("attempts", 1) or 1) for x in all_turns), 2),
        "validator_reject_rate": round(validator_reject_rate, 4),
        "reviewer_reject_rate": round(reviewer_reject_rate, 4),
        "retry_invocation_rate": round(retry_invocation_rate, 4),
        "retry_budget_exhausted_rate": round(retry_budget_exhausted_rate, 4),
        "quality_gate_pass_rate": round(quality_gate_pass_rate, 4),
        "persistent_write_block_rate": round(persistent_write_block_rate, 4),
        "tool_stage_success_rate": round(tool_success_rate, 4),
        "tool_stage_fail_rate": round(1.0 - tool_success_rate, 4),
        "compression_trigger_rate": round(compression_trigger_rate, 4),
    }


def _build_assertions(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    assertions: List[Dict[str, Any]] = []

    single = payload["single_turn_suite"]["results"]
    multi = payload["multi_turn_suite"]["results"]
    integration_turns = payload["integration_suite"]["main_turns"]
    reentry = payload["integration_suite"]["reentry_turn"]
    fault = payload.get("fault_injection_suite", {})
    metrics = payload["metrics"]

    expected_tools = {
        "rag_knowledge_search",
        "df_market_latest_price",
        "df_market_history_price",
        "df_market_price_advice",
        "df_place_profit_rank",
        "df_multi_item_compare",
        "df_profit_stability",
        "df_answer_composer",
    }

    selected_tools = {str(x.get("selected_tool", "")).strip() for x in integration_turns}
    assertions.append(
        {
            "name": "tool_coverage_complete",
            "passed": expected_tools.issubset(selected_tools),
            "detail": {"selected_tools": sorted(selected_tools), "expected_tools": sorted(expected_tools)},
        }
    )

    assertions.append(
        {
            "name": "single_turn_core_quality",
            "passed": (
                metrics["core"]["intent_accuracy"] >= 0.85
                and metrics["core"]["tool_accuracy"] >= 0.9
                and metrics["core"]["single_turn_success_rate"] >= 0.9
            ),
            "detail": metrics["core"],
        }
    )

    assertions.append(
        {
            "name": "kb_rag_quality",
            "passed": (
                metrics["kb_rag"]["kb_rag_route_accuracy"] >= 0.9
                and metrics["kb_rag"]["kb_rag_entity_recall_proxy"] >= 0.8
                and metrics["kb_rag"]["kb_rag_success_rate"] >= 0.9
            ),
            "detail": metrics["kb_rag"],
        }
    )

    assertions.append(
        {
            "name": "ltm_rag_resolution",
            "passed": (
                metrics["ltm_rag"]["ltm_entity_resolution_accuracy"] >= 0.8
                and metrics["ltm_rag"]["ltm_success_rate"] >= 0.9
            ),
            "detail": metrics["ltm_rag"],
        }
    )

    assertions.append(
        {
            "name": "review_retry_budget_safe",
            "passed": metrics["engineering"]["retry_budget_exhausted_rate"] == 0.0,
            "detail": {
                "retry_budget_exhausted_rate": metrics["engineering"]["retry_budget_exhausted_rate"],
                "validator_reject_rate": metrics["engineering"]["validator_reject_rate"],
                "reviewer_reject_rate": metrics["engineering"]["reviewer_reject_rate"],
            },
        }
    )

    assertions.append(
        {
            "name": "multi_turn_reentry_no_compare_failure",
            "passed": (
                ("请至少提供两个物品名称" not in str(reentry.get("final_answer", "")))
                and ("对比失败" not in str(reentry.get("final_answer", "")))
            ),
            "detail": {
                "reentry_query": reentry.get("query", ""),
                "reentry_tool": reentry.get("selected_tool", ""),
                "reentry_tool_query": reentry.get("tool_query", ""),
                "reentry_answer": str(reentry.get("final_answer", ""))[:500],
            },
        }
    )

    assertions.append(
        {
            "name": "user_isolation_effective",
            "passed": bool(payload["user_isolation_suite"].get("isolation_ok", False)),
            "detail": {
                "user_a_tool_query": payload["user_isolation_suite"]["user_a_reentry"].get("tool_query", ""),
                "user_b_tool_query": payload["user_isolation_suite"]["user_b_reentry"].get("tool_query", ""),
            },
        }
    )

    assertions.append(
        {
            "name": "persistent_memory_written",
            "passed": all(int(x.get("chat_turns", 0) or 0) > 0 for x in payload.get("postgres_counts", []) if "error" not in x),
            "detail": payload.get("postgres_counts", []),
        }
    )

    assertions.append(
        {
            "name": "fault_injection_retry_chain",
            "passed": (
                bool((fault.get("fault_injection_meta", {}) or {}).get("injected_failures", 0) >= 1)
                and bool(fault.get("validator_reject", False))
                and int(fault.get("retry_count_total", 0) or 0) >= 1
                and (not bool(fault.get("final_answer_failed", True)))
            ),
            "detail": {
                "query": fault.get("query", ""),
                "selected_tool": fault.get("selected_tool", ""),
                "tool_query": fault.get("tool_query", ""),
                "retry_count_total": fault.get("retry_count_total", 0),
                "validator_reject": fault.get("validator_reject", False),
                "retry_trace": fault.get("retry_trace", []),
                "fault_injection_meta": fault.get("fault_injection_meta", {}),
                "answer": str(fault.get("final_answer", ""))[:500],
            },
        }
    )

    # 报警型断言：审查重试机制至少在一次场景中工作（非强制失败）
    saw_retry = any(
        int(x.get("retry_count_total", 0) or 0) > 0
        for x in (single + integration_turns + [reentry, fault])
    )
    assertions.append(
        {
            "name": "retry_mechanism_observed",
            "passed": saw_retry,
            "detail": {"saw_retry": saw_retry},
            "severity": "warning",
        }
    )

    return assertions


def _brief_line(text: str, limit: int = 180) -> str:
    raw = str(text or "").strip().replace("\n", " ")
    return raw if len(raw) <= limit else (raw[:limit] + "...")


def _write_reports(payload: Dict[str, Any], assertions: List[Dict[str, Any]]) -> Dict[str, str]:
    docs_dir = PROJECT_ROOT / "docs"
    json_path = docs_dir / "SYSTEM_FULL_METRICS_RESULT.json"
    md_path = docs_dir / "SYSTEM_FULL_METRICS_REPORT.md"
    brief_path = docs_dir / "SYSTEM_FULL_METRICS_REPORT_BRIEF.md"

    payload["assertions"] = assertions
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    total = len(assertions)
    passed = sum(1 for x in assertions if bool(x.get("passed", False)))
    failed = total - passed

    lines: List[str] = []
    lines.append("# 系统全量功能与指标测试报告")
    lines.append("")
    lines.append("## 1. 测试元信息")
    lines.append(f"- 生成时间(UTC): `{payload['meta']['generated_at_utc']}`")
    lines.append(f"- 主用户: `{payload['meta']['user_id']}`")
    lines.append(f"- 主会话: `{payload['integration_suite']['main_session_id']}`")
    lines.append(f"- 断言: `PASS={passed}, FAIL={failed}, TOTAL={total}`")
    lines.append("")

    lines.append("## 2. 核心指标")
    for section in ("core", "kb_rag", "ltm_rag", "engineering"):
        lines.append(f"### {section}")
        for k, v in payload["metrics"].get(section, {}).items():
            lines.append(f"- `{k}`: `{v}`")
        lines.append("")

    lines.append("## 3. 单轮样本结果")
    for item in payload["single_turn_suite"]["results"]:
        lines.append(
            f"- `{item.get('case_id','')}` | tool=`{item.get('selected_tool','')}` | skill=`{item.get('selected_skill','')}` | intent_ok={item.get('intent_ok')} | tool_ok={item.get('tool_ok')} | entity_ok={item.get('entity_ok')} | answer_fail={item.get('final_answer_failed')}"
        )
    lines.append("")

    lines.append("## 4. 多轮样本结果")
    for item in payload["multi_turn_suite"]["results"]:
        lines.append(
            f"- `{item.get('case_id','')}` | target_tool=`{item.get('target',{}).get('selected_tool','')}` | tool_query_ok={item.get('tool_query_ok')} | answer_fail={item.get('final_answer_failed')}"
        )
    lines.append("")

    lines.append("## 5. 集成功能覆盖（全工具）")
    for idx, turn in enumerate(payload["integration_suite"]["main_turns"], 1):
        lines.append(
            f"- R{idx} | query=`{turn.get('query','')}` | tool=`{turn.get('selected_tool','')}` | gate={turn.get('memory_gate_score',0)} | recall_hits={turn.get('memory_recall_hits',0)} | retries={turn.get('retry_count_total',0)}"
        )
        lines.append(f"  - output: `{_brief_line(turn.get('final_answer',''), 220)}`")
    lines.append("")

    reentry = payload["integration_suite"]["reentry_turn"]
    lines.append("## 6. 跨会话重入")
    lines.append(f"- query: `{reentry.get('query','')}`")
    lines.append(f"- tool: `{reentry.get('selected_tool','')}` / tool_query=`{reentry.get('tool_query','')}`")
    lines.append(
        f"- gate=`{reentry.get('memory_gate_score',0)}` recall_hits=`{reentry.get('memory_recall_hits',0)}` retries=`{reentry.get('retry_count_total',0)}`"
    )
    lines.append(f"- output: `{_brief_line(reentry.get('final_answer',''), 240)}`")
    lines.append("")

    fault = payload.get("fault_injection_suite", {})
    lines.append("## 7. 故障注入与重试链路")
    lines.append(f"- query: `{fault.get('query','')}`")
    lines.append(f"- tool: `{fault.get('selected_tool','')}` / tool_query=`{fault.get('tool_query','')}`")
    lines.append(
        f"- retry_count_total=`{fault.get('retry_count_total',0)}` validator_reject=`{fault.get('validator_reject',False)}` reviewer_reject=`{fault.get('reviewer_reject',False)}`"
    )
    lines.append(f"- injected: `{(fault.get('fault_injection_meta',{}) or {}).get('injected_failures',0)}`")
    lines.append(f"- output: `{_brief_line(fault.get('final_answer',''), 220)}`")
    lines.append("")

    iso = payload["user_isolation_suite"]
    lines.append("## 8. user_id 隔离")
    lines.append(f"- user-A reentry query: `{iso['user_a_reentry'].get('query','')}`")
    lines.append(f"- user-A output: `{_brief_line(iso['user_a_reentry'].get('final_answer',''), 180)}`")
    lines.append(f"- user-B reentry query: `{iso['user_b_reentry'].get('query','')}`")
    lines.append(f"- user-B output: `{_brief_line(iso['user_b_reentry'].get('final_answer',''), 180)}`")
    lines.append(f"- isolation_ok: `{iso.get('isolation_ok', False)}`")
    lines.append("")

    lines.append("## 9. 持久化计数")
    for row in payload.get("postgres_counts", []):
        lines.append(f"- `{row}`")
    lines.append("")

    lines.append("## 10. 断言")
    for item in assertions:
        sev = str(item.get("severity", "normal"))
        lines.append(f"- `{item['name']}`: {'PASS' if item['passed'] else 'FAIL'} (severity={sev})")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    brief: List[str] = []
    brief.append("# 系统全量测试简报")
    brief.append("")
    brief.append(f"- 时间(UTC): `{payload['meta']['generated_at_utc']}`")
    brief.append(f"- 断言: `PASS={passed}, FAIL={failed}, TOTAL={total}`")
    brief.append(f"- Core overall_score: `{payload['metrics']['core']['overall_score']}`")
    brief.append(f"- KB RAG entity_recall_proxy: `{payload['metrics']['kb_rag']['kb_rag_entity_recall_proxy']}`")
    brief.append(f"- LTM RAG entity_resolution_accuracy: `{payload['metrics']['ltm_rag']['ltm_entity_resolution_accuracy']}`")
    brief.append(f"- Engineering retry_invocation_rate: `{payload['metrics']['engineering']['retry_invocation_rate']}`")
    brief.append(f"- Engineering quality_gate_pass_rate: `{payload['metrics']['engineering']['quality_gate_pass_rate']}`")
    brief.append(f"- Fault injection retries: `{payload.get('fault_injection_suite',{}).get('retry_count_total',0)}`")
    brief.append("")
    brief.append("## 关键输入输出")
    for idx, turn in enumerate(payload["integration_suite"]["main_turns"], 1):
        brief.append(f"- R{idx} 输入: `{turn.get('query','')}`")
        brief.append(f"  输出: `{_brief_line(turn.get('final_answer',''), 160)}`")
    brief.append(f"- 重入输入: `{reentry.get('query','')}`")
    brief.append(f"  重入输出: `{_brief_line(reentry.get('final_answer',''), 180)}`")

    brief_path.write_text("\n".join(brief), encoding="utf-8")
    return {"json": str(json_path), "md": str(md_path), "md_brief": str(brief_path)}


def main() -> None:
    cfg = DEFAULT_CONFIG
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    threshold = int(getattr(cfg, "memory_persistent_trigger_threshold", 2) or 2)

    benchmark_path = PROJECT_ROOT / "data/benchmarks/system_eval_cases.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))

    rag_service = RAGService(cfg)
    registry = ToolRegistry(rag_service=rag_service, config=cfg)
    store = PersistentMemoryStore(cfg)
    graph = build_multi_agent_graph(registry, persistent_store=store)

    try:
        reset_info = _reset_persistent_memory(store)

        # 1) 单轮基准
        single_results: List[Dict[str, Any]] = []
        single_manager = SessionMemoryManager()
        for i, case in enumerate(benchmark.get("single_turn_cases", []), 1):
            session_id = f"full-single-{ts}-{i:03d}"
            turn = _run_turn(
                graph=graph,
                memory_manager=single_manager,
                user_id="full-user-single",
                session_id=session_id,
                query=str(case.get("query", "")),
                include_pending_in_prompt=cfg.memory_include_pending_in_prompt,
                persistent_gate_threshold=threshold,
            )
            turn.update(
                {
                    "case_id": case.get("id", ""),
                    "expected_tool": case.get("expected_tool", ""),
                    "expected_skill": case.get("expected_skill", ""),
                    "intent_ok": _intent_match(
                        actual_intent=str(turn.get("intent", "")),
                        selected_tool=str(turn.get("selected_tool", "")),
                        expected_intents=[str(x) for x in case.get("expected_intents", [])],
                    ),
                    "tool_ok": str(turn.get("selected_tool", "")) == str(case.get("expected_tool", "")),
                    "skill_ok": (
                        True
                        if not str(case.get("expected_skill", ""))
                        else (str(turn.get("selected_skill", "")) == str(case.get("expected_skill", "")))
                    ),
                    "entity_ok": _entity_hit(
                        expected_entities=[str(x) for x in case.get("expected_entities", [])],
                        entities=[str(x) for x in (turn.get("understanding_entities", []) or [])],
                        tool_query=str(turn.get("tool_query", "")),
                        answer=str(turn.get("final_answer", "")),
                    ),
                    "answer_keyword_coverage": round(
                        _keyword_coverage(str(turn.get("final_answer", "")), [str(x) for x in case.get("answer_keywords", [])]),
                        4,
                    ),
                }
            )
            single_results.append(turn)

        # 2) 多轮基准
        multi_results: List[Dict[str, Any]] = []
        ltm_cases: List[Dict[str, Any]] = []
        for i, case in enumerate(benchmark.get("multi_turn_cases", []), 1):
            manager = SessionMemoryManager()
            session_id = f"full-multi-{ts}-{i:03d}"
            user_id = "full-user-multi"
            turns: List[Dict[str, Any]] = []
            steps = case.get("steps", [])
            for step in steps:
                turns.append(
                    _run_turn(
                        graph=graph,
                        memory_manager=manager,
                        user_id=user_id,
                        session_id=session_id,
                        query=str(step.get("query", "")),
                        include_pending_in_prompt=cfg.memory_include_pending_in_prompt,
                        persistent_gate_threshold=threshold,
                    )
                )

            target = turns[-1] if turns else {}
            expect = steps[-1] if steps else {}
            expected_q_contains = [str(x) for x in expect.get("expected_tool_query_contains", []) if str(x)]
            tool_query = str(target.get("tool_query", ""))
            tool_query_ok = all(x in tool_query for x in expected_q_contains)
            answer_cov = _keyword_coverage(str(target.get("final_answer", "")), [str(x) for x in expect.get("answer_keywords", [])])
            record = {
                "case_id": case.get("id", ""),
                "session_id": session_id,
                "turns": turns,
                "target": target,
                "tool_ok": (
                    True
                    if not expect.get("expected_tool")
                    else str(target.get("selected_tool", "")) == str(expect.get("expected_tool", ""))
                ),
                "skill_ok": (
                    True
                    if not expect.get("expected_skill")
                    else str(target.get("selected_skill", "")) == str(expect.get("expected_skill", ""))
                ),
                "tool_query_ok": tool_query_ok,
                "answer_keyword_coverage": round(answer_cov, 4),
                "final_answer_failed": bool(target.get("final_answer_failed", False)),
            }
            multi_results.append(record)

            query_text = str(expect.get("query", "") or "")
            if any(m in query_text for m in PRONOUN_MARKERS):
                ltm_cases.append(
                    {
                        "source": f"multi:{case.get('id','')}",
                        **target,
                        "entity_resolved": tool_query_ok,
                    }
                )

        # 3) 全工具集成会话（之前的完整测试）
        integration_queries = [
            "介绍一下非洲之心",
            "它现在什么价格",
            "它的历史价格",
            "它现在建议买吗",
            "再介绍一下海洋之泪并告诉我现在价格",
            "对比一下这两个物品",
            "分析碳纤维散射箭矢利润稳定性",
            "特勤处制造什么子弹利润最高",
            "特勤处四大分组利润top3",
        ]
        main_user_id = "full-user-main"
        main_session_id = f"full-main-{ts}"
        integration_manager = SessionMemoryManager()
        integration_turns: List[Dict[str, Any]] = []
        for q in integration_queries:
            integration_turns.append(
                _run_turn(
                    graph=graph,
                    memory_manager=integration_manager,
                    user_id=main_user_id,
                    session_id=main_session_id,
                    query=q,
                    include_pending_in_prompt=cfg.memory_include_pending_in_prompt,
                    persistent_gate_threshold=threshold,
                )
            )

        finalize_pre = integration_manager.stats(user_id=main_user_id, session_id=main_session_id)
        _finalize_session_memory(
            user_id=main_user_id,
            session_id=main_session_id,
            config=cfg,
            memory_manager=integration_manager,
            persistent_store=store,
        )
        finalize_post = integration_manager.stats(user_id=main_user_id, session_id=main_session_id)

        # 跨会话重入
        reentry_turn = _run_turn(
            graph=graph,
            memory_manager=SessionMemoryManager(),
            user_id=main_user_id,
            session_id=main_session_id,
            query="刚才两个物品里，哪个更适合买入，简短回答",
            include_pending_in_prompt=cfg.memory_include_pending_in_prompt,
            persistent_gate_threshold=threshold,
        )
        ltm_cases.append(
            {
                "source": "integration:reentry_compare",
                **reentry_turn,
                "entity_resolved": (
                    "非洲之心" in str(reentry_turn.get("tool_query", ""))
                    and "海洋之泪" in str(reentry_turn.get("tool_query", ""))
                ),
            }
        )

        # 4) user_id 隔离
        iso_session_id = f"full-iso-{ts}"
        iso_case = benchmark.get("user_isolation_case", {})
        user_a = iso_case.get("user_a", {})
        user_b = iso_case.get("user_b", {})

        iso_manager = SessionMemoryManager()
        user_a_warmup = _run_turn(
            graph=graph,
            memory_manager=iso_manager,
            user_id="user-A",
            session_id=iso_session_id,
            query=str(user_a.get("warmup", "介绍一下非洲之心")),
            include_pending_in_prompt=cfg.memory_include_pending_in_prompt,
            persistent_gate_threshold=threshold,
        )
        user_b_warmup = _run_turn(
            graph=graph,
            memory_manager=iso_manager,
            user_id="user-B",
            session_id=iso_session_id,
            query=str(user_b.get("warmup", "介绍一下QBZ95-1突击步枪")),
            include_pending_in_prompt=cfg.memory_include_pending_in_prompt,
            persistent_gate_threshold=threshold,
        )

        user_a_reentry = _run_turn(
            graph=graph,
            memory_manager=SessionMemoryManager(),
            user_id="user-A",
            session_id=iso_session_id,
            query=str(user_a.get("reentry", "它现在价格是多少")),
            include_pending_in_prompt=cfg.memory_include_pending_in_prompt,
            persistent_gate_threshold=threshold,
        )
        user_b_reentry = _run_turn(
            graph=graph,
            memory_manager=SessionMemoryManager(),
            user_id="user-B",
            session_id=iso_session_id,
            query=str(user_b.get("reentry", "它现在价格是多少")),
            include_pending_in_prompt=cfg.memory_include_pending_in_prompt,
            persistent_gate_threshold=threshold,
        )

        a_ok = _keyword_coverage(str(user_a_reentry.get("final_answer", "")), [str(x) for x in user_a.get("answer_keywords", [])]) > 0
        b_ok = _keyword_coverage(str(user_b_reentry.get("final_answer", "")), [str(x) for x in user_b.get("answer_keywords", [])]) > 0
        isolation_ok = bool(a_ok and b_ok)

        ltm_cases.append(
            {
                "source": "isolation:user_a_reentry",
                **user_a_reentry,
                "entity_resolved": "非洲之心" in str(user_a_reentry.get("tool_query", ""))
                or "非洲之心" in str(user_a_reentry.get("final_answer", "")),
            }
        )
        ltm_cases.append(
            {
                "source": "isolation:user_b_reentry",
                **user_b_reentry,
                "entity_resolved": "QBZ95" in str(user_b_reentry.get("tool_query", ""))
                or "QBZ95" in str(user_b_reentry.get("final_answer", "")),
            }
        )

        # 5) 故障注入：覆盖重试链路
        fault_injection_turn = _run_fault_injection_retry_case(
            graph=graph,
            registry=registry,
            include_pending_in_prompt=cfg.memory_include_pending_in_prompt,
            persistent_gate_threshold=threshold,
        )

        # 指标聚合
        core = _aggregate_core_metrics(single=single_results, multi=multi_results, isolation_ok=isolation_ok)
        kb_rag = _aggregate_kb_rag_metrics(single=single_results)
        ltm_rag = _aggregate_ltm_rag_metrics(ltm_cases=ltm_cases)

        all_turns = list(single_results)
        for item in multi_results:
            all_turns.extend(item.get("turns", []))
        all_turns.extend(integration_turns)
        all_turns.append(reentry_turn)
        all_turns.append(fault_injection_turn)
        all_turns.extend([user_a_warmup, user_b_warmup, user_a_reentry, user_b_reentry])

        engineering = _aggregate_engineering_metrics(all_turns=all_turns)

        payload: Dict[str, Any] = {
            "meta": {
                "generated_at_utc": _now_utc(),
                "user_id": main_user_id,
                "benchmark_file": str(benchmark_path),
                "memory_reset": reset_info,
                "tools": registry.list_tools(),
            },
            "single_turn_suite": {
                "case_count": len(single_results),
                "results": single_results,
            },
            "multi_turn_suite": {
                "case_count": len(multi_results),
                "results": multi_results,
            },
            "integration_suite": {
                "main_session_id": main_session_id,
                "main_turns": integration_turns,
                "finalize": {"pre": finalize_pre, "post": finalize_post},
                "reentry_turn": reentry_turn,
            },
            "fault_injection_suite": fault_injection_turn,
            "user_isolation_suite": {
                "session_id": iso_session_id,
                "user_a_warmup": user_a_warmup,
                "user_b_warmup": user_b_warmup,
                "user_a_reentry": user_a_reentry,
                "user_b_reentry": user_b_reentry,
                "isolation_ok": isolation_ok,
            },
            "ltm_rag_cases": ltm_cases,
            "postgres_counts": [
                _db_counts(store, user_id=main_user_id, session_id=main_session_id),
                _db_counts(store, user_id="user-A", session_id=iso_session_id),
                _db_counts(store, user_id="user-B", session_id=iso_session_id),
            ],
            "metrics": {
                "core": core,
                "kb_rag": kb_rag,
                "ltm_rag": ltm_rag,
                "engineering": engineering,
            },
        }

        assertions = _build_assertions(payload)
        report_paths = _write_reports(payload, assertions)
        all_pass = all(bool(x.get("passed", False)) for x in assertions if x.get("severity", "normal") != "warning")

        print(
            json.dumps(
                {
                    "status": "ok" if all_pass else "failed",
                    "all_pass": all_pass,
                    "assertions": {"total": len(assertions), "passed": sum(1 for x in assertions if x.get('passed'))},
                    "reports": report_paths,
                },
                ensure_ascii=False,
            )
        )
    finally:
        registry.close()


if __name__ == "__main__":
    main()
