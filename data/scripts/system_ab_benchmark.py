#!/usr/bin/env python3
"""全项目 A/B 基准测试：skills on/off，覆盖意图、工具、实体、回答、记忆与 user 隔离。"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    # 兼容 LLM 返回中文意图描述
    intent_text = raw + " " + canonical
    if "history" in " ".join(expected) and ("历史" in intent_text or "history" in intent_text.lower()):
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
) -> Dict[str, Any]:
    attempts = 0
    start = time.time()
    result: Dict[str, Any] = {}

    while True:
        attempts += 1
        patch = memory_manager.build_state_patch(
            user_id=user_id,
            session_id=session_id,
            include_pending_in_prompt=DEFAULT_CONFIG.memory_include_pending_in_prompt,
        )
        result = graph.invoke(_build_initial_state(query, session_id=session_id, user_id=user_id, memory_patch=patch))

        answer = str(result.get("final_answer", "") or "")
        tool_outputs = [str(x.get("output", "")) for x in (result.get("tool_results", []) or [])]
        failed = _is_failure(answer)
        transient = _is_transient_failure(answer) or any(_is_transient_failure(x) for x in tool_outputs)
        if (not failed) or attempts >= 3 or (not transient):
            break
        time.sleep(1.0 * attempts)

    elapsed = round(time.time() - start, 2)
    memory_manager.save_from_state(user_id=user_id, session_id=session_id, state=result)

    return {
        "query": query,
        "attempts": attempts,
        "elapsed_sec": elapsed,
        "intent": result.get("intent", ""),
        "selected_tool": result.get("selected_tool", ""),
        "selected_skill": result.get("selected_skill", ""),
        "tool_query": result.get("tool_query", ""),
        "understanding_entities": result.get("understanding_entities", []) or [],
        "memory_gate_score": int(result.get("memory_persistent_gate_score", 0) or 0),
        "memory_recall_hits": len(result.get("memory_persistent_hits", []) or []),
        "final_answer": result.get("final_answer", ""),
        "final_answer_failed": _is_failure(result.get("final_answer", "")),
    }


def _run_single_turn_suite(graph, manager: SessionMemoryManager, variant: str, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, case in enumerate(cases, 1):
        sid = f"ab-{variant}-st-{i:03d}"
        item = _run_turn(
            graph=graph,
            memory_manager=manager,
            user_id=f"ab-user-{variant}",
            session_id=sid,
            query=str(case.get("query", "")),
        )

        intent_ok = _intent_match(
            actual_intent=item.get("intent", ""),
            selected_tool=item.get("selected_tool", ""),
            expected_intents=case.get("expected_intents", []),
        )
        tool_ok = item["selected_tool"] == case.get("expected_tool", "")
        expected_skill = str(case.get("expected_skill", "") or "")
        if expected_skill:
            skill_ok = (item["selected_skill"] == expected_skill) if variant == "skills_on" else (item["selected_skill"] in {"", None})
        else:
            skill_ok = True
        entity_ok = _entity_hit(
            expected_entities=[str(x) for x in case.get("expected_entities", [])],
            entities=[str(x) for x in item.get("understanding_entities", [])],
            tool_query=str(item.get("tool_query", "")),
            answer=str(item.get("final_answer", "")),
        )
        ans_cov = _keyword_coverage(item.get("final_answer", ""), [str(x) for x in case.get("answer_keywords", [])])

        item.update(
            {
                "case_id": case.get("id", ""),
                "intent_ok": intent_ok,
                "tool_ok": tool_ok,
                "skill_ok": skill_ok,
                "entity_ok": entity_ok,
                "answer_keyword_coverage": round(ans_cov, 4),
            }
        )
        out.append(item)
    return out


def _run_multi_turn_suite(graph, variant: str, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for i, case in enumerate(cases, 1):
        manager = SessionMemoryManager()
        sid = f"ab-{variant}-mt-{i:03d}"
        uid = f"ab-user-{variant}"
        steps = case.get("steps", [])
        turns: List[Dict[str, Any]] = []
        for step in steps:
            turns.append(
                _run_turn(
                    graph=graph,
                    memory_manager=manager,
                    user_id=uid,
                    session_id=sid,
                    query=str(step.get("query", "")),
                )
            )
        target = turns[-1] if turns else {}
        expect = steps[-1] if steps else {}
        query_contains = [str(x) for x in expect.get("expected_tool_query_contains", [])]
        tq = str(target.get("tool_query", ""))
        tool_query_ok = all(x in tq for x in query_contains)

        tool_ok = (target.get("selected_tool", "") == expect.get("expected_tool", "")) if expect.get("expected_tool") else True
        expected_skill = str(expect.get("expected_skill", "") or "")
        if expected_skill:
            skill_ok = (target.get("selected_skill", "") == expected_skill) if variant == "skills_on" else (target.get("selected_skill", "") in {"", None})
        else:
            skill_ok = True
        answer_cov = _keyword_coverage(target.get("final_answer", ""), [str(x) for x in expect.get("answer_keywords", [])])

        records.append(
            {
                "case_id": case.get("id", ""),
                "session_id": sid,
                "turns": turns,
                "target": target,
                "tool_ok": tool_ok,
                "skill_ok": skill_ok,
                "tool_query_ok": tool_query_ok,
                "answer_keyword_coverage": round(answer_cov, 4),
                "final_answer_failed": bool(target.get("final_answer_failed", False)),
            }
        )
    return records


def _run_user_isolation(graph, variant: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    manager = SessionMemoryManager()
    sid = f"{payload.get('session_id_prefix','ab-iso')}-{variant}"

    a_warmup = _run_turn(graph=graph, memory_manager=manager, user_id="user-A", session_id=sid, query=str(payload['user_a']['warmup']))
    b_warmup = _run_turn(graph=graph, memory_manager=manager, user_id="user-B", session_id=sid, query=str(payload['user_b']['warmup']))

    a_reentry = _run_turn(
        graph=graph,
        memory_manager=SessionMemoryManager(),
        user_id="user-A",
        session_id=sid,
        query=str(payload['user_a']['reentry']),
    )
    b_reentry = _run_turn(
        graph=graph,
        memory_manager=SessionMemoryManager(),
        user_id="user-B",
        session_id=sid,
        query=str(payload['user_b']['reentry']),
    )

    a_ok = _keyword_coverage(a_reentry.get("final_answer", ""), [str(x) for x in payload["user_a"].get("answer_keywords", [])]) > 0
    b_ok = _keyword_coverage(b_reentry.get("final_answer", ""), [str(x) for x in payload["user_b"].get("answer_keywords", [])]) > 0

    return {
        "session_id": sid,
        "user_a_warmup": a_warmup,
        "user_b_warmup": b_warmup,
        "user_a_reentry": a_reentry,
        "user_b_reentry": b_reentry,
        "isolation_ok": bool(a_ok and b_ok),
    }


def _aggregate_metrics(single: List[Dict[str, Any]], multi: List[Dict[str, Any]], isolation: Dict[str, Any]) -> Dict[str, Any]:
    n = max(1, len(single))
    latencies = [float(x.get("elapsed_sec", 0) or 0) for x in single]
    retries = [int(x.get("attempts", 1) or 1) for x in single]

    intent_acc = sum(1 for x in single if x.get("intent_ok")) / n
    tool_acc = sum(1 for x in single if x.get("tool_ok")) / n
    skill_acc = sum(1 for x in single if x.get("skill_ok")) / n
    entity_acc = sum(1 for x in single if x.get("entity_ok")) / n
    keyword_cov = sum(float(x.get("answer_keyword_coverage", 0) or 0) for x in single) / n
    success_rate = sum(1 for x in single if not x.get("final_answer_failed")) / n

    mt_n = max(1, len(multi))
    mt_success = sum(1 for x in multi if (x.get("tool_ok") and x.get("skill_ok") and x.get("tool_query_ok") and not x.get("final_answer_failed"))) / mt_n
    mt_keyword_cov = sum(float(x.get("answer_keyword_coverage", 0) or 0) for x in multi) / mt_n

    isolation_ok = 1.0 if isolation.get("isolation_ok") else 0.0

    overall = (
        0.18 * intent_acc
        + 0.20 * tool_acc
        + 0.12 * skill_acc
        + 0.12 * entity_acc
        + 0.12 * keyword_cov
        + 0.10 * success_rate
        + 0.10 * mt_success
        + 0.04 * mt_keyword_cov
        + 0.02 * isolation_ok
    )

    p95 = sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)] if latencies else 0.0
    return {
        "intent_accuracy": round(intent_acc, 4),
        "tool_accuracy": round(tool_acc, 4),
        "skill_accuracy": round(skill_acc, 4),
        "entity_resolution_accuracy": round(entity_acc, 4),
        "answer_keyword_coverage": round(keyword_cov, 4),
        "single_turn_success_rate": round(success_rate, 4),
        "multi_turn_success_rate": round(mt_success, 4),
        "multi_turn_keyword_coverage": round(mt_keyword_cov, 4),
        "user_isolation_success": bool(isolation.get("isolation_ok")),
        "avg_latency_sec": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "p95_latency_sec": round(p95, 2),
        "avg_attempts": round(statistics.mean(retries), 2) if retries else 1.0,
        "overall_score": round(overall, 4),
    }


def _run_variant(variant_name: str, skills_enabled: bool, benchmark: Dict[str, Any]) -> Dict[str, Any]:
    os.environ["AGENT_SKILLS_ENABLED"] = "1" if skills_enabled else "0"

    rag = RAGService(DEFAULT_CONFIG)
    registry = ToolRegistry(rag_service=rag, config=DEFAULT_CONFIG)
    store = PersistentMemoryStore(DEFAULT_CONFIG)
    graph = build_multi_agent_graph(registry, persistent_store=store)
    reset_info = _reset_persistent_memory(store)

    start = time.time()
    try:
        manager = SessionMemoryManager()
        single = _run_single_turn_suite(
            graph=graph,
            manager=manager,
            variant=variant_name,
            cases=benchmark.get("single_turn_cases", []),
        )

        multi = _run_multi_turn_suite(
            graph=graph,
            variant=variant_name,
            cases=benchmark.get("multi_turn_cases", []),
        )

        isolation = _run_user_isolation(
            graph=graph,
            variant=variant_name,
            payload=benchmark.get("user_isolation_case", {}),
        )

        metrics = _aggregate_metrics(single=single, multi=multi, isolation=isolation)

        # 写库成功性（抽样 1 个主 session）
        db_meta = {"checked": True}
        try:
            conn = store._connect()  # noqa: SLF001
            if conn is not None:
                with conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT COUNT(*) FROM chat_turns;")
                        db_meta["chat_turns"] = int((cur.fetchone() or [0])[0] or 0)
                        cur.execute("SELECT COUNT(*) FROM memory_summaries;")
                        db_meta["memory_summaries"] = int((cur.fetchone() or [0])[0] or 0)
                        cur.execute("SELECT COUNT(*) FROM memory_facts;")
                        db_meta["memory_facts"] = int((cur.fetchone() or [0])[0] or 0)
            else:
                db_meta["checked"] = False
                db_meta["reason"] = "db_connect_failed"
        except Exception as exc:
            db_meta["checked"] = False
            db_meta["reason"] = str(exc)

        return {
            "variant": variant_name,
            "skills_enabled": skills_enabled,
            "duration_sec": round(time.time() - start, 2),
            "memory_reset": reset_info,
            "metrics": metrics,
            "single_turn_results": single,
            "multi_turn_results": multi,
            "user_isolation": isolation,
            "db_meta": db_meta,
        }
    finally:
        registry.close()


def _write_reports(result: Dict[str, Any]) -> Dict[str, str]:
    docs_dir = PROJECT_ROOT / "docs"
    json_path = docs_dir / "SYSTEM_AB_BENCHMARK_RESULT.json"
    md_path = docs_dir / "SYSTEM_AB_BENCHMARK_REPORT.md"
    brief_path = docs_dir / "SYSTEM_AB_BENCHMARK_REPORT_BRIEF.md"

    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    a = result["variants"]["A_skills_on"]["metrics"]
    b = result["variants"]["B_skills_off"]["metrics"]

    def diff(k: str) -> float:
        return round(float(a.get(k, 0) or 0) - float(b.get(k, 0) or 0), 4)

    lines = [
        "# 全项目 A/B 基准测试报告",
        "",
        "## 1. 测试范围",
        "- A 组：skills 开启（AGENT_SKILLS_ENABLED=1）",
        "- B 组：skills 关闭（AGENT_SKILLS_ENABLED=0）",
        "- 覆盖：意图识别、工具选择、实体解析、回答关键词覆盖、多轮代词/对比、user 隔离、持久化写入、时延。",
        "",
        "## 2. 核心指标对比",
        "| 指标 | A(skills on) | B(skills off) | Δ(A-B) |",
        "|---|---:|---:|---:|",
    ]

    keys = [
        "intent_accuracy",
        "tool_accuracy",
        "skill_accuracy",
        "entity_resolution_accuracy",
        "answer_keyword_coverage",
        "single_turn_success_rate",
        "multi_turn_success_rate",
        "multi_turn_keyword_coverage",
        "overall_score",
        "avg_latency_sec",
        "p95_latency_sec",
    ]
    for k in keys:
        lines.append(f"| {k} | {a.get(k)} | {b.get(k)} | {diff(k)} |")

    lines += [
        "",
        "## 3. 结论",
        f"- user 隔离：A={a.get('user_isolation_success')}, B={b.get('user_isolation_success')}",
        f"- 综合分：A={a.get('overall_score')}，B={b.get('overall_score')}，差值={diff('overall_score')}",
        "",
        "## 4. 结果文件",
        f"- 详细 JSON：`{json_path}`",
        f"- 详细报告：`{md_path}`",
        f"- 简版报告：`{brief_path}`",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    brief = [
        "# A/B 简报",
        "",
        f"- 时间(UTC): `{result['meta']['generated_at_utc']}`",
        f"- 总体结论：A综合分 `{a.get('overall_score')}` vs B `{b.get('overall_score')}`，差值 `{diff('overall_score')}`",
        f"- 意图准确率：A `{a.get('intent_accuracy')}` vs B `{b.get('intent_accuracy')}`",
        f"- 工具准确率：A `{a.get('tool_accuracy')}` vs B `{b.get('tool_accuracy')}`",
        f"- 多轮成功率：A `{a.get('multi_turn_success_rate')}` vs B `{b.get('multi_turn_success_rate')}`",
        f"- 平均时延(秒)：A `{a.get('avg_latency_sec')}` vs B `{b.get('avg_latency_sec')}`",
    ]
    brief_path.write_text("\n".join(brief), encoding="utf-8")

    return {"json": str(json_path), "md": str(md_path), "md_brief": str(brief_path)}


def main() -> None:
    benchmark_path = PROJECT_ROOT / "data/benchmarks/system_eval_cases.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))

    result = {
        "meta": {
            "generated_at_utc": _now_utc(),
            "benchmark_file": str(benchmark_path),
            "note": "A/B 对比仅差异于 skills 开关，其余配置保持一致。",
        },
        "variants": {},
    }

    result["variants"]["A_skills_on"] = _run_variant("skills_on", True, benchmark)
    result["variants"]["B_skills_off"] = _run_variant("skills_off", False, benchmark)

    report_paths = _write_reports(result)

    a_score = float(result["variants"]["A_skills_on"]["metrics"].get("overall_score", 0) or 0)
    b_score = float(result["variants"]["B_skills_off"]["metrics"].get("overall_score", 0) or 0)

    print(
        json.dumps(
            {
                "status": "ok",
                "a_overall": a_score,
                "b_overall": b_score,
                "delta": round(a_score - b_score, 4),
                "reports": report_paths,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
