#!/usr/bin/env python3
"""端到端集成测试：工具调用 + 长短期记忆 + user_id 隔离。"""

from __future__ import annotations

import json
import re
import shutil
import sys
import time
from dataclasses import asdict
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


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_failure_text(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return True
    return any(x in raw for x in FAIL_MARKERS)


def _find_compression(result: Dict[str, Any]) -> Dict[str, Any]:
    for msg in reversed(result.get("agent_messages", []) or []):
        if msg.get("message_type") != "memory_update":
            continue
        payload = msg.get("payload", {})
        if isinstance(payload, dict):
            compression = payload.get("compression")
            if isinstance(compression, dict):
                return compression
    return {}


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

    local_dirs = [
        Path("/data/delta_agent/data/memory/readable"),
        Path("/data/delta_agent/data/memory/exports"),
    ]
    for target in local_dirs:
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
) -> Dict[str, Any]:
    memory_patch = memory_manager.build_state_patch(
        user_id=user_id,
        session_id=session_id,
        include_pending_in_prompt=include_pending_in_prompt,
    )
    start = time.time()
    result = graph.invoke(_build_initial_state(query, session_id=session_id, user_id=user_id, memory_patch=memory_patch))
    elapsed = round(time.time() - start, 2)
    memory_manager.save_from_state(user_id=user_id, session_id=session_id, state=result)

    gate_score = int(result.get("memory_persistent_gate_score", 0) or 0)
    recall_hits = len(result.get("memory_persistent_hits", []) or [])
    tool_results = result.get("tool_results", []) or []

    return {
        "timestamp_utc": _now_utc(),
        "query": query,
        "elapsed_sec": elapsed,
        "intent": result.get("intent", ""),
        "flow_type": result.get("flow_type", ""),
        "selected_tool": result.get("selected_tool", ""),
        "tool_query": result.get("tool_query", ""),
        "tool_results": tool_results,
        "memory_gate_score": gate_score,
        "memory_gate_triggered": gate_score >= persistent_gate_threshold,
        "memory_recall_hits": recall_hits,
        "memory_recall_effective": recall_hits > 0,
        "memory_compression": _find_compression(result),
        "memory_stats_after": memory_manager.stats(user_id=user_id, session_id=session_id),
        "debug_steps": result.get("debug_steps", []) or [],
        "final_answer": result.get("final_answer", ""),
        "final_answer_failed": _is_failure_text(result.get("final_answer", "")),
        "tool_failures": [
            {
                "tool_name": str(item.get("tool_name", "")),
                "output": str(item.get("output", ""))[:500],
            }
            for item in tool_results
            if _is_failure_text(str(item.get("output", "")))
        ],
    }


def _db_counts_for_session(store: PersistentMemoryStore, user_id: str, session_id: str) -> Dict[str, Any]:
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


def _build_assertions(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []

    turns = payload["main_session"]["turns"]
    selected = [str(t.get("selected_tool", "")).strip() for t in turns]
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

    checks.append(
        {
            "name": "all_tools_covered_in_main_session",
            "passed": expected_tools.issubset(set(selected)),
            "detail": {"selected": selected, "expected": sorted(expected_tools)},
        }
    )

    checks.append(
        {
            "name": "main_session_no_failure_answer",
            "passed": all(not bool(t.get("final_answer_failed", False)) for t in turns),
            "detail": [
                {
                    "query": t.get("query", ""),
                    "tool": t.get("selected_tool", ""),
                    "failed": bool(t.get("final_answer_failed", False)),
                    "tool_failures": t.get("tool_failures", []),
                }
                for t in turns
            ],
        }
    )

    finalize = payload["main_session"]["finalize"]
    pre_merge = int(finalize["pre"].get("merge_count", 0) or 0)
    post_merge = int(finalize["post"].get("merge_count", 0) or 0)
    checks.append(
        {
            "name": "compression_or_finalize_flush_happened",
            "passed": any(bool(t.get("memory_compression", {}).get("triggered")) for t in turns) or post_merge > pre_merge,
            "detail": {"pre_merge": pre_merge, "post_merge": post_merge},
        }
    )

    reentry = payload["reentry_session"]["turn"]
    reentry_text = str(reentry.get("final_answer", ""))
    compare_count = 0
    m = re.search(r"多物品价格对比（共(\d+)个）", reentry_text)
    if m:
        try:
            compare_count = int(m.group(1))
        except Exception:
            compare_count = 0
    checks.append(
        {
            "name": "reentry_compare_resolved_entities",
            "passed": (
                "请至少提供两个物品名称" not in reentry_text
                and "对比失败" not in reentry_text
                and "未成功对比" not in reentry_text
                and compare_count >= 2
            ),
            "detail": {
                "query": reentry.get("query", ""),
                "selected_tool": reentry.get("selected_tool", ""),
                "memory_gate_score": reentry.get("memory_gate_score", 0),
                "memory_recall_hits": reentry.get("memory_recall_hits", 0),
                "compare_count": compare_count,
                "answer": reentry_text[:500],
            },
        }
    )

    checks.append(
        {
            "name": "reentry_answer_not_failed",
            "passed": not bool(reentry.get("final_answer_failed", False)),
            "detail": {
                "query": reentry.get("query", ""),
                "failed": reentry.get("final_answer_failed", False),
                "tool": reentry.get("selected_tool", ""),
            },
        }
    )

    isolation = payload["user_isolation"]
    a_reentry = isolation.get("user_a_reentry", {})
    b_reentry = isolation.get("user_b_reentry", {})
    a_text = str(a_reentry.get("final_answer", ""))
    b_text = str(b_reentry.get("final_answer", ""))
    checks.append(
        {
            "name": "user_isolation_effective",
            "passed": ("非洲之心" in a_text) and ("QBZ95-1" in b_text or "QBZ95" in b_text),
            "detail": {
                "user_a_query": a_reentry.get("query", ""),
                "user_a_tool_query": a_reentry.get("tool_query", ""),
                "user_a_answer": a_text[:400],
                "user_b_query": b_reentry.get("query", ""),
                "user_b_tool_query": b_reentry.get("tool_query", ""),
                "user_b_answer": b_text[:400],
            },
        }
    )

    db_counts = payload.get("postgres_counts", [])
    checks.append(
        {
            "name": "persistent_db_records_written",
            "passed": all((int(item.get("chat_turns", 0)) > 0) for item in db_counts if "error" not in item),
            "detail": db_counts,
        }
    )

    return checks


def _truncate_line(text: str, limit: int) -> str:
    raw = str(text or "").strip().replace("\n", " ")
    return raw if len(raw) <= limit else (raw[:limit] + "...")


def _write_reports(payload: Dict[str, Any], assertions: List[Dict[str, Any]]) -> Dict[str, str]:
    docs_dir = Path("/data/delta_agent/docs")
    json_path = docs_dir / "INTEGRATION_TEST_RESULT.json"
    md_path = docs_dir / "INTEGRATION_TEST_REPORT.md"
    brief_path = docs_dir / "INTEGRATION_TEST_REPORT_BRIEF.md"

    payload["assertions"] = assertions
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    total_assert = len(assertions)
    passed_assert = sum(1 for item in assertions if bool(item.get("passed", False)))
    failed_assert = total_assert - passed_assert

    full_lines: List[str] = []
    full_lines.append("# 集成测试报告")
    full_lines.append("")
    full_lines.append("## 1. 总览")
    full_lines.append(f"- 报告时间(UTC): `{payload['meta']['generated_at_utc']}`")
    full_lines.append(f"- 主用户: `{payload['meta']['user_id']}`")
    full_lines.append(f"- 主会话ID: `{payload['main_session']['session_id']}`")
    full_lines.append(f"- 重入会话ID: `{payload['reentry_session']['session_id']}`")
    full_lines.append(f"- 断言统计: `PASS={passed_assert}, FAIL={failed_assert}, TOTAL={total_assert}`")
    full_lines.append("")
    full_lines.append("## 2. 主会话明细")
    for idx, turn in enumerate(payload["main_session"]["turns"], 1):
        full_lines.append(f"### Round {idx}")
        full_lines.append(f"- 输入: `{turn.get('query', '')}`")
        full_lines.append(f"- 工具: `{turn.get('selected_tool', '')}`")
        full_lines.append(
            f"- 记忆门控: `score={turn.get('memory_gate_score', 0)}, gate_triggered={turn.get('memory_gate_triggered', False)}, hits={turn.get('memory_recall_hits', 0)}`"
        )
        full_lines.append(f"- 短期记忆状态: `{turn.get('memory_stats_after', {})}`")
        full_lines.append("- 输出:")
        full_lines.append("```text")
        full_lines.append(str(turn.get("final_answer", "")).strip())
        full_lines.append("```")
        full_lines.append("")

    full_lines.append("## 3. 会话归档与跨会话重入")
    full_lines.append(f"- 归档前: `{payload['main_session']['finalize']['pre']}`")
    full_lines.append(f"- 归档后: `{payload['main_session']['finalize']['post']}`")
    reentry = payload["reentry_session"]["turn"]
    full_lines.append(f"- 重入问题: `{reentry.get('query', '')}`")
    full_lines.append(
        f"- 重入门控: `score={reentry.get('memory_gate_score', 0)}, gate_triggered={reentry.get('memory_gate_triggered', False)}, hits={reentry.get('memory_recall_hits', 0)}`"
    )
    full_lines.append(f"- 重入工具: `{reentry.get('selected_tool', '')}`")
    full_lines.append("```text")
    full_lines.append(str(reentry.get("final_answer", "")).strip())
    full_lines.append("```")
    full_lines.append("")

    full_lines.append("## 4. user_id 隔离验证")
    isolation = payload["user_isolation"]
    full_lines.append(f"- user-A 预热输入: `{isolation.get('user_a_warmup', {}).get('query', '')}`")
    full_lines.append(f"- user-A 重入输入: `{isolation.get('user_a_reentry', {}).get('query', '')}`")
    full_lines.append(f"- user-A 输出摘要: `{_truncate_line(isolation.get('user_a_reentry', {}).get('final_answer', ''), 180)}`")
    full_lines.append(f"- user-B 预热输入: `{isolation.get('user_b_warmup', {}).get('query', '')}`")
    full_lines.append(f"- user-B 重入输入: `{isolation.get('user_b_reentry', {}).get('query', '')}`")
    full_lines.append(f"- user-B 输出摘要: `{_truncate_line(isolation.get('user_b_reentry', {}).get('final_answer', ''), 180)}`")
    full_lines.append("")

    full_lines.append("## 5. 持久化记录计数")
    for item in payload.get("postgres_counts", []):
        full_lines.append(f"- `{item}`")
    full_lines.append("")

    full_lines.append("## 6. 断言结果")
    for item in assertions:
        full_lines.append(f"- `{item['name']}`: {'PASS' if item['passed'] else 'FAIL'}")

    md_path.write_text("\n".join(full_lines), encoding="utf-8")

    brief_lines: List[str] = []
    brief_lines.append("# 集成测试简报")
    brief_lines.append("")
    brief_lines.append(f"- 时间(UTC): `{payload['meta']['generated_at_utc']}`")
    brief_lines.append(f"- 主会话: `{payload['main_session']['session_id']}`")
    brief_lines.append(f"- 重入会话: `{payload['reentry_session']['session_id']}`")
    brief_lines.append(f"- 断言: `PASS={passed_assert}, FAIL={failed_assert}, TOTAL={total_assert}`")
    brief_lines.append("")
    brief_lines.append("## 输入输出（摘要）")
    for idx, turn in enumerate(payload["main_session"]["turns"], 1):
        brief_lines.append(f"- R{idx} 输入: `{turn.get('query', '')}`")
        brief_lines.append(f"  输出: `{_truncate_line(turn.get('final_answer', ''), 140)}`")
    brief_lines.append(f"- 重入输入: `{reentry.get('query', '')}`")
    brief_lines.append(f"  重入输出: `{_truncate_line(reentry.get('final_answer', ''), 160)}`")
    brief_lines.append("")
    brief_lines.append("## 断言")
    for item in assertions:
        brief_lines.append(f"- `{item['name']}`: {'PASS' if item['passed'] else 'FAIL'}")

    brief_path.write_text("\n".join(brief_lines), encoding="utf-8")

    return {"json": str(json_path), "md": str(md_path), "md_brief": str(brief_path)}


def main() -> None:
    cfg = DEFAULT_CONFIG
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    threshold = int(getattr(cfg, "memory_persistent_trigger_threshold", 2) or 2)

    rag_service = RAGService(cfg)
    registry = ToolRegistry(rag_service=rag_service, config=cfg)
    store = PersistentMemoryStore(cfg)
    graph = build_multi_agent_graph(registry, persistent_store=store)

    manager = SessionMemoryManager()
    user_id = "it-user-0001"
    main_sid = f"integration-main-{ts}"

    try:
        reset_info = _reset_persistent_memory(store)

        main_inputs = [
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
        main_turns: List[Dict[str, Any]] = []
        for query in main_inputs:
            main_turns.append(
                _run_turn(
                    graph=graph,
                    memory_manager=manager,
                    user_id=user_id,
                    session_id=main_sid,
                    query=query,
                    include_pending_in_prompt=cfg.memory_include_pending_in_prompt,
                    persistent_gate_threshold=threshold,
                )
            )

        finalize_pre = manager.stats(user_id=user_id, session_id=main_sid)
        _finalize_session_memory(
            user_id=user_id,
            session_id=main_sid,
            config=cfg,
            memory_manager=manager,
            persistent_store=store,
        )
        finalize_post = manager.stats(user_id=user_id, session_id=main_sid)

        reentry_turn = _run_turn(
            graph=graph,
            memory_manager=SessionMemoryManager(),
            user_id=user_id,
            session_id=main_sid,
            query="刚才两个物品里，哪个更适合买入，简短回答",
            include_pending_in_prompt=cfg.memory_include_pending_in_prompt,
            persistent_gate_threshold=threshold,
        )

        iso_sid = f"integration-iso-{ts}"
        iso_manager = SessionMemoryManager()
        user_a_warmup = _run_turn(
            graph=graph,
            memory_manager=iso_manager,
            user_id="user-A",
            session_id=iso_sid,
            query="介绍一下非洲之心",
            include_pending_in_prompt=cfg.memory_include_pending_in_prompt,
            persistent_gate_threshold=threshold,
        )
        user_b_warmup = _run_turn(
            graph=graph,
            memory_manager=iso_manager,
            user_id="user-B",
            session_id=iso_sid,
            query="介绍一下QBZ95-1突击步枪",
            include_pending_in_prompt=cfg.memory_include_pending_in_prompt,
            persistent_gate_threshold=threshold,
        )
        user_a_reentry = _run_turn(
            graph=graph,
            memory_manager=SessionMemoryManager(),
            user_id="user-A",
            session_id=iso_sid,
            query="它现在价格是多少",
            include_pending_in_prompt=cfg.memory_include_pending_in_prompt,
            persistent_gate_threshold=threshold,
        )
        user_b_reentry = _run_turn(
            graph=graph,
            memory_manager=SessionMemoryManager(),
            user_id="user-B",
            session_id=iso_sid,
            query="它现在价格是多少",
            include_pending_in_prompt=cfg.memory_include_pending_in_prompt,
            persistent_gate_threshold=threshold,
        )

        cfg_snapshot = asdict(cfg)
        for key in ("neo4j_password", "memory_persistent_dsn", "df_api_token"):
            if key in cfg_snapshot and cfg_snapshot[key]:
                cfg_snapshot[key] = "***"

        payload = {
            "meta": {
                "generated_at_utc": _now_utc(),
                "user_id": user_id,
                "memory_reset": reset_info,
                "config_snapshot": cfg_snapshot,
                "tools": registry.list_tools(),
            },
            "main_session": {
                "session_id": main_sid,
                "turns": main_turns,
                "finalize": {"pre": finalize_pre, "post": finalize_post},
            },
            "reentry_session": {
                "session_id": main_sid,
                "turn": reentry_turn,
            },
            "user_isolation": {
                "session_id": iso_sid,
                "user_a_warmup": user_a_warmup,
                "user_b_warmup": user_b_warmup,
                "user_a_reentry": user_a_reentry,
                "user_b_reentry": user_b_reentry,
            },
            "postgres_counts": [
                _db_counts_for_session(store, user_id=user_id, session_id=main_sid),
                _db_counts_for_session(store, user_id="user-A", session_id=iso_sid),
                _db_counts_for_session(store, user_id="user-B", session_id=iso_sid),
            ],
        }

        assertions = _build_assertions(payload)
        report_paths = _write_reports(payload, assertions)
        all_passed = all(bool(x.get("passed", False)) for x in assertions)
        print(json.dumps({"status": "ok" if all_passed else "failed", "all_passed": all_passed, "reports": report_paths}, ensure_ascii=False))
    finally:
        registry.close()


if __name__ == "__main__":
    main()
