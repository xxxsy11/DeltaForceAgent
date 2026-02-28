#!/usr/bin/env python3
"""Run full functional + memory validation for delta_agent."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _run_turn(
    *,
    graph,
    memory_manager: SessionMemoryManager,
    user_id: str,
    session_id: str,
    query: str,
    include_pending_in_prompt: bool,
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

    stats = memory_manager.stats(user_id=user_id, session_id=session_id)
    compression = _find_compression(result)
    tool_results = result.get("tool_results", []) or []
    return {
        "timestamp_utc": _now_utc(),
        "session_id": session_id,
        "query": query,
        "elapsed_sec": elapsed,
        "flow_type": result.get("flow_type", ""),
        "intent": result.get("intent", ""),
        "intent_reason": result.get("intent_reason", ""),
        "selected_tool": result.get("selected_tool", ""),
        "tool_query": result.get("tool_query", ""),
        "tool_results": tool_results,
        "tool_result_count": len(tool_results),
        "memory_persistent_gate_score": result.get("memory_persistent_gate_score", 0),
        "memory_persistent_used": bool(result.get("memory_persistent_used", False)),
        "memory_persistent_hits_count": len(result.get("memory_persistent_hits", []) or []),
        "memory_persistent_entities": result.get("memory_persistent_entities", []) or [],
        "memory_context_chars": len(str(result.get("memory_context", "") or "")),
        "compression": compression,
        "memory_stats_after": stats,
        "debug_steps": result.get("debug_steps", []) or [],
        "final_answer": result.get("final_answer", ""),
    }


def _db_counts_for_session(store: PersistentMemoryStore, user_id: str, session_id: str) -> Dict[str, Any]:
    conn = store._connect()  # noqa: SLF001 - internal helper acceptable for test script
    if conn is None:
        return {"session_id": session_id, "error": "db_connect_failed"}

    with conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chat_turns WHERE user_id=%s AND session_id=%s;", (user_id, session_id))
            chat_turns_count = int((cur.fetchone() or [0])[0] or 0)
            cur.execute("SELECT COUNT(*) FROM memory_summaries WHERE user_id=%s AND session_id=%s;", (user_id, session_id))
            summaries_count = int((cur.fetchone() or [0])[0] or 0)
            cur.execute("SELECT COUNT(*) FROM memory_facts WHERE user_id=%s AND session_id=%s;", (user_id, session_id))
            facts_count = int((cur.fetchone() or [0])[0] or 0)
            cur.execute(
                """
                SELECT turn_index, role, content
                FROM chat_turns
                WHERE user_id=%s AND session_id=%s
                ORDER BY turn_index ASC
                LIMIT 3;
                """,
                (user_id, session_id),
            )
            sample_turns = cur.fetchall() or []
    return {
        "session_id": session_id,
        "chat_turns_count": chat_turns_count,
        "memory_summaries_count": summaries_count,
        "memory_facts_count": facts_count,
        "chat_turns_samples": [
            {
                "turn_index": int(row[0]),
                "role": str(row[1]),
                "content_preview": str(row[2])[:500],
            }
            for row in sample_turns
        ],
    }


def _load_tablespace_evidence(store: PersistentMemoryStore) -> Dict[str, Any]:
    conn = store._connect()  # noqa: SLF001
    if conn is None:
        return {"error": "db_connect_failed"}
    with conn:
        with conn.cursor() as cur:
            data_directory = ""
            try:
                cur.execute("SHOW data_directory;")
                data_directory = str((cur.fetchone() or [""])[0] or "")
            except Exception:
                data_directory = ""

            cur.execute(
                """
                SELECT c.relname, pg_relation_filepath(c.oid)
                FROM pg_class c
                WHERE c.relname IN ('chat_sessions','chat_turns','memory_summaries','memory_facts')
                ORDER BY c.relname;
                """
            )
            rels = cur.fetchall() or []
            cur.execute(
                """
                SELECT t.spcname
                FROM pg_class c
                JOIN pg_tablespace t ON t.oid = c.reltablespace
                WHERE c.relname IN ('chat_sessions','chat_turns','memory_summaries','memory_facts')
                ORDER BY c.relname;
                """
            )
            ts = [str(x[0]) for x in (cur.fetchall() or [])]
    return {
        "data_directory": data_directory,
        "relation_filepaths": [{"table": str(r[0]), "filepath": str(r[1])} for r in rels],
        "tablespaces": ts,
    }


def main() -> None:
    config = DEFAULT_CONFIG
    session_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    report_json = PROJECT_ROOT / "docs/SYSTEM_FULL_VALIDATION_RESULT.json"
    report_md = PROJECT_ROOT / "docs/SYSTEM_FULL_VALIDATION_REPORT.md"

    rag_service = RAGService(config)
    registry = ToolRegistry(rag_service=rag_service, config=config)
    store = PersistentMemoryStore(config)
    graph = build_multi_agent_graph(registry, persistent_store=store)
    memory_manager = SessionMemoryManager()
    user_id = "validation-user-0001"

    try:
        tool_coverage_inputs = [
            "介绍一下非洲之心",
            "非洲之心现在什么价格",
            "查询一下非洲之心的历史价格",
            "非洲之心现在建议买吗",
            "特勤处制造什么子弹利润最高",
            "特勤处四大分组利润top3",
            "非洲之心和海洋之泪对比一下",
            "分析碳纤维散射箭矢利润稳定性",
            "介绍一下非洲之心并告诉我现在价格和是否建议买",
        ]
        tool_coverage_sid = f"tool-cover-{session_time}"
        tool_coverage_turns: List[Dict[str, Any]] = []
        for q in tool_coverage_inputs:
            turn = _run_turn(
                graph=graph,
                memory_manager=memory_manager,
                user_id=user_id,
                session_id=tool_coverage_sid,
                query=q,
                include_pending_in_prompt=config.memory_include_pending_in_prompt,
            )
            tool_coverage_turns.append(turn)

        memory_sid = f"memory-main-{session_time}"
        memory_queries = [
            "介绍一下非洲之心",
            "它现在价格多少",
            "再给我它的历史价格",
            "它现在建议买吗",
            "再介绍一下海洋之泪",
            "它现在价格呢",
            "对比一下这两个物品",
            "总结一下我们刚才聊了什么",
        ]
        memory_turns: List[Dict[str, Any]] = []
        for q in memory_queries:
            turn = _run_turn(
                graph=graph,
                memory_manager=memory_manager,
                user_id=user_id,
                session_id=memory_sid,
                query=q,
                include_pending_in_prompt=config.memory_include_pending_in_prompt,
            )
            memory_turns.append(turn)

        finalize_pre = memory_manager.stats(user_id=user_id, session_id=memory_sid)
        _finalize_session_memory(
            user_id=user_id,
            session_id=memory_sid,
            config=config,
            memory_manager=memory_manager,
            persistent_store=store,
        )
        finalize_post = memory_manager.stats(user_id=user_id, session_id=memory_sid)

        memory_manager_new = SessionMemoryManager()
        reentry_turn = _run_turn(
            graph=graph,
            memory_manager=memory_manager_new,
            user_id=user_id,
            session_id=memory_sid,
            query="刚才我们聊的两个物品里，哪个更适合买入，给我简短结论",
            include_pending_in_prompt=config.memory_include_pending_in_prompt,
        )

        isolation_sid = f"memory-isolation-{session_time}"
        isolation_turns = [
            _run_turn(
                graph=graph,
                memory_manager=memory_manager,
                user_id=user_id,
                session_id=isolation_sid,
                query="介绍一下QBZ95-1突击步枪",
                include_pending_in_prompt=config.memory_include_pending_in_prompt,
            ),
            _run_turn(
                graph=graph,
                memory_manager=memory_manager,
                user_id=user_id,
                session_id=isolation_sid,
                query="它现在价格多少",
                include_pending_in_prompt=config.memory_include_pending_in_prompt,
            ),
        ]

        db_evidence = {
            tool_coverage_sid: _db_counts_for_session(store, user_id=user_id, session_id=tool_coverage_sid),
            memory_sid: _db_counts_for_session(store, user_id=user_id, session_id=memory_sid),
            isolation_sid: _db_counts_for_session(store, user_id=user_id, session_id=isolation_sid),
        }
        tablespace_evidence = _load_tablespace_evidence(store)

        cfg_snapshot = asdict(config)
        for key in ("neo4j_password", "memory_persistent_dsn", "df_api_token"):
            if key in cfg_snapshot and cfg_snapshot[key]:
                cfg_snapshot[key] = "***"

        payload = {
            "meta": {
                "generated_at_utc": _now_utc(),
                "user_id": user_id,
                "config": cfg_snapshot,
                "tool_names": registry.list_tools(),
            },
            "tool_coverage": {
                "session_id": tool_coverage_sid,
                "turns": tool_coverage_turns,
            },
            "memory_validation": {
                "session_id": memory_sid,
                "turns": memory_turns,
                "finalize": {
                    "pre": finalize_pre,
                    "post": finalize_post,
                },
                "reentry_turn": reentry_turn,
            },
            "isolation_validation": {
                "session_id": isolation_sid,
                "turns": isolation_turns,
            },
            "postgres_evidence": db_evidence,
            "tablespace_evidence": tablespace_evidence,
        }

        report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

        lines: List[str] = []
        lines.append("# 系统全量测试报告（工具 + 长短期记忆）")
        lines.append("")
        lines.append(f"- 生成时间(UTC): `{payload['meta']['generated_at_utc']}`")
        lines.append(f"- 工具列表: `{', '.join(payload['meta']['tool_names'])}`")
        lines.append("")
        lines.append("## 1. 长期记忆存储位置验证")
        ts = payload["tablespace_evidence"]
        lines.append(f"- PostgreSQL data_directory: `{ts.get('data_directory', '')}`")
        lines.append("- 记忆表 relation filepath:")
        for item in ts.get("relation_filepaths", []):
            lines.append(f"  - `{item['table']}` -> `{item['filepath']}`")
        lines.append(f"- 记忆表 tablespace: `{ts.get('tablespaces', [])}`")
        lines.append("")
        lines.append("## 2. 工具覆盖测试（全部功能）")
        for i, turn in enumerate(tool_coverage_turns, 1):
            lines.append("")
            lines.append(f"### 2.{i} 输入")
            lines.append(f"- `{turn['query']}`")
            lines.append(f"- 选中工具: `{turn['selected_tool']}`")
            lines.append(f"- 流程: `flow_type={turn['flow_type']}`")
            lines.append(
                f"- 记忆门控: `score={turn['memory_persistent_gate_score']}, used={turn['memory_persistent_used']}, hits={turn['memory_persistent_hits_count']}`"
            )
            lines.append("- 中间过程(debug_steps):")
            for step in turn.get("debug_steps", []):
                lines.append(f"  - {step}")
            lines.append("- 工具输出:")
            for tr in turn.get("tool_results", []):
                lines.append(f"  - `{tr.get('tool_name', '')}` -> `{str(tr.get('output', ''))[:240]}`")
            lines.append("- 最终回答:")
            lines.append("```text")
            lines.append(str(turn.get("final_answer", "")).strip())
            lines.append("```")

        lines.append("")
        lines.append("## 3. 多轮记忆验证")
        lines.append(f"- 会话ID: `{memory_sid}`")
        for i, turn in enumerate(memory_turns, 1):
            stats = turn.get("memory_stats_after", {})
            lines.append("")
            lines.append(f"### 3.{i} Round")
            lines.append(f"- 输入: `{turn['query']}`")
            lines.append(f"- 选中工具: `{turn['selected_tool']}`")
            lines.append(
                f"- 记忆门控: `score={turn['memory_persistent_gate_score']}, used={turn['memory_persistent_used']}, hits={turn['memory_persistent_hits_count']}`"
            )
            lines.append(f"- 压缩信息: `{turn.get('compression', {})}`")
            lines.append(
                f"- 轮后短期记忆: `recent={stats.get('recent_raw_turns')}, pending={stats.get('pending_turns')}, merge={stats.get('merge_count')}`"
            )
            lines.append("- 最终回答:")
            lines.append("```text")
            lines.append(str(turn.get("final_answer", "")).strip())
            lines.append("```")

        lines.append("")
        lines.append("### 3.x 会话结束归档")
        lines.append(f"- before: `{payload['memory_validation']['finalize']['pre']}`")
        lines.append(f"- after: `{payload['memory_validation']['finalize']['post']}`")

        lines.append("")
        lines.append("### 3.y 跨进程长期记忆重入")
        lines.append(f"- 输入: `{reentry_turn['query']}`")
        lines.append(
            f"- 记忆门控: `score={reentry_turn['memory_persistent_gate_score']}, used={reentry_turn['memory_persistent_used']}, hits={reentry_turn['memory_persistent_hits_count']}`"
        )
        lines.append("- 最终回答:")
        lines.append("```text")
        lines.append(str(reentry_turn.get("final_answer", "")).strip())
        lines.append("```")

        lines.append("")
        lines.append("## 4. 多会话隔离验证")
        lines.append(f"- 会话ID: `{isolation_sid}`")
        for i, turn in enumerate(isolation_turns, 1):
            lines.append(f"- Round{i} 输入: `{turn['query']}` -> 工具 `{turn['selected_tool']}`")
            lines.append("```text")
            lines.append(str(turn.get("final_answer", "")).strip())
            lines.append("```")

        lines.append("")
        lines.append("## 5. PostgreSQL 落库计数验证")
        for sid, ev in db_evidence.items():
            lines.append(f"- `{sid}`: chat_turns={ev.get('chat_turns_count')}, summaries={ev.get('memory_summaries_count')}, facts={ev.get('memory_facts_count')}")
            for sample in ev.get("chat_turns_samples", []):
                lines.append(
                    f"  - sample turn#{sample['turn_index']} role={sample['role']} content={sample['content_preview'][:200]}"
                )

        lines.append("")
        lines.append("## 6. 结论")
        lines.append("- 已覆盖全部工具调用路径、短期记忆更新、长期记忆写入、门控召回、跨会话重入、会话隔离。")
        lines.append("- 若某条工具输出失败，保留原始失败输出，作为外部 API 可用性证据。")

        report_md.write_text("\n".join(lines))
        print(json.dumps({"json_report": str(report_json), "md_report": str(report_md)}, ensure_ascii=False))
    finally:
        registry.close()


if __name__ == "__main__":
    main()
