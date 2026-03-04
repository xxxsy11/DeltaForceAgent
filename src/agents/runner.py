"""
Multi-Agent 运行入口
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import time
from typing import Any, Dict, Optional

from agents.graph import build_multi_agent_graph
from config import DEFAULT_CONFIG, GraphRAGConfig
from memory import PersistentMemoryStore, SessionMemoryManager
from memory.components import MemoryCompressionAgent, PersistentMemoryWriteNode
from services import RAGService
from tools import ToolRegistry

_GLOBAL_MEMORY_MANAGER = SessionMemoryManager()
_QUERY_RUNTIME_CACHE: Dict[int, "QueryRuntime"] = {}


@dataclass
class QueryRuntime:
    config: GraphRAGConfig
    registry: ToolRegistry
    persistent_store: PersistentMemoryStore
    graph: Any

    async def close_async(self) -> None:
        await self.registry.close_async()


def _build_runtime(config: GraphRAGConfig) -> QueryRuntime:
    rag_service = RAGService(config)
    registry = ToolRegistry(rag_service=rag_service, config=config)
    persistent_store = PersistentMemoryStore(config)
    graph = build_multi_agent_graph(registry, persistent_store=persistent_store)
    return QueryRuntime(
        config=config,
        registry=registry,
        persistent_store=persistent_store,
        graph=graph,
    )


def _get_query_runtime(config: GraphRAGConfig) -> QueryRuntime:
    cache_key = id(config)
    runtime = _QUERY_RUNTIME_CACHE.get(cache_key)
    if runtime is not None:
        return runtime
    runtime = _build_runtime(config)
    _QUERY_RUNTIME_CACHE[cache_key] = runtime
    return runtime


def _build_initial_state(
    query: str,
    session_id: str,
    user_id: str = "default_user",
    memory_patch: Optional[Dict[str, Any]] = None,
):
    input_received_at_utc = datetime.now(timezone.utc).isoformat()
    input_received_perf = time.perf_counter()
    base = {
        "user_id": user_id,
        "session_id": session_id,
        "user_query": query,
        "intent": "",
        "intent_reason": "",
        "flow_type": "simple",
        "plan_source": "",
        "requires_task_planning": False,
        "requires_specialist_analysis": False,
        "selected_tool": "",
        "tool_query": "",
        "task_plan": [],
        "tool_calls": [],
        "tool_results": [],
        "analysis_report": {},
        "agent_messages": [],
        "orchestration_meta": {
            "input_received_at_utc": input_received_at_utc,
            "input_received_perf": input_received_perf,
        },
        "memory_context": "",
        "memory_pending_digest": "",
        "memory_recent_raw": [],
        "memory_pending_buffer": [],
        "memory_rolling_summary": "",
        "memory_merge_count": 0,
        "memory_persistent_context": "",
        "memory_persistent_entities": [],
        "memory_persistent_hits": [],
        "memory_persistent_used": False,
        "memory_persistent_gate_score": 0,
        "memory_keyword_candidates": [],
        "memory_fact_candidates": [],
        "selected_skill": "",
        "skill_reason": "",
        "skill_confidence": 0.0,
        "skill_matched_by": [],
        "skill_locked_plan": False,
        "skill_tool_chain": [],
        "retry_count_total": 0,
        "retry_count_by_stage": {},
        "retry_budget_exhausted": False,
        "retry_trace": [],
        "retry_target_stage": "",
        "retry_reason": "",
        "force_replan": False,
        "force_reintent": False,
        "execution_attempt": 0,
        "summary_attempt": 0,
        "validation_result": {},
        "review_result": {},
        "quality_score": 1.0,
        "quality_gate_passed": True,
        "block_persistent_write": False,
        "last_failed_stage": "",
        "last_error_type": "",
        "last_error_code": "",
        "attempt_id": "",
        "intermediate_artifacts": {},
        "understanding_entities": [],
        "understanding_entity_count": 0,
        "understanding_confidence": 0.0,
        "understanding_compare_target_count": 2,
        "tool_output": "",
        "final_answer": "",
        "debug_steps": [],
    }
    if memory_patch:
        base.update(memory_patch)
    return base


async def _finalize_session_memory(
    user_id: str,
    session_id: str,
    config: GraphRAGConfig,
    memory_manager: SessionMemoryManager,
    persistent_store: PersistentMemoryStore,
) -> None:
    memory_patch = memory_manager.build_state_patch(
        user_id=user_id,
        session_id=session_id,
        include_pending_in_prompt=config.memory_include_pending_in_prompt,
    )
    recent = [dict(x) for x in memory_patch.get("memory_recent_raw", []) if isinstance(x, dict)]
    pending = [dict(x) for x in memory_patch.get("memory_pending_buffer", []) if isinstance(x, dict)]
    if not recent and not pending:
        return

    compression = MemoryCompressionAgent(config=config)
    flush_input = {
        "user_id": user_id,
        "session_id": session_id,
        "user_query": "",
        "final_answer": "",
        "memory_recent_raw": [],
        "memory_pending_buffer": pending + recent,
        "memory_rolling_summary": str(memory_patch.get("memory_rolling_summary", "") or ""),
        "memory_merge_count": int(memory_patch.get("memory_merge_count", 0) or 0),
        "memory_context": str(memory_patch.get("memory_context", "") or ""),
        "memory_force_compress": True,
        "agent_messages": [],
        "debug_steps": [],
    }
    compressed = await compression.run(flush_input)
    memory_manager.save_from_state(user_id=user_id, session_id=session_id, state=compressed)

    writer = PersistentMemoryWriteNode(store=persistent_store, config=config)
    await writer.run(
        {
            "user_id": user_id,
            "session_id": session_id,
            "user_query": "",
            "final_answer": "",
            "tool_results": [],
            "memory_merge_count": compressed.get("memory_merge_count", 0),
            "memory_rolling_summary": compressed.get("memory_rolling_summary", ""),
            "memory_fact_candidates": compressed.get("memory_fact_candidates", []),
            "agent_messages": compressed.get("agent_messages", []),
            "debug_steps": [],
        }
    )


async def run_agent_query(
    query: str,
    config: Optional[GraphRAGConfig] = None,
    session_id: str = "default",
    user_id: str = "default_user",
    memory_manager: Optional[SessionMemoryManager] = None,
) -> str:
    cfg = config or DEFAULT_CONFIG
    manager = memory_manager or _GLOBAL_MEMORY_MANAGER
    runtime = _get_query_runtime(cfg)

    memory_patch = manager.build_state_patch(
        user_id=user_id,
        session_id=session_id,
        include_pending_in_prompt=cfg.memory_include_pending_in_prompt,
    )
    result = await runtime.graph.ainvoke(
        _build_initial_state(query, session_id=session_id, user_id=user_id, memory_patch=memory_patch)
    )
    manager.save_from_state(user_id=user_id, session_id=session_id, state=result)
    return result.get("final_answer", "")


async def run_agent_interactive(config: Optional[GraphRAGConfig] = None):
    cfg = config or DEFAULT_CONFIG
    runtime = _build_runtime(cfg)
    memory_manager = SessionMemoryManager()
    user_id = "user-0001"
    session_id = memory_manager.next_session_id(prefix="chat")

    tool_names = ", ".join(runtime.registry.list_tools())
    print(f"\nMulti-Agent 模式已启动（当前工具: {tool_names}）")
    print(f"当前用户: {user_id}")
    print(f"当前会话: {session_id}")
    print("输入 'quit' 退出，'new session' 新会话，'switch user <id>' 切换用户，'clear memory' 清空当前会话记忆，'memory stats' 查看记忆状态。")
    try:
        while True:
            query = (await asyncio.to_thread(input, "\nAgent问题: ")).strip()
            if not query:
                continue

            lower = query.lower()
            if lower == "quit":
                await _finalize_session_memory(
                    user_id=user_id,
                    session_id=session_id,
                    config=cfg,
                    memory_manager=memory_manager,
                    persistent_store=runtime.persistent_store,
                )
                break

            if lower == "new session":
                await _finalize_session_memory(
                    user_id=user_id,
                    session_id=session_id,
                    config=cfg,
                    memory_manager=memory_manager,
                    persistent_store=runtime.persistent_store,
                )
                session_id = memory_manager.next_session_id(prefix="chat")
                print(f"已切换到新会话: {session_id}")
                continue

            if lower.startswith("switch user"):
                parts = query.split(maxsplit=2)
                if len(parts) < 3 or not parts[2].strip():
                    print("用法: switch user <id>")
                    continue
                new_user_id = parts[2].strip()
                await _finalize_session_memory(
                    user_id=user_id,
                    session_id=session_id,
                    config=cfg,
                    memory_manager=memory_manager,
                    persistent_store=runtime.persistent_store,
                )
                user_id = new_user_id
                session_id = memory_manager.next_session_id(prefix="chat")
                print(f"已切换用户: {user_id}")
                print(f"当前会话: {session_id}")
                continue

            if lower == "clear memory":
                await _finalize_session_memory(
                    user_id=user_id,
                    session_id=session_id,
                    config=cfg,
                    memory_manager=memory_manager,
                    persistent_store=runtime.persistent_store,
                )
                memory_manager.clear_session(user_id=user_id, session_id=session_id)
                print(f"已清空会话记忆: {session_id}")
                continue

            if lower == "memory stats":
                stats = memory_manager.stats(user_id=user_id, session_id=session_id)
                print(f"记忆状态: {stats}")
                continue

            memory_patch = memory_manager.build_state_patch(
                user_id=user_id,
                session_id=session_id,
                include_pending_in_prompt=cfg.memory_include_pending_in_prompt,
            )
            result = await runtime.graph.ainvoke(
                _build_initial_state(query, session_id=session_id, user_id=user_id, memory_patch=memory_patch)
            )
            memory_manager.save_from_state(user_id=user_id, session_id=session_id, state=result)
            orchestration_meta = result.get("orchestration_meta", {}) or {}
            input_to_tool_ms = orchestration_meta.get("first_tool_selected_latency_ms")
            selected_tool = str(orchestration_meta.get("latest_selected_tool", "") or result.get("selected_tool", ""))
            if input_to_tool_ms is not None:
                try:
                    input_to_tool_ms = float(input_to_tool_ms)
                    if input_to_tool_ms >= 1000:
                        cost = f"{input_to_tool_ms / 1000:.2f} 秒"
                    else:
                        cost = f"{input_to_tool_ms:.1f} 毫秒"
                    print(f"\n[路由] 工具选择完成：{selected_tool or 'none'}（输入到选工具耗时 {cost}）")
                except Exception:
                    pass
            print(f"\n回答:\n{result.get('final_answer', '')}")
    except KeyboardInterrupt:
        await _finalize_session_memory(
            user_id=user_id,
            session_id=session_id,
            config=cfg,
            memory_manager=memory_manager,
            persistent_store=runtime.persistent_store,
        )
        print("\n已中断，当前会话记忆已归档。")
    finally:
        await runtime.close_async()
