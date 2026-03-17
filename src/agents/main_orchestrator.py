"""主调度 Agent：负责整体流程上下文初始化与 A2A 起始消息。"""

from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Dict

from agents.message_payloads import build_orchestration_start_payload
from agents.message_utils import append_agent_message
from agents.state import AgentState


class MainOrchestratorAgent:
    """主流程控制器，不直接调用业务工具。"""

    async def run(self, state: AgentState) -> Dict:
        query = str(state.get("user_query", "")).strip()
        memory_context = str(state.get("memory_context", "") or "")
        now_utc = datetime.now(timezone.utc).isoformat()
        now_perf = time.perf_counter()
        orchestration_meta = dict(state.get("orchestration_meta", {}) or {})
        orchestration_meta.update(
            {
                "started_at_utc": now_utc,
                "started_at_perf": now_perf,
                "version": "main-plus-subagents-v1",
                "has_memory_context": bool(memory_context),
            }
        )
        return {
            "user_query": query,
            "orchestration_meta": orchestration_meta,
            "agent_messages": append_agent_message(
                state.get("agent_messages", []),
                from_agent="main_orchestrator",
                to_agent="intent_recognition",
                message_type="orchestration_start",
                payload=build_orchestration_start_payload(
                    query=query,
                    has_memory_context=bool(memory_context),
                    timestamp_utc=now_utc,
                ),
            ),
            "debug_steps": state.get("debug_steps", []) + ["main_orchestrator: start"],
        }
