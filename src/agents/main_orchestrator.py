"""主调度 Agent：负责整体流程上下文初始化与 A2A 起始消息。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict

from agents.state import AgentState


class MainOrchestratorAgent:
    """主流程控制器，不直接调用业务工具。"""

    def run(self, state: AgentState) -> Dict:
        query = str(state.get("user_query", "")).strip()
        now_utc = datetime.now(timezone.utc).isoformat()
        message = {
            "from_agent": "main_orchestrator",
            "to_agent": "intent_recognition",
            "message_type": "orchestration_start",
            "payload": {
                "query": query,
                "timestamp_utc": now_utc,
            },
        }
        return {
            "user_query": query,
            "orchestration_meta": {
                "started_at_utc": now_utc,
                "version": "main-plus-subagents-v1",
            },
            "agent_messages": state.get("agent_messages", []) + [message],
            "debug_steps": state.get("debug_steps", []) + ["main_orchestrator: start"],
        }
