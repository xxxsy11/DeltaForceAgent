"""任务规划子 Agent：仅在复杂问题下执行多工具规划。"""

from __future__ import annotations

from typing import Dict, List

from agents.state import AgentState
from agents.tool_planner import LLMToolPlanner
from tools import ToolRegistry


class TaskPlanningAgent:
    """复杂问题的二级任务规划（可替换为更专业模型）。"""

    def __init__(self, planner: LLMToolPlanner, registry: ToolRegistry):
        self.planner = planner
        self.registry = registry

    def run(self, state: AgentState) -> Dict:
        query = state.get("user_query", "")
        fallback_intent = state.get("intent", "")
        fallback_tool = state.get("selected_tool", "")

        if not state.get("requires_task_planning", False):
            return {
                "debug_steps": state.get("debug_steps", []) + ["task_planning: skipped"],
            }

        decision = self.planner.plan_with_hint(
            query=query,
            available_tools=self.registry.list_tools(),
            fallback_intent=fallback_intent,
            fallback_tool=fallback_tool,
            force_llm=True,
        )
        task_plan: List[Dict] = [
            {"tool_name": call.tool_name, "tool_query": call.tool_query}
            for call in decision.tool_calls
        ]

        if not task_plan and fallback_tool and fallback_tool != "none":
            task_plan = [{"tool_name": fallback_tool, "tool_query": query}]

        plan_source = "llm_task_planning" if "LLM" in decision.reason or "规划" in decision.reason else "fallback_task_planning"
        message = {
            "from_agent": "task_planning",
            "to_agent": "execution",
            "message_type": "task_plan",
            "payload": {
                "intent": decision.intent,
                "reason": decision.reason,
                "task_plan": task_plan,
            },
        }
        return {
            "intent": decision.intent,
            "intent_reason": decision.reason,
            "plan_source": plan_source,
            "task_plan": task_plan,
            "tool_calls": task_plan,
            "agent_messages": state.get("agent_messages", []) + [message],
            "debug_steps": state.get("debug_steps", []) + [f"task_planning: {plan_source}"],
        }
