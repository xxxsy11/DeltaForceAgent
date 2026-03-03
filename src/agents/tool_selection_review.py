"""工具选择审核 Agent：仅在异常重试路径触发，复核并重选工具。"""

from __future__ import annotations

from typing import Dict, List

from agents.state import AgentState
from agents.tool_planner import LLMToolPlanner
from tools import ToolRegistry


class ToolSelectionReviewAgent:
    """异常时触发的工具选择复核层（默认单次重试预算）。"""

    def __init__(self, planner: LLMToolPlanner, registry: ToolRegistry):
        self.planner = planner
        self.registry = registry

    @staticmethod
    def _normalize_compare_query(tool_name: str, tool_query: str, entities: List[str], compare_n: int) -> str:
        query = str(tool_query or "").strip()
        if tool_name != "df_multi_item_compare":
            return query
        uniq: List[str] = []
        for item in entities:
            name = str(item or "").strip()
            if name and name not in uniq:
                uniq.append(name)
        if len(uniq) < 2:
            return query
        target = max(2, int(compare_n or 2))
        return f"{'、'.join(uniq[:target])} 对比"

    def run(self, state: AgentState) -> Dict:
        query = str(state.get("user_query", "") or "").strip()
        if not query:
            return {"debug_steps": list(state.get("debug_steps", []) or []) + ["tool_selection_review: skipped(empty_query)"]}

        fallback_intent = str(state.get("intent", "") or "").strip()
        fallback_tool = str(state.get("selected_tool", "") or "").strip()
        review_query = str(state.get("tool_query", "") or "").strip() or query
        entities = [str(x).strip() for x in (state.get("understanding_entities", []) or []) if str(x).strip()]
        compare_n = int(state.get("understanding_compare_target_count", 2) or 2)

        decision = self.planner.plan_force_tool_selection(
            query=review_query,
            available_tools=self.registry.list_tools(),
            fallback_intent=fallback_intent,
            fallback_tool=fallback_tool,
        )

        task_plan: List[Dict] = []
        for call in decision.tool_calls:
            tool_name = str(call.tool_name or "").strip()
            tool_query = self._normalize_compare_query(
                tool_name=tool_name,
                tool_query=str(call.tool_query or "").strip(),
                entities=entities,
                compare_n=compare_n,
            )
            if not tool_name:
                continue
            if not tool_query:
                tool_query = query
            task_plan.append({"tool_name": tool_name, "tool_query": tool_query})

        if not task_plan and fallback_tool:
            task_plan = [{"tool_name": fallback_tool, "tool_query": review_query}]

        selected_tool = str(task_plan[0].get("tool_name", "none")) if task_plan else "none"
        selected_query = str(task_plan[0].get("tool_query", "")) if task_plan else ""

        message = {
            "from_agent": "tool_selection_review",
            "to_agent": "execution",
            "message_type": "tool_reselected",
            "payload": {
                "intent": decision.intent,
                "reason": decision.reason,
                "task_plan": task_plan,
            },
        }
        return {
            "intent": decision.intent or fallback_intent,
            "intent_reason": f"{decision.reason} (review)",
            "plan_source": "tool_selection_review",
            "selected_tool": selected_tool,
            "tool_query": selected_query,
            "task_plan": task_plan,
            "tool_calls": task_plan,
            "force_reintent": False,
            "force_replan": False,
            "agent_messages": list(state.get("agent_messages", []) or []) + [message],
            "debug_steps": list(state.get("debug_steps", []) or [])
            + [f"tool_selection_review: reselected={selected_tool},plan={len(task_plan)}"],
        }

