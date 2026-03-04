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

    @staticmethod
    def _build_compare_query(entities: List[str], target_count: int) -> str:
        uniq: List[str] = []
        for item in entities:
            value = str(item or "").strip()
            if value and value not in uniq:
                uniq.append(value)
        if len(uniq) < 2:
            return ""
        count = max(2, int(target_count or 2))
        return f"{'、'.join(uniq[:count])} 对比"

    @staticmethod
    def _normalize_tool_query(
        tool_name: str,
        tool_query: str,
        understanding_entities: List[str],
        compare_target_count: int,
    ) -> str:
        query = str(tool_query or "").strip()
        if tool_name != "df_multi_item_compare":
            return query
        # 比较工具必须拿到明确实体，否则“这两个物品”场景会直接失败。
        if "、" in query and len(query) >= 3:
            return query
        fallback = TaskPlanningAgent._build_compare_query(
            entities=understanding_entities,
            target_count=compare_target_count,
        )
        return fallback or query

    async def run(self, state: AgentState) -> Dict:
        query = state.get("user_query", "")
        planning_query = str(state.get("tool_query", "") or "").strip() or query
        fallback_intent = state.get("intent", "")
        fallback_tool = state.get("selected_tool", "")
        flow_type = str(state.get("flow_type", "") or "simple")
        understanding_entities = [
            str(x).strip()
            for x in (state.get("understanding_entities", []) or [])
            if str(x).strip()
        ]
        compare_target_count = int(state.get("understanding_compare_target_count", 2) or 2)

        force_replan = bool(state.get("force_replan", False))

        if not state.get("requires_task_planning", False) and not force_replan:
            return {
                "debug_steps": state.get("debug_steps", []) + ["task_planning: skipped"],
            }

        if bool(state.get("skill_locked_plan", False)) and not force_replan:
            return {
                "debug_steps": state.get("debug_steps", []) + ["task_planning: skipped(skill_locked_plan)"],
            }

        # 简单流在重试进入 task_planning 时，保持原工具不变，避免“单工具问题被误改道”。
        if flow_type != "complex" and fallback_tool and fallback_tool != "none":
            fallback_query = self._normalize_tool_query(
                tool_name=fallback_tool,
                tool_query=planning_query,
                understanding_entities=understanding_entities,
                compare_target_count=compare_target_count,
            )
            task_plan = [{"tool_name": fallback_tool, "tool_query": fallback_query}]
            message = {
                "from_agent": "task_planning",
                "to_agent": "execution",
                "message_type": "task_plan",
                "payload": {
                    "intent": fallback_intent or state.get("intent", ""),
                    "reason": "simple_flow_locked_tool",
                    "task_plan": task_plan,
                },
            }
            return {
                "intent": fallback_intent or state.get("intent", ""),
                "intent_reason": "simple_flow_locked_tool",
                "plan_source": "simple_flow_locked_tool",
                "task_plan": task_plan,
                "tool_calls": task_plan,
                "agent_messages": state.get("agent_messages", []) + [message],
                "force_replan": False,
                "debug_steps": state.get("debug_steps", []) + ["task_planning: simple_flow_locked_tool"],
            }

        decision = await self.planner.plan_with_hint_async(
            query=planning_query,
            available_tools=self.registry.list_tools(),
            fallback_intent=fallback_intent,
            fallback_tool=fallback_tool,
            force_llm=True,
        )
        task_plan: List[Dict] = []
        for call in decision.tool_calls:
            tool_name = str(call.tool_name or "").strip()
            tool_query = self._normalize_tool_query(
                tool_name=tool_name,
                tool_query=str(call.tool_query or "").strip(),
                understanding_entities=understanding_entities,
                compare_target_count=compare_target_count,
            )
            task_plan.append({"tool_name": tool_name, "tool_query": tool_query})

        if not task_plan:
            skill_chain = [x for x in (state.get("skill_tool_chain", []) or []) if isinstance(x, dict)]
            for call in skill_chain:
                tool_name = str(call.get("tool_name", "") or "").strip()
                if not tool_name:
                    continue
                tool_query = self._normalize_tool_query(
                    tool_name=tool_name,
                    tool_query=str(call.get("tool_query", "") or "").strip(),
                    understanding_entities=understanding_entities,
                    compare_target_count=compare_target_count,
                )
                task_plan.append({"tool_name": tool_name, "tool_query": tool_query})

        if not task_plan and fallback_tool and fallback_tool != "none":
            fallback_query = self._normalize_tool_query(
                tool_name=fallback_tool,
                tool_query=planning_query,
                understanding_entities=understanding_entities,
                compare_target_count=compare_target_count,
            )
            task_plan = [{"tool_name": fallback_tool, "tool_query": fallback_query}]

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
            "force_replan": False,
            "debug_steps": state.get("debug_steps", []) + [f"task_planning: {plan_source}"],
        }
