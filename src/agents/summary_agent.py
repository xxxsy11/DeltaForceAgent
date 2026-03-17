"""总结子 Agent：负责最终对用户输出。"""

from __future__ import annotations

from typing import Dict, List

from agents.message_utils import find_latest_message_payload
from agents.output_quality import EMPTY_RESULT_TEXT, is_failure_text
from agents.state import AgentState
from agents.tool_planner import LLMToolPlanner


class SummaryAgent:
    """消费分析报告，输出最终回答。"""

    def __init__(self, planner: LLMToolPlanner):
        self.planner = planner

    @staticmethod
    def _pick_report(state: AgentState) -> Dict:
        report = state.get("analysis_report") or {}
        if report:
            return report
        return find_latest_message_payload(
            state.get("agent_messages", []),
            message_type="analysis_report",
            to_agents={"summary", "responder"},
        )

    async def _compose_simple(self, tool_results: List[Dict], user_query: str) -> str:
        success = [item for item in tool_results if not is_failure_text(str(item.get("output", "")))]
        if len(success) == 1:
            return str(success[0].get("output", "")).strip()
        return (await self.planner.compose_answer_async(user_query=user_query, tool_results=tool_results)).strip()

    @staticmethod
    def _review_hint(state: AgentState) -> str:
        review = state.get("review_result", {}) or {}
        hints = review.get("hints", []) if isinstance(review, dict) else []
        if isinstance(hints, list) and hints:
            return "；".join([str(x).strip() for x in hints if str(x).strip()])
        return ""

    async def run(self, state: AgentState) -> Dict:
        report = self._pick_report(state)
        flow_type = str(state.get("flow_type", "simple")).strip().lower()
        tool_results = state.get("tool_results", [])
        user_query = state.get("user_query", "")
        memory_context = str(state.get("memory_context", "") or "").strip()
        retry_target = str(state.get("retry_target_stage", "") or "").strip()
        review_hint = self._review_hint(state)

        query_parts = [user_query]
        if memory_context:
            query_parts.append(f"[会话上下文]\n{memory_context}")
        if retry_target == "summary" and review_hint:
            query_parts.append(f"[审查修正要求]\n{review_hint}")
        composed_query = "\n\n".join([x for x in query_parts if str(x).strip()])

        if flow_type == "simple":
            answer = await self._compose_simple(tool_results=tool_results, user_query=composed_query)
        else:
            answer = await self.planner.compose_from_analysis_async(
                user_query=composed_query,
                analysis_report=report,
                tool_results=tool_results,
            )
            answer = str(answer or "").strip()

        if not answer:
            answer = EMPTY_RESULT_TEXT
        return {
            "final_answer": answer,
            "summary_attempt": int(state.get("summary_attempt", 0) or 0) + 1,
            "debug_steps": state.get("debug_steps", []) + ["summary: done"],
        }
