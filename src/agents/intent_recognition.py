"""意图识别子 Agent：负责简单/复杂流程边界判定。"""

from __future__ import annotations

from typing import Dict, Set

from agents.intent_analyzer import IntentAnalyzer
from agents.state import AgentState


class IntentRecognitionAgent:
    """一级意图识别与边界划分。"""

    SPECIALIST_INTENTS: Set[str] = {
        "market_compare_query",
        "profit_stability_query",
        "market_price_advice_query",
    }

    def __init__(self):
        self.analyzer = IntentAnalyzer()

    @staticmethod
    def _contains_complex_markers(query: str) -> bool:
        text = str(query or "")
        markers = (
            "对比", "比较", "综合", "并且", "同时", "顺便", "风险", "回撤", "稳定性", "建议",
            "收益", "亏损", "分析",
        )
        return any(token in text for token in markers)

    def run(self, state: AgentState) -> Dict:
        query = state.get("user_query", "")
        decision = self.analyzer.analyze(query)
        intent = decision.intent

        is_complex = self.analyzer.is_complex_intent(intent) or self._contains_complex_markers(query)
        flow_type = "complex" if is_complex else "simple"
        requires_task_planning = flow_type == "complex"
        requires_specialist = intent in self.SPECIALIST_INTENTS

        call = {"tool_name": decision.tool_name, "tool_query": decision.tool_query}
        task_plan = [] if decision.tool_name == "none" else [call]
        message = {
            "from_agent": "intent_recognition",
            "to_agent": "task_planning" if requires_task_planning else "execution",
            "message_type": "intent_result",
            "payload": {
                "intent": intent,
                "tool_name": decision.tool_name,
                "flow_type": flow_type,
                "reason": decision.reason,
            },
        }

        return {
            "intent": intent,
            "intent_reason": decision.reason,
            "flow_type": flow_type,
            "plan_source": "intent_rule",
            "requires_task_planning": requires_task_planning,
            "requires_specialist_analysis": requires_specialist,
            "selected_tool": decision.tool_name,
            "tool_query": decision.tool_query,
            "task_plan": task_plan,
            "tool_calls": task_plan,
            "agent_messages": state.get("agent_messages", []) + [message],
            "debug_steps": state.get("debug_steps", []) + [f"intent_recognition: {intent}/{flow_type}"],
        }
