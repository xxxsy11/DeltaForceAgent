"""执行子 Agent：负责工具执行与结构化结果产出。"""

from __future__ import annotations

from typing import Dict, List

from agents.state import AgentState, AgentToolResult
from tools import ToolRegistry


class ExecutionAgent:
    """执行 task_plan/tool_calls 并产出 analysis_report。"""

    FAILURE_MARKERS = ("工具调用失败", "查询失败", "未找到工具", "系统错误", "不可用")

    def __init__(self, registry: ToolRegistry, sell_fee_rate: float = 0.13):
        self.registry = registry
        self.sell_fee_rate = sell_fee_rate

    @classmethod
    def _is_failure(cls, text: str) -> bool:
        raw = str(text or "").strip()
        if not raw:
            return True
        return any(token in raw for token in cls.FAILURE_MARKERS)

    @staticmethod
    def _extract_key_lines(text: str, keywords: List[str], limit: int = 6) -> List[str]:
        lines = []
        for line in str(text or "").splitlines():
            raw = line.strip()
            if not raw:
                continue
            if any(keyword in raw for keyword in keywords):
                lines.append(raw)
            if len(lines) >= limit:
                break
        return lines

    def _build_analysis_report(self, state: AgentState, tool_results: List[AgentToolResult]) -> Dict:
        facts: List[str] = []
        recommendations: List[str] = []
        risks: List[str] = []
        successes = []
        failures = []

        for item in tool_results:
            output = str(item.get("output", "")).strip()
            tool_name = item.get("tool_name", "")
            if self._is_failure(output):
                failures.append({"tool": tool_name, "error": output[:500]})
                continue
            successes.append({"tool": tool_name})
            facts.extend(self._extract_key_lines(output, keywords=["价格", "净利润", "区间", "样本数", "评级"], limit=5))
            recommendations.extend(self._extract_key_lines(output, keywords=["建议", "结论", "推荐"], limit=4))
            risks.extend(self._extract_key_lines(output, keywords=["风险", "回撤", "谨慎", "失败"], limit=4))

        boundary = {
            "flow_type": state.get("flow_type", "simple"),
            "summary_mode": "direct" if state.get("flow_type", "simple") == "simple" else "llm_summary",
            "rule": "intent_boundary",
            "reason": [state.get("intent_reason", "")],
        }
        return {
            "query": state.get("user_query", ""),
            "intent": state.get("intent", ""),
            "route_reason": state.get("intent_reason", ""),
            "boundary": boundary,
            "plan_source": state.get("plan_source", ""),
            "used_tools": [item.get("tool_name", "") for item in tool_results],
            "successful_tools": successes,
            "failed_tools": failures,
            "facts": facts[:16],
            "recommendations": recommendations[:12],
            "risks": risks[:12],
            "assumptions": {
                "sell_fee_rate": self.sell_fee_rate,
                "sell_fee_rate_note": "卖出统一按13%手续费估算净收益",
            },
            "raw_tool_results": tool_results,
        }

    def run(self, state: AgentState) -> Dict:
        tool_calls = state.get("tool_calls") or state.get("task_plan") or []
        debug_steps = state.get("debug_steps", [])

        if not tool_calls:
            report = {
                "query": state.get("user_query", ""),
                "intent": state.get("intent", ""),
                "route_reason": state.get("intent_reason", ""),
                "boundary": {
                    "flow_type": state.get("flow_type", "simple"),
                    "summary_mode": "direct",
                    "rule": "no_tool",
                    "reason": ["未命中可用工具"],
                },
                "used_tools": [],
                "facts": [],
                "recommendations": [],
                "risks": ["未命中可用工具，建议补充问题信息。"],
                "assumptions": {
                    "sell_fee_rate": self.sell_fee_rate,
                    "sell_fee_rate_note": "卖出统一按13%手续费估算净收益",
                },
                "raw_tool_results": [],
            }
            message = {
                "from_agent": "execution",
                "to_agent": "summary",
                "message_type": "analysis_report",
                "payload": report,
            }
            return {
                "tool_results": [],
                "tool_output": "未命中工具。",
                "analysis_report": report,
                "agent_messages": state.get("agent_messages", []) + [message],
                "debug_steps": debug_steps + ["execution: no_tool"],
            }

        tool_results: List[AgentToolResult] = []
        for call in tool_calls:
            tool_name = call.get("tool_name", "")
            query = call.get("tool_query", "")
            output = self.registry.invoke(tool_name, query)
            tool_results.append(
                {
                    "tool_name": tool_name,
                    "tool_query": query,
                    "output": output,
                }
            )
            debug_steps = debug_steps + [f"execution: tool={tool_name}"]

        merged_output = "\n\n".join([f"[{r['tool_name']}]\n{r['output']}" for r in tool_results]).strip()
        report = self._build_analysis_report(state=state, tool_results=tool_results)

        next_agent = "specialist_analysis" if state.get("requires_specialist_analysis", False) else "summary"
        message = {
            "from_agent": "execution",
            "to_agent": next_agent,
            "message_type": "analysis_report",
            "payload": report,
        }
        return {
            "selected_tool": tool_calls[0].get("tool_name", "none"),
            "tool_query": tool_calls[0].get("tool_query", ""),
            "tool_results": tool_results,
            "tool_output": merged_output,
            "analysis_report": report,
            "agent_messages": state.get("agent_messages", []) + [message],
            "debug_steps": debug_steps + ["execution: done"],
        }
