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

    @staticmethod
    def _classify_error(output: str) -> Dict[str, str | bool]:
        text = str(output or "")
        if not text.strip():
            return {"error_code": "empty_output", "error_type": "empty", "retryable": True}
        if "HTTP 429" in text:
            return {"error_code": "http_429", "error_type": "upstream_rate_limit", "retryable": True}
        if "HTTP 5" in text or "HTTP 502" in text or "HTTP 503" in text:
            return {"error_code": "http_5xx", "error_type": "upstream_unavailable", "retryable": True}
        if "超时" in text:
            return {"error_code": "timeout", "error_type": "timeout", "retryable": True}
        if "请至少提供两个物品名称" in text:
            return {"error_code": "missing_compare_entities", "error_type": "missing_entities", "retryable": True}
        if "未能根据 objectName 匹配到交易物品ID" in text:
            return {"error_code": "object_not_matched", "error_type": "entity_not_found", "retryable": True}
        if "未找到工具" in text:
            return {"error_code": "tool_not_found", "error_type": "tool_config", "retryable": False}
        return {"error_code": "tool_failed", "error_type": "tool_failed", "retryable": False}

    def _build_tool_result(self, tool_name: str, query: str, output: str) -> AgentToolResult:
        failed = self._is_failure(output)
        if failed:
            error_meta = self._classify_error(output)
            return {
                "tool_name": tool_name,
                "tool_query": query,
                "output": output,
                "ok": False,
                "error_code": str(error_meta.get("error_code", "tool_failed") or "tool_failed"),
                "error_type": str(error_meta.get("error_type", "tool_failed") or "tool_failed"),
                "retryable": bool(error_meta.get("retryable", False)),
                "stage": "execution",
                "diagnostics": {
                    "output_len": len(str(output or "")),
                    "has_http_error": "HTTP" in str(output or ""),
                },
            }
        return {
            "tool_name": tool_name,
            "tool_query": query,
            "output": output,
            "ok": True,
            "error_code": "",
            "error_type": "",
            "retryable": False,
            "stage": "execution",
            "diagnostics": {
                "output_len": len(str(output or "")),
                "has_http_error": False,
            },
        }

    def _build_analysis_report(self, state: AgentState, tool_results: List[AgentToolResult]) -> Dict:
        facts: List[str] = []
        recommendations: List[str] = []
        risks: List[str] = []
        successes = []
        failures = []

        for item in tool_results:
            output = str(item.get("output", "")).strip()
            tool_name = item.get("tool_name", "")
            if not bool(item.get("ok", False)):
                failures.append(
                    {
                        "tool": tool_name,
                        "error": output[:500],
                        "error_type": item.get("error_type", ""),
                        "error_code": item.get("error_code", ""),
                        "retryable": bool(item.get("retryable", False)),
                    }
                )
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
            "skill": {
                "skill_id": state.get("selected_skill", ""),
                "skill_reason": state.get("skill_reason", ""),
                "skill_confidence": float(state.get("skill_confidence", 0.0) or 0.0),
                "skill_matched_by": list(state.get("skill_matched_by", []) or []),
            },
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
            tool_name = str(call.get("tool_name", "") or "").strip()
            query = str(call.get("tool_query", "") or "").strip()
            output = self.registry.invoke(tool_name, query)
            tool_results.append(self._build_tool_result(tool_name=tool_name, query=query, output=output))
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
