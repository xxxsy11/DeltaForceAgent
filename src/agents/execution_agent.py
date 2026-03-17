"""执行子 Agent：负责工具执行与结构化结果产出。"""

from __future__ import annotations

import asyncio
from typing import Dict, List

from agents.analysis_report_utils import build_empty_analysis_report, build_execution_analysis_report
from agents.message_payloads import build_analysis_report_payload
from agents.message_utils import append_agent_message
from agents.output_quality import MISSING_COMPARE_ENTITIES_TEXT, is_failure_text
from agents.state import AgentState, AgentToolResult
from tools import ToolRegistry


class ExecutionAgent:
    """执行 task_plan/tool_calls 并产出 analysis_report。"""

    DEFAULT_SELL_FEE_RATE = 0.13
    MAX_ERROR_OUTPUT_LEN = 500
    FACT_LINE_LIMIT = 5
    RECOMMEND_LINE_LIMIT = 4
    RISK_LINE_LIMIT = 4
    REPORT_FACT_LIMIT = 16
    REPORT_RECOMMEND_LIMIT = 12
    REPORT_RISK_LIMIT = 12
    DEFAULT_MAX_CONCURRENCY = 4

    def __init__(
        self,
        registry: ToolRegistry,
        sell_fee_rate: float = DEFAULT_SELL_FEE_RATE,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    ):
        self.registry = registry
        self.sell_fee_rate = sell_fee_rate
        self.max_concurrency = max(1, int(max_concurrency))

    @classmethod
    def _is_failure(cls, text: str) -> bool:
        return is_failure_text(text)

    @staticmethod
    def _extract_key_lines(
        text: str,
        keywords: List[str],
        limit: int = FACT_LINE_LIMIT,
    ) -> List[str]:
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
        if MISSING_COMPARE_ENTITIES_TEXT in text:
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
                        "error": output[: self.MAX_ERROR_OUTPUT_LEN],
                        "error_type": item.get("error_type", ""),
                        "error_code": item.get("error_code", ""),
                        "retryable": bool(item.get("retryable", False)),
                    }
                )
                continue
            successes.append({"tool": tool_name})
            facts.extend(
                self._extract_key_lines(
                    output,
                    keywords=["价格", "净利润", "区间", "样本数", "评级"],
                    limit=self.FACT_LINE_LIMIT,
                )
            )
            recommendations.extend(
                self._extract_key_lines(output, keywords=["建议", "结论", "推荐"], limit=self.RECOMMEND_LINE_LIMIT)
            )
            risks.extend(self._extract_key_lines(output, keywords=["风险", "回撤", "谨慎", "失败"], limit=self.RISK_LINE_LIMIT))

        return build_execution_analysis_report(
            state,
            used_tools=[item.get("tool_name", "") for item in tool_results],
            successful_tools=successes,
            failed_tools=failures,
            facts=facts[: self.REPORT_FACT_LIMIT],
            recommendations=recommendations[: self.REPORT_RECOMMEND_LIMIT],
            risks=risks[: self.REPORT_RISK_LIMIT],
            raw_tool_results=tool_results,
            sell_fee_rate=self.sell_fee_rate,
        )

    def _build_empty_execution_result(self, state: AgentState, debug_steps: List[str]) -> Dict:
        report = build_empty_analysis_report(
            state,
            sell_fee_rate=self.sell_fee_rate,
            risk_message="未命中可用工具，建议补充问题信息。",
        )
        return {
            "tool_results": [],
            "tool_output": "未命中工具。",
            "analysis_report": report,
            "agent_messages": append_agent_message(
                state.get("agent_messages", []),
                from_agent="execution",
                to_agent="summary",
                message_type="analysis_report",
                payload=build_analysis_report_payload(report),
            ),
            "debug_steps": debug_steps + ["execution: no_tool"],
        }

    @staticmethod
    def _merge_tool_output(tool_results: List[AgentToolResult]) -> str:
        return "\n\n".join([f"[{item['tool_name']}]\n{item['output']}" for item in tool_results]).strip()

    async def _invoke_tool_call(self, semaphore: asyncio.Semaphore, call: Dict) -> AgentToolResult:
        tool_name = str(call.get("tool_name", "") or "").strip()
        query = str(call.get("tool_query", "") or "").strip()
        async with semaphore:
            output = await self.registry.invoke_async(tool_name, query)
        return self._build_tool_result(tool_name=tool_name, query=query, output=output)

    async def _execute_tool_calls(self, tool_calls: List[Dict]) -> List[AgentToolResult]:
        semaphore = asyncio.Semaphore(self.max_concurrency)
        tasks = [self._invoke_tool_call(semaphore, call) for call in tool_calls]
        return await asyncio.gather(*tasks)

    async def run(self, state: AgentState) -> Dict:
        tool_calls = state.get("tool_calls") or state.get("task_plan") or []
        debug_steps = state.get("debug_steps", [])

        if not tool_calls:
            return self._build_empty_execution_result(state=state, debug_steps=debug_steps)

        tool_results = await self._execute_tool_calls(tool_calls)
        for item in tool_results:
            debug_steps = debug_steps + [f"execution: tool={item.get('tool_name', '')}"]

        merged_output = self._merge_tool_output(tool_results)
        report = self._build_analysis_report(state=state, tool_results=tool_results)

        next_agent = "specialist_analysis" if state.get("requires_specialist_analysis", False) else "summary"
        return {
            "selected_tool": tool_calls[0].get("tool_name", "none"),
            "tool_query": tool_calls[0].get("tool_query", ""),
            "tool_results": tool_results,
            "tool_output": merged_output,
            "analysis_report": report,
            "agent_messages": append_agent_message(
                state.get("agent_messages", []),
                from_agent="execution",
                to_agent=next_agent,
                message_type="analysis_report",
                payload=build_analysis_report_payload(report),
            ),
            "debug_steps": debug_steps + ["execution: done"],
        }
