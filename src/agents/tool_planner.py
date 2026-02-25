"""基于 LLM 的工具规划器（支持多工具调用）。"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI

from config import DEFAULT_CONFIG, GraphRAGConfig
from agents.intent_analyzer import IntentAnalyzer
from rag_modules.llm_utils import extract_text_content

logger = logging.getLogger(__name__)


@dataclass
class ToolCallPlan:
    tool_name: str
    tool_query: str


@dataclass
class PlannerDecision:
    intent: str
    tool_calls: List[ToolCallPlan]
    reason: str


class LLMToolPlanner:
    """用 LLM 做工具路由与多工具编排。"""

    def __init__(
        self,
        config: Optional[GraphRAGConfig] = None,
        max_tool_calls: int = 3,
        model_name: Optional[str] = None,
    ):
        self.config = config or DEFAULT_CONFIG
        self.max_tool_calls = max(1, max_tool_calls)
        self.model_name = (model_name or self.config.llm_model).strip()
        self.rule_fallback = IntentAnalyzer()
        self.llm = self._build_llm()

    def _build_llm(self):
        api_key = os.getenv("MOONSHOT_API_KEY", "").strip()
        if not api_key:
            logger.warning("MOONSHOT_API_KEY 未设置，工具规划将回退到规则路由。")
            return None

        try:
            return ChatOpenAI(
                model=self.model_name,
                temperature=0,
                max_tokens=512,
                api_key=api_key,
                base_url="https://api.moonshot.cn/v1",
                timeout=60,
            )
        except TypeError:
            return ChatOpenAI(
                model=self.model_name,
                temperature=0,
                max_tokens=512,
                openai_api_key=api_key,
                openai_api_base="https://api.moonshot.cn/v1",
                request_timeout=60,
            )

    @staticmethod
    def _extract_json_object(text: str) -> Dict:
        raw = (text or "").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass

        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def _sanitize_tool_calls(
        self,
        tool_calls: List[Dict],
        available_tools: List[str],
        user_query: str,
    ) -> List[ToolCallPlan]:
        valid_tools = set(available_tools)
        lock_query_tools = {
            "df_multi_item_compare",
            "df_profit_stability",
            "df_answer_composer",
            "df_place_profit_rank",
        }
        seen = set()
        plans: List[ToolCallPlan] = []

        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            tool_name = str(call.get("tool_name", "")).strip()
            if tool_name not in valid_tools:
                continue
            if tool_name in lock_query_tools:
                tool_query = user_query
            else:
                tool_query = str(call.get("tool_query", "")).strip() or user_query
            key = (tool_name, tool_query)
            if key in seen:
                continue
            seen.add(key)
            plans.append(ToolCallPlan(tool_name=tool_name, tool_query=tool_query))
            if len(plans) >= self.max_tool_calls:
                break

        return plans

    def _llm_plan(self, query: str, available_tools: List[str]) -> Optional[PlannerDecision]:
        if self.llm is None:
            return None

        prompt = f"""
你是多工具调度器。根据用户问题决定要调用哪些工具，并按顺序输出。

可用工具：
{json.dumps(available_tools, ensure_ascii=False)}

规则：
1) 只能从可用工具中选择。
2) 可选择 0~3 个工具。
3) 如果问题是多物品对比（对比/比较/哪个好），优先调用 df_multi_item_compare。
4) 如果问题是利润稳定性（稳不稳/波动/回撤/风险），优先调用 df_profit_stability。
5) 如果问题同时要求“介绍 + 价格或建议”，优先调用 df_answer_composer。
6) 如果问题包含“贵了还是便宜了 / 能不能卖 / 建不建议买 / 赚或亏多少”，优先调用 df_market_price_advice。
7) 如果问题包含“特勤处制造 / 利润最高 / 利润前三 / 枪械配件利润 / 子弹利润 / 药品针剂利润 / 防具利润”，优先调用 df_place_profit_rank。
8) 如果问题同时包含“资料介绍 + 实时价格”，应同时调用：
   - rag_knowledge_search（负责介绍、背景、属性）
   - df_market_latest_price（负责最新价格）
9) 如果问题同时包含“资料介绍 + 买卖建议/贵便宜判断”，应同时调用：
   - rag_knowledge_search
   - df_market_price_advice
10) 如果是历史价格，调用 df_market_history_price。
11) 输出必须是 JSON，不要 Markdown，不要额外解释。

输出 JSON 格式：
{{
  "intent": "简短意图",
  "reason": "简短原因",
  "tool_calls": [
    {{"tool_name":"工具名","tool_query":"传给工具的查询"}}
  ]
}}

用户问题：{query}
"""
        try:
            response = self.llm.invoke(prompt)
            text = extract_text_content(getattr(response, "content", response)).strip()
            payload = self._extract_json_object(text)
            if not payload:
                return None

            calls = payload.get("tool_calls", [])
            plans = self._sanitize_tool_calls(calls if isinstance(calls, list) else [], available_tools, query)
            if not plans:
                return None

            return PlannerDecision(
                intent=str(payload.get("intent", "llm_tool_plan")).strip() or "llm_tool_plan",
                tool_calls=plans,
                reason=str(payload.get("reason", "LLM 规划")).strip() or "LLM 规划",
            )
        except Exception as exc:
            logger.warning(f"LLM 工具规划失败，回退规则路由: {exc}")
            return None

    def _fallback_plan(self, query: str) -> PlannerDecision:
        decision = self.rule_fallback.analyze(query)
        return self._fallback_from_decision(decision, reason_suffix="(fallback)")

    @staticmethod
    def _has_complex_markers(query: str) -> bool:
        text = str(query or "")
        markers = (
            "对比", "比较", "哪个好", "哪一个", "稳不稳", "稳定性", "波动", "回撤", "风险",
            "建议", "贵了", "便宜了", "赚", "亏", "并且", "同时", "顺便", "综合",
            "以及", "并告诉", "分析",
        )
        return any(token in text for token in markers)

    def _fallback_from_decision(self, decision, reason_suffix: str) -> PlannerDecision:
        if decision.tool_name == "none":
            return PlannerDecision(intent=decision.intent, tool_calls=[], reason=decision.reason)
        return PlannerDecision(
            intent=decision.intent,
            tool_calls=[ToolCallPlan(tool_name=decision.tool_name, tool_query=decision.tool_query)],
            reason=f"{decision.reason}{reason_suffix}",
        )

    def plan(self, query: str, available_tools: List[str]) -> PlannerDecision:
        # 先做轻量规则判定：简单问题直接走规则路由，避免每次都走复杂规划。
        rule_decision = self.rule_fallback.analyze(query)
        simple_tools = {
            "df_market_latest_price",
            "df_market_history_price",
            "df_place_profit_rank",
            "rag_knowledge_search",
        }
        if (
            rule_decision.tool_name in simple_tools
            and len(str(query or "").strip()) <= 36
            and not self._has_complex_markers(query)
        ):
            return self._fallback_from_decision(rule_decision, reason_suffix="(fast-rule)")

        llm_decision = self._llm_plan(query=query, available_tools=available_tools)
        if llm_decision:
            return llm_decision
        return self._fallback_from_decision(rule_decision, reason_suffix="(fallback)")

    def plan_with_hint(
        self,
        query: str,
        available_tools: List[str],
        fallback_intent: Optional[str] = None,
        fallback_tool: Optional[str] = None,
        force_llm: bool = False,
    ) -> PlannerDecision:
        if not force_llm:
            return self.plan(query=query, available_tools=available_tools)

        llm_decision = self._llm_plan(query=query, available_tools=available_tools)
        if llm_decision:
            return llm_decision

        if fallback_tool and fallback_tool in set(available_tools):
            return PlannerDecision(
                intent=fallback_intent or "fallback_tool_plan",
                tool_calls=[ToolCallPlan(tool_name=fallback_tool, tool_query=query)],
                reason="强制LLM规划失败，使用意图识别回退",
            )
        return self._fallback_plan(query)

    @staticmethod
    def _is_failure(text: str) -> bool:
        raw = (text or "").strip()
        if not raw:
            return True
        markers = ("工具调用失败", "查询失败", "未找到工具", "系统错误", "未获得可用结果", "不可用")
        return any(token in raw for token in markers)

    def compose_answer(self, user_query: str, tool_results: List[Dict[str, str]]) -> str:
        if not tool_results:
            return "未获得可用结果。"
        if len(tool_results) == 1:
            return str(tool_results[0].get("output", "")).strip()

        fallback = "\n\n".join(
            [
                f"[{item.get('tool_name', 'tool')}]\n{item.get('output', '')}"
                for item in tool_results
            ]
        ).strip()

        success_items = [item for item in tool_results if not self._is_failure(str(item.get("output", "")))]
        failed_items = [item for item in tool_results if self._is_failure(str(item.get("output", "")))]

        # 有失败结果时不再交给 LLM 改写，避免误判成功/失败。
        if failed_items:
            parts = []
            if success_items:
                parts.append("已获取到以下结果：")
                for item in success_items:
                    parts.append(str(item.get("output", "")).strip())
            if failed_items:
                parts.append("以下部分暂时不可用：")
                for item in failed_items:
                    parts.append(f"- {item.get('tool_name', 'tool')}: {item.get('output', '').strip()}")
            return "\n".join(parts).strip() or fallback

        if self.llm is None:
            return fallback

        prompt = f"""
请根据用户问题和工具结果，给出一段合并后的最终回答。
要求：
1) 回答必须覆盖用户问题中的所有子任务。
2) 不要编造，严格依据工具结果。
3) 语言简洁自然。

用户问题：{user_query}
工具结果：{json.dumps(tool_results, ensure_ascii=False)}
"""
        try:
            response = self.llm.invoke(prompt)
            text = extract_text_content(getattr(response, "content", response)).strip()
            return text or fallback
        except Exception:
            return fallback

    def compose_from_analysis(self, user_query: str, analysis_report: Dict, tool_results: List[Dict[str, str]]) -> str:
        report = analysis_report or {}
        if not report and tool_results:
            return self.compose_answer(user_query=user_query, tool_results=tool_results)

        facts = report.get("facts", []) if isinstance(report.get("facts", []), list) else []
        recs = report.get("recommendations", []) if isinstance(report.get("recommendations", []), list) else []
        risks = report.get("risks", []) if isinstance(report.get("risks", []), list) else []
        used_tools = report.get("used_tools", []) if isinstance(report.get("used_tools", []), list) else []
        failed_tools = report.get("failed_tools", []) if isinstance(report.get("failed_tools", []), list) else []
        assumptions = report.get("assumptions", {}) if isinstance(report.get("assumptions", {}), dict) else {}

        if self.llm is None:
            lines = ["分析结论："]
            if facts:
                lines.append("关键事实：")
                lines.extend([f"- {item}" for item in facts[:6]])
            if recs:
                lines.append("建议：")
                lines.extend([f"- {item}" for item in recs[:4]])
            if risks:
                lines.append("风险：")
                lines.extend([f"- {item}" for item in risks[:4]])
            if assumptions:
                fee = assumptions.get("sell_fee_rate")
                if fee is not None:
                    lines.append(f"说明：卖出手续费按 {float(fee) * 100:.0f}% 估算。")
            if used_tools:
                lines.append("已调用工具：" + ", ".join([str(x) for x in used_tools]))
            if failed_tools:
                lines.append("失败工具：" + ", ".join([str(item.get("tool")) for item in failed_tools if isinstance(item, dict)]))
            return "\n".join(lines).strip()

        prompt = f"""
你是回答 Agent。请根据用户问题和“数据分析 Agent”的结构化报告，给出最终回答。
要求：
1) 先给结论，再给简要依据。
2) 明确说明手续费假设（若有）。
3) 若有失败工具，要诚实说明不确定性。
4) 语言简洁、可执行，不要编造。

用户问题：{user_query}
分析报告：{json.dumps(report, ensure_ascii=False)}
工具原始结果：{json.dumps(tool_results, ensure_ascii=False)}
"""
        try:
            response = self.llm.invoke(prompt)
            text = extract_text_content(getattr(response, "content", response)).strip()
            if text:
                return text
        except Exception:
            pass

        return self.compose_answer(user_query=user_query, tool_results=tool_results)
