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
from agents.local_qwen_runtime import LocalQwenChatModel
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

    LOCAL_MAX_NEW_TOKENS_DEFAULT = 384
    REMOTE_MAX_TOKENS = 512
    REMOTE_TIMEOUT_SECONDS = 60
    FAST_RULE_QUERY_MAX_CHARS = 36
    COMPOSE_FACT_MAX_LINES = 6
    COMPOSE_RECOMMEND_MAX_LINES = 4
    COMPOSE_RISK_MAX_LINES = 4

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
        self.planning_llm = self._build_planning_llm()

    def _build_llm(self):
        if bool(getattr(self.config, "agent_local_enabled", True)) and os.path.exists(self.model_name):
            adapter_path = str(getattr(self.config, "agent_tool_selection_adapter_path", "") or "").strip()
            device = str(getattr(self.config, "agent_local_device", "cpu") or "cpu").strip()
            max_new_tokens = int(
                getattr(self.config, "agent_local_max_new_tokens", self.LOCAL_MAX_NEW_TOKENS_DEFAULT)
                or self.LOCAL_MAX_NEW_TOKENS_DEFAULT
            )
            force_no_think = bool(getattr(self.config, "agent_local_no_think", True))
            logger.info(
                "tool_planner: use local selector model=%s adapter=%s device=%s",
                self.model_name,
                adapter_path or "<none>",
                device,
            )
            return LocalQwenChatModel(
                base_model_path=self.model_name,
                adapter_path=adapter_path,
                device=device,
                max_new_tokens=max_new_tokens,
                force_no_think=force_no_think,
            )

        api_key = os.getenv("MOONSHOT_API_KEY", "").strip()
        if not api_key:
            logger.warning("MOONSHOT_API_KEY 未设置，工具规划将回退到规则路由。")
            return None

        return ChatOpenAI(
            model=self.model_name,
            temperature=0,
            max_tokens=self.REMOTE_MAX_TOKENS,
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1",
            timeout=self.REMOTE_TIMEOUT_SECONDS,
        )

    def _build_planning_llm(self):
        """二级任务规划模型：优先使用 planning LoRA，若不存在回落到 selector。"""
        if not (bool(getattr(self.config, "agent_local_enabled", True)) and os.path.exists(self.model_name)):
            return self.llm
        adapter_path = str(getattr(self.config, "agent_planning_adapter_path", "") or "").strip()
        if (
            not adapter_path
            or not os.path.isdir(adapter_path)
            or not os.path.isfile(os.path.join(adapter_path, "adapter_config.json"))
        ):
            return self.llm
        device = str(getattr(self.config, "agent_local_device", "cpu") or "cpu").strip()
        max_new_tokens = int(
            getattr(self.config, "agent_local_max_new_tokens", self.LOCAL_MAX_NEW_TOKENS_DEFAULT)
            or self.LOCAL_MAX_NEW_TOKENS_DEFAULT
        )
        force_no_think = bool(getattr(self.config, "agent_local_no_think", True))
        logger.info(
            "tool_planner: use local planning adapter=%s device=%s",
            adapter_path,
            device,
        )
        return LocalQwenChatModel(
            base_model_path=self.model_name,
            adapter_path=adapter_path,
            device=device,
            max_new_tokens=max_new_tokens,
            force_no_think=force_no_think,
        )

    @staticmethod
    def _extract_json_object(text: str) -> Dict:
        raw = (text or "").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception as exc:
            logger.debug("tool_planner: direct json parse failed, fallback to regex: %s", exc)

        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except Exception as exc:
            logger.debug("tool_planner: regex json parse failed: %s", exc)
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

    def _llm_plan(self, query: str, available_tools: List[str], for_task_planning: bool = False) -> Optional[PlannerDecision]:
        llm_client = self.planning_llm if for_task_planning else self.llm
        if llm_client is None:
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
            response = llm_client.invoke(prompt)
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

    async def _invoke_llm_async(self, llm_client, prompt: str):
        return await llm_client.ainvoke(prompt)

    async def _llm_plan_async(
        self,
        query: str,
        available_tools: List[str],
        for_task_planning: bool = False,
    ) -> Optional[PlannerDecision]:
        llm_client = self.planning_llm if for_task_planning else self.llm
        if llm_client is None:
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
            response = await self._invoke_llm_async(llm_client, prompt)
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
            logger.warning("LLM 工具规划失败，回退规则路由: %s", exc)
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
            and len(str(query or "").strip()) <= self.FAST_RULE_QUERY_MAX_CHARS
            and not self._has_complex_markers(query)
        ):
            return self._fallback_from_decision(rule_decision, reason_suffix="(fast-rule)")

        llm_decision = self._llm_plan(query=query, available_tools=available_tools, for_task_planning=False)
        if llm_decision:
            return llm_decision
        return self._fallback_from_decision(rule_decision, reason_suffix="(fallback)")

    async def plan_async(self, query: str, available_tools: List[str]) -> PlannerDecision:
        rule_decision = self.rule_fallback.analyze(query)
        simple_tools = {
            "df_market_latest_price",
            "df_market_history_price",
            "df_place_profit_rank",
            "rag_knowledge_search",
        }
        if (
            rule_decision.tool_name in simple_tools
            and len(str(query or "").strip()) <= self.FAST_RULE_QUERY_MAX_CHARS
            and not self._has_complex_markers(query)
        ):
            return self._fallback_from_decision(rule_decision, reason_suffix="(fast-rule)")

        llm_decision = await self._llm_plan_async(
            query=query,
            available_tools=available_tools,
            for_task_planning=False,
        )
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

        llm_decision = self._llm_plan(query=query, available_tools=available_tools, for_task_planning=True)
        if llm_decision:
            return llm_decision

        if fallback_tool and fallback_tool in set(available_tools):
            return PlannerDecision(
                intent=fallback_intent or "fallback_tool_plan",
                tool_calls=[ToolCallPlan(tool_name=fallback_tool, tool_query=query)],
                reason="强制LLM规划失败，使用意图识别回退",
            )
        return self._fallback_plan(query)

    async def plan_with_hint_async(
        self,
        query: str,
        available_tools: List[str],
        fallback_intent: Optional[str] = None,
        fallback_tool: Optional[str] = None,
        force_llm: bool = False,
    ) -> PlannerDecision:
        if not force_llm:
            return await self.plan_async(query=query, available_tools=available_tools)

        llm_decision = await self._llm_plan_async(
            query=query,
            available_tools=available_tools,
            for_task_planning=True,
        )
        if llm_decision:
            return llm_decision

        if fallback_tool and fallback_tool in set(available_tools):
            return PlannerDecision(
                intent=fallback_intent or "fallback_tool_plan",
                tool_calls=[ToolCallPlan(tool_name=fallback_tool, tool_query=query)],
                reason="强制LLM规划失败，使用意图识别回退",
            )
        return self._fallback_plan(query)

    def plan_force_tool_selection(
        self,
        query: str,
        available_tools: List[str],
        fallback_intent: Optional[str] = None,
        fallback_tool: Optional[str] = None,
    ) -> PlannerDecision:
        """强制使用工具选择模型重选（异常审核路径）。"""
        llm_decision = self._llm_plan(query=query, available_tools=available_tools, for_task_planning=False)
        if llm_decision:
            return llm_decision

        if fallback_tool and fallback_tool in set(available_tools):
            return PlannerDecision(
                intent=fallback_intent or "tool_selection_review_fallback",
                tool_calls=[ToolCallPlan(tool_name=fallback_tool, tool_query=query)],
                reason="工具选择审核失败，回退到原工具",
            )
        return self._fallback_plan(query)

    async def plan_force_tool_selection_async(
        self,
        query: str,
        available_tools: List[str],
        fallback_intent: Optional[str] = None,
        fallback_tool: Optional[str] = None,
    ) -> PlannerDecision:
        llm_decision = await self._llm_plan_async(
            query=query,
            available_tools=available_tools,
            for_task_planning=False,
        )
        if llm_decision:
            return llm_decision

        if fallback_tool and fallback_tool in set(available_tools):
            return PlannerDecision(
                intent=fallback_intent or "tool_selection_review_fallback",
                tool_calls=[ToolCallPlan(tool_name=fallback_tool, tool_query=query)],
                reason="工具选择审核失败，回退到原工具",
            )
        return self._fallback_plan(query)

    @staticmethod
    def _is_failure(text: str) -> bool:
        raw = (text or "").strip()
        if not raw:
            return True
        markers = ("工具调用失败", "查询失败", "未找到工具", "系统错误", "未获得可用结果", "不可用", "请至少提供两个物品名称")
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
你是最终回答整合器。请根据用户问题和工具结果给出最终回答。
要求：
1) 先给结论，再给最多3条依据。
2) 严格引用工具结果，不得补充工具中没有的数值、区间或结论。
3) 若同一信息来源冲突：价格/历史/建议以市场工具输出为准，资料描述以知识检索为准。
4) 若某部分工具失败，明确写“该部分数据不可用”，不要猜测。
5) 回答控制在 6~12 行，避免冗长模板化。

用户问题：{user_query}
工具结果：{json.dumps(tool_results, ensure_ascii=False)}
"""
        try:
            response = self.llm.invoke(prompt)
            text = extract_text_content(getattr(response, "content", response)).strip()
            return text or fallback
        except Exception as exc:
            logger.warning("tool_planner: compose_answer llm failed, fallback to raw tool outputs: %s", exc)
            return fallback

    async def compose_answer_async(self, user_query: str, tool_results: List[Dict[str, str]]) -> str:
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
你是最终回答整合器。请根据用户问题和工具结果给出最终回答。
要求：
1) 先给结论，再给最多3条依据。
2) 严格引用工具结果，不得补充工具中没有的数值、区间或结论。
3) 若同一信息来源冲突：价格/历史/建议以市场工具输出为准，资料描述以知识检索为准。
4) 若某部分工具失败，明确写“该部分数据不可用”，不要猜测。
5) 回答控制在 6~12 行，避免冗长模板化。

用户问题：{user_query}
工具结果：{json.dumps(tool_results, ensure_ascii=False)}
"""
        try:
            response = await self._invoke_llm_async(self.llm, prompt)
            text = extract_text_content(getattr(response, "content", response)).strip()
            return text or fallback
        except Exception as exc:
            logger.warning("tool_planner: compose_answer llm failed, fallback to raw tool outputs: %s", exc)
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
                lines.extend([f"- {item}" for item in facts[: self.COMPOSE_FACT_MAX_LINES]])
            if recs:
                lines.append("建议：")
                lines.extend([f"- {item}" for item in recs[: self.COMPOSE_RECOMMEND_MAX_LINES]])
            if risks:
                lines.append("风险：")
                lines.extend([f"- {item}" for item in risks[: self.COMPOSE_RISK_MAX_LINES]])
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
你是回答 Agent。请根据用户问题和“数据分析 Agent”的结构化报告输出最终回答。
要求：
1) 结构固定：结论 -> 关键依据(最多3条) -> 不确定性(如有)。
2) 所有价格、区间、样本数、利润等数值必须来自工具原始结果，不得自造或改写数量级。
3) 若存在失败工具，必须单独列出“不可用部分”，且不要用猜测补齐。
4) 若报告含手续费假设，必须原样说明。
5) 文本简洁，避免大段背景描述。

用户问题：{user_query}
分析报告：{json.dumps(report, ensure_ascii=False)}
工具原始结果：{json.dumps(tool_results, ensure_ascii=False)}
"""
        try:
            response = self.llm.invoke(prompt)
            text = extract_text_content(getattr(response, "content", response)).strip()
            if text:
                return text
        except Exception as exc:
            logger.warning("tool_planner: compose_from_analysis llm failed, fallback to compose_answer: %s", exc)

        return self.compose_answer(user_query=user_query, tool_results=tool_results)

    async def compose_from_analysis_async(
        self,
        user_query: str,
        analysis_report: Dict,
        tool_results: List[Dict[str, str]],
    ) -> str:
        report = analysis_report or {}
        if not report and tool_results:
            return await self.compose_answer_async(user_query=user_query, tool_results=tool_results)

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
                lines.extend([f"- {item}" for item in facts[: self.COMPOSE_FACT_MAX_LINES]])
            if recs:
                lines.append("建议：")
                lines.extend([f"- {item}" for item in recs[: self.COMPOSE_RECOMMEND_MAX_LINES]])
            if risks:
                lines.append("风险：")
                lines.extend([f"- {item}" for item in risks[: self.COMPOSE_RISK_MAX_LINES]])
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
你是回答 Agent。请根据用户问题和“数据分析 Agent”的结构化报告输出最终回答。
要求：
1) 结构固定：结论 -> 关键依据(最多3条) -> 不确定性(如有)。
2) 所有价格、区间、样本数、利润等数值必须来自工具原始结果，不得自造或改写数量级。
3) 若存在失败工具，必须单独列出“不可用部分”，且不要用猜测补齐。
4) 若报告含手续费假设，必须原样说明。
5) 文本简洁，避免大段背景描述。

用户问题：{user_query}
分析报告：{json.dumps(report, ensure_ascii=False)}
工具原始结果：{json.dumps(tool_results, ensure_ascii=False)}
"""
        try:
            response = await self._invoke_llm_async(self.llm, prompt)
            text = extract_text_content(getattr(response, "content", response)).strip()
            if text:
                return text
        except Exception as exc:
            logger.warning("tool_planner: compose_from_analysis llm failed, fallback to compose_answer: %s", exc)

        return await self.compose_answer_async(user_query=user_query, tool_results=tool_results)
