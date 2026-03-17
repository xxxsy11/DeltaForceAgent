"""意图识别+主体识别子 Agent：统一完成工具选择与主体解析。"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError

from agents.intent_analyzer import IntentAnalyzer
from agents.message_payloads import build_intent_result_payload
from agents.message_utils import append_agent_message
from agents.state import AgentState
from config import DEFAULT_CONFIG, GraphRAGConfig
from rag_modules.llm_utils import extract_text_content
from skills import SkillRegistry, SkillPlan, SkillSelection

logger = logging.getLogger(__name__)


class IntentLLMResponse(BaseModel):
    intent: str = ""
    tool_name: str = ""
    reason: str = ""
    entities: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class IntentRecognitionAgent:
    """统一的 query understanding：识别意图、主体数量、工具与标准化 tool_query。"""

    LOCAL_MAX_NEW_TOKENS_DEFAULT = 384
    REMOTE_MAX_TOKENS = 512
    REMOTE_TIMEOUT_SECONDS = 60
    MAX_LLM_OUTPUT_CHARS = 12000
    MAX_ENTITY_CHARS = 24
    SHORT_ALNUM_ENTITY_MAX_CHARS = 4
    RECENT_MEMORY_ENTITY_MAX_ITEMS = 4
    MAX_COMPARE_ENTITY_COUNT = 6
    LATENCY_MS_MULTIPLIER = 1000

    SPECIALIST_INTENTS: Set[str] = {
        "market_compare_query",
        "profit_stability_query",
        "market_price_advice_query",
    }
    PRICE_TOOLS: Set[str] = {
        "df_market_latest_price",
        "df_market_history_price",
        "df_market_price_advice",
    }
    PRONOUN_MARKERS = (
        "这个",
        "那个",
        "它",
        "他",
        "她",
        "它们",
        "他们",
        "她们",
        "上一条",
        "刚才",
        "上面",
        "继续",
        "现在呢",
        "这两个",
        "两个物品",
        "两者",
        "二者",
        "这俩",
        "这三个",
        "三个物品",
        "三者",
    )
    COMPLEX_MARKERS = (
        "对比",
        "比较",
        "综合",
        "并且",
        "同时",
        "顺便",
        "风险",
        "回撤",
        "稳定性",
        "建议",
        "收益",
        "亏损",
        "分析",
    )
    TASK_PLANNING_MARKERS = (
        "并且",
        "同时",
        "顺便",
        "再",
        "然后",
        "分别",
        "对比",
        "比较",
    )
    DEFAULT_AVAILABLE_TOOLS: List[str] = [
        "rag_knowledge_search",
        "df_market_latest_price",
        "df_market_history_price",
        "df_market_price_advice",
        "df_place_profit_rank",
        "df_multi_item_compare",
        "df_profit_stability",
        "df_answer_composer",
    ]
    PROMPT_TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "prompts" / "intent_recognition_prompt.txt"
    DEFAULT_PROMPT_TEMPLATE = """
你是 query understanding 子代理。
任务：同时识别主体和选择工具。

可用工具：
{available_tools_json}

用户问题：{query}
会话可用主体候选：{memory_entities_json}

输出 JSON（不要额外文本）：
{{
  "intent": "...",
  "tool_name": "...",
  "reason": "...",
  "entities": ["实体1", "实体2"],
  "confidence": 0.0
}}

规则：
1) 价格类问题使用 df_market_latest_price / df_market_history_price / df_market_price_advice。
2) 对比问题使用 df_multi_item_compare。
3) 介绍+价格组合问题使用 df_answer_composer。
4) 代词问题（他/她/它/这两个）必须优先从“会话可用主体候选”中补全主体。
5) entities 仅保留 1~3 个“主体名词”（如 非洲之心/海洋之泪），不要输出描述句或规格（如“1x1格”）。
6) 对“这两个/这三个”这类问题，entities 必须返回对应数量的主体。
""".strip()

    def __init__(
        self,
        config: Optional[GraphRAGConfig] = None,
        available_tools: Optional[List[str]] = None,
    ):
        self.config = config or DEFAULT_CONFIG
        self.available_tools = list(available_tools or [])
        self.analyzer = IntentAnalyzer()
        self.llm = self._build_llm()
        self.skills_enabled = os.getenv("AGENT_SKILLS_ENABLED", "1").strip().lower() not in {"0", "false", "off", "no"}
        self.skill_registry = SkillRegistry(definitions_dir=None, available_tools=self.available_tools) if self.skills_enabled else None
        self._prompt_template: Optional[str] = None
        self._available_tool_set: Set[str] = set(self.available_tools or self.DEFAULT_AVAILABLE_TOOLS)

    def _available_tools_for_prompt(self) -> List[str]:
        return list(self.available_tools or self.DEFAULT_AVAILABLE_TOOLS)

    def _build_llm(self):
        model_name = str(self.config.agent_intent_model or self.config.llm_model).strip()
        if bool(getattr(self.config, "agent_local_enabled", True)) and os.path.exists(model_name):
            from agents.local_qwen_runtime import LocalQwenChatModel

            adapter_path = str(getattr(self.config, "agent_intent_adapter_path", "") or "").strip()
            device = str(getattr(self.config, "agent_local_device", "cpu") or "cpu").strip()
            max_new_tokens = int(
                getattr(self.config, "agent_local_max_new_tokens", self.LOCAL_MAX_NEW_TOKENS_DEFAULT)
                or self.LOCAL_MAX_NEW_TOKENS_DEFAULT
            )
            force_no_think = bool(getattr(self.config, "agent_local_no_think", True))
            logger.info(
                "intent_recognition: use local qwen model=%s adapter=%s device=%s",
                model_name,
                adapter_path or "<none>",
                device,
            )
            return LocalQwenChatModel(
                base_model_path=model_name,
                adapter_path=adapter_path,
                device=device,
                max_new_tokens=max_new_tokens,
                force_no_think=force_no_think,
                config=self.config,
            )

        api_key = os.getenv("MOONSHOT_API_KEY", "").strip()
        if not api_key:
            return None
        return ChatOpenAI(
            model=model_name,
            temperature=0,
            max_tokens=self.REMOTE_MAX_TOKENS,
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1",
            timeout=self.REMOTE_TIMEOUT_SECONDS,
        )

    @staticmethod
    def _contains_complex_markers(query: str) -> bool:
        text = str(query or "")
        return any(token in text for token in IntentRecognitionAgent.COMPLEX_MARKERS)

    @staticmethod
    def _has_pronoun(query: str) -> bool:
        text = str(query or "")
        return any(token in text for token in IntentRecognitionAgent.PRONOUN_MARKERS)

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        raw = str(text or "").strip()
        if len(raw) > IntentRecognitionAgent.MAX_LLM_OUTPUT_CHARS:
            raw = raw[: IntentRecognitionAgent.MAX_LLM_OUTPUT_CHARS]
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception as exc:
            logger.debug("intent_recognition: json parse failed, fallback to regex: %s", exc)
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except Exception as exc:
            logger.debug("intent_recognition: regex json parse failed: %s", exc)
            return {}

    def _load_prompt_template(self) -> str:
        if self._prompt_template:
            return self._prompt_template
        try:
            text = self.PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8").strip()
            self._prompt_template = text or self.DEFAULT_PROMPT_TEMPLATE
        except Exception:
            self._prompt_template = self.DEFAULT_PROMPT_TEMPLATE
        return self._prompt_template

    def _build_prompt(self, query: str, memory_entities: List[str], available_tools: List[str]) -> str:
        template = self._load_prompt_template()
        return template.format(
            available_tools_json=json.dumps(available_tools, ensure_ascii=False),
            query=query,
            memory_entities_json=json.dumps(memory_entities, ensure_ascii=False),
        )

    def _validate_llm_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        try:
            validated = IntentLLMResponse.model_validate(payload)
        except ValidationError as exc:
            logger.debug("intent_recognition: llm payload validation failed: %s", exc)
            return {}
        return validated.model_dump()

    @staticmethod
    def _normalize_entity(text: str) -> str:
        candidate = str(text or "").strip()
        if not candidate:
            return ""
        candidate = re.sub(r"[，,。；;！？!?：:\n]", " ", candidate)
        candidate = re.split(r"并告诉我|并告诉|并且告诉我|并且告诉|并说|并且说|并问|并且问", candidate)[0].strip()
        candidate = re.split(r"以及|并且|并|同时|和|与", candidate)[0].strip()
        candidate = re.sub(r"^(给我|帮我|请你|请|麻烦你|麻烦)\s*", "", candidate)
        candidate = re.sub(r"^(介绍一下|介绍|查询一下|查询|查一下|查下|告诉我|说说|分析一下|分析|看看)\s*", "", candidate)
        candidate = re.sub(r"(现在|当前|目前)?(什么|多少)?(价格|价位|历史价格|历史|资料|信息)?$", "", candidate).strip()
        candidate = re.sub(r"\s+", " ", candidate).strip()
        if not (1 < len(candidate) <= IntentRecognitionAgent.MAX_ENTITY_CHARS):
            return ""
        if not re.search(r"[\u4e00-\u9fffA-Za-z]", candidate):
            return ""
        if re.fullmatch(r"[\d\-_.+()（）]+", candidate):
            return ""
        if (
            len(candidate) <= IntentRecognitionAgent.SHORT_ALNUM_ENTITY_MAX_CHARS
            and re.search(r"\d", candidate)
            and not re.search(r"[\u4e00-\u9fff]", candidate)
        ):
            return ""

        invalid_tokens = (
            "价格",
            "历史",
            "建议",
            "对比",
            "比较",
            "分析",
            "多少",
            "什么",
            "现在",
            "当前",
            "目前",
            "物品",
            "东西",
            "结论",
            "依据",
            "样本",
            "更新时间",
            "回撤",
            "风险",
            "利润",
            "区间",
            "可交易",
            "不要",
            "重仓",
            "追买",
            "分批",
            "小仓位",
            "止盈",
            "减仓",
            "外观",
            "外观描述",
            "描述",
            "关联关系",
            "关键依据",
        )
        pronoun_prefix = (
            "他",
            "她",
            "它",
            "这个",
            "那个",
            "这两个",
            "两个",
            "两者",
            "二者",
        )
        if candidate.startswith(pronoun_prefix):
            return ""
        # 排除描述句/规格字段，避免把事实句误判成实体名
        sentence_markers = ("是", "属于", "为", "位于", "包含", "包括", "相关", "对应")
        if any(marker in candidate for marker in sentence_markers):
            return ""
        if re.search(r"\d+\s*[xX×]\s*\d+|\d+\s*格", candidate):
            return ""
        if any(token in candidate for token in invalid_tokens):
            return ""
        return candidate

    def _extract_entities_from_query(self, query: str) -> List[str]:
        text = str(query or "").strip()
        if not text:
            return []

        parts = re.split(r"[，,、]|以及|并且|同时|和|与", text)
        candidates: List[str] = []
        for part in parts:
            entity = self._normalize_entity(part)
            if entity:
                candidates.append(entity)

        if not candidates:
            entity = self._normalize_entity(text)
            if entity:
                candidates.append(entity)

        dedup: List[str] = []
        for item in candidates:
            if item not in dedup:
                dedup.append(item)
        return dedup[:3]

    def _extract_recent_memory_entities(
        self,
        state: AgentState,
        max_items: int = RECENT_MEMORY_ENTITY_MAX_ITEMS,
    ) -> List[str]:
        found: List[str] = []

        def _append_entity(value: str):
            entity = self._normalize_entity(value)
            if not entity:
                return
            if entity in found:
                return
            found.append(entity)

        pending = [x for x in (state.get("memory_pending_buffer", []) or []) if isinstance(x, dict)]
        recent_raw = [x for x in (state.get("memory_recent_raw", []) or []) if isinstance(x, dict)]
        merged = pending + recent_raw
        for item in merged:
            if item.get("role") != "user":
                continue
            _append_entity(str(item.get("content", "")))
            if len(found) >= max_items:
                return found[-max_items:]

        # rolling summary fallback (line-based heuristic)
        summary = str(state.get("memory_rolling_summary", "") or "")
        if summary:
            for line in summary.splitlines():
                _append_entity(line)
                if len(found) >= max_items:
                    return found[-max_items:]

        # 先使用长期召回组件直接抽好的实体
        persistent_entities = [str(x).strip() for x in state.get("memory_persistent_entities", []) or []]
        for item in persistent_entities:
            _append_entity(item)
            if len(found) >= max_items:
                return found[-max_items:]

        # 再从长期召回命中里抽（兜底），避免“这两个物品”丢主体
        persistent_hits = [x for x in (state.get("memory_persistent_hits", []) or []) if isinstance(x, dict)]

        def _hit_priority(hit: Dict[str, Any]) -> int:
            fact_type = str(hit.get("fact_type", "") or "")
            fact_key = str(hit.get("fact_key", "") or "")
            if fact_type == "compare_target" or "compare" in fact_key:
                return 0
            if fact_type in {"focus", "entity"}:
                return 1
            if fact_type == "market":
                return 3
            return 2

        for item in sorted(persistent_hits, key=_hit_priority):
            raw = str(item.get("fact_value", "") or "").strip()
            if not raw:
                continue
            # 结构化分隔优先
            for token in re.split(r"[，,、/|；;：:\s]+", raw):
                _append_entity(token)
                if len(found) >= max_items:
                    return found[-max_items:]
            # 对比文本兜底：1. 非洲之心｜...
            for match in re.findall(r"\d+\.\s*([^\s｜|]+)", raw):
                _append_entity(match)
                if len(found) >= max_items:
                    return found[-max_items:]

        # 最后从拼接上下文里再兜一层
        memory_context = str(state.get("memory_context", "") or "")
        if memory_context:
            for line in memory_context.splitlines():
                _append_entity(line)
                if len(found) >= max_items:
                    return found[-max_items:]

        return found[-max_items:]

    @staticmethod
    def _infer_compare_target_count(query: str) -> int:
        text = str(query or "")
        if any(token in text for token in ("这三个", "三个物品", "三者", "3个", "三个")):
            return 3
        if any(token in text for token in ("这两个", "两个物品", "两者", "二者", "这俩", "2个", "两个")):
            return 2
        match = re.search(r"(\d+)\s*个", text)
        if match:
            try:
                value = int(match.group(1))
                if value >= 2:
                    return min(value, IntentRecognitionAgent.MAX_COMPARE_ENTITY_COUNT)
            except Exception as exc:
                logger.debug("intent_recognition: compare_target_count parse failed: %s", exc)
        return 2

    def _build_tool_query(self, tool_name: str, query: str, entities: List[str]) -> str:
        if tool_name in self.PRICE_TOOLS:
            if entities:
                return f"objectName={entities[0]}"
            return query

        if tool_name == "df_multi_item_compare":
            if len(entities) >= 2:
                return f"{'、'.join(entities)} 对比"
            if len(entities) == 1:
                return entities[0]
            return query

        if tool_name == "df_profit_stability":
            if entities:
                return f"objectName={entities[0]}"
            return query

        if tool_name == "rag_knowledge_search":
            if self._has_pronoun(query) and entities:
                return entities[0]
            return query

        if tool_name == "df_answer_composer":
            if self._has_pronoun(query) and entities:
                return f"{entities[0]}；{query}"
            return query

        return query

    def _should_use_task_planning(self, query: str, tool_name: str, entity_count: int, is_complex: bool) -> bool:
        if not bool(getattr(self.config, "task_planning_enabled", False)):
            return False
        if not is_complex:
            return False
        if tool_name in {"none", "df_answer_composer"}:
            return False
        text = str(query or "")
        has_markers = any(token in text for token in self.TASK_PLANNING_MARKERS)
        return has_markers or entity_count >= 2

    def _llm_understand(self, query: str, memory_entities: List[str]) -> Dict[str, Any]:
        if self.llm is None:
            return {}

        available_tools = self._available_tools_for_prompt()

        prompt = self._build_prompt(query=query, memory_entities=memory_entities, available_tools=available_tools)
        try:
            response = self.llm.invoke(prompt)
            text = extract_text_content(getattr(response, "content", response)).strip()
            payload = self._extract_json(text)
            return self._validate_llm_payload(payload)
        except Exception as exc:
            logger.warning("intent_recognition: llm_understand failed, fallback to rule: %s", exc)
            return {}

    async def _llm_understand_async(self, query: str, memory_entities: List[str]) -> Dict[str, Any]:
        if self.llm is None:
            return {}

        available_tools = self._available_tools_for_prompt()

        prompt = self._build_prompt(query=query, memory_entities=memory_entities, available_tools=available_tools)
        try:
            response = await self.llm.ainvoke(prompt)
            text = extract_text_content(getattr(response, "content", response)).strip()
            payload = self._extract_json(text)
            return self._validate_llm_payload(payload)
        except Exception as exc:
            logger.warning("intent_recognition: llm_understand failed, fallback to rule: %s", exc)
            return {}

    def _resolve_decision_from_payload(
        self,
        state: AgentState,
        query: str,
        llm_payload: Dict[str, Any],
        memory_entities: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        base = self.analyzer.analyze(query)
        memory_entities = memory_entities if memory_entities is not None else self._extract_recent_memory_entities(state)
        query_entities = self._extract_entities_from_query(query)
        compare_target_count = self._infer_compare_target_count(query)

        llm_tool = str(llm_payload.get("tool_name", "")).strip()

        if llm_tool and self._available_tool_set and llm_tool not in self._available_tool_set:
            llm_tool = ""

        tool_name = llm_tool or base.tool_name
        intent = str(llm_payload.get("intent", "")).strip() or base.intent
        reason = str(llm_payload.get("reason", "")).strip() or base.reason

        # 规则优先：制造台/特勤处制造类问题由 IntentAnalyzer 锁定到制造利润工具。
        if base.tool_name == "df_place_profit_rank":
            tool_name = base.tool_name
            intent = base.intent
            reason = f"{base.reason}(rule_preferred)"

        entities = []
        for item in llm_payload.get("entities", []) if isinstance(llm_payload.get("entities", []), list) else []:
            norm = self._normalize_entity(str(item))
            if norm and norm not in entities:
                entities.append(norm)

        if not entities:
            entities = list(query_entities)

        if self._has_pronoun(query) and not entities:
            if tool_name == "df_multi_item_compare":
                if len(memory_entities) >= compare_target_count:
                    entities = memory_entities[:compare_target_count]
                else:
                    entities = list(memory_entities)
            else:
                entities = memory_entities[-1:] if memory_entities else []

        if tool_name == "df_multi_item_compare":
            if len(entities) < 2 and len(memory_entities) >= 2:
                entities = memory_entities[:2]
            elif len(entities) >= 2 and compare_target_count > 2:
                merged = []
                for item in list(entities) + list(memory_entities):
                    if item and item not in merged:
                        merged.append(item)
                entities = merged[:compare_target_count]

        confidence = llm_payload.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except Exception as exc:
            logger.debug("intent_recognition: invalid confidence value, fallback to 0.0: %s", exc)
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        tool_query = self._build_tool_query(tool_name=tool_name, query=query, entities=entities)
        return {
            "intent": intent,
            "tool_name": tool_name,
            "reason": reason,
            "entities": entities[:3],
            "entity_count": len(entities[:3]),
            "confidence": confidence,
            "tool_query": tool_query,
            "compare_target_count": compare_target_count,
        }

    def _resolve_decision(self, state: AgentState, query: str) -> Dict[str, Any]:
        memory_entities = self._extract_recent_memory_entities(state)
        llm_payload = self._llm_understand(query=query, memory_entities=memory_entities)
        return self._resolve_decision_from_payload(
            state=state,
            query=query,
            llm_payload=llm_payload,
            memory_entities=memory_entities,
        )

    async def _resolve_decision_async(self, state: AgentState, query: str) -> Dict[str, Any]:
        memory_entities = self._extract_recent_memory_entities(state)
        llm_payload = await self._llm_understand_async(query=query, memory_entities=memory_entities)
        return self._resolve_decision_from_payload(
            state=state,
            query=query,
            llm_payload=llm_payload,
            memory_entities=memory_entities,
        )

    async def run(self, state: AgentState) -> Dict:
        node_started_perf = time.perf_counter()
        query = str(state.get("user_query", "") or "").strip()
        resolved = await self._resolve_decision_async(state=state, query=query)

        intent = resolved["intent"]
        tool_name = resolved["tool_name"]
        tool_query = resolved["tool_query"]
        reason = resolved["reason"]
        entities = resolved["entities"]
        entity_count = resolved["entity_count"]
        confidence = resolved["confidence"]
        compare_target_count = int(resolved.get("compare_target_count", 2) or 2)

        if self.skills_enabled and self.skill_registry is not None:
            skill_selection = self.skill_registry.select(
                query=query,
                intent=intent,
                tool_name=tool_name,
                entity_count=entity_count,
                compare_target_count=compare_target_count,
            )
            skill = self.skill_registry.get(skill_selection.skill_id)
            skill_plan = self.skill_registry.build_plan(
                selection=skill_selection,
                query=query,
                entities=entities,
                compare_target_count=compare_target_count,
                fallback_tool=tool_name,
                fallback_query=tool_query,
            )
        else:
            skill_selection = SkillSelection(
                skill_id="",
                score=0,
                confidence=0.0,
                reason="skills_disabled",
                matched_by=[],
            )
            skill = None
            fallback_calls = []
            if tool_name and tool_name != "none":
                fallback_calls = [{"tool_name": tool_name, "tool_query": tool_query}]
            skill_plan = SkillPlan(skill_id="", tool_calls=fallback_calls, locked_plan=False)

        tool_calls = skill_plan.tool_calls
        if not tool_calls and tool_name and tool_name != "none":
            tool_calls = [{"tool_name": tool_name, "tool_query": tool_query}]

        selected_tool = "none"
        selected_query = ""
        if tool_calls:
            selected_tool = str(tool_calls[0].get("tool_name", "") or "").strip() or "none"
            selected_query = str(tool_calls[0].get("tool_query", "") or "").strip()

        is_complex = self.analyzer.is_complex_intent(intent) or self._contains_complex_markers(query)
        flow_type = "complex" if is_complex else "simple"
        if skill and skill.flow_type in {"simple", "complex"}:
            flow_type = skill.flow_type

        base_task_planning = self._should_use_task_planning(
            query=query,
            tool_name=selected_tool,
            entity_count=entity_count,
            is_complex=is_complex,
        )
        requires_task_planning = (
            bool(getattr(self.config, "task_planning_enabled", False))
            and (base_task_planning or bool(skill.requires_task_planning if skill else False))
            and not bool(skill_plan.locked_plan)
        )
        requires_specialist = bool(skill.requires_specialist_analysis if skill else False) or intent in self.SPECIALIST_INTENTS

        task_plan = [] if selected_tool == "none" else tool_calls

        node_finished_perf = time.perf_counter()
        tool_selected_at_utc = datetime.now(timezone.utc).isoformat()
        orchestration_meta = dict(state.get("orchestration_meta", {}) or {})
        input_perf = float(orchestration_meta.get("input_received_perf", 0.0) or 0.0)
        latest_latency_ms = None
        if input_perf > 0:
            latest_latency_ms = round(
                (node_finished_perf - input_perf) * self.LATENCY_MS_MULTIPLIER,
                2,
            )
        orchestration_meta.update(
            {
                "intent_node_latency_ms": round(
                    (node_finished_perf - node_started_perf) * self.LATENCY_MS_MULTIPLIER,
                    2,
                ),
                "latest_tool_selected_at_utc": tool_selected_at_utc,
                "latest_selected_tool": selected_tool,
                "latest_selected_tool_query": selected_query,
            }
        )
        if latest_latency_ms is not None:
            orchestration_meta["latest_tool_selected_latency_ms"] = latest_latency_ms
        if "first_tool_selected_at_utc" not in orchestration_meta:
            orchestration_meta["first_tool_selected_at_utc"] = tool_selected_at_utc
            orchestration_meta["first_selected_tool"] = selected_tool
            orchestration_meta["first_selected_tool_query"] = selected_query
            if latest_latency_ms is not None:
                orchestration_meta["first_tool_selected_latency_ms"] = latest_latency_ms

        return {
            "intent": intent,
            "intent_reason": reason,
            "flow_type": flow_type,
            "plan_source": "skill_planning" if skill_selection.skill_id else "query_understanding",
            "requires_task_planning": requires_task_planning,
            "requires_specialist_analysis": requires_specialist,
            "selected_tool": selected_tool,
            "tool_query": selected_query,
            "task_plan": task_plan,
            "tool_calls": task_plan,
            "understanding_entities": entities,
            "understanding_entity_count": entity_count,
            "understanding_confidence": confidence,
            "understanding_compare_target_count": compare_target_count,
            "selected_skill": skill_selection.skill_id,
            "skill_reason": skill_selection.reason,
            "skill_confidence": skill_selection.confidence,
            "skill_matched_by": list(skill_selection.matched_by),
            "skill_locked_plan": bool(skill_plan.locked_plan),
            "skill_tool_chain": task_plan,
            "orchestration_meta": orchestration_meta,
            "force_reintent": False,
            "agent_messages": append_agent_message(
                state.get("agent_messages", []),
                from_agent="intent_recognition",
                to_agent="execution",
                message_type="intent_result",
                payload=build_intent_result_payload(
                    intent=intent,
                    tool_name=selected_tool,
                    flow_type=flow_type,
                    reason=reason,
                    entities=entities,
                    entity_count=entity_count,
                    confidence=confidence,
                    compare_target_count=compare_target_count,
                    skill_id=skill_selection.skill_id,
                    skill_confidence=skill_selection.confidence,
                    skill_reason=skill_selection.reason,
                    skill_matched_by=list(skill_selection.matched_by),
                ),
            ),
            "debug_steps": state.get("debug_steps", []) + [
                f"intent_recognition: {intent}/{flow_type}/skill={skill_selection.skill_id or 'none'}/tool={selected_tool}/entities={entity_count}/compare_n={compare_target_count}"
            ],
        }
