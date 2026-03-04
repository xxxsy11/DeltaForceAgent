"""Memory compression agent for multi-turn session memory."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI

from agents.state import AgentState
from rag_modules.llm_utils import extract_text_content


DEFAULT_MARKET_TTL_HOURS = 24
DEFAULT_RECENT_RAW_LIMIT = 5
DEFAULT_PENDING_TURNS_TRIGGER = 2
DEFAULT_PENDING_TOKENS_TRIGGER = 500
DEFAULT_SUMMARY_MAX_TOKENS = 400
DEFAULT_REBASE_MERGES = 5
TOKEN_ESTIMATE_CHAR_DIVISOR = 2
SUMMARY_KEYWORD_DEFAULT_LIMIT = 12
SUMMARY_KEYWORD_MAX_CHARS = 32
FORMAT_TURNS_MAX_ITEMS = 16
HEURISTIC_SUMMARY_PENDING_TURNS = 8
HEURISTIC_SUMMARY_LINE_MAX_CHARS = 120
HEURISTIC_SUMMARY_TOTAL_MAX_CHARS = 1200
PENDING_DIGEST_MAX_LINES = 4
PENDING_DIGEST_LINE_MAX_CHARS = 100
RECENT_LINES_DEFAULT_MAX_ITEMS = 5
HEURISTIC_ENTITY_MIN_CHARS = 2
HEURISTIC_ENTITY_MAX_CHARS = 24
HEURISTIC_ENTITY_MAX_ITEMS = 8
QUERY_KEYWORD_REGEX_MAX_ITEMS = 8
FACT_VALUE_MAX_CHARS = 500
FACTS_MAX_ITEMS = 6
FACT_KEY_MAX_CHARS = 80
FACT_TYPE_MAX_CHARS = 32
FACT_KEYWORD_MAX_ITEMS = 10
FACT_DEFAULT_CONFIDENCE = 0.7
FACT_FOCUS_ENTITY_COUNT = 3
FACT_FOCUS_CONFIDENCE = 0.85
FACT_COMPARE_CONFIDENCE = 0.8
FACT_SUMMARY_CONFIDENCE = 0.75
FACT_MARKET_CONFIDENCE = 0.8
LLM_TIMEOUT_SECONDS = 60


class MemoryCompressionAgent:
    """Updates recent/pending buffers and compresses pending history when needed."""

    FAILURE_MARKERS = ("工具调用失败", "查询失败", "未找到工具", "系统错误", "不可用", "HTTP 5")

    def __init__(self, config):
        self.config = config
        self.enabled = bool(getattr(config, "memory_enabled", True))
        self.persistent_enabled = bool(getattr(config, "memory_persistent_enabled", False))
        self.recent_raw_limit = int(getattr(config, "memory_recent_raw_limit", DEFAULT_RECENT_RAW_LIMIT))
        self.pending_turns_trigger = int(
            getattr(config, "memory_pending_turns_trigger", DEFAULT_PENDING_TURNS_TRIGGER)
        )
        self.pending_tokens_trigger = int(
            getattr(config, "memory_pending_tokens_trigger", DEFAULT_PENDING_TOKENS_TRIGGER)
        )
        self.summary_max_tokens = int(getattr(config, "memory_summary_max_tokens", DEFAULT_SUMMARY_MAX_TOKENS))
        self.rebase_every_n_merges = int(getattr(config, "memory_rebase_every_n_merges", DEFAULT_REBASE_MERGES))
        self.drop_failed_tool_messages = bool(getattr(config, "memory_drop_failed_tool_messages", True))
        self.market_ttl_hours = int(
            getattr(config, "memory_persistent_market_ttl_hours", DEFAULT_MARKET_TTL_HOURS)
            or DEFAULT_MARKET_TTL_HOURS
        )
        self.model_name = str(getattr(config, "agent_memory_model", getattr(config, "llm_model", ""))).strip()
        self.llm = self._build_llm()

    def _build_llm(self):
        api_key = os.getenv("MOONSHOT_API_KEY", "").strip()
        if not api_key or not self.model_name:
            return None
        return ChatOpenAI(
            model=self.model_name,
            temperature=0,
            max_tokens=self.summary_max_tokens,
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1",
            timeout=LLM_TIMEOUT_SECONDS,
        )

    async def _invoke_llm_async(self, prompt: str):
        if self.llm is None:
            return None
        return await self.llm.ainvoke(prompt)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(str(text or "")) // TOKEN_ESTIMATE_CHAR_DIVISOR)

    @classmethod
    def _contains_failure_markers(cls, text: str) -> bool:
        raw = str(text or "")
        return any(marker in raw for marker in cls.FAILURE_MARKERS)

    def _estimate_pending_tokens(self, pending: List[Dict[str, str]]) -> int:
        total = 0
        for item in pending:
            total += self._estimate_tokens(item.get("content", ""))
        return total

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        raw = str(text or "").strip()
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

    @staticmethod
    def _normalize_keywords(items: List[Any], limit: int = SUMMARY_KEYWORD_DEFAULT_LIMIT) -> List[str]:
        keywords: List[str] = []
        for item in items or []:
            text = str(item or "").strip()
            if not text or len(text) > SUMMARY_KEYWORD_MAX_CHARS:
                continue
            if text in keywords:
                continue
            keywords.append(text)
            if len(keywords) >= limit:
                break
        return keywords

    @staticmethod
    def _format_turns(turns: List[Dict[str, str]], max_items: int = FORMAT_TURNS_MAX_ITEMS) -> str:
        lines: List[str] = []
        for item in turns[-max_items:]:
            role = "user" if item.get("role") == "user" else "assistant"
            content = str(item.get("content", "")).strip().replace("\n", " ")
            lines.append(f"[{role}] {content}")
        return "\n".join(lines)

    def _heuristic_summary(self, old_summary: str, pending: List[Dict[str, str]]) -> str:
        old = str(old_summary or "").strip()
        lines = []
        if old:
            lines.append(old)
        for item in pending[-HEURISTIC_SUMMARY_PENDING_TURNS:]:
            role = "用户" if item.get("role") == "user" else "助手"
            content = str(item.get("content", "")).strip().replace("\n", " ")
            if len(content) > HEURISTIC_SUMMARY_LINE_MAX_CHARS:
                content = content[:HEURISTIC_SUMMARY_LINE_MAX_CHARS] + "..."
            lines.append(f"{role}: {content}")
        merged = "\n".join(lines).strip()
        if len(merged) > HEURISTIC_SUMMARY_TOTAL_MAX_CHARS:
            return merged[-HEURISTIC_SUMMARY_TOTAL_MAX_CHARS:]
        return merged

    def _llm_merge_summary(self, old_summary: str, pending: List[Dict[str, str]]) -> str:
        if self.llm is None:
            return self._heuristic_summary(old_summary=old_summary, pending=pending)

        prompt = f"""
你是对话记忆压缩代理。请将旧摘要与新增对话压缩成新的记忆摘要。
要求：
1) 输出 JSON，字段：summary。
2) summary 控制在 8~12 行，保留用户目标、关键事实、未完成任务。
3) 标注不确定信息，不要把失败报错当成事实。
4) 不要输出多余解释。

旧摘要：
{old_summary}

新增对话：
{self._format_turns(pending)}
"""
        try:
            response = self.llm.invoke(prompt)
            text = extract_text_content(getattr(response, "content", response)).strip()
            parsed = self._parse_json(text)
            summary = str(parsed.get("summary", "")).strip()
            if summary:
                return summary
        except Exception:
            pass
        return self._heuristic_summary(old_summary=old_summary, pending=pending)

    async def _llm_merge_summary_async(self, old_summary: str, pending: List[Dict[str, str]]) -> str:
        if self.llm is None:
            return self._heuristic_summary(old_summary=old_summary, pending=pending)

        prompt = f"""
你是对话记忆压缩代理。请将旧摘要与新增对话压缩成新的记忆摘要。
要求：
1) 输出 JSON，字段：summary。
2) summary 控制在 8~12 行，保留用户目标、关键事实、未完成任务。
3) 标注不确定信息，不要把失败报错当成事实。
4) 不要输出多余解释。

旧摘要：
{old_summary}

新增对话：
{self._format_turns(pending)}
"""
        try:
            response = await self._invoke_llm_async(prompt)
            text = extract_text_content(getattr(response, "content", response)).strip()
            parsed = self._parse_json(text)
            summary = str(parsed.get("summary", "")).strip()
            if summary:
                return summary
        except Exception:
            pass
        return self._heuristic_summary(old_summary=old_summary, pending=pending)

    def _llm_rebase_summary(self, rolling_summary: str) -> str:
        if self.llm is None:
            return str(rolling_summary or "").strip()
        prompt = f"""
请将下面对话摘要进行一次重整，去重并压缩，保持事实准确、结构清晰。
输出 JSON: {{"summary": "..."}}

摘要内容：
{rolling_summary}
"""
        try:
            response = self.llm.invoke(prompt)
            text = extract_text_content(getattr(response, "content", response)).strip()
            parsed = self._parse_json(text)
            summary = str(parsed.get("summary", "")).strip()
            if summary:
                return summary
        except Exception:
            pass
        return str(rolling_summary or "").strip()

    async def _llm_rebase_summary_async(self, rolling_summary: str) -> str:
        if self.llm is None:
            return str(rolling_summary or "").strip()
        prompt = f"""
请将下面对话摘要进行一次重整，去重并压缩，保持事实准确、结构清晰。
输出 JSON: {{"summary": "..."}}

摘要内容：
{rolling_summary}
"""
        try:
            response = await self._invoke_llm_async(prompt)
            text = extract_text_content(getattr(response, "content", response)).strip()
            parsed = self._parse_json(text)
            summary = str(parsed.get("summary", "")).strip()
            if summary:
                return summary
        except Exception:
            pass
        return str(rolling_summary or "").strip()

    @staticmethod
    def _build_pending_digest(pending: List[Dict[str, str]], max_lines: int = PENDING_DIGEST_MAX_LINES) -> str:
        if not pending:
            return ""
        lines = []
        for item in pending[-max_lines:]:
            role = "用户" if item.get("role") == "user" else "助手"
            content = str(item.get("content", "")).strip().replace("\n", " ")
            if len(content) > PENDING_DIGEST_LINE_MAX_CHARS:
                content = content[:PENDING_DIGEST_LINE_MAX_CHARS] + "..."
            lines.append(f"- {role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _build_recent_lines(
        recent: List[Dict[str, str]],
        max_items: int = RECENT_LINES_DEFAULT_MAX_ITEMS,
    ) -> str:
        lines = []
        for item in recent[-max_items:]:
            role = "用户" if item.get("role") == "user" else "助手"
            lines.append(f"- {role}: {str(item.get('content', '')).strip()}")
        return "\n".join(lines)

    def _heuristic_extract_facts(
        self,
        state: AgentState,
        summary_text: str,
        pending_turns: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        def _extract_entity_candidates(raw_text: str) -> List[str]:
            text = str(raw_text or "")
            if not text:
                return []

            candidates: List[str] = []
            # 对比输出优先：1. 非洲之心｜...
            for match in re.findall(r"\d+\.\s*([^\s｜|]{2,24})", text):
                candidates.append(match.strip())

            for token in re.split(r"[，,、/|；;：:\n\s]+", text):
                item = str(token or "").strip()
                if len(item) < HEURISTIC_ENTITY_MIN_CHARS or len(item) > HEURISTIC_ENTITY_MAX_CHARS:
                    continue
                if re.search(r"\d+\s*[xX×]\s*\d+|\d+\s*格", item):
                    continue
                if any(
                    x in item
                    for x in (
                        "价格", "建议", "历史", "区间", "样本", "更新时间", "结论", "依据", "风险", "回撤", "利润", "分析", "对比", "比较", "可交易", "summary", "http",
                    )
                ):
                    continue
                if item in {"用户", "助手", "工具", "最新", "当前", "现在", "买入", "卖出", "特勤处"}:
                    continue
                candidates.append(item)

            dedup: List[str] = []
            for item in candidates:
                if item not in dedup:
                    dedup.append(item)
            return dedup[:HEURISTIC_ENTITY_MAX_ITEMS]

        entities = [str(x).strip() for x in state.get("understanding_entities", []) if str(x).strip()]
        query = str(state.get("user_query", "") or "").strip()
        final_answer = str(state.get("final_answer", "") or "").strip()
        pending_text = "\n".join(
            str(x.get("content", "") or "")
            for x in pending_turns[-SUMMARY_KEYWORD_DEFAULT_LIMIT:]
        )


        entity_pool: List[str] = []
        for item in entities + _extract_entity_candidates(query) + _extract_entity_candidates(final_answer) + _extract_entity_candidates(pending_text) + _extract_entity_candidates(summary_text):
            if item not in entity_pool:
                entity_pool.append(item)

        keyword_candidates = entity_pool + re.findall(r"[\u4e00-\u9fff]{2,8}", query)[:QUERY_KEYWORD_REGEX_MAX_ITEMS]
        keywords = self._normalize_keywords(keyword_candidates, limit=SUMMARY_KEYWORD_DEFAULT_LIMIT)

        facts: List[Dict[str, Any]] = []
        if entity_pool:
            facts.append(
                {
                    "fact_key": "last_focus_items",
                    "fact_value": "、".join(entity_pool[:FACT_FOCUS_ENTITY_COUNT]),
                    "fact_type": "focus",
                    "confidence": FACT_FOCUS_CONFIDENCE,
                    "keywords": entity_pool[:FACT_FOCUS_ENTITY_COUNT],
                    "ttl_hours": None,
                }
            )
        if len(entity_pool) >= 2:
            facts.append(
                {
                    "fact_key": "compare_target_items",
                    "fact_value": "、".join(entity_pool[:FACT_FOCUS_ENTITY_COUNT]),
                    "fact_type": "compare_target",
                    "confidence": FACT_COMPARE_CONFIDENCE,
                    "keywords": entity_pool[:FACT_FOCUS_ENTITY_COUNT],
                    "ttl_hours": None,
                }
            )

        if summary_text:
            facts.append(
                {
                    "fact_key": "compressed_memory_summary",
                    "fact_value": summary_text[:FACT_VALUE_MAX_CHARS],
                    "fact_type": "plan",
                    "confidence": FACT_SUMMARY_CONFIDENCE,
                    "keywords": keywords[:QUERY_KEYWORD_REGEX_MAX_ITEMS],
                    "ttl_hours": None,
                }
            )

        combined_text = f"{query}\n{final_answer}\n{pending_text}"
        if "价格" in combined_text:
            market_text = final_answer if final_answer and not self._contains_failure_markers(final_answer) else summary_text
            if market_text:
                facts.append(
                    {
                        "fact_key": "latest_price_observation",
                        "fact_value": market_text[:FACT_VALUE_MAX_CHARS],
                        "fact_type": "market",
                        "confidence": FACT_MARKET_CONFIDENCE,
                        "keywords": keywords[:QUERY_KEYWORD_REGEX_MAX_ITEMS],
                        "ttl_hours": self.market_ttl_hours,
                    }
                )
        return {"keywords": keywords, "facts": facts[:FACTS_MAX_ITEMS]}

    def _llm_extract_facts(
        self,
        state: AgentState,
        summary_text: str,
        pending_turns: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        if self.llm is None:
            return {}
        query = str(state.get("user_query", "") or "").strip()
        final_answer = str(state.get("final_answer", "") or "").strip()
        pending_text = "\n".join(
            str(x.get("content", "") or "")
            for x in pending_turns[-SUMMARY_KEYWORD_DEFAULT_LIMIT:]
        )
        pending_text = self._format_turns(pending_turns)

        prompt = f"""
你是记忆压缩后的事实提取器。请从摘要与新增对话中提取可复用记忆。
输出 JSON：
{{
  "keywords": ["..."],
  "facts": [
    {{
      "fact_key": "snake_case_key",
      "fact_value": "事实内容",
      "fact_type": "focus|entity|market|preference|constraint|plan",
      "confidence": 0.0,
      "keywords": ["..."],
      "ttl_hours": 24
    }}
  ]
}}

规则：
1) 仅提取可复用事实，最多 6 条。
2) 报错信息不要写入 facts。
3) market 类型默认 ttl_hours={self.market_ttl_hours}。
4) keywords 与 facts 分别返回，keywords 是检索词，facts 是结构化记忆。

用户问题：{query}
助手回答：{final_answer}
压缩后摘要：{summary_text}
压缩源对话：{pending_text}
"""
        try:
            response = self.llm.invoke(prompt)
            text = extract_text_content(getattr(response, "content", response)).strip()
            return self._parse_json(text)
        except Exception:
            return {}

    async def _llm_extract_facts_async(
        self,
        state: AgentState,
        summary_text: str,
        pending_turns: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        if self.llm is None:
            return {}
        query = str(state.get("user_query", "") or "").strip()
        final_answer = str(state.get("final_answer", "") or "").strip()
        pending_text = "\n".join(
            str(x.get("content", "") or "")
            for x in pending_turns[-SUMMARY_KEYWORD_DEFAULT_LIMIT:]
        )
        pending_text = self._format_turns(pending_turns)

        prompt = f"""
你是记忆压缩后的事实提取器。请从摘要与新增对话中提取可复用记忆。
输出 JSON：
{{
  "keywords": ["..."],
  "facts": [
    {{
      "fact_key": "snake_case_key",
      "fact_value": "事实内容",
      "fact_type": "focus|entity|market|preference|constraint|plan",
      "confidence": 0.0,
      "keywords": ["..."],
      "ttl_hours": 24
    }}
  ]
}}

规则：
1) 仅提取可复用事实，最多 6 条。
2) 报错信息不要写入 facts。
3) market 类型默认 ttl_hours={self.market_ttl_hours}。
4) keywords 与 facts 分别返回，keywords 是检索词，facts 是结构化记忆。

用户问题：{query}
助手回答：{final_answer}
压缩后摘要：{summary_text}
压缩源对话：{pending_text}
"""
        try:
            response = await self._invoke_llm_async(prompt)
            text = extract_text_content(getattr(response, "content", response)).strip()
            return self._parse_json(text)
        except Exception:
            return {}

    def _extract_facts_on_compression(
        self,
        state: AgentState,
        summary_text: str,
        pending_turns: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        if not self.persistent_enabled:
            return {"keywords": [], "facts": []}

        final_answer = str(state.get("final_answer", "") or "").strip()
        pending_text = "\n".join(
            str(x.get("content", "") or "")
            for x in pending_turns[-SUMMARY_KEYWORD_DEFAULT_LIMIT:]
        )
        allow_llm = bool(final_answer) and (not self._contains_failure_markers(final_answer))

        heuristic_payload = self._heuristic_extract_facts(
            state=state,
            summary_text=summary_text,
            pending_turns=pending_turns,
        )
        llm_payload: Dict[str, Any] = {}
        if allow_llm:
            llm_payload = self._llm_extract_facts(
                state=state,
                summary_text=summary_text,
                pending_turns=pending_turns,
            )

        merged_keywords = self._normalize_keywords(
            (llm_payload.get("keywords", []) if isinstance(llm_payload, dict) else [])
            + (heuristic_payload.get("keywords", []) if isinstance(heuristic_payload, dict) else []),
            limit=12,
        )

        raw_facts: List[Any] = []
        if isinstance(llm_payload, dict) and isinstance(llm_payload.get("facts", []), list):
            raw_facts.extend(llm_payload.get("facts", []))
        if isinstance(heuristic_payload, dict) and isinstance(heuristic_payload.get("facts", []), list):
            raw_facts.extend(heuristic_payload.get("facts", []))

        facts: List[Dict[str, Any]] = []
        seen = set()
        for item in raw_facts:
            if not isinstance(item, dict):
                continue
            fact_key = str(item.get("fact_key", "") or "").strip()
            fact_value = str(item.get("fact_value", "") or "").strip()
            if not fact_key or not fact_value:
                continue
            dedup_key = (fact_key, fact_value)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            fact_type = str(item.get("fact_type", "focus") or "focus").strip()
            try:
                confidence = float(item.get("confidence", FACT_DEFAULT_CONFIDENCE))
            except Exception:
                confidence = FACT_DEFAULT_CONFIDENCE
            ttl_hours = item.get("ttl_hours", None)
            if fact_type == "market" and not ttl_hours:
                ttl_hours = self.market_ttl_hours

            facts.append(
                {
                    "fact_key": fact_key[:FACT_KEY_MAX_CHARS],
                    "fact_value": fact_value[:FACT_VALUE_MAX_CHARS],
                    "fact_type": fact_type[:FACT_TYPE_MAX_CHARS],
                    "confidence": max(0.0, min(1.0, confidence)),
                    "keywords": self._normalize_keywords(item.get("keywords", []), limit=FACT_KEYWORD_MAX_ITEMS),
                    "ttl_hours": int(ttl_hours) if isinstance(ttl_hours, int) or str(ttl_hours).isdigit() else None,
                }
            )
            if len(facts) >= FACTS_MAX_ITEMS:
                break

        return {"keywords": merged_keywords, "facts": facts}

    async def _extract_facts_on_compression_async(
        self,
        state: AgentState,
        summary_text: str,
        pending_turns: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        if not self.persistent_enabled:
            return {"keywords": [], "facts": []}

        final_answer = str(state.get("final_answer", "") or "").strip()
        pending_text = "\n".join(
            str(x.get("content", "") or "")
            for x in pending_turns[-SUMMARY_KEYWORD_DEFAULT_LIMIT:]
        )
        allow_llm = bool(final_answer) and (not self._contains_failure_markers(final_answer))

        heuristic_payload = self._heuristic_extract_facts(
            state=state,
            summary_text=summary_text,
            pending_turns=pending_turns,
        )
        llm_payload: Dict[str, Any] = {}
        if allow_llm:
            llm_payload = await self._llm_extract_facts_async(
                state=state,
                summary_text=summary_text,
                pending_turns=pending_turns,
            )

        merged_keywords = self._normalize_keywords(
            (llm_payload.get("keywords", []) if isinstance(llm_payload, dict) else [])
            + (heuristic_payload.get("keywords", []) if isinstance(heuristic_payload, dict) else []),
            limit=12,
        )

        raw_facts: List[Any] = []
        if isinstance(llm_payload, dict) and isinstance(llm_payload.get("facts", []), list):
            raw_facts.extend(llm_payload.get("facts", []))
        if isinstance(heuristic_payload, dict) and isinstance(heuristic_payload.get("facts", []), list):
            raw_facts.extend(heuristic_payload.get("facts", []))

        facts: List[Dict[str, Any]] = []
        seen = set()
        for item in raw_facts:
            if not isinstance(item, dict):
                continue
            fact_key = str(item.get("fact_key", "") or "").strip()
            fact_value = str(item.get("fact_value", "") or "").strip()
            if not fact_key or not fact_value:
                continue
            dedup_key = (fact_key, fact_value)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            fact_type = str(item.get("fact_type", "focus") or "focus").strip()
            try:
                confidence = float(item.get("confidence", FACT_DEFAULT_CONFIDENCE))
            except Exception:
                confidence = FACT_DEFAULT_CONFIDENCE
            ttl_hours = item.get("ttl_hours", None)
            if fact_type == "market" and not ttl_hours:
                ttl_hours = self.market_ttl_hours

            facts.append(
                {
                    "fact_key": fact_key[:FACT_KEY_MAX_CHARS],
                    "fact_value": fact_value[:FACT_VALUE_MAX_CHARS],
                    "fact_type": fact_type[:FACT_TYPE_MAX_CHARS],
                    "confidence": max(0.0, min(1.0, confidence)),
                    "keywords": self._normalize_keywords(item.get("keywords", []), limit=FACT_KEYWORD_MAX_ITEMS),
                    "ttl_hours": int(ttl_hours) if isinstance(ttl_hours, int) or str(ttl_hours).isdigit() else None,
                }
            )
            if len(facts) >= FACTS_MAX_ITEMS:
                break

        return {"keywords": merged_keywords, "facts": facts}

    async def run(self, state: AgentState) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "debug_steps": state.get("debug_steps", []) + ["memory_compression: disabled"],
            }

        recent = [dict(x) for x in state.get("memory_recent_raw", []) if isinstance(x, dict)]
        pending = [dict(x) for x in state.get("memory_pending_buffer", []) if isinstance(x, dict)]
        rolling_summary = str(state.get("memory_rolling_summary", "") or "")
        merge_count = int(state.get("memory_merge_count", 0) or 0)

        user_query = str(state.get("user_query", "")).strip()
        final_answer = str(state.get("final_answer", "")).strip()

        if user_query:
            recent.append({"role": "user", "content": user_query})

        if final_answer and not (self.drop_failed_tool_messages and self._contains_failure_markers(final_answer)):
            recent.append({"role": "assistant", "content": final_answer})

        while len(recent) > max(1, self.recent_raw_limit):
            pending.append(recent.pop(0))

        pending_tokens = self._estimate_pending_tokens(pending)
        force_compress = bool(state.get("memory_force_compress", False))
        should_compress = (
            len(pending) >= max(1, self.pending_turns_trigger)
            or pending_tokens >= max(1, self.pending_tokens_trigger)
            or (force_compress and bool(pending))
        )

        compression_info = {
            "triggered": False,
            "reason": "",
            "pending_turns": len(pending),
            "pending_tokens": pending_tokens,
        }
        keyword_candidates: List[str] = []
        fact_candidates: List[Dict[str, Any]] = []

        if should_compress and pending:
            pending_snapshot = [dict(x) for x in pending]
            rolling_summary = await self._llm_merge_summary_async(old_summary=rolling_summary, pending=pending_snapshot)
            extraction_payload = await self._extract_facts_on_compression_async(
                state=state,
                summary_text=rolling_summary,
                pending_turns=pending_snapshot,
            )
            keyword_candidates = extraction_payload.get("keywords", []) if isinstance(extraction_payload, dict) else []
            fact_candidates = extraction_payload.get("facts", []) if isinstance(extraction_payload, dict) else []
            pending = []
            merge_count += 1
            compression_info = {
                "triggered": True,
                "reason": "force_flush" if force_compress else "turn_or_token_threshold",
                "pending_turns": 0,
                "pending_tokens": 0,
                "fact_count": len(fact_candidates),
            }

            if self.rebase_every_n_merges > 0 and merge_count % self.rebase_every_n_merges == 0:
                rolling_summary = await self._llm_rebase_summary_async(rolling_summary)
                compression_info["rebase"] = True
            else:
                compression_info["rebase"] = False

        pending_digest = self._build_pending_digest(pending)
        context_blocks: List[str] = []
        if rolling_summary:
            context_blocks.append(f"[历史摘要]\n{rolling_summary}")
        if pending_digest:
            context_blocks.append(f"[待压缩摘要]\n{pending_digest}")
        if recent:
            context_blocks.append("[最近对话]\n" + self._build_recent_lines(recent, max_items=self.recent_raw_limit))

        memory_context = "\n\n".join(context_blocks).strip()
        msg = {
            "from_agent": "memory_compression",
            "to_agent": "main_orchestrator",
            "message_type": "memory_update",
            "payload": {
                "compression": compression_info,
                "recent_raw_count": len(recent),
                "fact_count": len(fact_candidates),
            },
        }

        return {
            "memory_recent_raw": recent,
            "memory_pending_buffer": pending,
            "memory_rolling_summary": rolling_summary,
            "memory_merge_count": merge_count,
            "memory_pending_digest": pending_digest,
            "memory_context": memory_context,
            "memory_keyword_candidates": keyword_candidates,
            "memory_fact_candidates": fact_candidates,
            "agent_messages": state.get("agent_messages", []) + [msg],
            "debug_steps": state.get("debug_steps", []) + [
                f"memory_compression: triggered={compression_info.get('triggered', False)},facts={len(fact_candidates)}"
            ],
        }
