"""Long-term memory recall agent (Postgres + pgvector)."""

from __future__ import annotations

import re
from typing import Dict, List

from agents.state import AgentState
from memory.persistent_memory_store import PersistentMemoryStore, RecallResult


class PersistentMemoryRecallNode:
    """Conditionally recalls long-term memory and appends it to memory_context."""

    PRONOUN_MARKERS = ("它", "他", "她", "这两个", "这三个", "上次", "之前", "刚才", "那个")
    COMPLEX_MARKERS = ("对比", "比较", "建议", "风险", "历史", "趋势", "稳定", "回撤", "分析")

    def __init__(self, store: PersistentMemoryStore, config):
        self.store = store
        self.enabled = bool(getattr(config, "memory_persistent_enabled", False))
        self.threshold = int(getattr(config, "memory_persistent_trigger_threshold", 2) or 2)

    def _gate_score(self, query: str) -> int:
        text = str(query or "").strip()
        if not text:
            return 0
        score = 0
        if any(token in text for token in self.PRONOUN_MARKERS):
            score += 2
        if any(token in text for token in self.COMPLEX_MARKERS):
            score += 2
        if re.search(r"(上次|之前|历史|过去)", text):
            score += 1
        if re.search(r"\d+\s*个", text):
            score += 1
        return score

    def run(self, state: AgentState) -> Dict:
        if not self.enabled or not self.store.enabled:
            return {
                "debug_steps": state.get("debug_steps", []) + ["persistent_memory_recall: disabled"],
            }

        user_id = str(state.get("user_id", "default_user") or "default_user")
        session_id = str(state.get("session_id", "default") or "default")
        query = str(state.get("user_query", "") or "").strip()
        score = self._gate_score(query)

        recall_used = score >= self.threshold
        if recall_used:
            recall_result = self.store.recall(user_id=user_id, session_id=session_id, query=query)
        else:
            latest_summary = self.store.latest_summary(user_id=user_id, session_id=session_id)
            summary_context = f"[长期记忆摘要]\n{latest_summary}" if latest_summary else ""
            recall_result = RecallResult(
                context=summary_context,
                entities=[],
                hits=[],
                used=False,
                debug={
                    "enabled": True,
                    "user_id": user_id,
                    "latest_summary": bool(latest_summary),
                    "vector_hits": 0,
                    "bm25_hits": 0,
                    "fused_hits": 0,
                },
            )

        context_parts: List[str] = []
        existing_context = str(state.get("memory_context", "") or "").strip()
        if existing_context:
            context_parts.append(existing_context)
        if recall_result.context:
            context_parts.append(recall_result.context)

        merged_context = "\n\n".join([x for x in context_parts if x]).strip()
        message = {
            "from_agent": "persistent_memory_recall",
            "to_agent": "intent_recognition",
            "message_type": "persistent_recall",
            "payload": {
                "gate_score": score,
                "used_recall": recall_used,
                "hit_count": len(recall_result.hits),
                "debug": recall_result.debug,
            },
        }

        return {
            "memory_context": merged_context,
            "memory_persistent_context": recall_result.context,
            "memory_persistent_entities": recall_result.entities,
            "memory_persistent_hits": recall_result.hits,
            "memory_persistent_used": recall_used,
            "memory_persistent_gate_score": score,
            "agent_messages": state.get("agent_messages", []) + [message],
            "debug_steps": state.get("debug_steps", [])
            + [f"persistent_memory_recall: gate={score}, used={recall_used}, hits={len(recall_result.hits)}"],
        }
