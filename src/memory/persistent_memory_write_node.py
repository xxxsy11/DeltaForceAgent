"""Long-term memory write agent (full turns + extracted facts)."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from agents.state import AgentState
from memory.persistent import PersistentMemoryStore
from observability.langsmith import langsmith_trace

logger = logging.getLogger(__name__)

DEFAULT_MARKET_TTL_HOURS = 24
DEFAULT_LOCAL_OBSERVER_DIR = str(Path(__file__).resolve().parents[2] / "data" / "memory" / "readable")
TOOL_OUTPUT_PREVIEW_MAX_CHARS = 400
FACT_DEFAULT_CONFIDENCE = 0.7

class PersistentMemoryWriteNode:
    """Writes current turn and extracted facts into PostgreSQL long-term memory."""

    def __init__(self, store: PersistentMemoryStore, config):
        self.store = store
        self.config = config
        self.enabled = bool(getattr(config, "memory_persistent_enabled", False))
        self.default_market_ttl_hours = int(
            getattr(config, "memory_persistent_market_ttl_hours", DEFAULT_MARKET_TTL_HOURS)
            or DEFAULT_MARKET_TTL_HOURS
        )
        self.local_observer_enabled = bool(getattr(config, "memory_local_observer_enabled", True))
        self.local_observer_dir = str(
            getattr(config, "memory_local_observer_dir", DEFAULT_LOCAL_OBSERVER_DIR) or ""
        ).strip()

    @staticmethod
    def _now_utc() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _safe_session_id(session_id: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(session_id or "default"))
        return safe or "default"

    def _append_local_observer(self, event: Dict[str, Any], result: Dict[str, Any]) -> None:
        if not self.local_observer_enabled or not self.local_observer_dir:
            return
        user_id = self._safe_session_id(str(event.get("user_id", "default_user") or "default_user"))
        session_id = self._safe_session_id(str(event.get("session_id", "default") or "default"))
        session_dir = Path(self.local_observer_dir) / user_id / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        ts = self._now_utc()
        user_query = str(event.get("user_query", "") or "").strip()
        final_answer = str(event.get("final_answer", "") or "").strip()
        tool_results = event.get("tool_results", []) or []
        facts = event.get("facts", []) or []
        compression_triggered = bool(event.get("compression_triggered", False))
        merge_count = int(event.get("merge_count", 0) or 0)
        rolling_summary = str(event.get("rolling_summary", "") or "").strip()

        record = {
            "timestamp_utc": ts,
            "user_id": user_id,
            "session_id": session_id,
            "user_query": user_query,
            "assistant_answer": final_answer,
            "tool_results": tool_results,
            "compression_triggered": compression_triggered,
            "merge_count": merge_count,
            "summary_saved": bool(result.get("summary_saved", False)),
            "facts_saved": int(result.get("fact_saved", 0) or 0),
            "facts": facts,
            "rolling_summary": rolling_summary if compression_triggered else "",
        }
        with (session_dir / "memory_events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        md_lines: List[str] = []
        md_lines.append(f"## {ts}")
        if user_query:
            md_lines.append(f"- 用户：{user_query}")
        if final_answer:
            md_lines.append(f"- 助手：{final_answer}")
        if tool_results:
            md_lines.append("- 工具输出：")
            for item in tool_results:
                name = str(item.get("tool_name", "") or "").strip()
                output = str(item.get("output", "") or "").strip()
                md_lines.append(f"  - {name}: {output[:TOOL_OUTPUT_PREVIEW_MAX_CHARS]}")
        if compression_triggered:
            md_lines.append(f"- 记忆压缩：已触发（merge_count={merge_count}）")
            if rolling_summary:
                md_lines.append("- 摘要：")
                md_lines.append("```text")
                md_lines.append(rolling_summary)
                md_lines.append("```")
        if facts:
            md_lines.append(f"- 提取 facts：{len(facts)} 条")
            for fact in facts:
                if not isinstance(fact, dict):
                    continue
                md_lines.append(
                    f"  - [{fact.get('fact_type', 'fact')}] {fact.get('fact_key', '')}: {fact.get('fact_value', '')}"
                )
        md_lines.append("")
        with (session_dir / "memory_readable.md").open("a", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

    @staticmethod
    def _find_compression_info(state: AgentState) -> Dict[str, Any]:
        for msg in reversed(state.get("agent_messages", []) or []):
            if msg.get("message_type") != "memory_update":
                continue
            payload = msg.get("payload")
            if isinstance(payload, dict):
                compression = payload.get("compression")
                if isinstance(compression, dict):
                    return compression
        return {}

    def _build_event(self, state: AgentState, compression: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "user_id": str(state.get("user_id", "default_user") or "default_user"),
            "session_id": str(state.get("session_id", "default") or "default"),
            "user_query": str(state.get("user_query", "") or "").strip(),
            "final_answer": str(state.get("final_answer", "") or "").strip(),
            "tool_results": state.get("tool_results", []) or [],
            "compression_triggered": bool(compression.get("triggered")),
            "merge_count": int(state.get("memory_merge_count", 0) or 0),
            "rolling_summary": str(state.get("memory_rolling_summary", "") or "").strip(),
            "facts": state.get("memory_fact_candidates", []) or [],
        }

    async def _persist_event_async(self, event: Dict[str, Any]) -> Dict[str, Any]:
        user_id = str(event.get("user_id", "default_user") or "default_user")
        session_id = str(event.get("session_id", "default") or "default")
        user_query = str(event.get("user_query", "") or "")
        final_answer = str(event.get("final_answer", "") or "")
        tool_results = event.get("tool_results", []) or []
        turn_meta = await self.store.append_turns_async(
            user_id=user_id,
            session_id=session_id,
            user_query=user_query,
            assistant_answer=final_answer,
            tool_results=tool_results,
        )
        turn_ok = bool(int(turn_meta.get("ok", 0) or 0))
        if (user_query or final_answer or tool_results) and not turn_ok:
            return {"ok": False, "reason": "turn_write_failed", "fact_saved": 0, "summary_saved": False}

        summary_saved = False
        if bool(event.get("compression_triggered")):
            rolling_summary = str(event.get("rolling_summary", "") or "").strip()
            if rolling_summary:
                summary_saved = await self.store.save_summary_async(
                    user_id=user_id,
                    session_id=session_id,
                    merge_count=int(event.get("merge_count", 0) or 0),
                    summary_text=rolling_summary,
                    source_turn_start=int(turn_meta.get("start_turn_index", 0) or 0),
                    source_turn_end=int(turn_meta.get("end_turn_index", 0) or 0),
                )
            else:
                summary_saved = False

        facts = event.get("facts", []) or []
        fact_saved = 0
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            fact_type = str(fact.get("fact_type", "focus") or "focus")
            ttl_hours = fact.get("ttl_hours", None)
            if fact_type == "market" and not ttl_hours:
                ttl_hours = self.default_market_ttl_hours
            ok = await self.store.append_fact_async(
                user_id=user_id,
                session_id=session_id,
                fact_key=str(fact.get("fact_key", "") or ""),
                fact_value=str(fact.get("fact_value", "") or ""),
                fact_type=fact_type,
                keywords=[str(x).strip() for x in fact.get("keywords", []) if str(x).strip()],
                confidence=float(fact.get("confidence", FACT_DEFAULT_CONFIDENCE) or FACT_DEFAULT_CONFIDENCE),
                ttl_hours=int(ttl_hours) if isinstance(ttl_hours, int) or str(ttl_hours).isdigit() else None,
                source_turn_id=int(turn_meta.get("last_turn_id", 0) or 0),
            )
            if ok:
                fact_saved += 1
        return {"ok": True, "fact_saved": fact_saved, "summary_saved": summary_saved}

    async def run(self, state: AgentState) -> Dict[str, Any]:
        with langsmith_trace(
            self.config,
            name="memory:persistent_write",
            run_type="chain",
            inputs={
                "session_id": str(state.get("session_id", "") or ""),
                "user_id": str(state.get("user_id", "") or ""),
                "fact_count": len(state.get("memory_fact_candidates", []) or []),
            },
            tags=["memory", "persistent", "write"],
            metadata={"local_observer_enabled": self.local_observer_enabled},
        ) as span:
            if not self.enabled:
                result_state = {
                    "debug_steps": state.get("debug_steps", []) + ["persistent_memory_write: disabled"],
                }
            elif bool(state.get("block_persistent_write", False)):
                result_state = {
                    "debug_steps": state.get("debug_steps", []) + ["persistent_memory_write: blocked_by_quality_gate"],
                }
            else:
                try:
                    compression = self._find_compression_info(state=state)
                    event = self._build_event(state=state, compression=compression)
                    result = await self._persist_event_async(event)
                    if not result.get("ok", False):
                        raise RuntimeError(str(result.get("reason", "persistent_memory_write_failed")))
                    await asyncio.to_thread(self._append_local_observer, event=event, result=result)
                except Exception as exc:
                    logger.warning("长期记忆写入失败（已降级继续）: %s", exc)
                    result_state = {
                        "debug_steps": state.get("debug_steps", [])
                        + [f"persistent_memory_write: exception={exc}"],
                    }
                else:
                    result_state = {
                        "debug_steps": state.get("debug_steps", [])
                        + [
                            f"persistent_memory_write: turns_saved=1,facts_saved={int(result.get('fact_saved', 0) or 0)},summary_saved={bool(result.get('summary_saved', False))}"
                        ],
                    }
                    if span is not None:
                        span.end(
                            outputs={
                                "facts_saved": int(result.get("fact_saved", 0) or 0),
                                "summary_saved": bool(result.get("summary_saved", False)),
                            }
                        )
            return result_state
