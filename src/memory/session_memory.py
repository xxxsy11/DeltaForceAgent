"""In-memory session memory manager."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _estimate_tokens(text: str) -> int:
    raw = str(text or "")
    if not raw:
        return 0
    return max(1, len(raw) // 2)


def _estimate_turn_tokens(turns: List[Dict[str, str]]) -> int:
    total = 0
    for item in turns:
        total += _estimate_tokens(item.get("content", ""))
    return total


@dataclass
class SessionMemory:
    user_id: str
    session_id: str
    recent_raw: List[Dict[str, str]] = field(default_factory=list)
    pending_buffer: List[Dict[str, str]] = field(default_factory=list)
    rolling_summary: str = ""
    merge_count: int = 0
    updated_at_utc: str = field(default_factory=_utc_now)


class SessionMemoryManager:
    """Manages in-memory conversation state by session."""

    def __init__(self):
        self._sessions: Dict[str, SessionMemory] = {}
        self._session_seq = defaultdict(int)

    @staticmethod
    def _build_key(user_id: str, session_id: str) -> str:
        uid = str(user_id or "default_user").strip() or "default_user"
        sid = str(session_id or "default").strip() or "default"
        return f"{uid}::{sid}"

    def get_or_create(self, user_id: str, session_id: str) -> SessionMemory:
        uid = str(user_id or "default_user").strip() or "default_user"
        sid = str(session_id or "default").strip() or "default"
        key = self._build_key(user_id=uid, session_id=sid)
        if key not in self._sessions:
            self._sessions[key] = SessionMemory(user_id=uid, session_id=sid)
        return self._sessions[key]

    def clear_session(self, user_id: str, session_id: str) -> None:
        uid = str(user_id or "default_user").strip() or "default_user"
        sid = str(session_id or "default").strip() or "default"
        key = self._build_key(user_id=uid, session_id=sid)
        self._sessions[key] = SessionMemory(user_id=uid, session_id=sid)

    def next_session_id(self, prefix: str = "session") -> str:
        self._session_seq[prefix] += 1
        return f"{prefix}-{self._session_seq[prefix]:04d}"

    @staticmethod
    def _build_pending_digest(pending_buffer: List[Dict[str, str]], max_lines: int = 4) -> str:
        if not pending_buffer:
            return ""
        lines = []
        for item in pending_buffer[-max_lines:]:
            role = "用户" if item.get("role") == "user" else "助手"
            content = str(item.get("content", "")).strip().replace("\n", " ")
            if len(content) > 100:
                content = content[:100] + "..."
            lines.append(f"- {role}: {content}")
        return "\n".join(lines)

    def build_state_patch(self, user_id: str, session_id: str, include_pending_in_prompt: bool = True) -> Dict[str, Any]:
        memory = self.get_or_create(user_id=user_id, session_id=session_id)
        pending_digest = self._build_pending_digest(memory.pending_buffer)

        context_blocks = []
        if memory.rolling_summary:
            context_blocks.append(f"[历史摘要]\n{memory.rolling_summary}")
        if include_pending_in_prompt and pending_digest:
            context_blocks.append(f"[待压缩摘要]\n{pending_digest}")
        if memory.recent_raw:
            recent_lines = []
            for item in memory.recent_raw[-10:]:
                role = "用户" if item.get("role") == "user" else "助手"
                content = str(item.get("content", "")).strip()
                recent_lines.append(f"- {role}: {content}")
            context_blocks.append("[最近对话]\n" + "\n".join(recent_lines))

        context_text = "\n\n".join(context_blocks).strip()
        return {
            "user_id": memory.user_id,
            "session_id": memory.session_id,
            "memory_recent_raw": [dict(x) for x in memory.recent_raw],
            "memory_pending_buffer": [dict(x) for x in memory.pending_buffer],
            "memory_rolling_summary": memory.rolling_summary,
            "memory_merge_count": memory.merge_count,
            "memory_pending_digest": pending_digest,
            "memory_context": context_text,
        }

    def save_from_state(self, user_id: str, session_id: str, state: Dict[str, Any]) -> None:
        memory = self.get_or_create(user_id=user_id, session_id=session_id)
        memory.recent_raw = [dict(x) for x in state.get("memory_recent_raw", []) if isinstance(x, dict)]
        memory.pending_buffer = [dict(x) for x in state.get("memory_pending_buffer", []) if isinstance(x, dict)]
        memory.rolling_summary = str(state.get("memory_rolling_summary", "") or "")
        memory.merge_count = int(state.get("memory_merge_count", 0) or 0)
        memory.updated_at_utc = _utc_now()

    def stats(self, user_id: str, session_id: str) -> Dict[str, Any]:
        memory = self.get_or_create(user_id=user_id, session_id=session_id)
        return {
            "user_id": memory.user_id,
            "session_id": memory.session_id,
            "recent_raw_turns": len(memory.recent_raw),
            "pending_turns": len(memory.pending_buffer),
            "pending_tokens": _estimate_turn_tokens(memory.pending_buffer),
            "rolling_summary_chars": len(memory.rolling_summary),
            "merge_count": memory.merge_count,
            "updated_at_utc": memory.updated_at_utc,
        }
