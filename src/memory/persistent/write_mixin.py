"""Auto-split from persistent_memory_store.py."""

from __future__ import annotations

import json
import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional

from memory.persistent.types import _to_vector_literal, _utc_now

logger = logging.getLogger(__name__)

class PersistentWriteMixin:
    USER_TURN_QUALITY_SCORE = 1.0
    ASSISTANT_SUCCESS_SCORE = 1.0
    ASSISTANT_FAILED_SCORE = 0.25
    TOOL_SUCCESS_SCORE = 0.8
    TOOL_FAILED_SCORE = 0.2
    ASSISTANT_CONTENT_MAX_LEN = 4000
    TOOL_CONTENT_MAX_LEN = 2500
    DEFAULT_FACT_CONFIDENCE = 0.7

    def _next_turn_index(self, cur, user_id: str, session_id: str) -> int:
        cur.execute(
            "SELECT COALESCE(MAX(turn_index), 0) FROM chat_turns WHERE user_id = %s AND session_id = %s;",
            (user_id, session_id),
        )
        row = cur.fetchone()
        return int((row[0] if row else 0) or 0) + 1

    def upsert_session(self, user_id: str, session_id: str, meta: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return
        conn = self._connect()
        if conn is None:
            return
        meta_json = json.dumps(meta or {}, ensure_ascii=False)
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_sessions(user_id, session_id, meta)
                    VALUES (%s, %s, %s::jsonb)
                    ON CONFLICT (user_id, session_id)
                    DO UPDATE SET updated_at = NOW(), meta = chat_sessions.meta || EXCLUDED.meta;
                    """,
                    (user_id, session_id, meta_json),
                )

    def append_turns(
        self,
        user_id: str,
        session_id: str,
        user_query: str,
        assistant_answer: str,
        tool_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, int]:
        if not self.enabled:
            return {"last_turn_id": 0, "start_turn_index": 0, "end_turn_index": 0, "ok": 0}
        self.upsert_session(user_id=user_id, session_id=session_id)
        conn = self._connect()
        if conn is None:
            return {"last_turn_id": 0, "start_turn_index": 0, "end_turn_index": 0, "ok": 0}

        last_turn_id = 0
        start_idx = 0
        end_idx = 0
        inserted_turns = 0
        with conn:
            with conn.cursor() as cur:
                next_idx = self._next_turn_index(cur=cur, user_id=user_id, session_id=session_id)
                if user_query:
                    if start_idx == 0:
                        start_idx = next_idx
                    cur.execute(
                        """
                        INSERT INTO chat_turns(user_id, session_id, turn_index, role, content, quality_score, is_failed)
                        VALUES (%s, %s, %s, 'user', %s, %s, FALSE)
                        RETURNING id;
                        """,
                        (user_id, session_id, next_idx, user_query, self.USER_TURN_QUALITY_SCORE),
                    )
                    row = cur.fetchone()
                    last_turn_id = int(row[0]) if row else 0
                    next_idx += 1
                    inserted_turns += 1

                if assistant_answer:
                    if start_idx == 0:
                        start_idx = next_idx
                    lower = str(assistant_answer)
                    is_failed = any(x in lower for x in ("查询失败", "工具调用失败", "系统错误", "未找到工具"))
                    score = self.ASSISTANT_FAILED_SCORE if is_failed else self.ASSISTANT_SUCCESS_SCORE
                    cur.execute(
                        """
                        INSERT INTO chat_turns(user_id, session_id, turn_index, role, content, quality_score, is_failed)
                        VALUES (%s, %s, %s, 'assistant', %s, %s, %s)
                        RETURNING id;
                        """,
                        (
                            user_id,
                            session_id,
                            next_idx,
                            assistant_answer[: self.ASSISTANT_CONTENT_MAX_LEN],
                            score,
                            is_failed,
                        ),
                    )
                    row = cur.fetchone()
                    last_turn_id = int(row[0]) if row else last_turn_id
                    next_idx += 1
                    inserted_turns += 1

                for item in (tool_results or []):
                    name = str(item.get("tool_name", "") or "").strip()
                    output = str(item.get("output", "") or "").strip()
                    if not output:
                        continue
                    if start_idx == 0:
                        start_idx = next_idx
                    is_failed = any(x in output for x in ("查询失败", "工具调用失败", "系统错误", "未找到工具"))
                    score = self.TOOL_FAILED_SCORE if is_failed else self.TOOL_SUCCESS_SCORE
                    cur.execute(
                        """
                        INSERT INTO chat_turns(user_id, session_id, turn_index, role, content, tool_name, quality_score, is_failed)
                        VALUES (%s, %s, %s, 'tool', %s, %s, %s, %s);
                        """,
                        (
                            user_id,
                            session_id,
                            next_idx,
                            output[: self.TOOL_CONTENT_MAX_LEN],
                            name or None,
                            score,
                            is_failed,
                        ),
                    )
                    next_idx += 1
                    inserted_turns += 1

                if inserted_turns > 0:
                    end_idx = next_idx - 1
                else:
                    start_idx = 0
                    end_idx = 0
        return {
            "last_turn_id": last_turn_id,
            "start_turn_index": start_idx,
            "end_turn_index": end_idx,
            "inserted_turns": inserted_turns,
            "ok": 1,
        }

    def save_summary(
        self,
        user_id: str,
        session_id: str,
        merge_count: int,
        summary_text: str,
        source_turn_start: int,
        source_turn_end: int,
    ) -> bool:
        if not self.enabled or not summary_text:
            return False
        conn = self._connect()
        if conn is None:
            return False
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO memory_summaries(user_id, session_id, merge_count, summary_text, source_turn_start, source_turn_end)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, session_id, source_turn_end)
                    DO NOTHING;
                    """,
                    (user_id, session_id, merge_count, summary_text, source_turn_start, source_turn_end),
                )
        return True

    def append_fact(
        self,
        user_id: str,
        session_id: str,
        fact_key: str,
        fact_value: str,
        fact_type: str,
        keywords: Optional[List[str]] = None,
        confidence: float = DEFAULT_FACT_CONFIDENCE,
        ttl_hours: Optional[int] = None,
        source_turn_id: Optional[int] = None,
    ) -> bool:
        if not self.enabled:
            return False
        key = str(fact_key or "").strip()
        value = str(fact_value or "").strip()
        if not key or not value:
            return False
        conn = self._connect()
        if conn is None:
            return False
        ttl_until = None
        if ttl_hours and ttl_hours > 0:
            ttl_until = _utc_now() + timedelta(hours=int(ttl_hours))
        kw = [str(x).strip() for x in (keywords or []) if str(x).strip()]
        embedding = self._embed(f"{key} {value} {' '.join(kw)}")
        vector_literal = _to_vector_literal(embedding, self.vector_dim) if embedding else None

        with conn:
            with conn.cursor() as cur:
                inserted = False
                if vector_literal and bool(getattr(self, "_vector_sql_ready", True)):
                    try:
                        cur.execute(
                            """
                            INSERT INTO memory_facts(
                                user_id, session_id, fact_key, fact_value, fact_type, keywords, confidence,
                                is_active, ttl_until, source_turn_id, embedding
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, TRUE, %s, %s, %s::vector)
                            """,
                            (user_id, session_id, key, value, fact_type, kw, float(confidence), ttl_until, source_turn_id, vector_literal),
                        )
                        inserted = True
                    except Exception:
                        if bool(getattr(self, "_vector_sql_ready", True)):
                            logger.warning("长期记忆向量写入失败，回退非向量写入", exc_info=False)
                        self._vector_sql_ready = False

                if not inserted:
                    cur.execute(
                        """
                        INSERT INTO memory_facts(
                            user_id, session_id, fact_key, fact_value, fact_type, keywords, confidence,
                            is_active, ttl_until, source_turn_id
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, TRUE, %s, %s)
                        """,
                        (user_id, session_id, key, value, fact_type, kw, float(confidence), ttl_until, source_turn_id),
                    )
        return True

    def latest_summary(self, user_id: str, session_id: str) -> str:
        if not self.enabled:
            return ""
        conn = self._connect()
        if conn is None:
            return ""
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT summary_text
                    FROM memory_summaries
                    WHERE user_id = %s AND session_id = %s
                    ORDER BY merge_count DESC
                    LIMIT 1;
                    """,
                    (user_id, session_id),
                )
                row = cur.fetchone()
                return str(row[0]) if row else ""
