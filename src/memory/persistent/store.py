"""Persistent memory store (composed from mixins)."""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlparse

from memory.persistent.deps import psycopg
from memory.persistent.embedding_mixin import PersistentEmbeddingMixin
from memory.persistent.recall_mixin import PersistentRecallMixin
from memory.persistent.schema_mixin import PersistentSchemaMixin
from memory.persistent.write_mixin import PersistentWriteMixin
from observability.langsmith import langsmith_trace

logger = logging.getLogger(__name__)


class PersistentMemoryStore(
    PersistentSchemaMixin,
    PersistentEmbeddingMixin,
    PersistentWriteMixin,
    PersistentRecallMixin,
):
    DEFAULT_VECTOR_DIM = 512
    VECTOR_DIM_MIN = 32
    VECTOR_DIM_MAX = 4096
    DEFAULT_RECALL_TOP_K = 6
    DEFAULT_VECTOR_TOP_K = 20
    DEFAULT_BM25_TOP_K = 20
    DEFAULT_BM25_CANDIDATE_LIMIT = 200
    DEFAULT_RRF_K = 60
    DEFAULT_MARKET_TTL_HOURS = 24
    DEFAULT_CONNECT_TIMEOUT_SECONDS = 10

    def __init__(self, config):
        self.config = config
        self.enabled = bool(getattr(config, "memory_persistent_enabled", False))
        self.dsn = str(getattr(config, "memory_persistent_dsn", "") or "").strip()
        self.vector_dim = int(getattr(config, "memory_persistent_vector_dim", self.DEFAULT_VECTOR_DIM) or self.DEFAULT_VECTOR_DIM)
        if self.vector_dim < self.VECTOR_DIM_MIN or self.vector_dim > self.VECTOR_DIM_MAX:
            raise ValueError("memory_persistent_vector_dim 必须在 [32, 4096] 范围")
        self.recall_top_k = int(getattr(config, "memory_persistent_recall_top_k", self.DEFAULT_RECALL_TOP_K) or self.DEFAULT_RECALL_TOP_K)
        self.vector_top_k = int(getattr(config, "memory_persistent_vector_top_k", self.DEFAULT_VECTOR_TOP_K) or self.DEFAULT_VECTOR_TOP_K)
        self.bm25_top_k = int(getattr(config, "memory_persistent_bm25_top_k", self.DEFAULT_BM25_TOP_K) or self.DEFAULT_BM25_TOP_K)
        self.bm25_candidate_limit = int(
            getattr(config, "memory_persistent_bm25_candidate_limit", self.DEFAULT_BM25_CANDIDATE_LIMIT)
            or self.DEFAULT_BM25_CANDIDATE_LIMIT
        )
        self.rrf_k = int(getattr(config, "memory_persistent_rrf_k", self.DEFAULT_RRF_K) or self.DEFAULT_RRF_K)
        self.market_ttl_hours = int(
            getattr(config, "memory_persistent_market_ttl_hours", self.DEFAULT_MARKET_TTL_HOURS)
            or self.DEFAULT_MARKET_TTL_HOURS
        )
        self.connect_timeout_seconds = int(
            getattr(config, "memory_persistent_connect_timeout_seconds", self.DEFAULT_CONNECT_TIMEOUT_SECONDS)
            or self.DEFAULT_CONNECT_TIMEOUT_SECONDS
        )
        self.embedding_model_name = str(getattr(config, "embedding_model", "") or "").strip()

        self._embedder = None
        self._schema_ready = False
        self._vector_sql_ready = True
        self._connect_error_logged = False

        if self.enabled and psycopg is None:
            raise RuntimeError("memory_persistent_enabled=True 但 psycopg 未安装")

        if self.enabled and self.dsn:
            self._ensure_schema()

    def _connect(self):
        if not self.enabled or not self.dsn or psycopg is None:
            return None
        try:
            conn = psycopg.connect(self.dsn, autocommit=True, connect_timeout=self.connect_timeout_seconds)
            self._connect_error_logged = False
            return conn
        except Exception as exc:
            if not self._connect_error_logged:
                host = "unknown"
                port = "unknown"
                dbname = "unknown"
                try:
                    parsed = urlparse(self.dsn)
                    if parsed.hostname:
                        host = str(parsed.hostname)
                    if parsed.port:
                        port = str(parsed.port)
                    if parsed.path:
                        dbname = parsed.path.lstrip("/") or "unknown"
                except Exception:
                    pass
                logger.warning(
                    "无法连接长期记忆数据库 host=%s port=%s db=%s err=%s",
                    host,
                    port,
                    dbname,
                    str(exc),
                    exc_info=False,
                )
                self._connect_error_logged = True
            return None

    async def recall_async(self, user_id: str, session_id: str, query: str):
        with langsmith_trace(
            self.config,
            name="memory_store:recall",
            run_type="retriever",
            inputs={"user_id": user_id, "session_id": session_id, "query": str(query or "")[:240]},
            tags=["memory-store", "recall"],
        ) as span:
            result = await asyncio.to_thread(
                self.recall,
                user_id=user_id,
                session_id=session_id,
                query=query,
            )
            if span is not None:
                span.end(
                    outputs={
                        "used": bool(getattr(result, "used", False)),
                        "hit_count": len(getattr(result, "hits", []) or []),
                    }
                )
            return result

    async def latest_summary_async(self, user_id: str, session_id: str) -> str:
        with langsmith_trace(
            self.config,
            name="memory_store:latest_summary",
            run_type="retriever",
            inputs={"user_id": user_id, "session_id": session_id},
            tags=["memory-store", "summary"],
        ) as span:
            result = await asyncio.to_thread(self.latest_summary, user_id=user_id, session_id=session_id)
            if span is not None:
                span.end(outputs={"summary_present": bool(str(result or "").strip()), "summary_chars": len(str(result or ""))})
            return result

    async def append_turns_async(self, user_id: str, session_id: str, user_query: str, assistant_answer: str, tool_results):
        with langsmith_trace(
            self.config,
            name="memory_store:append_turns",
            run_type="tool",
            inputs={
                "user_id": user_id,
                "session_id": session_id,
                "user_query_preview": str(user_query or "")[:240],
                "tool_result_count": len(tool_results or []),
            },
            tags=["memory-store", "write", "turns"],
        ) as span:
            result = await asyncio.to_thread(
                self.append_turns,
                user_id=user_id,
                session_id=session_id,
                user_query=user_query,
                assistant_answer=assistant_answer,
                tool_results=tool_results,
            )
            if span is not None:
                span.end(outputs={"ok": bool(int((result or {}).get("ok", 0) or 0)), "last_turn_id": int((result or {}).get("last_turn_id", 0) or 0)})
            return result

    async def save_summary_async(
        self,
        user_id: str,
        session_id: str,
        merge_count: int,
        summary_text: str,
        source_turn_start: int,
        source_turn_end: int,
    ) -> bool:
        with langsmith_trace(
            self.config,
            name="memory_store:save_summary",
            run_type="tool",
            inputs={
                "user_id": user_id,
                "session_id": session_id,
                "merge_count": merge_count,
                "source_turn_start": source_turn_start,
                "source_turn_end": source_turn_end,
            },
            tags=["memory-store", "write", "summary"],
        ) as span:
            result = await asyncio.to_thread(
                self.save_summary,
                user_id=user_id,
                session_id=session_id,
                merge_count=merge_count,
                summary_text=summary_text,
                source_turn_start=source_turn_start,
                source_turn_end=source_turn_end,
            )
            if span is not None:
                span.end(outputs={"ok": bool(result), "summary_chars": len(str(summary_text or ""))})
            return result

    async def append_fact_async(
        self,
        user_id: str,
        session_id: str,
        fact_key: str,
        fact_value: str,
        fact_type: str,
        keywords,
        confidence: float,
        ttl_hours,
        source_turn_id: int,
    ) -> bool:
        with langsmith_trace(
            self.config,
            name="memory_store:append_fact",
            run_type="tool",
            inputs={
                "user_id": user_id,
                "session_id": session_id,
                "fact_key": fact_key,
                "fact_type": fact_type,
                "source_turn_id": source_turn_id,
            },
            tags=["memory-store", "write", "fact"],
        ) as span:
            result = await asyncio.to_thread(
                self.append_fact,
                user_id=user_id,
                session_id=session_id,
                fact_key=fact_key,
                fact_value=fact_value,
                fact_type=fact_type,
                keywords=keywords,
                confidence=confidence,
                ttl_hours=ttl_hours,
                source_turn_id=source_turn_id,
            )
            if span is not None:
                span.end(outputs={"ok": bool(result), "keyword_count": len(keywords or [])})
            return result
