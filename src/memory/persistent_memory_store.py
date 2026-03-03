"""PostgreSQL + pgvector backed long-term memory store."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

from rank_bm25 import BM25Okapi
from retrieval import rank_ids_by_score, weighted_reciprocal_rank_fusion

try:
    import psycopg
except ImportError:  # pragma: no cover - optional dependency
    psycopg = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _tokenize_text(text: str) -> List[str]:
    raw = str(text or "").strip().lower()
    if not raw:
        return []
    latin = re.findall(r"[a-z0-9_]+", raw)
    zh_chars = re.findall(r"[\u4e00-\u9fff]", raw)
    zh_bigram = ["".join(zh_chars[i : i + 2]) for i in range(max(0, len(zh_chars) - 1))]
    return latin + zh_chars + zh_bigram


def _to_vector_literal(values: Sequence[float], dim: int) -> str:
    arr = list(values[:dim])
    if len(arr) < dim:
        arr.extend([0.0] * (dim - len(arr)))
    return "[" + ",".join(f"{float(x):.8f}" for x in arr) + "]"


@dataclass
class RecallResult:
    context: str
    entities: List[str]
    hits: List[Dict[str, Any]]
    used: bool
    debug: Dict[str, Any]


class PersistentMemoryStore:
    """Long-term memory with full chat rollback + fact recall."""

    def __init__(self, config):
        self.config = config
        self.enabled = bool(getattr(config, "memory_persistent_enabled", False))
        self.dsn = str(getattr(config, "memory_persistent_dsn", "") or "").strip()
        self.vector_dim = int(getattr(config, "memory_persistent_vector_dim", 512) or 512)
        if self.vector_dim < 32 or self.vector_dim > 4096:
            raise ValueError("memory_persistent_vector_dim 必须在 [32, 4096] 范围")
        self.recall_top_k = int(getattr(config, "memory_persistent_recall_top_k", 6) or 6)
        self.vector_top_k = int(getattr(config, "memory_persistent_vector_top_k", 20) or 20)
        self.bm25_top_k = int(getattr(config, "memory_persistent_bm25_top_k", 20) or 20)
        self.bm25_candidate_limit = int(getattr(config, "memory_persistent_bm25_candidate_limit", 200) or 200)
        self.rrf_k = int(getattr(config, "memory_persistent_rrf_k", 60) or 60)
        self.market_ttl_hours = int(getattr(config, "memory_persistent_market_ttl_hours", 24) or 24)
        self.connect_timeout_seconds = int(getattr(config, "memory_persistent_connect_timeout_seconds", 10) or 10)
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
        except Exception:
            if not self._connect_error_logged:
                logger.warning("无法连接长期记忆数据库", exc_info=False)
                self._connect_error_logged = True
            return None

    def _ensure_schema(self) -> None:
        if self._schema_ready or not self.enabled:
            return
        conn = self._connect()
        if conn is None:
            return
        with conn:
            with conn.cursor() as cur:
                try:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                except Exception:
                    # 扩展通常由 DBA 预装；失败不阻断业务
                    logger.warning("vector 扩展创建失败，继续尝试使用已有扩展", exc_info=False)

                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        user_id TEXT NOT NULL DEFAULT 'default_user',
                        session_id TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        meta JSONB NOT NULL DEFAULT '{}'::jsonb,
                        PRIMARY KEY (user_id, session_id)
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_turns (
                        id BIGSERIAL PRIMARY KEY,
                        user_id TEXT NOT NULL DEFAULT 'default_user',
                        session_id TEXT NOT NULL,
                        turn_index INT NOT NULL,
                        role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'tool')),
                        content TEXT NOT NULL,
                        tool_name TEXT,
                        quality_score REAL NOT NULL DEFAULT 1.0,
                        is_failed BOOLEAN NOT NULL DEFAULT FALSE,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_summaries (
                        id BIGSERIAL PRIMARY KEY,
                        user_id TEXT NOT NULL DEFAULT 'default_user',
                        session_id TEXT NOT NULL,
                        merge_count INT NOT NULL,
                        summary_text TEXT NOT NULL,
                        source_turn_start INT NOT NULL DEFAULT 0,
                        source_turn_end INT NOT NULL DEFAULT 0,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS memory_facts (
                        id BIGSERIAL PRIMARY KEY,
                        user_id TEXT NOT NULL DEFAULT 'default_user',
                        session_id TEXT NOT NULL,
                        fact_key TEXT NOT NULL,
                        fact_value TEXT NOT NULL,
                        fact_type TEXT NOT NULL,
                        keywords TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
                        confidence REAL NOT NULL DEFAULT 0.7,
                        is_active BOOLEAN NOT NULL DEFAULT TRUE,
                        ttl_until TIMESTAMPTZ,
                        source_turn_id BIGINT,
                        embedding VECTOR({self.vector_dim}),
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
                # ---- schema migration: add user_id columns to old tables ----
                cur.execute(
                    """
                    ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS user_id TEXT;
                    UPDATE chat_sessions SET user_id = 'default_user' WHERE user_id IS NULL OR user_id = '';
                    ALTER TABLE chat_sessions ALTER COLUMN user_id SET NOT NULL;
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE chat_turns ADD COLUMN IF NOT EXISTS user_id TEXT;
                    UPDATE chat_turns SET user_id = 'default_user' WHERE user_id IS NULL OR user_id = '';
                    ALTER TABLE chat_turns ALTER COLUMN user_id SET NOT NULL;
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE memory_summaries ADD COLUMN IF NOT EXISTS user_id TEXT;
                    UPDATE memory_summaries SET user_id = 'default_user' WHERE user_id IS NULL OR user_id = '';
                    ALTER TABLE memory_summaries ALTER COLUMN user_id SET NOT NULL;
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE memory_facts ADD COLUMN IF NOT EXISTS user_id TEXT;
                    UPDATE memory_facts SET user_id = 'default_user' WHERE user_id IS NULL OR user_id = '';
                    ALTER TABLE memory_facts ALTER COLUMN user_id SET NOT NULL;
                    """
                )

                # ---- primary key migration for chat_sessions ----
                cur.execute(
                    """
                    DO $$
                    DECLARE pkey_name TEXT;
                    BEGIN
                      SELECT conname INTO pkey_name
                      FROM pg_constraint
                      WHERE conrelid = 'chat_sessions'::regclass
                        AND contype = 'p'
                      LIMIT 1;
                      IF pkey_name IS NOT NULL THEN
                        EXECUTE format('ALTER TABLE chat_sessions DROP CONSTRAINT %I', pkey_name);
                      END IF;
                      ALTER TABLE chat_sessions ADD PRIMARY KEY (user_id, session_id);
                    EXCEPTION WHEN duplicate_object THEN
                      NULL;
                    END $$;
                    """
                )

                # ---- remove old uniqueness constraints ----
                cur.execute(
                    """
                    DO $$
                    DECLARE cname TEXT;
                    BEGIN
                      SELECT conname INTO cname
                      FROM pg_constraint
                      WHERE conrelid = 'memory_summaries'::regclass
                        AND contype = 'u'
                        AND pg_get_constraintdef(oid) LIKE '%(session_id, merge_count)%'
                      LIMIT 1;
                      IF cname IS NOT NULL THEN
                        EXECUTE format('ALTER TABLE memory_summaries DROP CONSTRAINT %I', cname);
                      END IF;
                    END $$;
                    """
                )
                cur.execute(
                    """
                    DO $$
                    DECLARE cname TEXT;
                    BEGIN
                      SELECT conname INTO cname
                      FROM pg_constraint
                      WHERE conrelid = 'memory_facts'::regclass
                        AND contype = 'u'
                        AND pg_get_constraintdef(oid) LIKE '%(session_id, fact_key)%'
                      LIMIT 1;
                      IF cname IS NOT NULL THEN
                        EXECUTE format('ALTER TABLE memory_facts DROP CONSTRAINT %I', cname);
                      END IF;
                    END $$;
                    """
                )

                # ---- create indexes after migration ----
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chat_turns_user_session_turn ON chat_turns(user_id, session_id, turn_index DESC);"
                )
                cur.execute(
                    "DROP INDEX IF EXISTS idx_memory_summaries_session_turn_end_uniq;"
                )
                cur.execute(
                    """
                    CREATE UNIQUE INDEX idx_memory_summaries_session_turn_end_uniq
                    ON memory_summaries(user_id, session_id, source_turn_end);
                    """
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memory_facts_user_session_active ON memory_facts(user_id, session_id, is_active);"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memory_facts_user_session_key ON memory_facts(user_id, session_id, fact_key);"
                )
                cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_facts_user_ttl ON memory_facts(user_id, session_id, ttl_until);")

        self._schema_ready = True

    def _load_embedder(self):
        if self._embedder is not None or SentenceTransformer is None:
            return self._embedder
        try:
            self._embedder = SentenceTransformer(self.embedding_model_name)
        except Exception:
            logger.warning("长期记忆嵌入模型加载失败，向量召回将跳过", exc_info=False)
            self._embedder = False
        return self._embedder

    def _embed(self, text: str) -> Optional[List[float]]:
        embedder = self._load_embedder()
        if not embedder or embedder is False:
            return None
        try:
            vec = embedder.encode(str(text or ""), normalize_embeddings=True)
            return [float(x) for x in vec]
        except Exception:
            return None

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
        with conn:
            with conn.cursor() as cur:
                next_idx = self._next_turn_index(cur=cur, user_id=user_id, session_id=session_id)
                start_idx = next_idx
                if user_query:
                    cur.execute(
                        """
                        INSERT INTO chat_turns(user_id, session_id, turn_index, role, content, quality_score, is_failed)
                        VALUES (%s, %s, %s, 'user', %s, 1.0, FALSE)
                        RETURNING id;
                        """,
                        (user_id, session_id, next_idx, user_query),
                    )
                    row = cur.fetchone()
                    last_turn_id = int(row[0]) if row else 0
                    next_idx += 1

                if assistant_answer:
                    lower = str(assistant_answer)
                    is_failed = any(x in lower for x in ("查询失败", "工具调用失败", "系统错误", "未找到工具"))
                    score = 0.25 if is_failed else 1.0
                    cur.execute(
                        """
                        INSERT INTO chat_turns(user_id, session_id, turn_index, role, content, quality_score, is_failed)
                        VALUES (%s, %s, %s, 'assistant', %s, %s, %s)
                        RETURNING id;
                        """,
                        (user_id, session_id, next_idx, assistant_answer[:4000], score, is_failed),
                    )
                    row = cur.fetchone()
                    last_turn_id = int(row[0]) if row else last_turn_id
                    next_idx += 1

                for item in (tool_results or []):
                    name = str(item.get("tool_name", "") or "").strip()
                    output = str(item.get("output", "") or "").strip()
                    if not output:
                        continue
                    is_failed = any(x in output for x in ("查询失败", "工具调用失败", "系统错误", "未找到工具"))
                    score = 0.2 if is_failed else 0.8
                    cur.execute(
                        """
                        INSERT INTO chat_turns(user_id, session_id, turn_index, role, content, tool_name, quality_score, is_failed)
                        VALUES (%s, %s, %s, 'tool', %s, %s, %s, %s);
                        """,
                        (user_id, session_id, next_idx, output[:2500], name or None, score, is_failed),
                    )
                    next_idx += 1

                end_idx = max(start_idx, next_idx - 1)
        return {"last_turn_id": last_turn_id, "start_turn_index": start_idx, "end_turn_index": end_idx, "ok": 1}

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
        confidence: float = 0.7,
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
                if vector_literal:
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
                else:
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

    def _fetch_bm25_candidates(self, user_id: str, session_id: str) -> List[Dict[str, Any]]:
        conn = self._connect()
        if conn is None:
            return []
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, fact_key, fact_value, fact_type, keywords, confidence, updated_at
                    FROM memory_facts
                    WHERE user_id = %s
                      AND session_id = %s
                      AND is_active = TRUE
                      AND (ttl_until IS NULL OR ttl_until > NOW())
                    ORDER BY updated_at DESC
                    LIMIT %s;
                    """,
                    (user_id, session_id, self.bm25_candidate_limit),
                )
                rows = cur.fetchall() or []
        docs: List[Dict[str, Any]] = []
        for row in rows:
            docs.append(
                {
                    "id": int(row[0]),
                    "fact_key": str(row[1] or ""),
                    "fact_value": str(row[2] or ""),
                    "fact_type": str(row[3] or ""),
                    "keywords": list(row[4] or []),
                    "confidence": float(row[5] or 0.7),
                    "updated_at": row[6],
                }
            )
        return docs

    def _bm25_rank(self, user_id: str, session_id: str, query: str) -> List[Dict[str, Any]]:
        candidates = self._fetch_bm25_candidates(user_id=user_id, session_id=session_id)
        if not candidates:
            return []
        docs_tokens = [
            _tokenize_text(
                " ".join(
                    [
                        item.get("fact_key", ""),
                        item.get("fact_value", ""),
                        " ".join(item.get("keywords", [])),
                        item.get("fact_type", ""),
                    ]
                )
            )
            for item in candidates
        ]
        query_tokens = _tokenize_text(query)
        if not query_tokens:
            return candidates[: self.bm25_top_k]

        bm25 = BM25Okapi(docs_tokens)
        scores = bm25.get_scores(query_tokens)
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )[: self.bm25_top_k]
        result: List[Dict[str, Any]] = []
        for item, score in ranked:
            row = dict(item)
            row["_bm25"] = float(score)
            result.append(row)
        return result

    def _vector_rank(self, user_id: str, session_id: str, query: str) -> List[Dict[str, Any]]:
        conn = self._connect()
        if conn is None:
            return []
        embedding = self._embed(query)
        if not embedding:
            return []
        vector_literal = _to_vector_literal(embedding, self.vector_dim)
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, fact_key, fact_value, fact_type, keywords, confidence, updated_at,
                               embedding <=> %s::vector AS dist
                        FROM memory_facts
                        WHERE user_id = %s
                          AND session_id = %s
                          AND is_active = TRUE
                          AND embedding IS NOT NULL
                          AND (ttl_until IS NULL OR ttl_until > NOW())
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s;
                        """,
                        (vector_literal, user_id, session_id, vector_literal, self.vector_top_k),
                    )
                    rows = cur.fetchall() or []
        except Exception:
            if self._vector_sql_ready:
                logger.warning("长期记忆向量检索失败，回退 BM25", exc_info=False)
                self._vector_sql_ready = False
            return []

        hits: List[Dict[str, Any]] = []
        for row in rows:
            hits.append(
                {
                    "id": int(row[0]),
                    "fact_key": str(row[1] or ""),
                    "fact_value": str(row[2] or ""),
                    "fact_type": str(row[3] or ""),
                    "keywords": list(row[4] or []),
                    "confidence": float(row[5] or 0.7),
                    "updated_at": row[6],
                    "_distance": float(row[7] or 0.0),
                }
            )
        return hits

    def _rrf_fuse(
        self,
        vector_hits: List[Dict[str, Any]],
        bm25_hits: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        by_id: Dict[str, Dict[str, Any]] = {}
        for item in vector_hits:
            by_id[str(item["id"])] = dict(item)
        for item in bm25_hits:
            by_id.setdefault(str(item["id"]), dict(item))

        score_map = weighted_reciprocal_rank_fusion(
            ranked_lists={
                "vector": [str(x["id"]) for x in vector_hits],
                "bm25": [str(x["id"]) for x in bm25_hits],
            },
            weights={"vector": 1.0, "bm25": 1.0},
            rrf_k=self.rrf_k,
        )
        ranked_ids = rank_ids_by_score(ids=by_id.keys(), scores=score_map)
        top = [by_id[item_id] for item_id in ranked_ids[: self.recall_top_k]]
        for item in top:
            item["_rrf"] = float(score_map.get(str(item["id"]), 0.0))
        return top

    @staticmethod
    def _extract_entities_from_hits(hits: Iterable[Dict[str, Any]]) -> List[str]:
        entities: List[str] = []
        invalid_tokens = {
            "结论", "依据", "建议", "价格", "历史", "样本", "区间", "风险", "回撤", "利润", "更新时间", "可交易",
            "介绍", "分析", "资料", "信息", "当前", "现在", "最高", "最低",
        }
        sentence_markers = ("是", "属于", "为", "位于", "包含", "包括", "相关", "对应")

        def _is_valid_entity(token: str) -> bool:
            t = str(token or "").strip()
            if not t:
                return False
            if len(t) < 2 or len(t) > 24:
                return False
            if any(marker in t for marker in sentence_markers):
                return False
            if any(x in t for x in invalid_tokens):
                return False
            if re.search(r"\d+\s*[xX×]\s*\d+|\d+\s*格", t):
                return False
            return True

        for item in hits:
            if item.get("fact_type") not in {"entity", "focus", "compare_target"}:
                continue
            raw = str(item.get("fact_value", "") or "").strip()
            for token in re.split(r"[，,、/|；;：:。.!?？\s]+", raw):
                t = token.strip()
                if not _is_valid_entity(t):
                    continue
                if t not in entities:
                    entities.append(t)
                if len(entities) >= 6:
                    return entities
        return entities


    @staticmethod
    def _extract_entities_from_text(text: str) -> List[str]:
        tokens: List[str] = []
        invalid_tokens = {
            "价格", "建议", "历史", "样本", "区间", "风险", "回撤", "利润", "更新时间",
            "介绍", "分析", "资料", "信息", "当前", "现在", "最新", "对比", "比较", "物品",
            "不要", "重仓", "追买", "分批", "小仓位", "止盈", "减仓",
        }
        for token in re.split(r"[，,、/|；;：:\s]+|以及|并且|并|和|与", str(text or "")):
            item = str(token or "").strip()
            if len(item) < 2 or len(item) > 24:
                continue
            if any(x in item for x in invalid_tokens):
                continue
            if re.search(r"\d+\s*[xX×]\s*\d+|\d+\s*格", item):
                continue
            if not re.search(r"[\u4e00-\u9fffA-Za-z]", item):
                continue
            if re.fullmatch(r"[\d\-_.+()（）]+", item):
                continue
            tokens.append(item)
        dedup: List[str] = []
        for item in tokens:
            if item not in dedup:
                dedup.append(item)
        return dedup[:8]

    def _recent_turn_entities(self, user_id: str, session_id: str, limit: int = 6) -> List[str]:
        conn = self._connect()
        if conn is None:
            return []

        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT role, content
                    FROM (
                        SELECT role, content, turn_index
                        FROM chat_turns
                        WHERE user_id = %s AND session_id = %s
                        ORDER BY turn_index DESC
                        LIMIT 30
                    ) t
                    ORDER BY t.turn_index ASC;
                    """,
                    (user_id, session_id),
                )
                rows = cur.fetchall() or []

        entities: List[str] = []

        # 1) 用户历史问题优先（最稳定）
        for role, content in rows:
            role_name = str(role or "")
            text = str(content or "")
            if role_name != "user":
                continue

            m = re.search(r"(?:介绍一下|介绍|查询一下|查询|查一下|分析一下|分析|告诉我|帮我看)\s*([^\s，,。；;]+)", text)
            if m:
                candidate = m.group(1).strip()
                for item in self._extract_entities_from_text(candidate):
                    if item not in entities:
                        entities.append(item)
            for item in self._extract_entities_from_text(text):
                if item not in entities:
                    entities.append(item)

        # 2) 对比工具输出补充（只吃 compare 输出，避免“基础属性”这类噪声）
        for role, content in rows:
            role_name = str(role or "")
            text = str(content or "")
            if role_name != "tool" or "多物品价格对比" not in text:
                continue
            for match in re.findall(r"\d+\.\s*([^\s｜|]+)", text):
                for item in self._extract_entities_from_text(match):
                    if item not in entities:
                        entities.append(item)

        return entities[:limit]

    def recall(self, user_id: str, session_id: str, query: str) -> RecallResult:
        if not self.enabled:
            return RecallResult(context="", entities=[], hits=[], used=False, debug={"enabled": False})

        latest_summary = self.latest_summary(user_id=user_id, session_id=session_id)
        vector_hits = self._vector_rank(user_id=user_id, session_id=session_id, query=query)
        bm25_hits = self._bm25_rank(user_id=user_id, session_id=session_id, query=query)
        fused = self._rrf_fuse(vector_hits=vector_hits, bm25_hits=bm25_hits)

        blocks: List[str] = []
        if latest_summary:
            blocks.append(f"[长期记忆摘要]\n{latest_summary}")
        if fused:
            lines = []
            for item in fused:
                lines.append(
                    f"- [{item.get('fact_type', 'fact')}] {item.get('fact_key', '')}: {item.get('fact_value', '')}"
                )
            blocks.append("[长期记忆召回]\n" + "\n".join(lines))

        fused_entities = self._extract_entities_from_hits(fused)
        turn_entities = self._recent_turn_entities(user_id=user_id, session_id=session_id, limit=6)
        entities: List[str] = []
        for item in turn_entities + fused_entities:
            if item and item not in entities:
                entities.append(item)

        if turn_entities:
            blocks.append("[长期记忆近轮实体]\n" + "、".join(turn_entities))

        context = "\n\n".join(blocks).strip()
        return RecallResult(
            context=context,
            entities=entities,
            hits=fused,
            used=bool(fused or turn_entities),
            debug={
                "enabled": True,
                "user_id": user_id,
                "session_id": session_id,
                "latest_summary": bool(latest_summary),
                "vector_hits": len(vector_hits),
                "bm25_hits": len(bm25_hits),
                "fused_hits": len(fused),
                "recent_turn_entities": len(turn_entities),
            },
        )
