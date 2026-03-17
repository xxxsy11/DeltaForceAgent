"""Auto-split from persistent_memory_store.py."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional

from rank_bm25 import BM25Okapi
from retrieval import rank_ids_by_score, weighted_reciprocal_rank_fusion

from memory.persistent.types import RecallResult, _to_vector_literal, _tokenize_text


logger = logging.getLogger(__name__)

class PersistentRecallMixin:
    DEFAULT_FACT_CONFIDENCE = 0.7
    MAX_ENTITY_CHARS = 24
    MAX_ENTITIES_FROM_HITS = 6
    MAX_ENTITIES_FROM_TEXT = 8
    MAX_RECENT_TURN_ENTITIES = 6
    RECENT_TURN_SCAN_LIMIT = 30

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
                    "confidence": float(row[5] or self.DEFAULT_FACT_CONFIDENCE),
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
        if not bool(getattr(self, "_vector_sql_ready", True)):
            return []
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
                    "confidence": float(row[5] or self.DEFAULT_FACT_CONFIDENCE),
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
            if len(t) < 2 or len(t) > PersistentRecallMixin.MAX_ENTITY_CHARS:
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
                if len(entities) >= PersistentRecallMixin.MAX_ENTITIES_FROM_HITS:
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
            if len(item) < 2 or len(item) > PersistentRecallMixin.MAX_ENTITY_CHARS:
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
        return dedup[: PersistentRecallMixin.MAX_ENTITIES_FROM_TEXT]

    def _recent_turn_entities(
        self,
        user_id: str,
        session_id: str,
        limit: int = MAX_RECENT_TURN_ENTITIES,
    ) -> List[str]:
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
                        LIMIT %s
                    ) t
                    ORDER BY t.turn_index ASC;
                    """,
                    (user_id, session_id, self.RECENT_TURN_SCAN_LIMIT),
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
        turn_entities = self._recent_turn_entities(
            user_id=user_id,
            session_id=session_id,
            limit=self.MAX_RECENT_TURN_ENTITIES,
        )
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
