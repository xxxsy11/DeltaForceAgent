"""Auto-split from persistent_memory_store.py."""

from __future__ import annotations

import logging
from typing import List, Optional

from memory.persistent.deps import SentenceTransformer

logger = logging.getLogger(__name__)

class PersistentEmbeddingMixin:
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

