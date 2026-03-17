"""Persistent memory shared types and helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence


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
