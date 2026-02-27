"""Shared retrieval fusion utilities."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional


def weighted_reciprocal_rank_fusion(
    ranked_lists: Dict[str, Iterable[str]],
    weights: Optional[Dict[str, float]] = None,
    rrf_k: int = 60,
) -> Dict[str, float]:
    """
    Compute weighted RRF scores for multiple ranked id lists.

    ranked_lists:
      {
        "source_a": ["id1", "id2", ...],
        "source_b": ["id3", "id1", ...],
      }
    """
    if rrf_k <= 0:
        raise ValueError("rrf_k must be > 0")

    weights = weights or {}
    scores: Dict[str, float] = {}

    for source_name, ids in ranked_lists.items():
        weight = float(weights.get(source_name, 1.0))
        if weight <= 0:
            continue
        for rank, item_id in enumerate(ids, start=1):
            key = str(item_id)
            if not key:
                continue
            scores[key] = scores.get(key, 0.0) + weight / (float(rrf_k) + float(rank))
    return scores


def rank_ids_by_score(ids: Iterable[str], scores: Dict[str, float]) -> List[str]:
    unique_ids: List[str] = []
    seen = set()
    for item_id in ids:
        key = str(item_id)
        if not key or key in seen:
            continue
        seen.add(key)
        unique_ids.append(key)
    return sorted(unique_ids, key=lambda x: scores.get(x, 0.0), reverse=True)

