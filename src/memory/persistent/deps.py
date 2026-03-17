"""Persistent memory optional dependencies."""

from __future__ import annotations

try:
    import psycopg
except ImportError:  # pragma: no cover
    psycopg = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None

__all__ = ["psycopg", "SentenceTransformer"]
