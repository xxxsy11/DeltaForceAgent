"""LangGraph Agent Server entrypoint for Studio graph monitoring."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")
os.environ.setdefault("RAG_RUN_MODE", "agent")

from agents.runner import _build_runtime  # noqa: E402
from config import DEFAULT_CONFIG  # noqa: E402


_RUNTIME = _build_runtime(DEFAULT_CONFIG)
graph = _RUNTIME.graph

