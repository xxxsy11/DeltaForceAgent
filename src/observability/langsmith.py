"""LangSmith tracing helpers for LangGraph runs."""

from __future__ import annotations

from contextvars import ContextVar
import logging
import os
import uuid
from datetime import datetime, timezone
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Mapping, Optional

if TYPE_CHECKING:
    from config import GraphRAGConfig

try:
    from langsmith.client import Client
except Exception:  # pragma: no cover
    Client = None

logger = logging.getLogger(__name__)
_ROOT_RUN: ContextVar[Any] = ContextVar("langsmith_root_run", default=None)
_CURRENT_RUN: ContextVar[Any] = ContextVar("langsmith_current_run", default=None)
_WARNED_KEYS: set[str] = set()


def _warn_once(key: str, message: str, *args: Any) -> None:
    if key in _WARNED_KEYS:
        return
    _WARNED_KEYS.add(key)
    logger.warning(message, *args)


@dataclass
class _ManualRunHandle:
    client: Any
    project_name: str
    run_id: uuid.UUID
    trace_id: uuid.UUID
    dotted_order: str
    closed: bool = False

    def end(self, *, outputs: Optional[dict] = None, error: Optional[str] = None) -> None:
        if self.closed:
            return
        try:
            self.client.update_run(
                self.run_id,
                outputs=outputs,
                error=error,
                end_time=datetime.now(timezone.utc),
            )
        except Exception as exc:
            _warn_once("langsmith_update_run_failed", "LangSmith update_run 失败，已降级为本地继续运行: %s", exc)
        finally:
            self.closed = True


def _enabled(config: Optional["GraphRAGConfig"]) -> bool:
    if config is None:
        enabled = os.getenv("LANGSMITH_ENABLED", "0").strip().lower() not in {"0", "false", "off", "no"}
        api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
        return bool(enabled and api_key)
    return bool(config.langsmith_enabled and config.langsmith_api_key)


def _parse_tags(raw: str) -> List[str]:
    tags: List[str] = []
    for item in str(raw or "").split(","):
        value = item.strip()
        if value:
            tags.append(value)
    return tags


def _get_client(config: Optional["GraphRAGConfig"]):
    if Client is None or not _enabled(config):
        return None
    try:
        if config is None:
            return Client(
                api_key=os.getenv("LANGSMITH_API_KEY", "").strip() or None,
                api_url=os.getenv("LANGSMITH_ENDPOINT", "").strip() or None,
            )
        return Client(
            api_key=config.langsmith_api_key or None,
            api_url=config.langsmith_endpoint or None,
        )
    except Exception as exc:
        _warn_once("langsmith_client_init_failed", "LangSmith client 初始化失败，已降级为无追踪: %s", exc)
        return None


def _project_name(config: Optional["GraphRAGConfig"]) -> str:
    if config is None:
        return os.getenv("LANGSMITH_PROJECT", "").strip()
    return config.langsmith_project


def _new_dotted_segment(run_id: uuid.UUID) -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + str(run_id)


def configure_langsmith_env(config: "GraphRAGConfig") -> bool:
    """Apply LangSmith env vars lazily before building the graph/runtime."""
    if not _enabled(config):
        return False

    os.environ["LANGSMITH_API_KEY"] = config.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = config.langsmith_project
    os.environ["LANGSMITH_ENDPOINT"] = config.langsmith_endpoint
    os.environ["LANGSMITH_TRACING"] = "true" if config.langsmith_tracing_v2 else "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if config.langsmith_tracing_v2 else "false"
    return True


def build_langsmith_run_config(
    config: "GraphRAGConfig",
    *,
    query: str,
    user_id: str,
    session_id: str,
) -> Dict[str, Any]:
    """Build per-run metadata passed to graph.ainvoke(..., config=...)."""
    if not _enabled(config):
        return {}

    run_name = f"{config.langsmith_run_prefix}:{user_id}:{session_id}"
    tags = _parse_tags(config.langsmith_tags)
    tags.extend(
        [
            f"user:{user_id}",
            f"session:{session_id}",
            f"mode:{config.run_mode}",
        ]
    )
    metadata = {
        "user_id": user_id,
        "session_id": session_id,
        "run_mode": config.run_mode,
        "graph_name": config.langsmith_graph_name,
        "agent_local_enabled": bool(config.agent_local_enabled),
        "task_planning_enabled": bool(config.task_planning_enabled),
        "memory_persistent_enabled": bool(config.memory_persistent_enabled),
        "query_length": len(str(query or "")),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return {
        "run_name": run_name,
        "tags": tags,
        "metadata": metadata,
        "configurable": {
            "thread_id": session_id,
            "user_id": user_id,
        },
    }


def _end_run(run: Any, *, outputs: Optional[dict] = None, error: Optional[str] = None) -> None:
    if run is None:
        return
    if getattr(run, "closed", False):
        return
    try:
        run.end(outputs=outputs, error=error)
    except Exception as exc:
        _warn_once("langsmith_end_run_failed", "LangSmith end_run 失败，已忽略: %s", exc)


@contextmanager
def langsmith_root_run(
    config: Optional["GraphRAGConfig"],
    *,
    name: str,
    inputs: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    run_type: str = "chain",
) -> Iterator[Any]:
    if not _enabled(config):
        yield None
        return

    project_name = _project_name(config)
    client = _get_client(config)
    if client is None:
        yield None
        return
    run_id = uuid.uuid4()
    dotted_order = _new_dotted_segment(run_id)
    try:
        client.create_run(
            id=run_id,
            project_name=project_name,
            name=name,
            run_type=run_type,
            inputs=dict(inputs or {}),
            start_time=datetime.now(timezone.utc),
            tags=list(tags or []),
            extra={"metadata": dict(metadata or {})},
            trace_id=run_id,
            dotted_order=dotted_order,
        )
    except Exception as exc:
        _warn_once("langsmith_create_root_failed", "LangSmith create_run(root) 失败，已降级为无追踪: %s", exc)
        yield None
        return
    run = _ManualRunHandle(
        client=client,
        project_name=project_name,
        run_id=run_id,
        trace_id=run_id,
        dotted_order=dotted_order,
    )
    root_token = _ROOT_RUN.set(run)
    current_token = _CURRENT_RUN.set(run)
    try:
        yield run
    except Exception as exc:
        _end_run(run, error=str(exc))
        raise
    finally:
        _end_run(run)
        _CURRENT_RUN.reset(current_token)
        _ROOT_RUN.reset(root_token)


@contextmanager
def langsmith_span(
    config: Optional["GraphRAGConfig"],
    *,
    name: str,
    run_type: str = "chain",
    inputs: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Iterator[Any]:
    parent = _CURRENT_RUN.get() or _ROOT_RUN.get()
    if not (_enabled(config) and parent is not None):
        yield None
        return

    client = _get_client(config)
    if client is None:
        yield None
        return
    run_id = uuid.uuid4()
    dotted_order = f"{parent.dotted_order}.{_new_dotted_segment(run_id)}"
    try:
        client.create_run(
            id=run_id,
            project_name=parent.project_name,
            name=name,
            run_type=run_type,
            inputs=dict(inputs or {}),
            start_time=datetime.now(timezone.utc),
            tags=list(tags or []),
            extra={"metadata": dict(metadata or {})},
            parent_run_id=parent.run_id,
            trace_id=parent.trace_id,
            dotted_order=dotted_order,
        )
    except Exception as exc:
        _warn_once("langsmith_create_span_failed", "LangSmith create_run(span) 失败，已降级为无追踪: %s", exc)
        yield None
        return
    run = _ManualRunHandle(
        client=client,
        project_name=parent.project_name,
        run_id=run_id,
        trace_id=parent.trace_id,
        dotted_order=dotted_order,
    )
    current_token = _CURRENT_RUN.set(run)
    try:
        yield run
    except Exception as exc:
        _end_run(run, error=str(exc))
        raise
    finally:
        _end_run(run)
        _CURRENT_RUN.reset(current_token)


def langsmith_trace(
    config: Optional["GraphRAGConfig"],
    *,
    name: str,
    run_type: str = "chain",
    inputs: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Iterator[Any]:
    """Backward-compatible alias using manual nested RunTree spans."""
    return langsmith_span(
        config,
        name=name,
        run_type=run_type,
        inputs=inputs,
        tags=tags,
        metadata=dict(metadata or {}),
    )
