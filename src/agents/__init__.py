"""Multi-Agent 编排层。"""

import asyncio
from typing import Optional

from config import GraphRAGConfig

__all__ = ["run_agent_query", "run_agent_interactive"]


async def run_agent_query(query: str, *, user_id: str = "user-0001", session_id: str = "chat-0001", config: Optional[GraphRAGConfig] = None) -> str:
    """Lazy import runner to avoid package-import side effects."""
    from .runner import run_agent_query as _run_agent_query

    return await _run_agent_query(query, user_id=user_id, session_id=session_id, config=config)


def run_agent_interactive(config: Optional[GraphRAGConfig] = None) -> None:
    """Lazy import runner to avoid package-import side effects."""
    from .runner import run_agent_interactive as _run_agent_interactive

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_run_agent_interactive(config=config))
        return
    raise RuntimeError(
        "run_agent_interactive() cannot be called inside an active event loop. "
        "Use `await agents.runner.run_agent_interactive(...)` instead."
    )
