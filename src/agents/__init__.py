"""
Multi-Agent 编排层
"""

def run_agent_query(*args, **kwargs):
    from .runner import run_agent_query as _run_agent_query
    return _run_agent_query(*args, **kwargs)


def run_agent_interactive(*args, **kwargs):
    from .runner import run_agent_interactive as _run_agent_interactive
    return _run_agent_interactive(*args, **kwargs)

__all__ = ["run_agent_query", "run_agent_interactive"]
