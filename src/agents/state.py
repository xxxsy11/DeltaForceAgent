"""
LangGraph 状态定义
"""

from typing import List, TypedDict


class AgentState(TypedDict):
    user_query: str
    intent: str
    selected_tool: str
    tool_query: str
    tool_output: str
    final_answer: str
    debug_steps: List[str]
