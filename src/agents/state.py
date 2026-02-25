"""LangGraph 状态定义（支持 A2A 通讯）。"""

from typing import Any, Dict, List, TypedDict


class AgentToolCall(TypedDict):
    tool_name: str
    tool_query: str


class AgentToolResult(TypedDict):
    tool_name: str
    tool_query: str
    output: str


class AgentMessage(TypedDict):
    from_agent: str
    to_agent: str
    message_type: str
    payload: Dict[str, Any]


class AgentState(TypedDict):
    user_query: str
    intent: str
    intent_reason: str
    flow_type: str
    plan_source: str
    requires_task_planning: bool
    requires_specialist_analysis: bool
    selected_tool: str
    tool_query: str
    tool_calls: List[AgentToolCall]
    task_plan: List[AgentToolCall]
    tool_results: List[AgentToolResult]
    analysis_report: Dict[str, Any]
    agent_messages: List[AgentMessage]
    orchestration_meta: Dict[str, Any]
    tool_output: str
    final_answer: str
    debug_steps: List[str]
