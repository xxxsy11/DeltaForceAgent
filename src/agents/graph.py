"""
Multi-Agent LangGraph：
- Intent 节点做意图分析
- Planner 节点选择工具
- Tool 节点执行
- Responder 输出
"""

from typing import Dict

from langgraph.graph import END, START, StateGraph

from tools import ToolRegistry
from agents.intent_analyzer import IntentAnalyzer
from agents.state import AgentState


def _intent_node(state: AgentState, analyzer: IntentAnalyzer) -> Dict:
    decision = analyzer.analyze(state.get("user_query", ""))
    return {
        "intent": decision.intent,
        "selected_tool": decision.tool_name,
        "tool_query": decision.tool_query,
        "debug_steps": state.get("debug_steps", []) + [f"intent: {decision.reason}"],
    }


def _tool_exec_node(state: AgentState, registry: ToolRegistry) -> Dict:
    tool_name = state.get("selected_tool", "none")
    query = state.get("tool_query", "")
    if tool_name == "none":
        return {
            "tool_output": "未提供有效问题。",
            "debug_steps": state.get("debug_steps", []) + ["tool: none"],
        }

    output = registry.invoke(tool_name, query)
    return {
        "tool_output": output,
        "debug_steps": state.get("debug_steps", []) + [f"tool: {tool_name}"],
    }


def _response_node(state: AgentState) -> Dict:
    output = (state.get("tool_output") or "").strip()
    if not output:
        output = "未获得可用结果。"
    return {
        "final_answer": output,
        "debug_steps": state.get("debug_steps", []) + ["responder: done"],
    }


def _route_after_intent(state: AgentState) -> str:
    if state.get("selected_tool") == "none":
        return "responder"
    return "tool_exec"


def build_multi_agent_graph(registry: ToolRegistry):
    analyzer = IntentAnalyzer()
    builder = StateGraph(AgentState)
    builder.add_node("intent", lambda s: _intent_node(s, analyzer))
    builder.add_node("tool_exec", lambda s: _tool_exec_node(s, registry))
    builder.add_node("responder", _response_node)

    builder.add_edge(START, "intent")
    builder.add_conditional_edges(
        "intent",
        _route_after_intent,
        {
            "tool_exec": "tool_exec",
            "responder": "responder",
        },
    )
    builder.add_edge("tool_exec", "responder")
    builder.add_edge("responder", END)
    return builder.compile()
