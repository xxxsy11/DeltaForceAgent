"""LangGraph 主Agent+子Agent A2A 编排。"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.execution_agent import ExecutionAgent
from agents.intent_recognition import IntentRecognitionAgent
from agents.main_orchestrator import MainOrchestratorAgent
from agents.specialist_analysis import SpecialistAnalysisAgent
from agents.state import AgentState
from agents.summary_agent import SummaryAgent
from agents.task_planning import TaskPlanningAgent
from agents.tool_planner import LLMToolPlanner
from memory import PersistentMemoryStore
from memory.components import (
    MemoryCompressionAgent,
    PersistentMemoryRecallNode,
    PersistentMemoryWriteNode,
)
from tools import ToolRegistry


def _route_after_intent(state: AgentState) -> str:
    if not state.get("tool_calls"):
        return "summary"
    if state.get("flow_type") == "complex" and state.get("requires_task_planning", False):
        return "task_planning"
    return "execution"


def _route_after_execution(state: AgentState) -> str:
    if state.get("flow_type") == "complex" and state.get("requires_specialist_analysis", False):
        return "specialist_analysis"
    return "summary"


def build_multi_agent_graph(registry: ToolRegistry, persistent_store: PersistentMemoryStore | None = None):
    config = registry.config
    store = persistent_store or PersistentMemoryStore(config)

    orchestrator = MainOrchestratorAgent()
    persistent_recall = PersistentMemoryRecallNode(store=store, config=config)
    intent_agent = IntentRecognitionAgent(config=config, available_tools=registry.list_tools())
    task_planner = TaskPlanningAgent(
        planner=LLMToolPlanner(config=config, model_name=config.agent_planner_model),
        registry=registry,
    )
    execution = ExecutionAgent(registry=registry, sell_fee_rate=0.13)
    specialist = SpecialistAnalysisAgent(model_name=config.agent_specialist_model)
    summary = SummaryAgent(planner=LLMToolPlanner(config=config, model_name=config.agent_summary_model))
    memory_compression = MemoryCompressionAgent(config=config)
    persistent_write = PersistentMemoryWriteNode(store=store, config=config)

    builder = StateGraph(AgentState)
    builder.add_node("main_orchestrator", orchestrator.run)
    builder.add_node("persistent_memory_recall", persistent_recall.run)
    builder.add_node("intent_recognition", intent_agent.run)
    builder.add_node("task_planning", task_planner.run)
    builder.add_node("execution", execution.run)
    builder.add_node("specialist_analysis", specialist.run)
    builder.add_node("summary", summary.run)
    builder.add_node("memory_compression", memory_compression.run)
    builder.add_node("persistent_memory_write", persistent_write.run)

    builder.add_edge(START, "main_orchestrator")
    builder.add_edge("main_orchestrator", "persistent_memory_recall")
    builder.add_edge("persistent_memory_recall", "intent_recognition")
    builder.add_conditional_edges(
        "intent_recognition",
        _route_after_intent,
        {
            "task_planning": "task_planning",
            "execution": "execution",
            "summary": "summary",
        },
    )
    builder.add_edge("task_planning", "execution")
    builder.add_conditional_edges(
        "execution",
        _route_after_execution,
        {
            "specialist_analysis": "specialist_analysis",
            "summary": "summary",
        },
    )
    builder.add_edge("specialist_analysis", "summary")
    builder.add_edge("summary", "memory_compression")
    builder.add_edge("memory_compression", "persistent_memory_write")
    builder.add_edge("persistent_memory_write", END)
    return builder.compile()
