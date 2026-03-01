"""LangGraph 主Agent+子Agent A2A 编排。"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.answer_reviewer import AnswerReviewerAgent
from agents.execution_agent import ExecutionAgent
from agents.intent_recognition import IntentRecognitionAgent
from agents.main_orchestrator import MainOrchestratorAgent
from agents.retry_router import RetryRouterAgent
from agents.specialist_analysis import SpecialistAnalysisAgent
from agents.state import AgentState
from agents.summary_agent import SummaryAgent
from agents.task_planning import TaskPlanningAgent
from agents.tool_output_validator import ToolOutputValidatorAgent
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


def _route_after_validator(state: AgentState) -> str:
    validation = state.get("validation_result", {}) or {}
    if bool(validation.get("retry_requested", False)):
        return "retry_router"
    if state.get("flow_type") == "complex" and state.get("requires_specialist_analysis", False):
        return "specialist_analysis"
    return "summary"


def _route_after_reviewer(state: AgentState) -> str:
    review = state.get("review_result", {}) or {}
    if bool(review.get("retry_requested", False)):
        return "retry_router"
    return "memory_compression"


def _route_after_retry_router(state: AgentState) -> str:
    if bool(state.get("retry_budget_exhausted", False)):
        return "memory_compression"
    target = str(state.get("retry_target_stage", "") or "").strip()
    if target in {"intent_recognition", "task_planning", "execution", "specialist_analysis", "summary"}:
        return target
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
    validator = ToolOutputValidatorAgent(config=config)
    specialist = SpecialistAnalysisAgent(model_name=config.agent_specialist_model)
    summary = SummaryAgent(planner=LLMToolPlanner(config=config, model_name=config.agent_summary_model))
    reviewer = AnswerReviewerAgent(config=config)
    retry_router = RetryRouterAgent(config=config)
    memory_compression = MemoryCompressionAgent(config=config)
    persistent_write = PersistentMemoryWriteNode(store=store, config=config)

    builder = StateGraph(AgentState)
    builder.add_node("main_orchestrator", orchestrator.run)
    builder.add_node("persistent_memory_recall", persistent_recall.run)
    builder.add_node("intent_recognition", intent_agent.run)
    builder.add_node("task_planning", task_planner.run)
    builder.add_node("execution", execution.run)
    builder.add_node("tool_output_validator", validator.run)
    builder.add_node("specialist_analysis", specialist.run)
    builder.add_node("summary", summary.run)
    builder.add_node("answer_reviewer", reviewer.run)
    builder.add_node("retry_router", retry_router.run)
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
    builder.add_edge("execution", "tool_output_validator")
    builder.add_conditional_edges(
        "tool_output_validator",
        _route_after_validator,
        {
            "retry_router": "retry_router",
            "specialist_analysis": "specialist_analysis",
            "summary": "summary",
        },
    )
    builder.add_edge("specialist_analysis", "summary")
    builder.add_edge("summary", "answer_reviewer")
    builder.add_conditional_edges(
        "answer_reviewer",
        _route_after_reviewer,
        {
            "retry_router": "retry_router",
            "memory_compression": "memory_compression",
        },
    )
    builder.add_conditional_edges(
        "retry_router",
        _route_after_retry_router,
        {
            "intent_recognition": "intent_recognition",
            "task_planning": "task_planning",
            "execution": "execution",
            "specialist_analysis": "specialist_analysis",
            "summary": "summary",
            "memory_compression": "memory_compression",
        },
    )
    builder.add_edge("memory_compression", "persistent_memory_write")
    builder.add_edge("persistent_memory_write", END)
    return builder.compile()
