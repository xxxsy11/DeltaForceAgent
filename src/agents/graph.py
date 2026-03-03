"""LangGraph 主Agent+子Agent A2A 编排。"""

from __future__ import annotations

import time
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
from agents.tool_selection_review import ToolSelectionReviewAgent
from agents.tool_output_validator import ToolOutputValidatorAgent
from agents.tool_planner import LLMToolPlanner
from memory import PersistentMemoryStore
from memory.components import (
    MemoryCompressionAgent,
    PersistentMemoryRecallNode,
    PersistentMemoryWriteNode,
)
from tools import ToolRegistry


def _with_stage_trace(stage_name: str, fn, enable_trace: bool):
    def _wrapped(state: AgentState):
        orchestration_meta = dict(state.get("orchestration_meta", {}) or {})
        if enable_trace:
            prev_name = str(orchestration_meta.get("last_stage_name", "") or "").strip()
            prev_ms = orchestration_meta.get("last_stage_elapsed_ms")
            if prev_name and prev_ms is not None:
                try:
                    print(f"[Stage] -> {stage_name} | 上一步 {prev_name}: {float(prev_ms):.2f} ms", flush=True)
                except Exception:
                    print(f"[Stage] -> {stage_name}", flush=True)
            else:
                print(f"[Stage] -> {stage_name}", flush=True)

        started = time.perf_counter()
        update = fn(state) or {}
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

        out_meta = dict(orchestration_meta)
        update_meta = update.get("orchestration_meta")
        if isinstance(update_meta, dict):
            out_meta.update(update_meta)
        timings = dict(out_meta.get("stage_timings_ms", {}) or {})
        timings[stage_name] = elapsed_ms
        out_meta["stage_timings_ms"] = timings
        out_meta["last_stage_name"] = stage_name
        out_meta["last_stage_elapsed_ms"] = elapsed_ms
        update["orchestration_meta"] = out_meta

        existing_debug = update.get("debug_steps")
        if isinstance(existing_debug, list):
            debug_steps = existing_debug
        else:
            debug_steps = list(state.get("debug_steps", []) or [])
        debug_steps.append(f"stage_timing: {stage_name}={elapsed_ms}ms")
        update["debug_steps"] = debug_steps

        if enable_trace:
            print(f"[Stage] {stage_name} done: {elapsed_ms:.2f} ms", flush=True)
        return update

    return _wrapped


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
    if target in {"intent_recognition", "tool_selection_review", "task_planning", "execution", "specialist_analysis", "summary"}:
        return target
    return "summary"


def build_multi_agent_graph(registry: ToolRegistry, persistent_store: PersistentMemoryStore | None = None):
    config = registry.config
    stage_trace_enabled = bool(getattr(config, "agent_stage_trace_enabled", True))
    store = persistent_store or PersistentMemoryStore(config)

    orchestrator = MainOrchestratorAgent()
    persistent_recall = PersistentMemoryRecallNode(store=store, config=config)
    intent_agent = IntentRecognitionAgent(config=config, available_tools=registry.list_tools())
    tool_selection_review = ToolSelectionReviewAgent(
        planner=LLMToolPlanner(config=config, model_name=config.agent_planner_model),
        registry=registry,
    )
    task_planner = TaskPlanningAgent(
        planner=LLMToolPlanner(config=config, model_name=config.agent_planner_model),
        registry=registry,
    )
    execution = ExecutionAgent(registry=registry, sell_fee_rate=float(getattr(config, "sell_fee_rate", 0.13) or 0.13))
    validator = ToolOutputValidatorAgent(config=config)
    specialist = SpecialistAnalysisAgent(model_name=config.agent_specialist_model)
    summary = SummaryAgent(planner=LLMToolPlanner(config=config, model_name=config.agent_summary_model))
    reviewer = AnswerReviewerAgent(config=config)
    retry_router = RetryRouterAgent(config=config)
    memory_compression = MemoryCompressionAgent(config=config)
    persistent_write = PersistentMemoryWriteNode(store=store, config=config)

    builder = StateGraph(AgentState)
    builder.add_node("main_orchestrator", _with_stage_trace("main_orchestrator", orchestrator.run, stage_trace_enabled))
    builder.add_node("persistent_memory_recall", _with_stage_trace("persistent_memory_recall", persistent_recall.run, stage_trace_enabled))
    builder.add_node("intent_recognition", _with_stage_trace("intent_recognition", intent_agent.run, stage_trace_enabled))
    builder.add_node("tool_selection_review", _with_stage_trace("tool_selection_review", tool_selection_review.run, stage_trace_enabled))
    builder.add_node("task_planning", _with_stage_trace("task_planning", task_planner.run, stage_trace_enabled))
    builder.add_node("execution", _with_stage_trace("execution", execution.run, stage_trace_enabled))
    builder.add_node("tool_output_validator", _with_stage_trace("tool_output_validator", validator.run, stage_trace_enabled))
    builder.add_node("specialist_analysis", _with_stage_trace("specialist_analysis", specialist.run, stage_trace_enabled))
    builder.add_node("summary", _with_stage_trace("summary", summary.run, stage_trace_enabled))
    builder.add_node("answer_reviewer", _with_stage_trace("answer_reviewer", reviewer.run, stage_trace_enabled))
    builder.add_node("retry_router", _with_stage_trace("retry_router", retry_router.run, stage_trace_enabled))
    builder.add_node("memory_compression", _with_stage_trace("memory_compression", memory_compression.run, stage_trace_enabled))
    builder.add_node("persistent_memory_write", _with_stage_trace("persistent_memory_write", persistent_write.run, stage_trace_enabled))

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
            "tool_selection_review": "tool_selection_review",
            "task_planning": "task_planning",
            "execution": "execution",
            "specialist_analysis": "specialist_analysis",
            "summary": "summary",
            "memory_compression": "memory_compression",
        },
    )
    builder.add_edge("tool_selection_review", "execution")
    builder.add_edge("memory_compression", "persistent_memory_write")
    builder.add_edge("persistent_memory_write", END)
    return builder.compile()
