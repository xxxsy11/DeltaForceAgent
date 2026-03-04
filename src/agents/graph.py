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

DEFAULT_SELL_FEE_RATE = 0.13
MS_PER_SECOND = 1000

STAGE_LABELS = {
    "main_orchestrator": "主流程编排",
    "persistent_memory_recall": "长期记忆召回",
    "intent_recognition": "意图识别",
    "tool_selection_review": "工具选择复核",
    "task_planning": "任务规划",
    "execution": "工具执行",
    "tool_output_validator": "结果校验",
    "specialist_analysis": "专业分析",
    "summary": "回答生成",
    "answer_reviewer": "回答审查",
    "retry_router": "重试路由",
    "memory_compression": "记忆压缩",
    "persistent_memory_write": "长期记忆写入",
}


def _format_elapsed_cn(elapsed_ms: float) -> str:
    value = float(elapsed_ms or 0.0)
    if value >= MS_PER_SECOND:
        return f"{value / MS_PER_SECOND:.2f} 秒"
    return f"{value:.2f} 毫秒"


def _build_stage_detail(stage_name: str, state: AgentState, update: dict) -> str:
    if stage_name == "intent_recognition":
        intent = str(update.get("intent") or state.get("intent") or "").strip()
        selected_tool = str(
            update.get("selected_tool")
            or state.get("selected_tool")
            or ""
        ).strip()
        if not selected_tool:
            tool_calls = update.get("tool_calls") or state.get("tool_calls") or []
            if tool_calls:
                selected_tool = str((tool_calls[0] or {}).get("tool_name", "")).strip()
        parts = []
        if intent:
            parts.append(f"意图={intent}")
        if selected_tool:
            parts.append(f"工具={selected_tool}")
        return "，".join(parts)

    if stage_name == "task_planning":
        calls = update.get("tool_calls") or state.get("tool_calls") or []
        if not calls:
            return ""
        first_tool = str((calls[0] or {}).get("tool_name", "")).strip()
        return f"计划工具数={len(calls)}，首工具={first_tool or 'none'}"

    if stage_name == "execution":
        results = update.get("tool_results") or state.get("tool_results") or []
        if not isinstance(results, list) or not results:
            return ""
        success = 0
        for item in results:
            if isinstance(item, dict) and bool(item.get("ok", False)):
                success += 1
        failed = len(results) - success
        return f"执行成功={success}，失败={failed}"

    if stage_name == "tool_output_validator":
        validation = update.get("validation_result") or state.get("validation_result") or {}
        if not isinstance(validation, dict):
            return ""
        passed = bool(validation.get("passed", False))
        reason = str(validation.get("reason", "") or "").strip()
        return f"校验={'通过' if passed else '未通过'}" + (f"，原因={reason}" if reason else "")

    if stage_name == "retry_router":
        target = str(update.get("retry_target_stage") or state.get("retry_target_stage") or "").strip()
        if target:
            return f"回退目标={STAGE_LABELS.get(target, target)}"
    return ""


def _with_stage_trace_async(stage_name: str, fn, enable_trace: bool):
    async def _wrapped(state: AgentState):
        orchestration_meta = dict(state.get("orchestration_meta", {}) or {})
        stage_label = STAGE_LABELS.get(stage_name, stage_name)
        if enable_trace:
            print(f"\n[流程] 开始：{stage_label}", flush=True)

        started = time.perf_counter()
        update = await fn(state)
        update = update or {}
        elapsed_ms = round((time.perf_counter() - started) * MS_PER_SECOND, 2)

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
            detail = _build_stage_detail(stage_name=stage_name, state=state, update=update)
            if detail:
                print(f"[流程] 完成：{stage_label}（用时 {_format_elapsed_cn(elapsed_ms)}，{detail}）", flush=True)
            else:
                print(f"[流程] 完成：{stage_label}（用时 {_format_elapsed_cn(elapsed_ms)}）", flush=True)
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


def build_multi_agent_graph(
    registry: ToolRegistry,
    persistent_store: PersistentMemoryStore | None = None,
):
    """
    异步版本 LangGraph，调用端使用 graph.ainvoke(...)。
    """
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
    sell_fee_rate = float(getattr(config, "sell_fee_rate", DEFAULT_SELL_FEE_RATE) or DEFAULT_SELL_FEE_RATE)
    execution = ExecutionAgent(registry=registry, sell_fee_rate=sell_fee_rate)
    validator = ToolOutputValidatorAgent(config=config)
    specialist = SpecialistAnalysisAgent(model_name=config.agent_specialist_model)
    summary = SummaryAgent(planner=LLMToolPlanner(config=config, model_name=config.agent_summary_model))
    reviewer = AnswerReviewerAgent(config=config)
    retry_router = RetryRouterAgent(config=config)
    memory_compression = MemoryCompressionAgent(config=config)
    persistent_write = PersistentMemoryWriteNode(store=store, config=config)

    builder = StateGraph(AgentState)
    builder.add_node("main_orchestrator", _with_stage_trace_async("main_orchestrator", orchestrator.run, stage_trace_enabled))
    builder.add_node("persistent_memory_recall", _with_stage_trace_async("persistent_memory_recall", persistent_recall.run, stage_trace_enabled))
    builder.add_node("intent_recognition", _with_stage_trace_async("intent_recognition", intent_agent.run, stage_trace_enabled))
    builder.add_node("tool_selection_review", _with_stage_trace_async("tool_selection_review", tool_selection_review.run, stage_trace_enabled))
    builder.add_node("task_planning", _with_stage_trace_async("task_planning", task_planner.run, stage_trace_enabled))
    builder.add_node("execution", _with_stage_trace_async("execution", execution.run, stage_trace_enabled))
    builder.add_node("tool_output_validator", _with_stage_trace_async("tool_output_validator", validator.run, stage_trace_enabled))
    builder.add_node("specialist_analysis", _with_stage_trace_async("specialist_analysis", specialist.run, stage_trace_enabled))
    builder.add_node("summary", _with_stage_trace_async("summary", summary.run, stage_trace_enabled))
    builder.add_node("answer_reviewer", _with_stage_trace_async("answer_reviewer", reviewer.run, stage_trace_enabled))
    builder.add_node("retry_router", _with_stage_trace_async("retry_router", retry_router.run, stage_trace_enabled))
    builder.add_node("memory_compression", _with_stage_trace_async("memory_compression", memory_compression.run, stage_trace_enabled))
    builder.add_node("persistent_memory_write", _with_stage_trace_async("persistent_memory_write", persistent_write.run, stage_trace_enabled))

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
