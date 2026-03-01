"""LangGraph 状态定义（支持 A2A 通讯）。"""

from typing import Any, Dict, List, TypedDict


class AgentToolCall(TypedDict):
    tool_name: str
    tool_query: str


class AgentToolResult(TypedDict):
    tool_name: str
    tool_query: str
    output: str
    ok: bool
    error_code: str
    error_type: str
    retryable: bool
    stage: str
    diagnostics: Dict[str, Any]


class AgentMessage(TypedDict):
    from_agent: str
    to_agent: str
    message_type: str
    payload: Dict[str, Any]


class AgentState(TypedDict):
    user_id: str
    session_id: str
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
    memory_context: str
    memory_pending_digest: str
    memory_recent_raw: List[Dict[str, str]]
    memory_pending_buffer: List[Dict[str, str]]
    memory_rolling_summary: str
    memory_merge_count: int
    memory_persistent_context: str
    memory_persistent_entities: List[str]
    memory_persistent_hits: List[Dict[str, Any]]
    memory_persistent_used: bool
    memory_persistent_gate_score: int
    memory_keyword_candidates: List[str]
    memory_fact_candidates: List[Dict[str, Any]]
    selected_skill: str
    skill_reason: str
    skill_confidence: float
    skill_matched_by: List[str]
    skill_locked_plan: bool
    skill_tool_chain: List[AgentToolCall]

    # 审查与重试状态
    retry_count_total: int
    retry_count_by_stage: Dict[str, int]
    retry_budget_exhausted: bool
    retry_trace: List[Dict[str, Any]]
    retry_target_stage: str
    retry_reason: str
    force_replan: bool
    force_reintent: bool
    execution_attempt: int
    summary_attempt: int
    validation_result: Dict[str, Any]
    review_result: Dict[str, Any]
    quality_score: float
    quality_gate_passed: bool
    block_persistent_write: bool
    last_failed_stage: str
    last_error_type: str
    last_error_code: str
    attempt_id: str
    intermediate_artifacts: Dict[str, Any]

    understanding_entities: List[str]
    understanding_entity_count: int
    understanding_confidence: float
    understanding_compare_target_count: int
    tool_output: str
    final_answer: str
    debug_steps: List[str]
