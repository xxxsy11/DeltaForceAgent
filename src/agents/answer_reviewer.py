"""回答审查 Agent：对最终答案做质量审核并决定是否退回重试。"""

from __future__ import annotations

from typing import Any, Dict, List

from agents.output_quality import (
    EMPTY_RESULT_TEXT,
    MISSING_COMPARE_ENTITIES_TEXT,
    has_success_tool_result,
    is_failure_text,
)
from agents.retry_shared import append_retry_trace
from agents.state import AgentState


class AnswerReviewerAgent:
    """Review final answer quality with deterministic rules."""

    SCORE_PASS = 1.0
    SCORE_EMPTY = 0.0
    SCORE_TOO_SHORT = 0.35
    SCORE_COMPARE_ENTITY_UNRESOLVED = 0.15
    SCORE_CONFLICT_WITH_TOOL = 0.3
    DEFAULT_MIN_ANSWER_CHARS = 12

    def __init__(self, config):
        self.min_answer_chars = int(getattr(config, "answer_min_chars", self.DEFAULT_MIN_ANSWER_CHARS) or self.DEFAULT_MIN_ANSWER_CHARS)
        # 兼容旧字段 `summary_retry_max`，优先使用当前配置键 `retry_max_summary`。
        max_summary_retry = getattr(config, "retry_max_summary", getattr(config, "summary_retry_max", 1))
        self.max_summary_retry = max(0, int(max_summary_retry or 1))
        self.max_replan_retry = max(0, int(getattr(config, "replan_retry_max", 1) or 1))

    @staticmethod
    def _has_success_tool(results: List[Dict[str, Any]]) -> bool:
        return has_success_tool_result(results)

    @staticmethod
    def _contains_failure_text(text: str) -> bool:
        return is_failure_text(text, extra_markers=(MISSING_COMPARE_ENTITIES_TEXT, EMPTY_RESULT_TEXT))

    def _need_compare_entities(self, state: AgentState, answer: str) -> bool:
        tool_names = {str(x.get("tool_name", "") or "") for x in (state.get("tool_results", []) or []) if isinstance(x, dict)}
        if "df_multi_item_compare" not in tool_names:
            return False
        if MISSING_COMPARE_ENTITIES_TEXT in answer:
            return True
        return False

    async def run(self, state: AgentState) -> Dict[str, Any]:
        answer = str(state.get("final_answer", "") or "").strip()
        debug_steps = list(state.get("debug_steps", []) or [])
        tool_results = [x for x in (state.get("tool_results", []) or []) if isinstance(x, dict)]
        retry_count_by_stage = dict(state.get("retry_count_by_stage", {}) or {})

        review: Dict[str, Any] = {
            "passed": True,
            "score": self.SCORE_PASS,
            "reason": "ok",
            "retry_requested": False,
            "target_stage": "",
            "hints": [],
        }

        if not answer:
            review.update(
                {
                    "passed": False,
                    "score": self.SCORE_EMPTY,
                    "reason": "empty_answer",
                    "retry_requested": int(retry_count_by_stage.get("summary", 0) or 0) < self.max_summary_retry,
                    "target_stage": "summary",
                    "hints": ["答案为空，请基于已有工具结果重新组织完整回答。"],
                }
            )
        elif len(answer) < self.min_answer_chars:
            review.update(
                {
                    "passed": False,
                    "score": self.SCORE_TOO_SHORT,
                    "reason": "too_short",
                    "retry_requested": int(retry_count_by_stage.get("summary", 0) or 0) < self.max_summary_retry,
                    "target_stage": "summary",
                    "hints": ["答案过短，请补齐结论与关键依据。"],
                }
            )
        elif self._need_compare_entities(state, answer):
            review.update(
                {
                    "passed": False,
                    "score": self.SCORE_COMPARE_ENTITY_UNRESOLVED,
                    "reason": "compare_entity_unresolved",
                    "retry_requested": int(retry_count_by_stage.get("task_planning", 0) or 0) < self.max_replan_retry,
                    "target_stage": "task_planning",
                    "hints": ["对比对象未解析，请基于上下文实体重建 compare 工具查询。"],
                }
            )
        elif self._contains_failure_text(answer) and self._has_success_tool(tool_results):
            review.update(
                {
                    "passed": False,
                    "score": self.SCORE_CONFLICT_WITH_TOOL,
                    "reason": "answer_conflicts_with_successful_tools",
                    "retry_requested": int(retry_count_by_stage.get("summary", 0) or 0) < self.max_summary_retry,
                    "target_stage": "summary",
                    "hints": ["工具已有可用结果，请不要输出失败文案，改为基于可用结果作答。"],
                }
            )

        trace = list(state.get("retry_trace", []) or [])
        if not review["passed"]:
            trace = append_retry_trace(
                trace,
                stage="answer_reviewer",
                reason=str(review["reason"] or ""),
                retry_requested=bool(review["retry_requested"]),
                target_stage=str(review["target_stage"] or ""),
            )

        return {
            "review_result": review,
            "quality_score": float(review["score"]),
            "quality_gate_passed": bool(review["passed"]),
            "retry_target_stage": review["target_stage"] if review["retry_requested"] else "",
            "retry_reason": review["reason"] if review["retry_requested"] else "",
            "last_failed_stage": "summary" if not review["passed"] else "",
            "retry_trace": trace,
            "block_persistent_write": bool(state.get("block_persistent_write", False) or (not review["passed"])),
            "debug_steps": debug_steps
            + [
                (
                    "answer_reviewer: pass"
                    if review["passed"]
                    else f"answer_reviewer: fail(reason={review['reason']},retry={review['retry_requested']},target={review['target_stage'] or 'none'})"
                )
            ],
        }
