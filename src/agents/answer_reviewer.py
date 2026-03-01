"""回答审查 Agent：对最终答案做质量审核并决定是否退回重试。"""

from __future__ import annotations

from typing import Any, Dict, List

from agents.state import AgentState


class AnswerReviewerAgent:
    """Review final answer quality with deterministic rules."""

    FAILURE_MARKERS = (
        "查询失败",
        "工具调用失败",
        "未找到工具",
        "系统错误",
        "请至少提供两个物品名称",
        "未获得可用结果",
    )

    def __init__(self, config):
        self.min_answer_chars = int(getattr(config, "answer_min_chars", 12) or 12)
        self.max_summary_retry = int(getattr(config, "summary_retry_max", 1) or 1)
        self.max_replan_retry = int(getattr(config, "replan_retry_max", 1) or 1)

    @staticmethod
    def _has_success_tool(results: List[Dict[str, Any]]) -> bool:
        for item in results:
            ok = item.get("ok")
            if isinstance(ok, bool):
                if ok:
                    return True
                continue
            output = str(item.get("output", "") or "")
            if output and not any(token in output for token in AnswerReviewerAgent.FAILURE_MARKERS):
                return True
        return False

    @staticmethod
    def _contains_failure_text(text: str) -> bool:
        return any(token in text for token in AnswerReviewerAgent.FAILURE_MARKERS)

    def _need_compare_entities(self, state: AgentState, answer: str) -> bool:
        tool_names = {str(x.get("tool_name", "") or "") for x in (state.get("tool_results", []) or []) if isinstance(x, dict)}
        if "df_multi_item_compare" not in tool_names:
            return False
        if "请至少提供两个物品名称" in answer:
            return True
        return False

    def run(self, state: AgentState) -> Dict[str, Any]:
        answer = str(state.get("final_answer", "") or "").strip()
        debug_steps = list(state.get("debug_steps", []) or [])
        tool_results = [x for x in (state.get("tool_results", []) or []) if isinstance(x, dict)]
        retry_count_by_stage = dict(state.get("retry_count_by_stage", {}) or {})

        review: Dict[str, Any] = {
            "passed": True,
            "score": 1.0,
            "reason": "ok",
            "retry_requested": False,
            "target_stage": "",
            "hints": [],
        }

        if not answer:
            review.update(
                {
                    "passed": False,
                    "score": 0.0,
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
                    "score": 0.35,
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
                    "score": 0.15,
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
                    "score": 0.3,
                    "reason": "answer_conflicts_with_successful_tools",
                    "retry_requested": int(retry_count_by_stage.get("summary", 0) or 0) < self.max_summary_retry,
                    "target_stage": "summary",
                    "hints": ["工具已有可用结果，请不要输出失败文案，改为基于可用结果作答。"],
                }
            )

        trace = list(state.get("retry_trace", []) or [])
        if not review["passed"]:
            trace.append(
                {
                    "stage": "answer_reviewer",
                    "reason": review["reason"],
                    "retry_requested": bool(review["retry_requested"]),
                    "target_stage": review["target_stage"],
                }
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
