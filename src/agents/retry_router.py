"""重试路由 Agent：统一管理重试预算、路由目标与重试元数据。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from agents.state import AgentState


class RetryRouterAgent:
    """Central retry controller for deterministic rollback/retry routing."""

    STAGE_DEFAULT_BUDGET = {
        "intent_recognition": 1,
        "tool_selection_review": 1,
        "task_planning": 1,
        "execution": 1,
        "specialist_analysis": 1,
        "summary": 1,
    }

    def __init__(self, config):
        self.max_total_retry = int(getattr(config, "retry_max_total", 3) or 3)
        self.max_by_stage = {
            stage: int(getattr(config, f"retry_max_{stage}", default) or default)
            for stage, default in self.STAGE_DEFAULT_BUDGET.items()
        }

    @staticmethod
    def _now_utc() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _resolve_target_stage(self, state: AgentState) -> str:
        target = str(state.get("retry_target_stage", "") or "").strip()
        if target:
            return target

        validation = state.get("validation_result", {}) or {}
        if bool(validation.get("retry_requested", False)):
            return str(validation.get("target_stage", "") or "execution")

        review = state.get("review_result", {}) or {}
        if bool(review.get("retry_requested", False)):
            return str(review.get("target_stage", "") or "summary")

        failed_stage = str(state.get("last_failed_stage", "") or "")
        return failed_stage or "summary"

    def _with_budget_check(self, state: AgentState, target_stage: str) -> Dict[str, Any]:
        retry_total = int(state.get("retry_count_total", 0) or 0)
        by_stage = dict(state.get("retry_count_by_stage", {}) or {})
        stage_used = int(by_stage.get(target_stage, 0) or 0)

        stage_budget = int(self.max_by_stage.get(target_stage, 1) or 1)
        exceeds = retry_total >= self.max_total_retry or stage_used >= stage_budget
        if exceeds:
            return {
                "allowed": False,
                "retry_count_total": retry_total,
                "retry_count_by_stage": by_stage,
            }

        by_stage[target_stage] = stage_used + 1
        return {
            "allowed": True,
            "retry_count_total": retry_total + 1,
            "retry_count_by_stage": by_stage,
        }

    def _clear_for_stage(self, target_stage: str) -> Dict[str, Any]:
        if target_stage == "intent_recognition":
            return {
                "force_reintent": True,
                "force_replan": False,
                "task_plan": [],
                "tool_calls": [],
                "tool_results": [],
                "analysis_report": {},
                "tool_output": "",
                "final_answer": "",
            }
        if target_stage == "task_planning":
            return {
                "force_replan": True,
                "task_plan": [],
                "tool_calls": [],
                "tool_results": [],
                "analysis_report": {},
                "tool_output": "",
                "final_answer": "",
            }
        if target_stage == "tool_selection_review":
            return {
                "force_reintent": False,
                "force_replan": False,
                "task_plan": [],
                "tool_calls": [],
                "tool_results": [],
                "analysis_report": {},
                "tool_output": "",
                "final_answer": "",
            }
        if target_stage == "execution":
            return {
                "tool_results": [],
                "analysis_report": {},
                "tool_output": "",
                "final_answer": "",
            }
        if target_stage == "specialist_analysis":
            return {
                "final_answer": "",
            }
        # summary
        return {
            "final_answer": "",
        }

    def run(self, state: AgentState) -> Dict[str, Any]:
        target_stage = self._resolve_target_stage(state)
        decision = self._with_budget_check(state=state, target_stage=target_stage)
        debug_steps = list(state.get("debug_steps", []) or [])
        retry_trace = list(state.get("retry_trace", []) or [])

        if not decision["allowed"]:
            reason = str(state.get("retry_reason", "") or "retry_budget_exhausted")
            retry_trace.append(
                {
                    "stage": "retry_router",
                    "target_stage": target_stage,
                    "accepted": False,
                    "reason": reason,
                    "at_utc": self._now_utc(),
                }
            )
            fallback_answer = str(state.get("final_answer", "") or "").strip()
            if not fallback_answer:
                fallback_answer = "当前请求在多次重试后仍未通过质量审查，请补充更明确的主体或稍后重试。"
            return {
                "final_answer": fallback_answer,
                "retry_budget_exhausted": True,
                "retry_target_stage": "",
                "retry_reason": reason,
                "retry_count_total": decision["retry_count_total"],
                "retry_count_by_stage": decision["retry_count_by_stage"],
                "retry_trace": retry_trace,
                "block_persistent_write": True,
                "debug_steps": debug_steps + [f"retry_router: denied(target={target_stage},budget_exhausted=True)"],
            }

        patch = self._clear_for_stage(target_stage)
        retry_trace.append(
            {
                "stage": "retry_router",
                "target_stage": target_stage,
                "accepted": True,
                "reason": str(state.get("retry_reason", "") or "quality_retry"),
                "at_utc": self._now_utc(),
                "retry_count_total": decision["retry_count_total"],
            }
        )
        return {
            **patch,
            "validation_result": {},
            "review_result": {},
            "quality_gate_passed": False,
            "quality_score": float(state.get("quality_score", 0.0) or 0.0),
            "block_persistent_write": False,
            "retry_budget_exhausted": False,
            "retry_target_stage": target_stage,
            "retry_count_total": decision["retry_count_total"],
            "retry_count_by_stage": decision["retry_count_by_stage"],
            "retry_trace": retry_trace,
            "execution_attempt": int(state.get("execution_attempt", 0) or 0)
            + (1 if target_stage == "execution" else 0),
            "summary_attempt": int(state.get("summary_attempt", 0) or 0)
            + (1 if target_stage == "summary" else 0),
            "debug_steps": debug_steps
            + [
                f"retry_router: accepted(target={target_stage},total={decision['retry_count_total']},stage={decision['retry_count_by_stage'].get(target_stage, 0)})"
            ],
        }
