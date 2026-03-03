"""工具输出审查 Agent：校验执行阶段结果质量并决定是否触发重试。"""

from __future__ import annotations

from typing import Any, Dict, List

from agents.state import AgentState


class ToolOutputValidatorAgent:
    """Rule-first validator for execution outputs."""

    FAILURE_MARKERS = ("工具调用失败", "查询失败", "未找到工具", "系统错误", "不可用")

    def __init__(self, config):
        self.max_execution_retry = int(getattr(config, "execution_retry_max", 1) or 1)
        self.max_replan_retry = int(getattr(config, "replan_retry_max", 1) or 1)
        self.max_reintent_retry = int(getattr(config, "retry_max_intent_recognition", 1) or 1)
        self.max_tool_selection_review_retry = int(getattr(config, "retry_max_tool_selection_review", 1) or 1)

    @staticmethod
    def _is_failed_result(item: Dict[str, Any]) -> bool:
        ok = item.get("ok")
        if isinstance(ok, bool):
            return not ok
        output = str(item.get("output", "") or "").strip()
        if not output:
            return True
        return any(token in output for token in ToolOutputValidatorAgent.FAILURE_MARKERS)

    @staticmethod
    def _is_retryable(item: Dict[str, Any]) -> bool:
        retryable = item.get("retryable")
        if isinstance(retryable, bool):
            return retryable
        output = str(item.get("output", "") or "")
        return "HTTP 5" in output or "429" in output or "超时" in output

    @staticmethod
    def _needs_replan(item: Dict[str, Any]) -> bool:
        text = str(item.get("output", "") or "")
        if "请至少提供两个物品名称" in text:
            return True
        if "未能根据 objectName 匹配到交易物品ID" in text:
            return True
        error_type = str(item.get("error_type", "") or "")
        return error_type in {"missing_entities", "entity_not_found"}

    def run(self, state: AgentState) -> Dict[str, Any]:
        tool_calls = state.get("tool_calls", []) or []
        results = state.get("tool_results", []) or []
        debug_steps = list(state.get("debug_steps", []) or [])

        if not tool_calls:
            validation = {
                "passed": True,
                "reason": "no_tool_call",
                "retry_requested": False,
                "target_stage": "",
                "failure_count": 0,
                "success_count": 0,
            }
            return {
                "validation_result": validation,
                "quality_gate_passed": True,
                "quality_score": float(state.get("quality_score", 1.0) or 1.0),
                "retry_target_stage": "",
                "retry_reason": "",
                "last_failed_stage": "",
                "block_persistent_write": False,
                "debug_steps": debug_steps + ["tool_output_validator: pass(no_tool_call)"],
            }

        failed = [item for item in results if self._is_failed_result(item)]
        success_count = max(0, len(results) - len(failed))
        failure_count = len(failed)
        quality_score = 1.0 if not results else max(0.0, min(1.0, success_count / len(results)))

        if failure_count == 0:
            validation = {
                "passed": True,
                "reason": "all_tools_success",
                "retry_requested": False,
                "target_stage": "",
                "failure_count": 0,
                "success_count": success_count,
            }
            return {
                "validation_result": validation,
                "quality_gate_passed": True,
                "quality_score": quality_score,
                "retry_target_stage": "",
                "retry_reason": "",
                "last_failed_stage": "",
                "block_persistent_write": False,
                "debug_steps": debug_steps + ["tool_output_validator: pass(all_tools_success)"],
            }

        retry_count_by_stage = dict(state.get("retry_count_by_stage", {}) or {})
        execution_retry_used = int(retry_count_by_stage.get("execution", 0) or 0)
        planning_retry_used = int(retry_count_by_stage.get("task_planning", 0) or 0)
        tool_selection_review_used = int(retry_count_by_stage.get("tool_selection_review", 0) or 0)

        needs_replan = any(self._needs_replan(item) for item in failed)
        retryable = any(self._is_retryable(item) for item in failed)

        target_stage = ""
        reason = "tool_failure"
        retry_requested = False

        if needs_replan:
            # 复杂链路才走 task_planning 重排；简单问题优先回到 intent_recognition 做主体重解。
            is_complex_flow = bool(state.get("flow_type") == "complex") and bool(state.get("requires_task_planning", False))
            if (not is_complex_flow) and tool_selection_review_used < self.max_tool_selection_review_retry:
                retry_requested = True
                target_stage = "tool_selection_review"
                reason = "tool_selection_audit_retry"
            elif is_complex_flow and planning_retry_used < self.max_replan_retry:
                retry_requested = True
                target_stage = "task_planning"
                reason = "tool_query_invalid_or_entity_missing"
            else:
                reintent_used = int(retry_count_by_stage.get("intent_recognition", 0) or 0)
                if reintent_used < self.max_reintent_retry:
                    retry_requested = True
                    target_stage = "intent_recognition"
                    reason = "entity_resolution_retry"
        elif retryable and execution_retry_used < self.max_execution_retry:
            retry_requested = True
            target_stage = "execution"
            reason = "transient_tool_failure"

        validation = {
            "passed": False,
            "reason": reason,
            "retry_requested": retry_requested,
            "target_stage": target_stage,
            "failure_count": failure_count,
            "success_count": success_count,
            "failed_tools": [
                {
                    "tool_name": item.get("tool_name", ""),
                    "error_type": item.get("error_type", ""),
                    "error_code": item.get("error_code", ""),
                    "retryable": self._is_retryable(item),
                }
                for item in failed
            ],
        }

        block_write = failure_count == len(results)
        return {
            "validation_result": validation,
            "quality_gate_passed": False,
            "quality_score": quality_score,
            "retry_target_stage": target_stage,
            "retry_reason": reason,
            "last_failed_stage": "execution",
            "retry_trace": list(state.get("retry_trace", []) or [])
            + [
                {
                    "stage": "tool_output_validator",
                    "reason": reason,
                    "retry_requested": retry_requested,
                    "target_stage": target_stage,
                    "failure_count": failure_count,
                    "success_count": success_count,
                }
            ],
            "block_persistent_write": bool(state.get("block_persistent_write", False) or block_write),
            "debug_steps": debug_steps
            + [
                f"tool_output_validator: fail(reason={reason},retry={retry_requested},target={target_stage or 'none'},failures={failure_count})"
            ],
        }
