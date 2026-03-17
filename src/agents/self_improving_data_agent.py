"""Self-Improving 数据采集 Agent（面向 Tool-Planning Agentic RL）。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_openai import ChatOpenAI

from agents.state import AgentState
from observability.langsmith import langsmith_trace
from rag_modules.llm_utils import extract_text_content

MEMORY_CONTEXT_MAX_CHARS = 2000
USER_QUERY_HASH_SLICE = 12
LATENCY_DIV_FLOOR = 1e-6
MS_PER_SECOND = 1000.0
RETRY_PENALTY_CAP = 3
LLM_SCORE_MIN = 0.0
LLM_SCORE_MAX = 10.0
DEFAULT_LLM_CONFIDENCE = 0.6
MAX_LLM_JUDGE_OUTPUT_CHARS = 8000
TRACE_REASON_PREVIEW_CHARS = 320

logger = logging.getLogger(__name__)


@dataclass
class SelfImproveConfig:
    enabled: bool
    output_dir: str
    collect_only_with_tools: bool
    reward_tool_match: float
    reward_args_ok: float
    reward_exec_success: float
    reward_quality_pass: float
    reward_plan_trigger_correct: float
    penalty_overplan: float
    penalty_underplan: float
    reward_plan_coverage: float
    reward_dependency_consistency: float
    reward_plan_exec_alignment: float
    penalty_redundancy: float
    reward_efficiency: float
    reward_terminal_success: float
    reward_terminal_partial: float
    penalty_terminal_fail: float
    reward_recovery_success: float
    penalty_blind_retry: float
    reasonable_max_steps: int
    penalty_retry: float
    penalty_budget_exhausted: float
    penalty_latency_over_s: float
    latency_budget_s: float
    reward_rule_weight: float
    reward_llm_weight: float
    llm_judge_enabled: bool
    llm_judge_model: str
    llm_judge_timeout_seconds: int
    llm_judge_max_tokens: int
    llm_judge_temperature: float
    llm_weight_overall: float
    llm_weight_planning_quality: float
    llm_weight_dependency_consistency: float
    llm_weight_argument_quality: float
    llm_weight_execution_consistency: float
    llm_weight_result_quality: float
    llm_hard_adjustment_cap: float


class SelfImprovingDataAgent:
    """将线上运行轨迹转换为可训练的 Agentic RL 样本。"""
    PROMPT_TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "prompts" / "self_improve_llm_judge_prompt.txt"
    DEFAULT_PROMPT_TEMPLATE = """
你是 DeltaAgent 的“自进化奖励评估器（LLM Judge）”。
请基于给定运行轨迹，为“复杂任务规划质量”打分。

目标：
1) 识别规划是否正确触发（该规划是否规划，不该规划是否避免过度规划）
2) 评估规划链路是否完整、依赖是否连贯、参数是否可执行
3) 评估执行结果与用户目标对齐程度
4) 在不违背事实的前提下，给出稳定可复用的评分

强约束：
- 只能依据输入 JSON，不得编造不存在的工具结果
- 输出必须是单个 JSON 对象，不要 Markdown，不要解释段落
- 所有分数区间必须在 0~10（允许小数）
- confidence 区间必须在 0~1

评分标尺（0~10）：
- 0~2：严重错误（规划缺失/工具明显错选/结果不对题）
- 3~4：较差（有部分动作但无法满足目标）
- 5~6：一般（基本可用但规划或参数质量一般）
- 7~8：良好（规划合理，执行与目标一致）
- 9~10：优秀（规划完整，依赖清晰，结果高质量且高可复用）

请输出严格 JSON：
{{
  "overall_score": 0.0,
  "dimension_scores": {{
    "planning_quality": 0.0,
    "dependency_consistency": 0.0,
    "argument_quality": 0.0,
    "execution_consistency": 0.0,
    "result_quality": 0.0
  }},
  "hard_rule_adjustment": {{
    "bonus": 0.0,
    "penalty": 0.0,
    "violations": []
  }},
  "confidence": 0.0,
  "brief_reason": ""
}}

轨迹输入：
{trajectory_json}
""".strip()

    INTENT_TOOL_HINTS: Dict[str, Tuple[str, ...]] = {
        "knowledge_query": ("rag_knowledge_search",),
        "market_latest_price": ("df_market_latest_price", "df_answer_composer"),
        "market_history_price": ("df_market_history_price", "df_answer_composer"),
        "market_price_advice": ("df_market_price_advice", "df_answer_composer"),
        "place_profit_rank": ("df_place_profit_rank",),
        "multi_item_compare": ("df_multi_item_compare", "df_answer_composer"),
        "profit_stability": ("df_profit_stability", "df_answer_composer"),
        "comprehensive_answer": ("df_answer_composer",),
    }

    def __init__(self, config: Any):
        self.config = config
        self.cfg = SelfImproveConfig(
            enabled=bool(getattr(config, "self_improve_enabled", False)),
            output_dir=str(getattr(config, "self_improve_output_dir", "data/self_improve/raw_trajectories") or ""),
            collect_only_with_tools=bool(getattr(config, "self_improve_collect_only_with_tools", True)),
            reward_tool_match=float(getattr(config, "self_improve_reward_tool_match", 1.2) or 1.2),
            reward_args_ok=float(getattr(config, "self_improve_reward_args_ok", 1.0) or 1.0),
            reward_exec_success=float(getattr(config, "self_improve_reward_exec_success", 1.5) or 1.5),
            reward_quality_pass=float(getattr(config, "self_improve_reward_quality_pass", 1.0) or 1.0),
            reward_plan_trigger_correct=float(
                getattr(config, "self_improve_reward_plan_trigger_correct", 1.5) or 1.5
            ),
            penalty_overplan=float(getattr(config, "self_improve_penalty_overplan", 0.8) or 0.8),
            penalty_underplan=float(getattr(config, "self_improve_penalty_underplan", 1.8) or 1.8),
            reward_plan_coverage=float(getattr(config, "self_improve_reward_plan_coverage", 2.0) or 2.0),
            reward_dependency_consistency=float(
                getattr(config, "self_improve_reward_dependency_consistency", 1.5) or 1.5
            ),
            reward_plan_exec_alignment=float(
                getattr(config, "self_improve_reward_plan_exec_alignment", 1.5) or 1.5
            ),
            penalty_redundancy=float(getattr(config, "self_improve_penalty_redundancy", 1.2) or 1.2),
            reward_efficiency=float(getattr(config, "self_improve_reward_efficiency", 0.8) or 0.8),
            reward_terminal_success=float(
                getattr(config, "self_improve_reward_terminal_success", 4.0) or 4.0
            ),
            reward_terminal_partial=float(
                getattr(config, "self_improve_reward_terminal_partial", 1.5) or 1.5
            ),
            penalty_terminal_fail=float(getattr(config, "self_improve_penalty_terminal_fail", 2.0) or 2.0),
            reward_recovery_success=float(
                getattr(config, "self_improve_reward_recovery_success", 1.2) or 1.2
            ),
            penalty_blind_retry=float(getattr(config, "self_improve_penalty_blind_retry", 1.2) or 1.2),
            reasonable_max_steps=int(getattr(config, "self_improve_reasonable_max_steps", 3) or 3),
            penalty_retry=float(getattr(config, "self_improve_penalty_retry", 0.7) or 0.7),
            penalty_budget_exhausted=float(getattr(config, "self_improve_penalty_budget_exhausted", 1.5) or 1.5),
            penalty_latency_over_s=float(getattr(config, "self_improve_penalty_latency_over_s", 0.3) or 0.3),
            latency_budget_s=float(getattr(config, "self_improve_latency_budget_s", 8.0) or 8.0),
            reward_rule_weight=float(getattr(config, "self_improve_reward_rule_weight", 1.0) or 1.0),
            reward_llm_weight=float(getattr(config, "self_improve_reward_llm_weight", 0.8) or 0.8),
            llm_judge_enabled=bool(getattr(config, "self_improve_llm_judge_enabled", True)),
            llm_judge_model=str(getattr(config, "self_improve_llm_judge_model", "kimi-k2-0711-preview") or ""),
            llm_judge_timeout_seconds=int(getattr(config, "self_improve_llm_judge_timeout_seconds", 30) or 30),
            llm_judge_max_tokens=int(getattr(config, "self_improve_llm_judge_max_tokens", 600) or 600),
            llm_judge_temperature=float(getattr(config, "self_improve_llm_judge_temperature", 0.0) or 0.0),
            llm_weight_overall=float(getattr(config, "self_improve_llm_weight_overall", 0.6) or 0.6),
            llm_weight_planning_quality=float(
                getattr(config, "self_improve_llm_weight_planning_quality", 1.0) or 1.0
            ),
            llm_weight_dependency_consistency=float(
                getattr(config, "self_improve_llm_weight_dependency_consistency", 0.8) or 0.8
            ),
            llm_weight_argument_quality=float(
                getattr(config, "self_improve_llm_weight_argument_quality", 0.8) or 0.8
            ),
            llm_weight_execution_consistency=float(
                getattr(config, "self_improve_llm_weight_execution_consistency", 0.9) or 0.9
            ),
            llm_weight_result_quality=float(getattr(config, "self_improve_llm_weight_result_quality", 1.0) or 1.0),
            llm_hard_adjustment_cap=float(getattr(config, "self_improve_llm_hard_adjustment_cap", 2.0) or 2.0),
        )
        self._prompt_template = ""
        self.llm_judge = self._build_llm_judge()

    def _build_llm_judge(self):
        if not self.cfg.llm_judge_enabled:
            return None
        api_key = os.getenv("MOONSHOT_API_KEY", "").strip()
        if not api_key:
            logger.warning("self_improve_data: llm judge enabled but MOONSHOT_API_KEY missing, fallback to rule-only.")
            return None
        model_name = str(self.cfg.llm_judge_model or "").strip()
        if not model_name:
            logger.warning("self_improve_data: llm judge enabled but llm_judge_model empty, fallback to rule-only.")
            return None
        return ChatOpenAI(
            model=model_name,
            temperature=self.cfg.llm_judge_temperature,
            max_tokens=self.cfg.llm_judge_max_tokens,
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1",
            timeout=self.cfg.llm_judge_timeout_seconds,
        )

    @staticmethod
    def _now_utc() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _safe_id(text: str, fallback: str) -> str:
        value = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text or ""))
        value = value.strip("_")
        return value or fallback

    @staticmethod
    def _clip_text(text: str, max_chars: int) -> str:
        raw = str(text or "")
        if len(raw) <= max_chars:
            return raw
        return raw[:max_chars]

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _clip_score_0_10(value: Any) -> float:
        try:
            score = float(value)
        except Exception:
            return 5.0
        if score < LLM_SCORE_MIN:
            return LLM_SCORE_MIN
        if score > LLM_SCORE_MAX:
            return LLM_SCORE_MAX
        return score

    @staticmethod
    def _centered_from_0_10(score_0_10: float) -> float:
        score = float(score_0_10)
        return (score - 5.0) / 5.0

    @staticmethod
    def _extract_json_payload(text: str) -> Dict[str, Any]:
        raw = str(text or "").strip()
        if len(raw) > MAX_LLM_JUDGE_OUTPUT_CHARS:
            raw = raw[:MAX_LLM_JUDGE_OUTPUT_CHARS]
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def _load_prompt_template(self) -> str:
        if self._prompt_template:
            return self._prompt_template
        try:
            text = self.PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8").strip()
            self._prompt_template = text or self.DEFAULT_PROMPT_TEMPLATE
        except Exception:
            self._prompt_template = self.DEFAULT_PROMPT_TEMPLATE
        return self._prompt_template

    def _build_llm_trajectory_payload(
        self,
        state: AgentState,
        selected_tool: str,
        tool_calls: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
        rule_components: Dict[str, float],
        rule_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        meta = state.get("orchestration_meta", {}) or {}
        stage_timings_ms = meta.get("stage_timings_ms", {}) or {}
        total_timing_ms = 0.0
        for value in stage_timings_ms.values():
            total_timing_ms += self._to_float(value, 0.0)
        query = str(state.get("user_query", "") or "")
        payload = {
            "task": {
                "user_query": query,
                "intent": str(state.get("intent", "") or ""),
                "flow_type": str(state.get("flow_type", "") or ""),
                "requires_task_planning": bool(state.get("requires_task_planning", False)),
                "entities": [str(x) for x in (state.get("understanding_entities", []) or []) if str(x).strip()],
            },
            "action": {
                "selected_tool": selected_tool,
                "tool_calls": tool_calls,
                "plan_source": str(state.get("plan_source", "") or ""),
            },
            "outcome": {
                "tool_results": tool_results,
                "quality_gate_passed": bool(state.get("quality_gate_passed", False)),
                "retry_count_total": int(state.get("retry_count_total", 0) or 0),
                "retry_budget_exhausted": bool(state.get("retry_budget_exhausted", False)),
                "terminal_status": str(rule_meta.get("terminal_status", "fail") or "fail"),
                "stage_timings_ms": stage_timings_ms,
                "total_latency_ms": round(total_timing_ms, 2),
            },
            "rule_score_reference": {
                "rule_total": round(sum(rule_components.values()), 6),
                "components": {k: round(float(v), 6) for k, v in rule_components.items()},
                "should_plan": bool(rule_meta.get("should_plan", False)),
                "expected_steps": int(rule_meta.get("expected_steps", 1) or 1),
                "planned_steps": int(rule_meta.get("planned_steps", 0) or 0),
                "executed_steps": int(rule_meta.get("executed_steps", 0) or 0),
            },
        }
        return payload

    def _build_llm_judge_prompt(self, trajectory_payload: Dict[str, Any]) -> str:
        template = self._load_prompt_template()
        return template.replace("{trajectory_json}", json.dumps(trajectory_payload, ensure_ascii=False))

    async def _compute_llm_judge_reward(
        self,
        state: AgentState,
        selected_tool: str,
        tool_calls: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
        rule_components: Dict[str, float],
        rule_meta: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
        if self.llm_judge is None:
            return 0.0, {}, {"llm_judge_used": False, "llm_judge_reason": "disabled_or_missing_api_key"}

        trajectory_payload = self._build_llm_trajectory_payload(
            state=state,
            selected_tool=selected_tool,
            tool_calls=tool_calls,
            tool_results=tool_results,
            rule_components=rule_components,
            rule_meta=rule_meta,
        )
        prompt = self._build_llm_judge_prompt(trajectory_payload=trajectory_payload)
        try:
            with langsmith_trace(
                self.config,
                name="self_improve:llm_judge",
                run_type="llm",
                inputs={
                    "selected_tool": selected_tool,
                    "tool_call_count": len(tool_calls),
                    "tool_result_count": len(tool_results),
                    "rule_total": round(sum(rule_components.values()), 6),
                },
                tags=["self-improve", "llm-judge"],
                metadata={"model_name": self.cfg.llm_judge_model},
            ) as span:
                resp = await self.llm_judge.ainvoke(prompt)
                raw_text = extract_text_content(getattr(resp, "content", resp)).strip()
                parsed = self._extract_json_payload(raw_text)
                if not parsed:
                    return 0.0, {}, {"llm_judge_used": False, "llm_judge_reason": "parse_failed", "raw": raw_text[:512]}

                dims = parsed.get("dimension_scores", {}) or {}
                overall_0_10 = self._clip_score_0_10(parsed.get("overall_score", 5.0))
                planning_0_10 = self._clip_score_0_10(dims.get("planning_quality", 5.0))
                dependency_0_10 = self._clip_score_0_10(dims.get("dependency_consistency", 5.0))
                argument_0_10 = self._clip_score_0_10(dims.get("argument_quality", 5.0))
                execution_0_10 = self._clip_score_0_10(dims.get("execution_consistency", 5.0))
                result_0_10 = self._clip_score_0_10(dims.get("result_quality", 5.0))
                confidence = self._to_float(parsed.get("confidence", DEFAULT_LLM_CONFIDENCE), DEFAULT_LLM_CONFIDENCE)
                if confidence < 0:
                    confidence = 0.0
                if confidence > 1:
                    confidence = 1.0
                confidence_factor = 0.5 + 0.5 * confidence

                hard_adj = parsed.get("hard_rule_adjustment", {}) or {}
                hard_bonus = self._to_float(hard_adj.get("bonus", 0.0), 0.0)
                hard_penalty = self._to_float(hard_adj.get("penalty", 0.0), 0.0)
                hard_delta = hard_bonus - hard_penalty
                cap = max(0.1, float(self.cfg.llm_hard_adjustment_cap))
                if hard_delta > cap:
                    hard_delta = cap
                if hard_delta < -cap:
                    hard_delta = -cap

                llm_components = {
                    "llm_overall": self.cfg.llm_weight_overall * self._centered_from_0_10(overall_0_10),
                    "llm_planning_quality": self.cfg.llm_weight_planning_quality
                    * self._centered_from_0_10(planning_0_10),
                    "llm_dependency_consistency": self.cfg.llm_weight_dependency_consistency
                    * self._centered_from_0_10(dependency_0_10),
                    "llm_argument_quality": self.cfg.llm_weight_argument_quality * self._centered_from_0_10(argument_0_10),
                    "llm_execution_consistency": self.cfg.llm_weight_execution_consistency
                    * self._centered_from_0_10(execution_0_10),
                    "llm_result_quality": self.cfg.llm_weight_result_quality * self._centered_from_0_10(result_0_10),
                    "llm_hard_adjustment": hard_delta,
                }
                llm_total = sum(llm_components.values()) * confidence_factor
                llm_components["llm_confidence_factor"] = confidence_factor
                llm_components = {k: round(float(v), 6) for k, v in llm_components.items()}

                llm_meta = {
                    "llm_judge_used": True,
                    "llm_judge_model": self.cfg.llm_judge_model,
                    "llm_judge_confidence": confidence,
                    "llm_reason": str(parsed.get("brief_reason", "") or ""),
                    "llm_violations": hard_adj.get("violations", []) if isinstance(hard_adj.get("violations"), list) else [],
                    "llm_raw": parsed,
                }
                if span is not None:
                    span.end(
                        outputs={
                            "overall_score": overall_0_10,
                            "confidence": confidence,
                            "brief_reason": str(parsed.get("brief_reason", "") or "")[:TRACE_REASON_PREVIEW_CHARS],
                        }
                    )
                return round(llm_total, 6), llm_components, llm_meta
        except Exception as exc:
            return 0.0, {}, {"llm_judge_used": False, "llm_judge_reason": f"invoke_failed:{exc}"}

    def _build_sample_ids(self, state: AgentState) -> Tuple[str, str]:
        user_id = self._safe_id(str(state.get("user_id", "user")), "user")
        session_id = self._safe_id(str(state.get("session_id", "session")), "session")
        query = str(state.get("user_query", "") or "")
        digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:USER_QUERY_HASH_SLICE]
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        sample_id = f"{user_id}-{session_id}-{ts}-{digest}"
        episode_id = f"{user_id}-{session_id}"
        return sample_id, episode_id

    def _reward_tool_match(self, state: AgentState, selected_tool: str) -> float:
        if not selected_tool:
            return 0.0
        intent = str(state.get("intent", "") or "").strip()
        candidates = self.INTENT_TOOL_HINTS.get(intent, ())
        if not candidates:
            return 0.0
        return self.cfg.reward_tool_match if selected_tool in candidates else -self.cfg.reward_tool_match

    def _reward_args_ok(self, tool_calls: List[Dict[str, Any]]) -> float:
        if not tool_calls:
            return 0.0
        valid_count = 0
        for call in tool_calls:
            name = str(call.get("tool_name", "") or "").strip()
            query = str(call.get("tool_query", "") or "").strip()
            if name and query:
                valid_count += 1
        ratio = valid_count / max(1, len(tool_calls))
        return self.cfg.reward_args_ok * ratio

    def _reward_exec_success(self, tool_results: List[Dict[str, Any]]) -> float:
        if not tool_results:
            return 0.0
        success = 0
        for item in tool_results:
            if bool(item.get("ok", False)):
                success += 1
        ratio = success / max(1, len(tool_results))
        return self.cfg.reward_exec_success * ratio

    def _penalty_retry(self, state: AgentState) -> float:
        retry_total = int(state.get("retry_count_total", 0) or 0)
        return self.cfg.penalty_retry * min(retry_total, RETRY_PENALTY_CAP)

    def _penalty_budget_exhausted(self, state: AgentState) -> float:
        if bool(state.get("retry_budget_exhausted", False)):
            return self.cfg.penalty_budget_exhausted
        return 0.0

    def _penalty_latency(self, state: AgentState) -> float:
        meta = state.get("orchestration_meta", {}) or {}
        stage_timings = meta.get("stage_timings_ms", {}) or {}
        total_ms = 0.0
        for value in stage_timings.values():
            total_ms += self._to_float(value, 0.0)
        total_sec = total_ms / MS_PER_SECOND
        over = max(0.0, total_sec - self.cfg.latency_budget_s)
        if over <= 0:
            return 0.0
        return self.cfg.penalty_latency_over_s * (over / max(self.cfg.latency_budget_s, LATENCY_DIV_FLOOR))

    @staticmethod
    def _is_complex_task(state: AgentState) -> bool:
        return bool(str(state.get("flow_type", "") or "").strip() == "complex")

    def _plan_trigger_reward(self, should_plan: bool, did_plan: bool) -> float:
        if should_plan and did_plan:
            return self.cfg.reward_plan_trigger_correct
        if should_plan and not did_plan:
            return -self.cfg.penalty_underplan
        if (not should_plan) and did_plan:
            return -self.cfg.penalty_overplan
        return self.cfg.reward_plan_trigger_correct * 0.5

    def _expected_steps(self, state: AgentState, should_plan: bool) -> int:
        if should_plan:
            compare_count = int(state.get("understanding_compare_target_count", 0) or 0)
            if compare_count >= 2:
                return min(self.cfg.reasonable_max_steps, max(2, compare_count))
            return min(self.cfg.reasonable_max_steps, 2)
        return 1

    @staticmethod
    def _alignment_ratio(planned_names: List[str], executed_names: List[str]) -> float:
        if not planned_names:
            return 0.0
        # 顺序一致性：按最短前缀比较
        n = min(len(planned_names), len(executed_names))
        if n <= 0:
            return 0.0
        hit = 0
        for idx in range(n):
            if planned_names[idx] == executed_names[idx]:
                hit += 1
        return hit / max(1, len(planned_names))

    @staticmethod
    def _dependency_consistency_ratio(tool_calls: List[Dict[str, Any]], entities: List[str]) -> float:
        if not tool_calls:
            return 0.0
        if not entities:
            return 0.0
        normalized_entities = [str(x).strip() for x in entities if str(x).strip()]
        if not normalized_entities:
            return 0.0
        hit = 0
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            query = str(call.get("tool_query", "") or "")
            if any(ent in query for ent in normalized_entities):
                hit += 1
        return hit / max(1, len(tool_calls))

    def _terminal_status(self, quality_pass: bool, tool_results: List[Dict[str, Any]]) -> str:
        if quality_pass and tool_results and all(bool(x.get("ok", False)) for x in tool_results):
            return "success"
        success_count = 0
        for x in tool_results:
            if bool(x.get("ok", False)):
                success_count += 1
        if quality_pass or success_count > 0:
            return "partial"
        return "fail"

    def _compute_rule_reward(self, state: AgentState) -> Tuple[float, Dict[str, float], Dict[str, Any], str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        tool_calls = [x for x in (state.get("tool_calls", []) or []) if isinstance(x, dict)]
        tool_results = [x for x in (state.get("tool_results", []) or []) if isinstance(x, dict)]
        selected_tool = str(state.get("selected_tool", "") or "").strip()
        if not selected_tool and tool_calls:
            selected_tool = str((tool_calls[0] or {}).get("tool_name", "") or "").strip()

        should_plan = self._is_complex_task(state=state)
        did_plan = bool(state.get("requires_task_planning", False) or len(tool_calls) > 1)
        expected_steps = self._expected_steps(state=state, should_plan=should_plan)
        plan_steps = len(tool_calls)

        plan_coverage = min(1.0, plan_steps / max(1, expected_steps))
        planned_tool_names = [str((x or {}).get("tool_name", "") or "").strip() for x in tool_calls]
        executed_tool_names = [str((x or {}).get("tool_name", "") or "").strip() for x in tool_results]
        order_alignment = self._alignment_ratio(planned_names=planned_tool_names, executed_names=executed_tool_names)
        dependency_consistency = self._dependency_consistency_ratio(
            tool_calls=tool_calls,
            entities=[str(x) for x in (state.get("understanding_entities", []) or [])],
        )
        plan_exec_alignment = 0.0
        if planned_tool_names:
            overlap = 0
            executed_set = set(executed_tool_names)
            for name in planned_tool_names:
                if name in executed_set:
                    overlap += 1
            plan_exec_alignment = overlap / max(1, len(planned_tool_names))

        redundancy_over = max(0, plan_steps - expected_steps)
        quality_pass = bool(state.get("quality_gate_passed", False))
        terminal_status = self._terminal_status(quality_pass=quality_pass, tool_results=tool_results)
        terminal_score = 0.0
        if terminal_status == "success":
            terminal_score = self.cfg.reward_terminal_success
        elif terminal_status == "partial":
            terminal_score = self.cfg.reward_terminal_partial
        else:
            terminal_score = -self.cfg.penalty_terminal_fail

        retry_total = int(state.get("retry_count_total", 0) or 0)
        recovery_score = 0.0
        if retry_total > 0 and terminal_status == "success":
            recovery_score = self.cfg.reward_recovery_success
        elif retry_total > 0 and terminal_status != "success":
            recovery_score = -self.cfg.penalty_blind_retry * min(retry_total, RETRY_PENALTY_CAP)

        efficiency_score = 0.0
        if terminal_status == "success":
            efficiency_ratio = max(0.0, 1.0 - (redundancy_over / max(1, expected_steps)))
            efficiency_score = self.cfg.reward_efficiency * efficiency_ratio

        quality_pass = bool(state.get("quality_gate_passed", False))
        components = {
            "tool_match": self._reward_tool_match(state=state, selected_tool=selected_tool),
            "plan_trigger": self._plan_trigger_reward(should_plan=should_plan, did_plan=did_plan),
            "plan_coverage": self.cfg.reward_plan_coverage * plan_coverage,
            "plan_order": self.cfg.reward_plan_exec_alignment * order_alignment,
            "dependency_consistency": self.cfg.reward_dependency_consistency * dependency_consistency,
            "plan_exec_alignment": self.cfg.reward_plan_exec_alignment * plan_exec_alignment,
            "redundancy_penalty": -(self.cfg.penalty_redundancy * redundancy_over),
            "args_ok": self._reward_args_ok(tool_calls=tool_calls),
            "exec_success": self._reward_exec_success(tool_results=tool_results),
            "quality_pass": self.cfg.reward_quality_pass if quality_pass else -self.cfg.reward_quality_pass,
            "terminal": terminal_score,
            "recovery": recovery_score,
            "efficiency": efficiency_score,
            "retry_penalty": -self._penalty_retry(state=state),
            "budget_exhausted_penalty": -self._penalty_budget_exhausted(state=state),
            "latency_penalty": -self._penalty_latency(state=state),
        }
        total = round(sum(components.values()), 6)
        meta = {
            "should_plan": should_plan,
            "did_plan": did_plan,
            "expected_steps": expected_steps,
            "planned_steps": plan_steps,
            "executed_steps": len(tool_results),
            "terminal_status": terminal_status,
        }
        return total, components, meta, selected_tool, tool_calls, tool_results

    async def _build_dataset_row(self, state: AgentState) -> Dict[str, Any]:
        sample_id, episode_id = self._build_sample_ids(state)
        (
            rule_total,
            rule_components,
            reward_meta,
            selected_tool,
            tool_calls,
            tool_results,
        ) = self._compute_rule_reward(state=state)
        llm_total, llm_components, llm_meta = await self._compute_llm_judge_reward(
            state=state,
            selected_tool=selected_tool,
            tool_calls=tool_calls,
            tool_results=tool_results,
            rule_components=rule_components,
            rule_meta=reward_meta,
        )
        reward_components = dict(rule_components)
        reward_components.update(llm_components)
        reward_components["rule_total"] = round(rule_total, 6)
        reward_components["llm_total"] = round(llm_total, 6)
        reward_components["weighted_rule_total"] = round(self.cfg.reward_rule_weight * rule_total, 6)
        reward_components["weighted_llm_total"] = round(self.cfg.reward_llm_weight * llm_total, 6)
        reward_total = round(
            (self.cfg.reward_rule_weight * rule_total) + (self.cfg.reward_llm_weight * llm_total),
            6,
        )

        failure_tags: List[str] = []
        if bool(reward_meta.get("should_plan", False)) and not bool(reward_meta.get("did_plan", False)):
            failure_tags.append("underplanning")
        if (not bool(reward_meta.get("should_plan", False))) and bool(reward_meta.get("did_plan", False)):
            failure_tags.append("overplanning")
        if int(reward_meta.get("planned_steps", 0) or 0) > int(reward_meta.get("expected_steps", 1) or 1):
            failure_tags.append("redundant_steps")
        if float((reward_components.get("dependency_consistency", 0.0) or 0.0)) < 0.2:
            failure_tags.append("weak_dependency")
        if int(state.get("retry_count_total", 0) or 0) > 0 and str(reward_meta.get("terminal_status", "")) != "success":
            failure_tags.append("blind_retry")
        if bool(state.get("retry_budget_exhausted", False)):
            failure_tags.append("retry_budget_exhausted")

        row = {
            "sample_id": sample_id,
            "episode_id": episode_id,
            "created_at_utc": self._now_utc(),
            "state": {
                "user_id": str(state.get("user_id", "default_user") or "default_user"),
                "session_id": str(state.get("session_id", "default_session") or "default_session"),
                "user_query": str(state.get("user_query", "") or ""),
                "memory_context": self._clip_text(
                    str(state.get("memory_context", "") or ""),
                    MEMORY_CONTEXT_MAX_CHARS,
                ),
                "intent": str(state.get("intent", "") or ""),
                "flow_type": str(state.get("flow_type", "") or ""),
                "requires_task_planning": bool(state.get("requires_task_planning", False)),
                "retry_count_used": int(state.get("retry_count_total", 0) or 0),
            },
            "action": {
                "selected_tool": selected_tool,
                "tool_calls": tool_calls,
                "requires_task_planning": bool(state.get("requires_task_planning", False)),
                "plan_source": str(state.get("plan_source", "") or ""),
            },
            "outcome": {
                "tool_results": tool_results,
                "validation_result": state.get("validation_result", {}) or {},
                "review_result": state.get("review_result", {}) or {},
                "quality_gate_passed": bool(state.get("quality_gate_passed", False)),
                "retry_count_total": int(state.get("retry_count_total", 0) or 0),
                "retry_budget_exhausted": bool(state.get("retry_budget_exhausted", False)),
                "stage_timings_ms": (state.get("orchestration_meta", {}) or {}).get("stage_timings_ms", {}) or {},
                "terminal_status": str(reward_meta.get("terminal_status", "") or "fail"),
                "should_plan": bool(reward_meta.get("should_plan", False)),
                "planned_steps": int(reward_meta.get("planned_steps", 0) or 0),
                "executed_steps": int(reward_meta.get("executed_steps", 0) or 0),
                "expected_steps": int(reward_meta.get("expected_steps", 1) or 1),
                "failure_tags": failure_tags,
            },
            "reward": {
                "total": reward_total,
                "components": reward_components,
                "meta": {
                    "rule_weight": self.cfg.reward_rule_weight,
                    "llm_weight": self.cfg.reward_llm_weight,
                    "llm_judge_used": bool(llm_meta.get("llm_judge_used", False)),
                    "llm_judge_model": str(llm_meta.get("llm_judge_model", "") or ""),
                    "llm_judge_confidence": self._to_float(llm_meta.get("llm_judge_confidence", 0.0), 0.0),
                    "llm_reason": str(llm_meta.get("llm_reason", "") or ""),
                    "llm_violations": llm_meta.get("llm_violations", []),
                    "llm_judge_status": str(llm_meta.get("llm_judge_reason", "ok") or "ok"),
                },
            },
        }
        return row

    @staticmethod
    def _row_has_tool_action(row: Dict[str, Any]) -> bool:
        action = row.get("action", {}) or {}
        selected_tool = str(action.get("selected_tool", "") or "").strip()
        tool_calls = action.get("tool_calls", []) or []
        return bool(selected_tool or tool_calls)

    def _append_row(self, row: Dict[str, Any]) -> None:
        output_dir = Path(self.cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        path = output_dir / f"tool_planning_trajectory_{day}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    async def run(self, state: AgentState) -> Dict[str, Any]:
        debug_steps = list(state.get("debug_steps", []) or [])
        with langsmith_trace(
            self.config,
            name="self_improve:collect_trajectory",
            run_type="chain",
            inputs={
                "session_id": str(state.get("session_id", "") or ""),
                "user_id": str(state.get("user_id", "") or ""),
                "selected_tool": str(state.get("selected_tool", "") or ""),
                "retry_count_total": int(state.get("retry_count_total", 0) or 0),
            },
            tags=["self-improve", "trajectory"],
            metadata={"enabled": self.cfg.enabled, "collect_only_with_tools": self.cfg.collect_only_with_tools},
        ) as span:
            if not self.cfg.enabled:
                result = {"debug_steps": debug_steps + ["self_improve_data: disabled"]}
            else:
                row = await self._build_dataset_row(state=state)
                if self.cfg.collect_only_with_tools and not self._row_has_tool_action(row):
                    result = {"debug_steps": debug_steps + ["self_improve_data: skipped(no_tool_action)"]}
                else:
                    try:
                        await asyncio.to_thread(self._append_row, row=row)
                    except Exception as exc:
                        result = {"debug_steps": debug_steps + [f"self_improve_data: exception={exc}"]}
                    else:
                        reward = (row.get("reward", {}) or {}).get("total", 0.0)
                        result = {
                            "self_improve_sample_id": str(row.get("sample_id", "") or ""),
                            "self_improve_episode_id": str(row.get("episode_id", "") or ""),
                            "self_improve_reward": float(reward or 0.0),
                            "self_improve_reward_components": dict((row.get("reward", {}) or {}).get("components", {}) or {}),
                            "self_improve_dataset_row": row,
                            "debug_steps": debug_steps + [f"self_improve_data: saved(sample={row.get('sample_id','')},reward={reward})"],
                        }
            if span is not None:
                dataset_row = result.get("self_improve_dataset_row", {}) or {}
                reward_payload = dataset_row.get("reward", {}) or {}
                outcome = dataset_row.get("outcome", {}) or {}
                span.end(
                    outputs={
                        "sample_id": str(result.get("self_improve_sample_id", "") or ""),
                        "reward_total": reward_payload.get("total", 0.0),
                        "failure_tags": (outcome.get("failure_tags", []) or [])[:8],
                    }
                )
            return result
