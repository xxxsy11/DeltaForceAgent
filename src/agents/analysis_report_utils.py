"""analysis_report 共享 schema 与 builder。"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping


def _string_list(values: Iterable[Any] | None) -> list[str]:
    return [str(item).strip() for item in (values or []) if str(item).strip()]


def _dict_list(values: Iterable[Any] | None) -> list[dict]:
    return [dict(item) for item in (values or []) if isinstance(item, Mapping)]


def build_analysis_boundary(*, flow_type: str, route_reason: str, rule: str) -> Dict[str, Any]:
    normalized_flow = str(flow_type or "simple").strip() or "simple"
    return {
        "flow_type": normalized_flow,
        "summary_mode": "direct" if normalized_flow == "simple" else "llm_summary",
        "rule": str(rule or "").strip(),
        "reason": _string_list([route_reason]),
    }


def build_analysis_skill_section(state: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "skill_id": str(state.get("selected_skill", "") or "").strip(),
        "skill_reason": str(state.get("skill_reason", "") or "").strip(),
        "skill_confidence": float(state.get("skill_confidence", 0.0) or 0.0),
        "skill_matched_by": _string_list(state.get("skill_matched_by", [])),
    }


def build_analysis_assumptions(*, sell_fee_rate: float) -> Dict[str, Any]:
    fee = float(sell_fee_rate or 0.0)
    return {
        "sell_fee_rate": fee,
        "sell_fee_rate_note": f"卖出统一按{fee * 100:.0f}%手续费估算净收益",
    }


def normalize_analysis_report(report: Mapping[str, Any] | None) -> Dict[str, Any]:
    data = dict(report or {})
    normalized = {
        "query": str(data.get("query", "") or "").strip(),
        "intent": str(data.get("intent", "") or "").strip(),
        "route_reason": str(data.get("route_reason", "") or "").strip(),
        "boundary": dict(data.get("boundary", {}) or {}),
        "plan_source": str(data.get("plan_source", "") or "").strip(),
        "skill": dict(data.get("skill", {}) or {}),
        "used_tools": _string_list(data.get("used_tools", [])),
        "successful_tools": _dict_list(data.get("successful_tools", [])),
        "failed_tools": _dict_list(data.get("failed_tools", [])),
        "facts": _string_list(data.get("facts", [])),
        "recommendations": _string_list(data.get("recommendations", [])),
        "risks": _string_list(data.get("risks", [])),
        "assumptions": dict(data.get("assumptions", {}) or {}),
        "raw_tool_results": _dict_list(data.get("raw_tool_results", [])),
    }
    if isinstance(data.get("specialist"), Mapping):
        normalized["specialist"] = dict(data.get("specialist", {}) or {})
    return normalized


def build_execution_analysis_report(
    state: Mapping[str, Any],
    *,
    used_tools: Iterable[str],
    successful_tools: Iterable[Mapping[str, Any]],
    failed_tools: Iterable[Mapping[str, Any]],
    facts: Iterable[str],
    recommendations: Iterable[str],
    risks: Iterable[str],
    raw_tool_results: Iterable[Mapping[str, Any]],
    sell_fee_rate: float,
    boundary_rule: str = "intent_boundary",
) -> Dict[str, Any]:
    return normalize_analysis_report(
        {
            "query": state.get("user_query", ""),
            "intent": state.get("intent", ""),
            "route_reason": state.get("intent_reason", ""),
            "boundary": build_analysis_boundary(
                flow_type=str(state.get("flow_type", "simple") or "simple"),
                route_reason=str(state.get("intent_reason", "") or ""),
                rule=boundary_rule,
            ),
            "plan_source": state.get("plan_source", ""),
            "skill": build_analysis_skill_section(state),
            "used_tools": list(used_tools),
            "successful_tools": list(successful_tools),
            "failed_tools": list(failed_tools),
            "facts": list(facts),
            "recommendations": list(recommendations),
            "risks": list(risks),
            "assumptions": build_analysis_assumptions(sell_fee_rate=sell_fee_rate),
            "raw_tool_results": list(raw_tool_results),
        }
    )


def build_empty_analysis_report(
    state: Mapping[str, Any],
    *,
    sell_fee_rate: float,
    risk_message: str,
) -> Dict[str, Any]:
    return build_execution_analysis_report(
        state,
        used_tools=[],
        successful_tools=[],
        failed_tools=[],
        facts=[],
        recommendations=[],
        risks=[risk_message],
        raw_tool_results=[],
        sell_fee_rate=sell_fee_rate,
        boundary_rule="no_tool",
    )
