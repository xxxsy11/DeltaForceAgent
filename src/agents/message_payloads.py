"""Agent 间消息 payload builder。"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

from agents.analysis_report_utils import normalize_analysis_report


def _normalize_tool_plan(task_plan: Iterable[Mapping[str, Any]] | None) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for item in task_plan or []:
        if not isinstance(item, Mapping):
            continue
        tool_name = str(item.get("tool_name", "") or "").strip()
        tool_query = str(item.get("tool_query", "") or "").strip()
        if not tool_name:
            continue
        normalized.append({"tool_name": tool_name, "tool_query": tool_query})
    return normalized


def build_orchestration_start_payload(*, query: str, has_memory_context: bool, timestamp_utc: str) -> Dict[str, Any]:
    return {
        "query": str(query or "").strip(),
        "has_memory_context": bool(has_memory_context),
        "timestamp_utc": str(timestamp_utc or "").strip(),
    }


def build_task_plan_payload(*, intent: str, reason: str, task_plan: Iterable[Mapping[str, Any]] | None) -> Dict[str, Any]:
    return {
        "intent": str(intent or "").strip(),
        "reason": str(reason or "").strip(),
        "task_plan": _normalize_tool_plan(task_plan),
    }


def build_intent_result_payload(
    *,
    intent: str,
    tool_name: str,
    flow_type: str,
    reason: str,
    entities: Iterable[str] | None,
    entity_count: int,
    confidence: float,
    compare_target_count: int,
    skill_id: str,
    skill_confidence: float,
    skill_reason: str,
    skill_matched_by: Iterable[str] | None,
) -> Dict[str, Any]:
    return {
        "intent": str(intent or "").strip(),
        "tool_name": str(tool_name or "").strip(),
        "flow_type": str(flow_type or "").strip(),
        "reason": str(reason or "").strip(),
        "entities": [str(item).strip() for item in (entities or []) if str(item).strip()],
        "entity_count": int(entity_count or 0),
        "confidence": float(confidence or 0.0),
        "compare_target_count": int(compare_target_count or 0),
        "skill_id": str(skill_id or "").strip(),
        "skill_confidence": float(skill_confidence or 0.0),
        "skill_reason": str(skill_reason or "").strip(),
        "skill_matched_by": [str(item).strip() for item in (skill_matched_by or []) if str(item).strip()],
    }


def build_analysis_report_payload(report: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return normalize_analysis_report(report)


def build_specialist_analysis_payload(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    data = dict(payload or {})
    insights = data.get("insights")
    if isinstance(insights, list):
        data["insights"] = [str(item).strip() for item in insights if str(item).strip()]
    else:
        data["insights"] = []
    data["enabled"] = bool(data.get("enabled", False))
    data["model"] = str(data.get("model", "") or "").strip()
    data["focus"] = str(data.get("focus", "") or "").strip()
    data["confidence"] = str(data.get("confidence", "") or "").strip()
    return data
