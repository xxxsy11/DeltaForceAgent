"""Skill models for agent-level capability routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SkillDefinition:
    skill_id: str
    display_name: str
    description: str
    intent_hints: List[str] = field(default_factory=list)
    tool_hints: List[str] = field(default_factory=list)
    query_keywords_any: List[str] = field(default_factory=list)
    query_keywords_all: List[str] = field(default_factory=list)
    default_chain: List[str] = field(default_factory=list)
    flow_type: str = "simple"
    requires_task_planning: bool = False
    requires_specialist_analysis: bool = False
    locked_plan: bool = True
    min_entities: int = 0
    max_entities: int = 6
    min_compare_targets: int = 2

    @classmethod
    def from_dict(cls, payload: Dict) -> "SkillDefinition":
        return cls(
            skill_id=str(payload.get("skill_id", "")).strip(),
            display_name=str(payload.get("display_name", "")).strip(),
            description=str(payload.get("description", "")).strip(),
            intent_hints=[str(x).strip() for x in payload.get("intent_hints", []) if str(x).strip()],
            tool_hints=[str(x).strip() for x in payload.get("tool_hints", []) if str(x).strip()],
            query_keywords_any=[str(x).strip() for x in payload.get("query_keywords_any", []) if str(x).strip()],
            query_keywords_all=[str(x).strip() for x in payload.get("query_keywords_all", []) if str(x).strip()],
            default_chain=[str(x).strip() for x in payload.get("default_chain", []) if str(x).strip()],
            flow_type=str(payload.get("flow_type", "simple") or "simple").strip().lower(),
            requires_task_planning=bool(payload.get("requires_task_planning", False)),
            requires_specialist_analysis=bool(payload.get("requires_specialist_analysis", False)),
            locked_plan=bool(payload.get("locked_plan", True)),
            min_entities=max(0, int(payload.get("min_entities", 0) or 0)),
            max_entities=max(1, int(payload.get("max_entities", 6) or 6)),
            min_compare_targets=max(2, int(payload.get("min_compare_targets", 2) or 2)),
        )


@dataclass
class SkillSelection:
    skill_id: str
    score: int
    confidence: float
    reason: str
    matched_by: List[str] = field(default_factory=list)


@dataclass
class SkillPlan:
    skill_id: str
    tool_calls: List[Dict[str, str]]
    locked_plan: bool
