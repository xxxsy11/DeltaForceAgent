"""Skill registry, matching, and tool-chain planning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from skills.models import SkillDefinition, SkillPlan, SkillSelection


class SkillRegistry:
    def __init__(self, definitions_dir: Optional[str], available_tools: Iterable[str]):
        self.available_tools = set(str(x).strip() for x in (available_tools or []) if str(x).strip())
        self.definitions = self._load_definitions(definitions_dir)
        self.tool_to_skill = self._build_tool_to_skill()

    @staticmethod
    def _default_definitions_dir() -> Path:
        return Path(__file__).resolve().parent / "definitions"

    def _load_definitions(self, definitions_dir: Optional[str]) -> Dict[str, SkillDefinition]:
        root = Path(definitions_dir).resolve() if definitions_dir else self._default_definitions_dir()
        result: Dict[str, SkillDefinition] = {}
        if not root.exists():
            return result

        for path in sorted(root.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                skill = SkillDefinition.from_dict(payload)
                if not skill.skill_id:
                    continue
                if not skill.display_name:
                    skill.display_name = skill.skill_id
                result[skill.skill_id] = skill
            except Exception:
                continue
        return result

    def _build_tool_to_skill(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for skill in self.definitions.values():
            for tool in skill.tool_hints:
                mapping.setdefault(tool, skill.skill_id)
        return mapping

    def get(self, skill_id: str) -> Optional[SkillDefinition]:
        return self.definitions.get(str(skill_id or "").strip())

    @staticmethod
    def _count_hits(text: str, keywords: List[str]) -> int:
        raw = str(text or "")
        return sum(1 for token in keywords if token and token in raw)

    def select(
        self,
        *,
        query: str,
        intent: str,
        tool_name: str,
        entity_count: int,
        compare_target_count: int,
    ) -> SkillSelection:
        text = str(query or "")
        best_id = ""
        best_score = -1
        best_reasons: List[str] = []

        for skill in self.definitions.values():
            score = 0
            reasons: List[str] = []

            if intent and intent in skill.intent_hints:
                score += 6
                reasons.append("intent")
            if tool_name and tool_name in skill.tool_hints:
                score += 4
                reasons.append("tool")

            hit_any = self._count_hits(text, skill.query_keywords_any)
            if hit_any > 0:
                score += min(hit_any, 3)
                reasons.append(f"kw_any={hit_any}")

            if skill.query_keywords_all and all(token in text for token in skill.query_keywords_all):
                score += 2
                reasons.append("kw_all")

            if entity_count >= skill.min_entities:
                score += 1
                reasons.append("entity_count")

            if skill.skill_id == "market_multi_item_compare" and compare_target_count >= skill.min_compare_targets:
                score += 2
                reasons.append("compare_count")

            if score > best_score:
                best_score = score
                best_id = skill.skill_id
                best_reasons = reasons

        if best_id and best_score > 0:
            confidence = min(0.99, max(0.0, best_score / 12.0))
            return SkillSelection(
                skill_id=best_id,
                score=best_score,
                confidence=confidence,
                reason=f"skill_match:{best_id}",
                matched_by=best_reasons,
            )

        fallback_id = self.tool_to_skill.get(tool_name, "")
        if fallback_id:
            return SkillSelection(
                skill_id=fallback_id,
                score=3,
                confidence=0.4,
                reason=f"fallback_tool:{tool_name}",
                matched_by=["fallback_tool"],
            )

        # 兜底：优先知识技能
        generic = "knowledge_profile"
        if generic in self.definitions:
            return SkillSelection(
                skill_id=generic,
                score=1,
                confidence=0.2,
                reason="fallback_generic",
                matched_by=["fallback"],
            )

        return SkillSelection(skill_id="", score=0, confidence=0.0, reason="no_skill", matched_by=[])

    @staticmethod
    def _build_query_for_tool(tool_name: str, query: str, entities: List[str], compare_target_count: int) -> str:
        clean_entities = [str(x).strip() for x in entities if str(x).strip()]
        if tool_name in {"df_market_latest_price", "df_market_history_price", "df_market_price_advice", "df_profit_stability"}:
            if clean_entities:
                return f"objectName={clean_entities[0]}"
            return str(query or "").strip()

        if tool_name == "df_multi_item_compare":
            if len(clean_entities) >= 2:
                n = max(2, int(compare_target_count or 2))
                return f"{'、'.join(clean_entities[:n])} 对比"
            return str(query or "").strip()

        if tool_name == "df_answer_composer":
            if clean_entities and any(token in str(query or "") for token in ("它", "他", "她", "这个", "那个", "这两个", "这三个")):
                return f"{clean_entities[0]}；{query}"
            return str(query or "").strip()

        if tool_name == "rag_knowledge_search":
            if clean_entities and any(token in str(query or "") for token in ("它", "他", "她", "这个", "那个")):
                return clean_entities[0]
            return str(query or "").strip()

        return str(query or "").strip()

    def build_plan(
        self,
        *,
        selection: SkillSelection,
        query: str,
        entities: List[str],
        compare_target_count: int,
        fallback_tool: str,
        fallback_query: str,
    ) -> SkillPlan:
        skill = self.get(selection.skill_id)
        calls: List[Dict[str, str]] = []

        if skill is not None:
            for tool_name in skill.default_chain:
                if tool_name not in self.available_tools:
                    continue
                tool_query = self._build_query_for_tool(
                    tool_name=tool_name,
                    query=query,
                    entities=entities,
                    compare_target_count=compare_target_count,
                )
                calls.append({"tool_name": tool_name, "tool_query": tool_query})

        if not calls and fallback_tool and fallback_tool in self.available_tools and fallback_tool != "none":
            calls.append(
                {
                    "tool_name": fallback_tool,
                    "tool_query": self._build_query_for_tool(
                        tool_name=fallback_tool,
                        query=fallback_query or query,
                        entities=entities,
                        compare_target_count=compare_target_count,
                    ),
                }
            )

        locked_plan = bool(skill.locked_plan) if skill else False
        return SkillPlan(skill_id=selection.skill_id, tool_calls=calls, locked_plan=locked_plan)
