"""专业分析子 Agent：仅在特定复杂任务下补充深度洞察。"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI

from agents.state import AgentState
from rag_modules.llm_utils import extract_text_content


class SpecialistAnalysisAgent:
    """对价格/利润类复杂问题提供额外数据解读（可替换专业模型）。"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.llm = self._build_llm()

    def _build_llm(self):
        api_key = os.getenv("MOONSHOT_API_KEY", "").strip()
        if not api_key:
            return None
        try:
            return ChatOpenAI(
                model=self.model_name,
                temperature=0,
                max_tokens=512,
                api_key=api_key,
                base_url="https://api.moonshot.cn/v1",
                timeout=60,
            )
        except TypeError:
            return ChatOpenAI(
                model=self.model_name,
                temperature=0,
                max_tokens=512,
                openai_api_key=api_key,
                openai_api_base="https://api.moonshot.cn/v1",
                request_timeout=60,
            )

    @staticmethod
    def _extract_json(text: str) -> Dict:
        raw = (text or "").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _heuristic_insights(report: Dict) -> Dict:
        facts = report.get("facts", []) if isinstance(report.get("facts"), list) else []
        risks = report.get("risks", []) if isinstance(report.get("risks"), list) else []
        recs = report.get("recommendations", []) if isinstance(report.get("recommendations"), list) else []
        insights = []
        if facts:
            insights.append(f"关键事实：{facts[0]}")
        if recs:
            insights.append(f"执行建议：{recs[0]}")
        if risks:
            insights.append(f"主要风险：{risks[0]}")
        return {
            "insights": insights[:3],
            "focus": "风险收益平衡",
            "confidence": "medium",
            "model": "heuristic",
        }

    def _llm_insights(self, query: str, report: Dict) -> Dict:
        if self.llm is None:
            return self._heuristic_insights(report)

        prompt = f"""
你是量化分析子代理。请基于分析报告给出专业化洞察。
要求：
1) 输出 JSON。
2) 字段必须包含：insights(数组), focus(字符串), confidence(字符串)。
3) insights 最多 3 条，每条一句话，强调可执行性。
4) 不编造，不输出报告外的数据。

用户问题：{query}
分析报告：{json.dumps(report, ensure_ascii=False)}
"""
        try:
            resp = self.llm.invoke(prompt)
            text = extract_text_content(getattr(resp, "content", resp)).strip()
            parsed = self._extract_json(text)
            if parsed.get("insights"):
                parsed["model"] = self.model_name
                return parsed
        except Exception:
            pass
        return self._heuristic_insights(report)

    def run(self, state: AgentState) -> Dict:
        report = state.get("analysis_report") or {}
        if not state.get("requires_specialist_analysis", False):
            return {
                "debug_steps": state.get("debug_steps", []) + ["specialist_analysis: skipped"],
            }

        specialist = self._llm_insights(query=state.get("user_query", ""), report=report)
        updated = dict(report)
        updated["specialist"] = {
            "enabled": True,
            "model": specialist.get("model", self.model_name),
            "focus": specialist.get("focus", "风险收益平衡"),
            "confidence": specialist.get("confidence", "medium"),
            "insights": specialist.get("insights", [])[:3],
        }
        msg = {
            "from_agent": "specialist_analysis",
            "to_agent": "summary",
            "message_type": "specialist_analysis",
            "payload": updated.get("specialist", {}),
        }
        return {
            "analysis_report": updated,
            "agent_messages": state.get("agent_messages", []) + [msg],
            "debug_steps": state.get("debug_steps", []) + ["specialist_analysis: done"],
        }
