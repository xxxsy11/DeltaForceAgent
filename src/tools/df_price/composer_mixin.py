"""DF 价格工具综合回答能力。"""

from __future__ import annotations

from typing import Dict, List


class DFPriceComposerMixin:
    @staticmethod
    def _contains_any(text: str, keywords: List[str]) -> bool:
        lowered = text.lower()
        return any((k in text) or (k in lowered) for k in keywords)

    @staticmethod
    def _sanitize_knowledge_text(text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        noise_markers = (
            "挂单", "成交", "游戏币", "价格", "价位", "报价", "行情", "区间", "回撤", "涨跌", "建议买", "建议卖",
            "建议", "是否建议", "能不能卖", "贵了", "便宜", "万元", "万游戏币",
        )
        cleaned_lines: List[str] = []
        for line in raw.splitlines():
            item = line.strip()
            if not item:
                continue
            if any(marker in item for marker in noise_markers):
                continue
            cleaned_lines.append(item)
        cleaned = "\n".join(cleaned_lines).strip()
        return cleaned or raw

    def _format_answer_composer_result(self, query: str, params: Dict[str, str]) -> str:
        if self.rag_service is None:
            return "综合回答工具不可用：RAG 服务未注入。"

        primary_name = self._extract_primary_item_name(query=query, params=params)
        price_query = primary_name or query

        knowledge_text = ""
        try:
            # 介绍信息与行情信息解耦，减少知识回答中的价格幻觉
            knowledge_query = query
            if primary_name:
                knowledge_query = f"介绍一下{primary_name}，仅给资料介绍，不要价格和买卖建议"
            knowledge = self.rag_service.query(question=knowledge_query, explain_routing=False)
            knowledge_text = self._sanitize_knowledge_text(str(knowledge.get("answer", "")).strip())
        except Exception as exc:
            knowledge_text = f"知识检索失败：{exc}"

        if any(token in knowledge_text for token in ("Error code: 429", "engine_overloaded", "生成回答时出现错误")):
            knowledge_text = "资料介绍暂不可用（知识模型限流），已返回可用行情数据。"

        latest_text = ""
        advice_text = ""
        if primary_name:
            latest_result = self.price_service.get_latest_price({"objectName": price_query})
            latest_info = self._extract_latest_price_info(latest_result)
            history_result = self.price_service.get_history_price({"objectName": price_query})
            is_match = latest_info.get("success") and self._is_name_consistent(primary_name, str(latest_info.get("object_name") or ""))
            if is_match:
                latest_text = self._format_latest_result(latest_result)
            else:
                latest_text = f"{primary_name} 价格查询失败：未命中准确物品。"

            if history_result.get("success") and is_match:
                advice_text = self._format_price_advice_result(
                    query=price_query,
                    latest_result=latest_result,
                    history_result=history_result,
                )
            else:
                advice_text = self._format_history_result(history_result)

        need_profit_rank = self._contains_any(
            query,
            ["制造", "特勤处", "利润", "枪械配件", "子弹", "药品", "针剂", "防具台", "技术中心", "工作台", "制药台"],
        )
        profit_rank_text = self._format_place_profit_rank_result(query=query, params=params) if need_profit_rank else ""

        sections: List[str] = []
        if knowledge_text:
            sections.append("【资料介绍】\n" + knowledge_text)
        if latest_text:
            sections.append("【实时价格】\n" + latest_text)
        if advice_text:
            sections.append("【买卖建议】\n" + advice_text)
        if profit_rank_text:
            sections.append("【制造利润】\n" + profit_rank_text)

        if not sections:
            return "未获得可用结果。"
        return "\n\n".join(sections).strip()
