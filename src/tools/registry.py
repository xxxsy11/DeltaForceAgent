"""
工具注册中心
"""
from typing import Dict, Optional

from services import RAGService
from services.df_price_service import DFPriceService
from config import DEFAULT_CONFIG, GraphRAGConfig
from tools.df_price_tools import (
    build_df_answer_composer_tool,
    build_df_history_price_tool,
    build_df_latest_price_tool,
    build_df_multi_item_compare_tool,
    build_df_place_profit_rank_tool,
    build_df_profit_stability_tool,
    build_df_price_advice_tool,
)
from tools.rag_knowledge_tool import build_rag_knowledge_tool


class ToolRegistry:
    def __init__(self, rag_service: Optional[RAGService] = None, config: Optional[GraphRAGConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.rag_service = rag_service or RAGService()
        self.price_service = DFPriceService.from_config(self.config)
        latest_tool = build_df_latest_price_tool(self.price_service)
        history_tool = build_df_history_price_tool(self.price_service)
        advice_tool = build_df_price_advice_tool(self.price_service)
        place_profit_rank_tool = build_df_place_profit_rank_tool(self.price_service)
        multi_item_compare_tool = build_df_multi_item_compare_tool(self.price_service)
        profit_stability_tool = build_df_profit_stability_tool(self.price_service)
        answer_composer_tool = build_df_answer_composer_tool(self.price_service, self.rag_service)
        self.tools: Dict[str, object] = {
            "rag_knowledge_search": build_rag_knowledge_tool(self.rag_service),
            "df_market_latest_price": latest_tool,
            "df_market_history_price": history_tool,
            "df_market_price_advice": advice_tool,
            "df_place_profit_rank": place_profit_rank_tool,
            "df_multi_item_compare": multi_item_compare_tool,
            "df_profit_stability": profit_stability_tool,
            "df_answer_composer": answer_composer_tool,
        }

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self.tools

    def list_tools(self):
        return list(self.tools.keys())

    async def invoke_async(self, tool_name: str, query: str) -> str:
        if not self.has_tool(tool_name):
            return f"未找到工具: {tool_name}"
        tool = self.tools[tool_name]
        try:
            return await tool.ainvoke({"query": query})
        except Exception as exc:
            return f"工具调用失败({tool_name}): {exc}"

    async def close_async(self):
        await self.rag_service.close_async()
