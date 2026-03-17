"""
工具注册中心
"""
from typing import Dict, List, Optional

from langchain_core.tools import BaseTool
from services import RAGService
from services.df_price_service import DFPriceService
from config import DEFAULT_CONFIG, GraphRAGConfig
from observability.langsmith import langsmith_trace
from tools.df_price import (
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
        self.rag_service = rag_service or RAGService(self.config)
        self.price_service = DFPriceService.from_config(self.config)
        self.tools: Dict[str, BaseTool] = self._build_tools()

    def _register_tool(self, tools: Dict[str, BaseTool], tool_name: str, tool: BaseTool) -> None:
        normalized_name = str(tool_name or "").strip()
        actual_name = str(getattr(tool, "name", "") or "").strip()
        if not normalized_name:
            raise ValueError("工具名不能为空")
        if normalized_name != actual_name:
            raise ValueError(f"工具注册名与工具对象名不一致: registry={normalized_name}, tool={actual_name}")
        if normalized_name in tools:
            raise ValueError(f"重复注册工具: {normalized_name}")
        tools[normalized_name] = tool

    def _build_tools(self) -> Dict[str, BaseTool]:
        tools: Dict[str, BaseTool] = {}
        self._register_tool(tools, "rag_knowledge_search", build_rag_knowledge_tool(self.rag_service))
        self._register_tool(tools, "df_market_latest_price", build_df_latest_price_tool(self.price_service))
        self._register_tool(tools, "df_market_history_price", build_df_history_price_tool(self.price_service))
        self._register_tool(tools, "df_market_price_advice", build_df_price_advice_tool(self.price_service))
        self._register_tool(tools, "df_place_profit_rank", build_df_place_profit_rank_tool(self.price_service))
        self._register_tool(tools, "df_multi_item_compare", build_df_multi_item_compare_tool(self.price_service))
        self._register_tool(tools, "df_profit_stability", build_df_profit_stability_tool(self.price_service))
        self._register_tool(
            tools,
            "df_answer_composer",
            build_df_answer_composer_tool(self.price_service, self.rag_service),
        )
        return tools

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self.tools

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        return self.tools.get(tool_name)

    def list_tools(self) -> List[str]:
        return list(self.tools.keys())

    def list_langchain_tools(self) -> List[BaseTool]:
        return list(self.tools.values())

    async def invoke_async(self, tool_name: str, query: str) -> str:
        tool = self.get_tool(tool_name)
        if tool is None:
            return f"未找到工具: {tool_name}"
        try:
            with langsmith_trace(
                self.config,
                name=f"tool:{tool_name}",
                run_type="tool",
                inputs={"tool_name": tool_name, "query": query},
                tags=["tool", tool_name],
                metadata={"tool_name": tool_name},
            ):
                return await tool.ainvoke({"query": query})
        except Exception as exc:
            return f"工具调用失败({tool_name}): {exc}"

    async def close_async(self):
        await self.rag_service.close_async()
