"""
工具层：封装可被 Agent 调用的能力。
"""

from .registry import ToolRegistry
from .rag_knowledge_tool import build_rag_knowledge_tool
from .df_price_tools import (
    DFPriceTools,
    build_df_answer_composer_tool,
    build_df_history_price_tool,
    build_df_latest_price_tool,
    build_df_multi_item_compare_tool,
    build_df_place_profit_rank_tool,
    build_df_profit_stability_tool,
    build_df_price_advice_tool,
)

__all__ = [
    "ToolRegistry",
    "build_rag_knowledge_tool",
    "build_df_latest_price_tool",
    "build_df_history_price_tool",
    "build_df_price_advice_tool",
    "build_df_place_profit_rank_tool",
    "build_df_multi_item_compare_tool",
    "build_df_profit_stability_tool",
    "build_df_answer_composer_tool",
    "DFPriceTools",
]
