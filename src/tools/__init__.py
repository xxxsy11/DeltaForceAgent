"""
工具层：封装可被 Agent 调用的能力。
"""

from .registry import ToolRegistry

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


def __getattr__(name: str):
    if name == "build_rag_knowledge_tool":
        from .rag_knowledge_tool import build_rag_knowledge_tool

        return build_rag_knowledge_tool

    if name in {
        "DFPriceTools",
        "build_df_latest_price_tool",
        "build_df_history_price_tool",
        "build_df_price_advice_tool",
        "build_df_place_profit_rank_tool",
        "build_df_multi_item_compare_tool",
        "build_df_profit_stability_tool",
        "build_df_answer_composer_tool",
    }:
        from .df_price import (
            DFPriceTools,
            build_df_answer_composer_tool,
            build_df_history_price_tool,
            build_df_latest_price_tool,
            build_df_multi_item_compare_tool,
            build_df_place_profit_rank_tool,
            build_df_profit_stability_tool,
            build_df_price_advice_tool,
        )

        mapping = {
            "DFPriceTools": DFPriceTools,
            "build_df_latest_price_tool": build_df_latest_price_tool,
            "build_df_history_price_tool": build_df_history_price_tool,
            "build_df_price_advice_tool": build_df_price_advice_tool,
            "build_df_place_profit_rank_tool": build_df_place_profit_rank_tool,
            "build_df_multi_item_compare_tool": build_df_multi_item_compare_tool,
            "build_df_profit_stability_tool": build_df_profit_stability_tool,
            "build_df_answer_composer_tool": build_df_answer_composer_tool,
        }
        return mapping[name]

    raise AttributeError(f"module 'tools' has no attribute {name!r}")
