"""Delta Force 价格工具入口（聚合 mixins）。"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.tools import tool

from services.df_price_service import DFPriceService
from tools.df_price.base_mixin import DFPriceBaseMixin
from tools.df_price.compare_mixin import DFPriceCompareMixin
from tools.df_price.composer_mixin import DFPriceComposerMixin
from tools.df_price.market_mixin import DFPriceMarketMixin
from tools.df_price.profit_mixin import DFPriceProfitMixin


class DFPriceTools(
    DFPriceBaseMixin,
    DFPriceMarketMixin,
    DFPriceCompareMixin,
    DFPriceProfitMixin,
    DFPriceComposerMixin,
):
    """封装最新价、历史价、建议、对比、利润与综合回答工具。"""

    def __init__(self, price_service: DFPriceService, rag_service: Optional[Any] = None):
        self.price_service = price_service
        self.rag_service = rag_service

    def build_latest_price_tool(self):
        @tool("df_market_latest_price")
        def df_market_latest_price(query: str) -> str:
            """
            查询物品最新价格。
            输入建议：
            - JSON: {"id":"14060000003"} 或 {"objectName":"非洲之心"}
            - 或文本: "id=14060000003" / "非洲之心最新价格"
            """
            params = self._parse_query_to_params(query)
            safe_params, error = self._sanitize_common_params(params)
            if error:
                return f"参数校验失败：{error}"
            result = self.price_service.get_latest_price(safe_params)
            return self._format_latest_result(result)

        return df_market_latest_price

    def build_history_price_tool(self):
        @tool("df_market_history_price")
        def df_market_history_price(query: str) -> str:
            """
            查询物品历史价格。
            输入建议：
            - JSON: {"id":"14060000003","startTime":"2026-02-20","endTime":"2026-02-24"}
            - JSON: {"objectName":"非洲之心","startTime":"2026-02-20","endTime":"2026-02-24"}
            - 或文本: "非洲之心 2026-02-20 到 2026-02-24 历史价格"
            具体参数名以 API 文档为准，工具会原样透传你提供的字段。
            """
            params = self._parse_query_to_params(query)
            safe_params, error = self._sanitize_common_params(params)
            if error:
                return f"参数校验失败：{error}"
            result = self.price_service.get_history_price(safe_params)
            return self._format_history_result(result)

        return df_market_history_price

    def build_price_advice_tool(self):
        @tool("df_market_price_advice")
        def df_market_price_advice(query: str) -> str:
            """
            基于“当前价格 + 历史区间”给出买卖建议与盈亏测算。
            输入建议：
            - "非洲之心现在能不能卖"
            - "非洲之心现在贵了还是便宜了"
            - "非洲之心现在建议买吗 成本价=12500000"
            """
            params = self._parse_query_to_params(query)
            safe_params, error = self._sanitize_common_params(params)
            if error:
                return f"参数校验失败：{error}"
            latest_result = self.price_service.get_latest_price(safe_params)
            history_result = self.price_service.get_history_price(safe_params)
            if not history_result.get("success"):
                return self._format_history_result(history_result)
            return self._format_price_advice_result(query, latest_result, history_result)

        return df_market_price_advice

    def build_place_profit_rank_tool(self):
        @tool("df_place_profit_rank")
        def df_place_profit_rank(query: str) -> str:
            """
            查询特勤处制造利润榜（四个分组支持 Top1/Top3）：
            - 技术中心（枪械/配件）
            - 工作台（子弹）
            - 制药台（药品/针剂/维修工具包）
            - 防具台（头盔/护甲/胸挂/背包）

            用法示例：
            - "特勤处现在制造什么利润最高"
            - "制造什么子弹利润最高"
            - "技术中心枪械配件利润前三"
            """
            params = self._parse_query_to_params(query)
            safe_params, error = self._sanitize_common_params(params, strict_entity=False)
            if error:
                return f"参数校验失败：{error}"
            return self._format_place_profit_rank_result(query=query, params=safe_params)

        return df_place_profit_rank

    def build_multi_item_compare_tool(self):
        @tool("df_multi_item_compare")
        def df_multi_item_compare(query: str) -> str:
            """
            对比多个物品的价格区间与性价比，给出相对优先级。
            用法示例：
            - "非洲之心、海洋之泪、腾龙突击步枪对比一下"
            - "比较 非洲之心 和 海洋之泪 哪个更值得买"
            """
            params = self._parse_query_to_params(query)
            safe_params, error = self._sanitize_common_params(params, strict_entity=False)
            if error:
                return f"参数校验失败：{error}"
            return self._format_multi_item_compare_result(query=query, params=safe_params)

        return df_multi_item_compare

    def build_profit_stability_tool(self):
        @tool("df_profit_stability")
        def df_profit_stability(query: str) -> str:
            """
            分析制造利润稳定性（波动、正利润占比、最大回撤、趋势）。
            用法示例：
            - "分析 碳纤维散射箭矢 的利润稳定性"
            - "objectId=37270500002 的利润稳不稳"
            """
            params = self._parse_query_to_params(query)
            safe_params, error = self._sanitize_common_params(params)
            if error:
                return f"参数校验失败：{error}"
            return self._format_profit_stability_result(query=query, params=safe_params)

        return df_profit_stability

    def build_answer_composer_tool(self):
        @tool("df_answer_composer")
        def df_answer_composer(query: str) -> str:
            """
            综合回答工具：自动组合资料介绍、实时价格、买卖建议、制造利润信息。
            用法示例：
            - "介绍一下非洲之心并告诉我现在价格和是否建议买"
            - "介绍腾龙突击步枪，并给制造利润建议"
            """
            params = self._parse_query_to_params(query)
            safe_params, error = self._sanitize_common_params(params, strict_entity=False)
            if error:
                return f"参数校验失败：{error}"
            return self._format_answer_composer_result(query=query, params=safe_params)

        return df_answer_composer


def build_df_latest_price_tool(price_service: DFPriceService):
    return DFPriceTools(price_service).build_latest_price_tool()


def build_df_history_price_tool(price_service: DFPriceService):
    return DFPriceTools(price_service).build_history_price_tool()


def build_df_price_advice_tool(price_service: DFPriceService):
    return DFPriceTools(price_service).build_price_advice_tool()


def build_df_place_profit_rank_tool(price_service: DFPriceService):
    return DFPriceTools(price_service).build_place_profit_rank_tool()


def build_df_multi_item_compare_tool(price_service: DFPriceService):
    return DFPriceTools(price_service).build_multi_item_compare_tool()


def build_df_profit_stability_tool(price_service: DFPriceService):
    return DFPriceTools(price_service).build_profit_stability_tool()


def build_df_answer_composer_tool(price_service: DFPriceService, rag_service: Any):
    return DFPriceTools(price_service, rag_service=rag_service).build_answer_composer_tool()
