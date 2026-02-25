"""意图分析器：路由到 RAG 或价格查询工具。"""

from dataclasses import dataclass


@dataclass
class IntentDecision:
    intent: str
    tool_name: str
    tool_query: str
    reason: str


class IntentAnalyzer:
    """
    当前版本使用轻量规则路由：
    - 空问题 -> none
    - 价格相关 -> 价格工具
    - 其余问题 -> rag_knowledge_search

    后续有新工具时，在这里增加分流逻辑。
    """

    rag_keywords = {
        "三角洲", "delta", "武器", "枪", "弹药", "配件", "地图", "房卡", "干员",
        "资料", "知识", "关系", "对比", "分析", "查询", "推荐", "怎么搭配",
    }
    price_keywords = {
        "价格", "价位", "行情", "交易行", "最新价", "实时价", "多少钱", "物价",
    }
    history_keywords = {
        "历史", "走势", "曲线", "某天", "某日", "之前", "过去", "一周", "一天", "时间段",
    }
    advice_keywords = {
        "能不能卖", "建议买", "建议卖", "建不建议买", "建不建议卖", "贵了", "便宜了", "高位", "低位", "赚", "亏",
    }
    place_profit_keywords = {
        "特勤处", "制造", "利润", "净利润", "最赚钱", "利润最高", "利润前三", "制造台", "技术中心", "工作台", "制药台", "防具台",
    }
    place_profit_category_keywords = {
        "枪械配件", "枪械", "配件", "子弹", "弹药", "药品", "针剂", "维修工具包", "头盔", "护甲", "胸挂", "背包",
    }
    compare_keywords = {
        "对比", "比较", "哪个好", "哪一个好", "谁更好", "谁更值得买", "横向比较",
    }
    stability_keywords = {
        "稳不稳", "稳定性", "波动", "回撤", "风险", "利润稳定", "利润波动",
    }
    composer_keywords = {
        "介绍", "并告诉", "并且告诉", "顺便", "同时告诉", "综合分析",
    }
    simple_intents = {
        "knowledge_query",
        "general_query",
        "market_price_latest_query",
        "market_price_history_query",
        "place_profit_query",
    }
    complex_intents = {
        "answer_composer_query",
        "market_compare_query",
        "profit_stability_query",
        "market_price_advice_query",
    }

    @classmethod
    def is_complex_intent(cls, intent: str) -> bool:
        return intent in cls.complex_intents

    def _is_price_query(self, text: str) -> bool:
        lowered = text.lower()
        return any(keyword in text or keyword in lowered for keyword in self.price_keywords)

    def _is_history_query(self, text: str) -> bool:
        lowered = text.lower()
        if any(keyword in text or keyword in lowered for keyword in self.history_keywords):
            return True
        return "starttime" in lowered or "endtime" in lowered or "start_date" in lowered or "end_date" in lowered

    def _is_advice_query(self, text: str) -> bool:
        lowered = text.lower()
        return any(keyword in text or keyword in lowered for keyword in self.advice_keywords)

    def _is_place_profit_query(self, text: str) -> bool:
        lowered = text.lower()
        hit_base = any(keyword in text or keyword in lowered for keyword in self.place_profit_keywords)
        hit_category = any(keyword in text or keyword in lowered for keyword in self.place_profit_category_keywords)
        return hit_base and hit_category or ("制造" in text and "利润" in text)

    def _is_compare_query(self, text: str) -> bool:
        lowered = text.lower()
        return any(keyword in text or keyword in lowered for keyword in self.compare_keywords)

    def _is_stability_query(self, text: str) -> bool:
        lowered = text.lower()
        return any(keyword in text or keyword in lowered for keyword in self.stability_keywords) and "利润" in text

    def _is_composer_query(self, text: str) -> bool:
        lowered = text.lower()
        has_intro = any(keyword in text or keyword in lowered for keyword in self.composer_keywords)
        has_price_or_advice = self._is_price_query(text) or self._is_advice_query(text)
        return has_intro and has_price_or_advice

    def analyze(self, query: str) -> IntentDecision:
        text = (query or "").strip()
        if not text:
            return IntentDecision(
                intent="empty",
                tool_name="none",
                tool_query="",
                reason="空问题",
            )

        if self._is_composer_query(text):
            return IntentDecision(
                intent="answer_composer_query",
                tool_name="df_answer_composer",
                tool_query=text,
                reason="命中综合回答意图",
            )

        if self._is_compare_query(text):
            return IntentDecision(
                intent="market_compare_query",
                tool_name="df_multi_item_compare",
                tool_query=text,
                reason="命中多物品对比意图",
            )

        if self._is_stability_query(text):
            return IntentDecision(
                intent="profit_stability_query",
                tool_name="df_profit_stability",
                tool_query=text,
                reason="命中利润稳定性分析意图",
            )

        if self._is_place_profit_query(text):
            return IntentDecision(
                intent="place_profit_query",
                tool_name="df_place_profit_rank",
                tool_query=text,
                reason="命中特勤处制造利润查询意图",
            )

        if self._is_price_query(text):
            if self._is_advice_query(text):
                return IntentDecision(
                    intent="market_price_advice_query",
                    tool_name="df_market_price_advice",
                    tool_query=text,
                    reason="命中价格策略分析意图",
                )
            if self._is_history_query(text):
                return IntentDecision(
                    intent="market_price_history_query",
                    tool_name="df_market_history_price",
                    tool_query=text,
                    reason="命中价格历史查询意图",
                )
            return IntentDecision(
                intent="market_price_latest_query",
                tool_name="df_market_latest_price",
                tool_query=text,
                reason="命中价格实时查询意图",
            )

        if self._is_advice_query(text):
            return IntentDecision(
                intent="market_price_advice_query",
                tool_name="df_market_price_advice",
                tool_query=text,
                reason="命中价格策略分析意图",
            )

        lowered = text.lower()
        if any(keyword in text or keyword in lowered for keyword in self.rag_keywords):
            return IntentDecision(
                intent="knowledge_query",
                tool_name="rag_knowledge_search",
                tool_query=text,
                reason="命中资料/知识关键词",
            )

        # 默认也走 RAG，保证可回答性
        return IntentDecision(
            intent="general_query",
            tool_name="rag_knowledge_search",
            tool_query=text,
            reason="当前仅接入 RAG 工具，默认走知识查询",
        )
