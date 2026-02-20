"""意图分析器：当前仅路由到 RAG 工具。"""

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
    - 其余问题 -> rag_knowledge_search

    后续有新工具时，在这里增加分流逻辑。
    """

    rag_keywords = {
        "三角洲", "delta", "武器", "枪", "弹药", "配件", "地图", "房卡", "干员",
        "资料", "知识", "关系", "对比", "分析", "查询", "推荐", "怎么搭配",
    }

    def analyze(self, query: str) -> IntentDecision:
        text = (query or "").strip()
        if not text:
            return IntentDecision(
                intent="empty",
                tool_name="none",
                tool_query="",
                reason="空问题",
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
