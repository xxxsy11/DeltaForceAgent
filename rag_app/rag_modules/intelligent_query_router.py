"""
智能查询路由器
根据查询特点自动选择最适合的检索策略
"""

import json
import logging
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    """搜索策略枚举"""
    HYBRID_TRADITIONAL = "hybrid_traditional"
    GRAPH_RAG = "graph_rag"
    COMBINED = "combined"


@dataclass
class QueryAnalysis:
    """查询分析结果"""
    query_complexity: float
    relationship_intensity: float
    reasoning_required: bool
    entity_count: int
    recommended_strategy: SearchStrategy
    confidence: float
    reasoning: str


class IntelligentQueryRouter:
    """智能查询路由器"""

    def __init__(self,
                 traditional_retrieval,
                 graph_rag_retrieval,
                 llm_client,
                 config):
        self.traditional_retrieval = traditional_retrieval
        self.graph_rag_retrieval = graph_rag_retrieval
        self.llm_client = llm_client
        self.config = config

        self.route_stats = {
            "traditional_count": 0,
            "graph_rag_count": 0,
            "combined_count": 0,
            "total_queries": 0
        }

    def analyze_query(self, query: str) -> QueryAnalysis:
        """深度分析查询特征，决定最佳检索策略"""
        logger.info(f"分析查询特征: {query}")

        analysis_prompt = f"""
        作为RAG系统的查询分析专家，请深度分析以下查询的特征（领域：DeltaForce知识图谱）：

        查询：{query}

        请从以下维度分析：

        1. 查询复杂度 (0-1)：
           - 0.0-0.3: 简单信息查找
           - 0.4-0.7: 中等复杂度
           - 0.8-1.0: 高复杂度推理

        2. 关系密集度 (0-1)：
           - 0.0-0.3: 单一实体信息
           - 0.4-0.7: 实体间关系
           - 0.8-1.0: 复杂关系网络

        3. 推理需求：
           - 是否需要多跳推理？
           - 是否需要因果分析？
           - 是否需要对比分析？

        4. 实体识别：
           - 查询中包含多少个明确实体？
           - 实体类型是什么？

        基于分析推荐检索策略：
        - hybrid_traditional: 适合简单直接的信息查找
        - graph_rag: 适合复杂关系推理和知识发现
        - combined: 需要两种策略结合

        返回JSON格式：
        {{
            "query_complexity": 0.6,
            "relationship_intensity": 0.8,
            "reasoning_required": true,
            "entity_count": 3,
            "recommended_strategy": "graph_rag",
            "confidence": 0.85,
            "reasoning": "该查询涉及多个实体间的复杂关系，需要图结构推理"
        }}
        """

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1,
                max_tokens=800
            )

            result = json.loads(response.choices[0].message.content.strip())

            analysis = QueryAnalysis(
                query_complexity=result.get("query_complexity", 0.5),
                relationship_intensity=result.get("relationship_intensity", 0.5),
                reasoning_required=result.get("reasoning_required", False),
                entity_count=result.get("entity_count", 1),
                recommended_strategy=SearchStrategy(result.get("recommended_strategy", "hybrid_traditional")),
                confidence=result.get("confidence", 0.5),
                reasoning=result.get("reasoning", "默认分析")
            )

            logger.info(f"查询分析完成: {analysis.recommended_strategy.value} (置信度: {analysis.confidence:.2f})")
            return analysis

        except Exception as e:
            logger.error(f"查询分析失败: {e}")
            return self._rule_based_analysis(query)

    def _rule_based_analysis(self, query: str) -> QueryAnalysis:
        """基于规则的降级分析"""
        complexity_keywords = ["为什么", "如何", "关系", "影响", "原因", "比较", "区别", "关联", "路径"]
        relation_keywords = ["配", "搭配", "组合", "相关", "联系", "连接", "属于", "包含", "需要"]

        complexity = sum(1 for kw in complexity_keywords if kw in query) / len(complexity_keywords)
        relation_intensity = sum(1 for kw in relation_keywords if kw in query) / len(relation_keywords)

        if complexity > 0.3 or relation_intensity > 0.3:
            strategy = SearchStrategy.GRAPH_RAG
        else:
            strategy = SearchStrategy.HYBRID_TRADITIONAL

        return QueryAnalysis(
            query_complexity=complexity,
            relationship_intensity=relation_intensity,
            reasoning_required=complexity > 0.3,
            entity_count=len(query.split()),
            recommended_strategy=strategy,
            confidence=0.6,
            reasoning="基于规则的简单分析"
        )

    def route_query(self, query: str, top_k: int = 5) -> Tuple[List[Document], QueryAnalysis]:
        """智能路由查询到最适合的检索引擎"""
        logger.info(f"开始智能路由: {query}")

        analysis = self.analyze_query(query)
        self._update_route_stats(analysis.recommended_strategy)

        documents = []

        try:
            if analysis.recommended_strategy == SearchStrategy.HYBRID_TRADITIONAL:
                logger.info("使用传统混合检索")
                documents = self.traditional_retrieval.hybrid_search(query, top_k)

            elif analysis.recommended_strategy == SearchStrategy.GRAPH_RAG:
                logger.info("使用图RAG检索")
                documents = self.graph_rag_retrieval.graph_rag_search(query, top_k)

            elif analysis.recommended_strategy == SearchStrategy.COMBINED:
                logger.info("使用组合检索策略")
                documents = self._combined_search(query, top_k)

            documents = self._post_process_results(documents, analysis)

            logger.info(f"路由完成，返回 {len(documents)} 个结果")
            return documents, analysis

        except Exception as e:
            logger.error(f"查询路由失败: {e}")
            documents = self.traditional_retrieval.hybrid_search(query, top_k)
            return documents, analysis

    def _combined_search(self, query: str, top_k: int) -> List[Document]:
        """组合搜索策略"""
        traditional_k = max(1, top_k // 2)
        graph_k = top_k - traditional_k

        traditional_docs = self.traditional_retrieval.hybrid_search(query, traditional_k)
        graph_docs = self.graph_rag_retrieval.graph_rag_search(query, graph_k)

        combined_docs = []
        seen_contents = set()

        max_len = max(len(traditional_docs), len(graph_docs))
        for i in range(max_len):
            if i < len(graph_docs):
                doc = graph_docs[i]
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    doc.metadata["search_source"] = "graph_rag"
                    combined_docs.append(doc)

            if i < len(traditional_docs):
                doc = traditional_docs[i]
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    doc.metadata["search_source"] = "traditional"
                    combined_docs.append(doc)

        return combined_docs[:top_k]

    def _post_process_results(self, documents: List[Document], analysis: QueryAnalysis) -> List[Document]:
        """结果后处理"""
        for doc in documents:
            doc.metadata.update({
                "route_strategy": analysis.recommended_strategy.value,
                "query_complexity": analysis.query_complexity,
                "route_confidence": analysis.confidence
            })

        return documents

    def _update_route_stats(self, strategy: SearchStrategy):
        """更新路由统计"""
        self.route_stats["total_queries"] += 1

        if strategy == SearchStrategy.HYBRID_TRADITIONAL:
            self.route_stats["traditional_count"] += 1
        elif strategy == SearchStrategy.GRAPH_RAG:
            self.route_stats["graph_rag_count"] += 1
        elif strategy == SearchStrategy.COMBINED:
            self.route_stats["combined_count"] += 1

    def get_route_statistics(self) -> Dict[str, Any]:
        """获取路由统计信息"""
        total = self.route_stats["total_queries"]
        if total == 0:
            return self.route_stats

        return {
            **self.route_stats,
            "traditional_ratio": self.route_stats["traditional_count"] / total,
            "graph_rag_ratio": self.route_stats["graph_rag_count"] / total,
            "combined_ratio": self.route_stats["combined_count"] / total
        }

    def explain_routing_decision(self, query: str) -> str:
        """解释路由决策过程"""
        analysis = self.analyze_query(query)

        explanation = f"""
        查询路由分析报告

        查询：{query}

        特征分析：
        - 复杂度：{analysis.query_complexity:.2f} ({'简单' if analysis.query_complexity < 0.4 else '中等' if analysis.query_complexity < 0.8 else '复杂'})
        - 关系密集度：{analysis.relationship_intensity:.2f} ({'单一实体' if analysis.relationship_intensity < 0.4 else '实体关系' if analysis.relationship_intensity < 0.8 else '复杂关系网络'})
        - 推理需求：{'是' if analysis.reasoning_required else '否'}
        - 实体数量：{analysis.entity_count}

        推荐策略：{analysis.recommended_strategy.value}
        置信度：{analysis.confidence:.2f}

        决策理由：{analysis.reasoning}
        """

        return explanation
