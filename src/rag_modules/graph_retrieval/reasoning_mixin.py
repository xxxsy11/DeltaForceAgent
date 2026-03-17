"""Auto-split from graph_rag_retrieval.py."""

from __future__ import annotations

from collections import Counter
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from rag_modules.graph_retrieval.types import GraphPath, KnowledgeSubgraph


logger = logging.getLogger(__name__)

TOP_RELATIONS_MAX_ITEMS = 3
REASONING_MAX_PATTERNS = 5
GRAPH_DENSITY_THRESHOLD = 0.25
QUERY_TERM_MIN_CHARS = 2
QUERY_HIT_SCORE_MAX = 0.6
QUERY_HIT_SCORE_STEP = 0.2
RELATION_MARKER_BONUS = 0.25
REASONING_KEYWORD_BONUS = 0.05

class GraphReasoningMixin:
    def graph_structure_reasoning(self, subgraph: KnowledgeSubgraph, query: str) -> List[str]:
        """
        基于图结构的推理：这是图RAG的智能之处
        不仅检索信息，还能进行逻辑推理
        """
        reasoning_chains = []
        
        try:
            # 1. 识别推理模式
            reasoning_patterns = self._identify_reasoning_patterns(subgraph)
            
            # 2. 构建推理链
            for pattern in reasoning_patterns:
                chain = self._build_reasoning_chain(pattern, subgraph)
                if chain:
                    reasoning_chains.append(chain)
            
            # 3. 验证推理链的可信度
            validated_chains = self._validate_reasoning_chains(reasoning_chains, query)
            subgraph.reasoning_chains = validated_chains
            
            logger.info(f"图结构推理完成，生成 {len(validated_chains)} 条推理链")
            return validated_chains
            
        except Exception as e:
            logger.error(f"图结构推理失败: {e}")
            return []

    def _paths_to_documents(self, paths: List[GraphPath], query: str) -> List[Document]:
        """将图路径转换为Document对象"""
        documents = []
        
        for i, path in enumerate(paths):
            # 构建路径描述
            path_desc = self._build_path_description(path)
            
            doc = Document(
                page_content=path_desc,
                metadata={
                    "search_type": "graph_path",
                    "path_length": path.path_length,
                    "relevance_score": path.relevance_score,
                    "path_type": path.path_type,
                    "node_count": len(path.nodes),
                    "relationship_count": len(path.relationships),
                    "recipe_name": path.nodes[0].get("name", "图结构结果") if path.nodes else "图结构结果",
                    "entity_name": path.nodes[0].get("name", "图结构结果") if path.nodes else "图结构结果"
                }
            )
            documents.append(doc)
            
        return documents

    def _subgraph_to_documents(self, subgraph: KnowledgeSubgraph, 
                              reasoning_chains: List[str], query: str) -> List[Document]:
        """将知识子图转换为Document对象"""
        documents = []
        
        # 子图整体描述
        subgraph_desc = self._build_subgraph_description(subgraph)
        if reasoning_chains:
            reasoning_lines = [f"{idx}. {chain}" for idx, chain in enumerate(reasoning_chains, start=1)]
            subgraph_desc = f"{subgraph_desc}\n\n推理要点:\n" + "\n".join(reasoning_lines)
        
        doc = Document(
            page_content=subgraph_desc,
            metadata={
                "search_type": "knowledge_subgraph",
                "node_count": len(subgraph.connected_nodes),
                "relationship_count": len(subgraph.relationships),
                "graph_density": subgraph.graph_metrics.get("density", 0.0),
                "reasoning_chains": reasoning_chains,
                "recipe_name": subgraph.central_nodes[0].get("name", "知识子图") if subgraph.central_nodes else "知识子图",
                "entity_name": subgraph.central_nodes[0].get("name", "知识子图") if subgraph.central_nodes else "知识子图"
            }
        )
        documents.append(doc)
        
        return documents

    def _build_path_description(self, path: GraphPath) -> str:
        """构建路径的自然语言描述"""
        if not path.nodes:
            return "空路径"
            
        desc_parts = []
        for i, node in enumerate(path.nodes):
            desc_parts.append(node.get("name", f"节点{i}"))
            if i < len(path.relationships):
                rel_type = path.relationships[i].get("type", "相关")
                desc_parts.append(f" --{rel_type}--> ")
        
        return "".join(desc_parts)

    def _build_subgraph_description(self, subgraph: KnowledgeSubgraph) -> str:
        """构建子图的自然语言描述"""
        central_names = [node.get("name", "未知") for node in subgraph.central_nodes]
        node_count = len(subgraph.connected_nodes)
        rel_count = len(subgraph.relationships)

        header = f"关于 {', '.join(central_names)} 的知识网络，包含 {node_count} 个相关概念和 {rel_count} 个关系。"

        if not subgraph.central_nodes:
            return header

        focus = subgraph.central_nodes[0]
        details = []
        labels = focus.get("labels") or []
        if labels:
            details.append(f"实体类型: {', '.join([str(x) for x in labels])}")
        for field in ("desc", "description", "class", "difficulty", "typeName", "caliber"):
            value = focus.get(field)
            if value not in (None, ""):
                details.append(f"{field}: {value}")
        for field in ("weight", "number", "tradable", "mapRestriction"):
            value = focus.get(field)
            if value not in (None, ""):
                details.append(f"{field}: {value}")

        if details:
            return header + "\n核心实体信息:\n- " + "\n- ".join(details)
        return header

    def _rank_by_graph_relevance(self, documents: List[Document], query: str) -> List[Document]:
        """基于图结构相关性排序"""
        return sorted(documents, 
                     key=lambda x: x.metadata.get("relevance_score", 0.0), 
                     reverse=True)

    def _analyze_query_complexity(self, query: str) -> float:
        """分析查询复杂度"""
        complexity_indicators = ["什么", "如何", "为什么", "哪些", "关系", "影响", "原因"]
        score = sum(1 for indicator in complexity_indicators if indicator in query)
        return min(score / len(complexity_indicators), 1.0)

    def _query_terms(self, query: str) -> List[str]:
        terms = re.findall(r"[\u4e00-\u9fffA-Za-z0-9_]+", query or "")
        deduped = []
        seen = set()
        for term in terms:
            normalized = term.strip().lower()
            if len(normalized) < QUERY_TERM_MIN_CHARS or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(term)
        return deduped

    def _relation_type_counter(self, subgraph: KnowledgeSubgraph) -> Counter:
        relation_types = []
        for rel in subgraph.relationships:
            rel_type = str(rel.get("type", "")).strip()
            if rel_type:
                relation_types.append(rel_type)
        return Counter(relation_types)

    def _identify_reasoning_patterns(self, subgraph: KnowledgeSubgraph) -> List[str]:
        """识别可解释的图推理模式（基于真实关系分布）。"""
        if not subgraph.central_nodes and not subgraph.connected_nodes:
            return []

        relation_counter = self._relation_type_counter(subgraph)
        patterns: List[str] = []

        if any(rel.startswith("OF_") for rel in relation_counter):
            patterns.append("type_classification")
        if {"HAS_LEVEL", "HAS_DIFFICULTY"} & set(relation_counter.keys()):
            patterns.append("level_constraint")
        if {"CAN_ATTACH", "USES_AMMO", "HAS_SKILL"} & set(relation_counter.keys()):
            patterns.append("capability_dependency")

        node_count = max(1, len(subgraph.connected_nodes))
        rel_count = len(subgraph.relationships)
        density = float(subgraph.graph_metrics.get("density", 0.0))
        if rel_count >= node_count or density > GRAPH_DENSITY_THRESHOLD:
            patterns.append("dense_association")

        if subgraph.central_nodes and subgraph.connected_nodes:
            patterns.append("hub_coverage")

        if relation_counter and "relation_summary" not in patterns:
            patterns.append("relation_summary")

        return patterns[:REASONING_MAX_PATTERNS]

    def _build_reasoning_chain(self, pattern: str, subgraph: KnowledgeSubgraph) -> Optional[str]:
        """基于图统计构建可读推理链。"""
        central_names = [node.get("name", "未知实体") for node in subgraph.central_nodes if node.get("name")]
        central_desc = "、".join(central_names[:2]) if central_names else "核心实体"
        relation_counter = self._relation_type_counter(subgraph)
        top_relations = relation_counter.most_common(TOP_RELATIONS_MAX_ITEMS)
        top_relation_text = "、".join([f"{rel}({cnt})" for rel, cnt in top_relations]) if top_relations else "无显著关系"

        if pattern == "type_classification":
            of_relations = [rel for rel, _ in top_relations if rel.startswith("OF_")]
            if not of_relations:
                return None
            return f"类型归属推理：{central_desc} 与类型节点主要通过 {', '.join(of_relations)} 连接，类别归属是回答关键。"

        if pattern == "level_constraint":
            return f"等级约束推理：图中存在 HAS_LEVEL/HAS_DIFFICULTY 关系，说明 {central_desc} 相关结论应结合等级或难度条件。"

        if pattern == "capability_dependency":
            return f"能力依赖推理：{central_desc} 的关键能力关系集中在 CAN_ATTACH/USES_AMMO/HAS_SKILL，适合用于搭配与可用性判断。"

        if pattern == "dense_association":
            density = float(subgraph.graph_metrics.get("density", 0.0))
            return f"结构密度推理：局部网络关系较密（density={density:.3f}），应优先参考多关系共同指向而非单条边。"

        if pattern == "hub_coverage":
            connected_names = [
                node.get("name", "")
                for node in subgraph.connected_nodes[:REASONING_MAX_PATTERNS]
                if node.get("name")
            ]
            if not connected_names:
                return None
            return (
                f"中心扩散推理：{central_desc} 的相关节点包括 "
                f"{'、'.join(connected_names[:TOP_RELATIONS_MAX_ITEMS])}，可作为证据覆盖范围。"
            )

        if pattern == "relation_summary":
            return f"关系分布推理：当前子图主要关系为 {top_relation_text}。"

        return None

    def _validate_reasoning_chains(self, chains: List[str], query: str) -> List[str]:
        """对推理链做去重、相关性排序和截断。"""
        if not chains:
            return []

        unique_chains = []
        seen = set()
        for chain in chains:
            text = (chain or "").strip()
            if not text:
                continue
            signature = text.lower()
            if signature in seen:
                continue
            seen.add(signature)
            unique_chains.append(text)

        query_terms = self._query_terms(query)
        relation_markers = ("HAS_", "OF_", "CAN_ATTACH", "USES_AMMO", "HAS_LEVEL", "HAS_DIFFICULTY")

        scored: List[Tuple[float, str]] = []
        for chain in unique_chains:
            score = 0.0
            lowered = chain.lower()
            term_hits = sum(1 for term in query_terms if term.lower() in lowered)
            score += min(QUERY_HIT_SCORE_MAX, term_hits * QUERY_HIT_SCORE_STEP)
            if any(marker.lower() in lowered for marker in relation_markers):
                score += RELATION_MARKER_BONUS
            if "推理" in chain:
                score += REASONING_KEYWORD_BONUS
            scored.append((score, chain))

        scored.sort(key=lambda item: item[0], reverse=True)
        validated = [chain for _, chain in scored[:REASONING_MAX_PATTERNS]]
        return validated
