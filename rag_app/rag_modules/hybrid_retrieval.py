"""
混合检索模块
支持实体级、主题级和向量检索
"""

import json
import logging
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from neo4j import GraphDatabase
from .graph_indexing import GraphIndexingModule

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """检索结果数据结构"""
    content: str
    node_id: str
    node_type: str
    relevance_score: float
    retrieval_level: str
    metadata: Dict[str, Any]


class HybridRetrievalModule:
    """混合检索模块"""

    def __init__(self, config, milvus_module, data_module, llm_client):
        self.config = config
        self.milvus_module = milvus_module
        self.data_module = data_module
        self.llm_client = llm_client
        self.driver = None
        self.bm25_retriever = None

        self.graph_indexing = GraphIndexingModule(config, llm_client)
        self.graph_indexed = False

    def initialize(self, chunks: List[Document]):
        logger.info("初始化混合检索模块...")
        self.driver = GraphDatabase.driver(
            self.config.neo4j_uri,
            auth=(self.config.neo4j_user, self.config.neo4j_password)
        )

        if chunks:
            self.bm25_retriever = BM25Retriever.from_documents(chunks)
            logger.info(f"BM25检索器初始化完成，文档数量: {len(chunks)}")

        self._build_graph_index()

    def _build_graph_index(self):
        if self.graph_indexed:
            return

        logger.info("开始构建图索引...")
        try:
            nodes = self.data_module.nodes
            self.graph_indexing.create_entity_key_values(nodes)

            relationships = self._extract_relationships_from_graph()
            self.graph_indexing.create_relation_key_values(relationships)
            self.graph_indexing.deduplicate_entities_and_relations()

            self.graph_indexed = True
            stats = self.graph_indexing.get_statistics()
            logger.info(f"图索引构建完成: {stats}")

        except Exception as e:
            logger.error(f"构建图索引失败: {e}")

    def _extract_relationships_from_graph(self) -> List[Tuple[str, str, str]]:
        relationships = []
        try:
            with self.driver.session() as session:
                query = """
                MATCH (source)-[r]->(target)
                WHERE source.nodeId IS NOT NULL AND target.nodeId IS NOT NULL
                RETURN source.nodeId as source_id, type(r) as relation_type, target.nodeId as target_id
                LIMIT 2000
                """
                result = session.run(query)
                for record in result:
                    relationships.append(
                        (record["source_id"], record["relation_type"], record["target_id"])
                    )
        except Exception as e:
            logger.error(f"提取图关系失败: {e}")
        return relationships

    def extract_query_keywords(self, query: str) -> Tuple[List[str], List[str]]:
        """提取查询关键词：实体级 + 主题级"""
        prompt = f"""
        你是DeltaForce知识助手。请从问题中提取关键词，分为两类：
        1) entity_keywords：具体实体名称（地图、区域、房卡、装备、枪械、配件、弹药、收集品）
        2) topic_keywords：主题/类别/属性（等级、稀有度、装备类型、枪械类型、配件类型、区域类型）

        问题：{query}

        返回 JSON：
        {{"entity_keywords": ["..."], "topic_keywords": ["..."]}}
        """
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=400
            )
            result = json.loads(response.choices[0].message.content.strip())
            return result.get("entity_keywords", []), result.get("topic_keywords", [])
        except Exception as e:
            logger.error(f"关键词提取失败: {e}")
            keywords = query.split()
            return keywords[:3], keywords[3:6] if len(keywords) > 3 else keywords

    def entity_level_retrieval(self, entity_keywords: List[str], top_k: int = 5) -> List[RetrievalResult]:
        results: List[RetrievalResult] = []

        for keyword in entity_keywords:
            entities = self.graph_indexing.get_entities_by_key(keyword)
            for entity in entities:
                neighbors = self._get_node_neighbors(entity.metadata["node_id"], max_neighbors=3)
                enhanced_content = entity.value_content
                if neighbors:
                    enhanced_content += f"\n相关信息: {', '.join(neighbors)}"

                results.append(RetrievalResult(
                    content=enhanced_content,
                    node_id=entity.metadata["node_id"],
                    node_type=entity.entity_type,
                    relevance_score=0.9,
                    retrieval_level="entity",
                    metadata={
                        "entity_name": entity.entity_name,
                        "entity_type": entity.entity_type,
                        "matched_keyword": keyword
                    }
                ))

        if len(results) < top_k:
            results.extend(self._neo4j_entity_level_search(entity_keywords, top_k - len(results)))

        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:top_k]

    def _neo4j_entity_level_search(self, keywords: List[str], limit: int) -> List[RetrievalResult]:
        results: List[RetrievalResult] = []
        if not keywords:
            return results

        try:
            with self.driver.session() as session:
                cypher_query = """
                UNWIND $keywords as keyword
                MATCH (node)
                WHERE node.nodeId IS NOT NULL AND (
                    (node.name IS NOT NULL AND toString(node.name) CONTAINS keyword) OR
                    (node.typeName IS NOT NULL AND toString(node.typeName) CONTAINS keyword) OR
                    (node.difficulty IS NOT NULL AND toString(node.difficulty) CONTAINS keyword) OR
                    (node.caliber IS NOT NULL AND toString(node.caliber) CONTAINS keyword) OR
                    (node.colorName IS NOT NULL AND toString(node.colorName) CONTAINS keyword)
                )
                RETURN node.nodeId as node_id,
                       COALESCE(node.name, node.typeName, node.difficulty, node.caliber, node.colorName, toString(node.level)) as name,
                       labels(node) as labels,
                       properties(node) as props
                ORDER BY name
                LIMIT $limit
                """
                result = session.run(cypher_query, {"keywords": keywords, "limit": limit})

                for record in result:
                    props = record.get("props") or {}
                    content_parts = [f"名称: {record['name']}"]
                    for k, v in props.items():
                        if k in ("nodeId", "name"):
                            continue
                        if v is None or v == "":
                            continue
                        content_parts.append(f"{k}: {v}")

                    results.append(RetrievalResult(
                        content="\n".join(content_parts),
                        node_id=record["node_id"],
                        node_type=record["labels"][0] if record["labels"] else "Node",
                        relevance_score=0.6,
                        retrieval_level="entity",
                        metadata={
                            "name": record["name"],
                            "labels": record["labels"],
                            "source": "neo4j_fallback"
                        }
                    ))
        except Exception as e:
            logger.error(f"Neo4j补充检索失败: {e}")

        return results

    def topic_level_retrieval(self, topic_keywords: List[str], top_k: int = 5) -> List[RetrievalResult]:
        results: List[RetrievalResult] = []

        for keyword in topic_keywords:
            relations = self.graph_indexing.get_relations_by_key(keyword)
            for relation in relations:
                source_entity = self.graph_indexing.entity_kv_store.get(relation.source_entity)
                target_entity = self.graph_indexing.entity_kv_store.get(relation.target_entity)
                if not source_entity or not target_entity:
                    continue

                content_parts = [
                    f"主题: {keyword}",
                    relation.value_content,
                    f"相关实体: {source_entity.entity_name}",
                    f"关联对象: {target_entity.entity_name}"
                ]

                results.append(RetrievalResult(
                    content="\n".join(content_parts),
                    node_id=relation.source_entity,
                    node_type=source_entity.entity_type,
                    relevance_score=0.85,
                    retrieval_level="topic",
                    metadata={
                        "relation_id": relation.relation_id,
                        "relation_type": relation.relation_type,
                        "source_name": source_entity.entity_name,
                        "target_name": target_entity.entity_name,
                        "matched_keyword": keyword
                    }
                ))

        if len(results) < top_k:
            results.extend(self._neo4j_topic_level_search(topic_keywords, top_k - len(results)))

        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:top_k]

    def _neo4j_topic_level_search(self, keywords: List[str], limit: int) -> List[RetrievalResult]:
        results: List[RetrievalResult] = []
        if not keywords:
            return results

        try:
            with self.driver.session() as session:
                cypher_query = """
                UNWIND $keywords as keyword
                MATCH (n)
                WHERE n.nodeId IS NOT NULL AND (
                    (n.name IS NOT NULL AND toString(n.name) CONTAINS keyword) OR
                    (n.typeName IS NOT NULL AND toString(n.typeName) CONTAINS keyword)
                )
                RETURN n.nodeId as node_id,
                       COALESCE(n.name, n.typeName, n.difficulty, n.caliber, n.colorName, toString(n.level)) as name,
                       labels(n) as labels,
                       keyword as matched_keyword
                ORDER BY name
                LIMIT $limit
                """
                result = session.run(cypher_query, {"keywords": keywords, "limit": limit})

                for record in result:
                    results.append(RetrievalResult(
                        content=f"名称: {record['name']}",
                        node_id=record["node_id"],
                        node_type=record["labels"][0] if record["labels"] else "Node",
                        relevance_score=0.65,
                        retrieval_level="topic",
                        metadata={
                            "name": record["name"],
                            "matched_keyword": record["matched_keyword"],
                            "source": "neo4j_fallback"
                        }
                    ))
        except Exception as e:
            logger.error(f"Neo4j主题级检索失败: {e}")

        return results

    def dual_level_retrieval(self, query: str, top_k: int = 5) -> List[Document]:
        entity_keywords, topic_keywords = self.extract_query_keywords(query)

        entity_results = self.entity_level_retrieval(entity_keywords, top_k)
        topic_results = self.topic_level_retrieval(topic_keywords, top_k)

        all_results = entity_results + topic_results
        seen_nodes = set()
        unique_results = []
        for result in sorted(all_results, key=lambda x: x.relevance_score, reverse=True):
            if result.node_id not in seen_nodes:
                seen_nodes.add(result.node_id)
                unique_results.append(result)

        documents = []
        for result in unique_results[:top_k]:
            entry_name = result.metadata.get("name") or result.metadata.get("entity_name", "未知条目")
            doc = Document(
                page_content=result.content,
                metadata={
                    "node_id": result.node_id,
                    "node_type": result.node_type,
                    "retrieval_level": result.retrieval_level,
                    "relevance_score": result.relevance_score,
                    "entity_name": entry_name,
                    "search_type": "dual_level",
                    **result.metadata
                }
            )
            documents.append(doc)

        return documents

    def vector_search_enhanced(self, query: str, top_k: int = 5) -> List[Document]:
        try:
            vector_docs = self.milvus_module.similarity_search(query, k=top_k * 2)
            enhanced_docs = []

            for result in vector_docs:
                content = result.get("text", "")
                metadata = result.get("metadata", {})
                node_id = metadata.get("node_id")

                if node_id:
                    neighbors = self._get_node_neighbors(node_id)
                    if neighbors:
                        content += f"\n相关信息: {', '.join(neighbors[:3])}"

                entry_name = metadata.get("entity_name", "未知条目")
                vector_score = result.get("score", 0.0)

                doc = Document(
                    page_content=content,
                    metadata={
                        **metadata,
                        "entity_name": entry_name,
                        "score": vector_score,
                        "search_type": "vector_enhanced"
                    }
                )
                enhanced_docs.append(doc)

            return enhanced_docs[:top_k]

        except Exception as e:
            logger.error(f"增强向量检索失败: {e}")
            return []

    def _get_node_neighbors(self, node_id: str, max_neighbors: int = 3) -> List[str]:
        try:
            with self.driver.session() as session:
                query = """
                MATCH (n {nodeId: $node_id})-[r]-(neighbor)
                RETURN neighbor.name as name
                LIMIT $limit
                """
                result = session.run(query, {"node_id": node_id, "limit": max_neighbors})
                return [record["name"] for record in result if record["name"]]
        except Exception as e:
            logger.error(f"获取邻居节点失败: {e}")
            return []

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Document]:
        logger.info(f"开始混合检索: {query}")

        dual_docs = self.dual_level_retrieval(query, top_k)
        vector_docs = self.vector_search_enhanced(query, top_k)

        merged_docs = []
        seen_doc_ids = set()
        max_len = max(len(dual_docs), len(vector_docs))
        origin_len = len(dual_docs) + len(vector_docs)

        for i in range(max_len):
            if i < len(dual_docs):
                doc = dual_docs[i]
                doc_id = doc.metadata.get("node_id", hash(doc.page_content))
                if doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    doc.metadata["search_method"] = "dual_level"
                    doc.metadata["round_robin_order"] = len(merged_docs)
                    doc.metadata["final_score"] = doc.metadata.get("relevance_score", 0.0)
                    merged_docs.append(doc)

            if i < len(vector_docs):
                doc = vector_docs[i]
                doc_id = doc.metadata.get("node_id", hash(doc.page_content))
                if doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    doc.metadata["search_method"] = "vector_enhanced"
                    doc.metadata["round_robin_order"] = len(merged_docs)
                    vector_score = doc.metadata.get("score", 0.0)
                    similarity_score = max(0.0, 1.0 - vector_score) if vector_score <= 1.0 else 0.0
                    doc.metadata["final_score"] = similarity_score
                    merged_docs.append(doc)

        final_docs = merged_docs[:top_k]
        logger.info(f"Round-robin合并：从总共{origin_len}个结果合并为{len(final_docs)}个文档")
        return final_docs

    def close(self):
        if self.driver:
            self.driver.close()
            logger.info("Neo4j连接已关闭")
