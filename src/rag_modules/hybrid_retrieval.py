"""
混合检索模块（Delta Force）
- 实体级：基于图索引的精确实体匹配
- 主题级：基于关系与类型的主题检索
- 向量检索：Milvus + 邻居增强
"""

import json
import logging
import re
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from neo4j import GraphDatabase
from .graph_indexing import GraphIndexingModule
from .llm_utils import invoke_llm_text

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """检索结果数据结构"""
    content: str
    node_id: str
    node_type: str
    relevance_score: float
    retrieval_level: str  # 'entity' or 'topic'
    metadata: Dict[str, Any]


class HybridRetrievalModule:
    """混合检索模块"""

    def __init__(self, config, milvus_module, data_module, llm_client):
        self.config = config
        self.milvus_module = milvus_module
        self.data_module = data_module
        self.llm_client = llm_client
        self.driver = None
        self.database = getattr(config, "neo4j_database", "neo4j")
        self.bm25_retriever = None

        self.graph_indexing = GraphIndexingModule(config, llm_client)
        self.graph_indexed = False
        self.hybrid_dual_weight = float(getattr(config, "hybrid_dual_weight", 0.55))
        self.hybrid_vector_weight = float(getattr(config, "hybrid_vector_weight", 0.45))
        self.rrf_k = int(getattr(config, "rrf_k", 60))
        self.entity_contains_min_len = int(getattr(config, "entity_contains_min_len", 3))

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
            with self.driver.session(database=self.database) as session:
                query = """
                MATCH (source)-[r]->(target)
                WHERE source.nodeId IS NOT NULL AND target.nodeId IS NOT NULL
                RETURN source.nodeId as source_id,
                       type(r) as relation_type,
                       target.nodeId as target_id,
                       labels(source) as source_labels,
                       labels(target) as target_labels
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
        你是三角洲行动知识助手。请从问题中提取关键词，分为两类：
        1) entity_keywords：具体实体名称（地图、区域、房卡、装备、枪械、配件、弹药、收集品）
        2) topic_keywords：主题/类别/属性（等级、稀有度、装备类型、枪械类型、配件类型、区域类型）

        问题：{query}

        返回 JSON：
        {{"entity_keywords": ["..."], "topic_keywords": ["..."]}}
        """
        try:
            llm_text = invoke_llm_text(
                llm_client=self.llm_client,
                prompt=prompt,
                model=self.config.llm_model,
                temperature=0.1,
                max_tokens=400,
            )
            payload = self._extract_json_payload(llm_text)
            entity_keywords = self._normalize_keywords(payload.get("entity_keywords", []))
            topic_keywords = self._normalize_keywords(payload.get("topic_keywords", []))
            if not entity_keywords and not topic_keywords:
                return self._fallback_keywords(query)
            return entity_keywords, topic_keywords
        except Exception as e:
            logger.error(f"关键词提取失败: {e}")
            return self._fallback_keywords(query)

    def _extract_json_payload(self, content: str) -> Dict[str, Any]:
        raw = (content or "").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass

        fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()
        if fenced and fenced != raw:
            try:
                parsed = json.loads(fenced)
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                pass

        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _fallback_keywords(self, query: str) -> Tuple[List[str], List[str]]:
        # 中文/英文统一分词回退，避免 query.split() 在中文场景失效
        tokens = re.findall(r"[A-Za-z0-9_+\-]+|[\u4e00-\u9fff]{2,6}", query or "")
        tokens = self._normalize_keywords(tokens)
        if not tokens:
            return [], []

        entity_candidates = [
            token for token in tokens
            if re.search(r"[A-Za-z0-9]", token) or 2 <= len(token) <= 4
        ]
        entity_keywords = entity_candidates[:4] if entity_candidates else tokens[:2]
        entity_set = set(entity_keywords)
        topic_keywords = [token for token in tokens if token not in entity_set][:4]
        if not topic_keywords:
            topic_keywords = tokens[:4]
        return entity_keywords, topic_keywords

    def _normalize_keywords(self, keywords: List[str]) -> List[str]:
        normalized: List[str] = []
        seen = set()
        for keyword in keywords:
            if keyword is None:
                continue
            key = str(keyword).strip()
            if not key:
                continue
            lowered = key.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            normalized.append(key)
        return normalized

    def _lookup_entities_from_graph_index(self, keyword: str, max_results: int = 10) -> List[Tuple[Any, float, str]]:
        """基于索引键分层匹配实体（精确 > 前缀 > 包含）。"""
        keyword_lower = keyword.lower()
        scored_entities: Dict[str, Tuple[Any, float, str]] = {}

        for index_key, entity_ids in self.graph_indexing.key_to_entities.items():
            candidate_key = str(index_key).strip()
            if not candidate_key:
                continue

            candidate_lower = candidate_key.lower()
            match_score = 0.0
            match_mode = ""
            if candidate_lower == keyword_lower:
                match_score = 1.0
                match_mode = "exact"
            elif candidate_lower.startswith(keyword_lower) or keyword_lower.startswith(candidate_lower):
                match_score = 0.85
                match_mode = "prefix"
            elif len(keyword_lower) >= self.entity_contains_min_len and keyword_lower in candidate_lower:
                match_score = 0.65
                match_mode = "contains"
            else:
                continue

            for entity_id in entity_ids:
                entity = self.graph_indexing.entity_kv_store.get(entity_id)
                if not entity:
                    continue
                previous = scored_entities.get(entity_id)
                if previous and previous[1] >= match_score:
                    continue
                scored_entities[entity_id] = (entity, match_score, match_mode)

        ranked = sorted(
            scored_entities.values(),
            key=lambda item: item[1],
            reverse=True
        )
        return ranked[:max_results]

    def entity_level_retrieval(self, entity_keywords: List[str], top_k: int = 5) -> List[RetrievalResult]:
        results: List[RetrievalResult] = []

        normalized_keywords = self._normalize_keywords(entity_keywords)
        for keyword in normalized_keywords:
            candidates = self._lookup_entities_from_graph_index(keyword, max_results=top_k * 2)
            for entity, match_score, match_mode in candidates:
                neighbors = self._get_node_neighbors(entity.metadata["node_id"], max_neighbors=3)
                enhanced_content = entity.value_content
                if neighbors:
                    enhanced_content += f"\n相关信息: {', '.join(neighbors)}"

                results.append(RetrievalResult(
                    content=enhanced_content,
                    node_id=entity.metadata["node_id"],
                    node_type=entity.entity_type,
                    relevance_score=min(1.0, 0.7 + (0.3 * match_score)),
                    retrieval_level="entity",
                    metadata={
                        "entity_name": entity.entity_name,
                        "entity_type": entity.entity_type,
                        "matched_keyword": keyword,
                        "match_mode": match_mode
                    }
                ))

        if len(results) < top_k:
            results.extend(self._neo4j_entity_level_search(normalized_keywords, top_k - len(results)))

        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:top_k]

    def _neo4j_entity_level_search(self, keywords: List[str], limit: int) -> List[RetrievalResult]:
        results: List[RetrievalResult] = []
        if not keywords:
            return results

        try:
            with self.driver.session(database=self.database) as session:
                cypher_query = """
                UNWIND $keywords as keyword
                WITH trim(keyword) as keyword
                WHERE keyword <> ""
                MATCH (node)
                WHERE node.nodeId IS NOT NULL
                WITH keyword, node,
                     CASE
                       WHEN toLower(toString(node.nodeId)) = toLower(keyword)
                         OR toLower(COALESCE(toString(node.name), "")) = toLower(keyword)
                         OR toLower(COALESCE(toString(node.typeName), "")) = toLower(keyword)
                         OR toLower(COALESCE(toString(node.difficulty), "")) = toLower(keyword)
                         OR toLower(COALESCE(toString(node.caliber), "")) = toLower(keyword)
                         OR toLower(COALESCE(toString(node.colorName), "")) = toLower(keyword)
                       THEN 1.0
                       WHEN any(alias IN COALESCE(node.aliases, []) WHERE toLower(toString(alias)) = toLower(keyword))
                       THEN 0.95
                       WHEN toLower(COALESCE(toString(node.name), "")) STARTS WITH toLower(keyword)
                         OR toLower(COALESCE(toString(node.typeName), "")) STARTS WITH toLower(keyword)
                       THEN 0.80
                       WHEN size(keyword) >= $contains_min_len AND (
                            toLower(COALESCE(toString(node.name), "")) CONTAINS toLower(keyword)
                         OR toLower(COALESCE(toString(node.typeName), "")) CONTAINS toLower(keyword)
                         OR toLower(COALESCE(toString(node.difficulty), "")) CONTAINS toLower(keyword)
                         OR toLower(COALESCE(toString(node.caliber), "")) CONTAINS toLower(keyword)
                         OR toLower(COALESCE(toString(node.colorName), "")) CONTAINS toLower(keyword)
                       )
                       THEN 0.60
                       ELSE 0.0
                     END AS match_score
                WHERE match_score > 0.0
                RETURN node.nodeId as node_id,
                       COALESCE(node.name, node.typeName, node.difficulty, node.caliber, node.colorName, toString(node.level)) as name,
                       labels(node) as labels,
                       properties(node) as props,
                       keyword as matched_keyword,
                       match_score
                ORDER BY match_score DESC, name
                LIMIT $limit
                """
                result = session.run(
                    cypher_query,
                    {
                        "keywords": keywords,
                        "limit": limit,
                        "contains_min_len": self.entity_contains_min_len,
                    }
                )

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
                        relevance_score=float(record.get("match_score", 0.6)),
                        retrieval_level="entity",
                        metadata={
                            "name": record["name"],
                            "labels": record["labels"],
                            "source": "neo4j_fallback",
                            "matched_keyword": record.get("matched_keyword", ""),
                            "match_mode": "neo4j_ranked"
                        }
                    ))
        except Exception as e:
            logger.error(f"Neo4j补充检索失败: {e}")

        return results

    def topic_level_retrieval(self, topic_keywords: List[str], top_k: int = 5) -> List[RetrievalResult]:
        results: List[RetrievalResult] = []
        normalized_keywords = self._normalize_keywords(topic_keywords)

        for keyword in normalized_keywords:
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
            results.extend(self._neo4j_topic_level_search(normalized_keywords, top_k - len(results)))

        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:top_k]

    def _neo4j_topic_level_search(self, keywords: List[str], limit: int) -> List[RetrievalResult]:
        results: List[RetrievalResult] = []
        if not keywords:
            return results

        try:
            with self.driver.session(database=self.database) as session:
                cypher_query = """
                UNWIND $keywords as keyword
                WITH trim(keyword) as keyword
                WHERE keyword <> ""
                MATCH (n)
                WHERE n.nodeId IS NOT NULL AND (
                    toLower(COALESCE(toString(n.name), "")) = toLower(keyword) OR
                    toLower(COALESCE(toString(n.typeName), "")) = toLower(keyword) OR
                    toLower(COALESCE(toString(n.name), "")) STARTS WITH toLower(keyword) OR
                    toLower(COALESCE(toString(n.typeName), "")) STARTS WITH toLower(keyword) OR
                    (size(keyword) >= $contains_min_len AND (
                        toLower(COALESCE(toString(n.name), "")) CONTAINS toLower(keyword) OR
                        toLower(COALESCE(toString(n.typeName), "")) CONTAINS toLower(keyword)
                    ))
                )
                RETURN n.nodeId as node_id,
                       COALESCE(n.name, n.typeName, n.difficulty, n.caliber, n.colorName, toString(n.level)) as name,
                       labels(n) as labels,
                       keyword as matched_keyword
                ORDER BY name
                LIMIT $limit
                """
                result = session.run(
                    cypher_query,
                    {
                        "keywords": keywords,
                        "limit": limit,
                        "contains_min_len": self.entity_contains_min_len,
                    }
                )

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
                    "recipe_name": entry_name,
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

                entry_name = metadata.get("recipe_name", metadata.get("entity_name", "未知条目"))
                vector_score = result.get("score", 0.0)

                doc = Document(
                    page_content=content,
                    metadata={
                        **metadata,
                        "recipe_name": entry_name,
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
            with self.driver.session(database=self.database) as session:
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

        fused_docs = self._fuse_retrieval_results(dual_docs, vector_docs, top_k)
        logger.info(
            "融合排序完成：dual=%s, vector=%s, final=%s",
            len(dual_docs),
            len(vector_docs),
            len(fused_docs),
        )
        return fused_docs

    def _doc_identity(self, doc: Document) -> str:
        metadata = doc.metadata or {}
        chunk_id = metadata.get("chunk_id")
        if chunk_id:
            return f"chunk::{chunk_id}"

        record_id = metadata.get("id")
        if record_id:
            return f"id::{record_id}"

        node_id = metadata.get("node_id")
        normalized_content = (doc.page_content or "").strip()
        if node_id:
            return f"node::{node_id}::content::{hash(normalized_content[:256])}"

        return f"content::{hash(normalized_content)}"

    def _normalize_score_map(self, score_map: Dict[str, float]) -> Dict[str, float]:
        if not score_map:
            return {}
        values = list(score_map.values())
        max_score = max(values)
        min_score = min(values)
        if max_score - min_score < 1e-9:
            return {key: (1.0 if value > 0 else 0.0) for key, value in score_map.items()}
        return {key: (value - min_score) / (max_score - min_score) for key, value in score_map.items()}

    def _fuse_retrieval_results(self, dual_docs: List[Document], vector_docs: List[Document], top_k: int) -> List[Document]:
        doc_store: Dict[str, Document] = {}
        dual_rank: Dict[str, int] = {}
        vector_rank: Dict[str, int] = {}

        for rank, doc in enumerate(dual_docs):
            doc_id = self._doc_identity(doc)
            if doc_id not in doc_store:
                doc_store[doc_id] = doc
            dual_rank.setdefault(doc_id, rank + 1)

        for rank, doc in enumerate(vector_docs):
            doc_id = self._doc_identity(doc)
            if doc_id not in doc_store:
                doc_store[doc_id] = doc
            vector_rank.setdefault(doc_id, rank + 1)

        scored_docs: List[Tuple[float, Document]] = []
        k_rrf = float(self.rrf_k)
        for doc_id, doc in doc_store.items():
            dual_rrf = (
                self.hybrid_dual_weight / (k_rrf + dual_rank[doc_id])
                if doc_id in dual_rank else 0.0
            )
            vector_rrf = (
                self.hybrid_vector_weight / (k_rrf + vector_rank[doc_id])
                if doc_id in vector_rank else 0.0
            )
            final_score = dual_rrf + vector_rrf

            doc.metadata["dual_rrf_score"] = dual_rrf
            doc.metadata["vector_rrf_score"] = vector_rrf
            doc.metadata["rrf_k"] = self.rrf_k
            doc.metadata["final_score"] = final_score
            doc.metadata["search_method"] = (
                "hybrid_fused"
                if doc_id in dual_rank and doc_id in vector_rank
                else ("dual_level" if doc_id in dual_rank else "vector_enhanced")
            )
            scored_docs.append((final_score, doc))

        scored_docs.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]

    def close(self):
        if self.driver:
            self.driver.close()
            logger.info("Neo4j连接已关闭")
