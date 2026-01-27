"""
图RAG检索模块
基于图结构的知识推理和检索
"""

import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from langchain_core.documents import Document
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """查询类型枚举"""
    ENTITY_RELATION = "entity_relation"
    MULTI_HOP = "multi_hop"
    SUBGRAPH = "subgraph"
    PATH_FINDING = "path_finding"
    CLUSTERING = "clustering"


@dataclass
class GraphQuery:
    """图查询结构"""
    query_type: QueryType
    source_entities: List[str]
    target_entities: List[str] = None
    relation_types: List[str] = None
    max_depth: int = 2
    max_nodes: int = 50
    constraints: Dict[str, Any] = None


@dataclass
class GraphPath:
    """图路径结构"""
    nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    path_length: int
    relevance_score: float
    path_type: str


@dataclass
class KnowledgeSubgraph:
    """知识子图结构"""
    central_nodes: List[Dict[str, Any]]
    connected_nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    graph_metrics: Dict[str, float]
    reasoning_chains: List[List[str]]


class GraphRAGRetrieval:
    """图RAG检索系统"""

    def __init__(self, config, llm_client):
        self.config = config
        self.llm_client = llm_client
        self.driver = None

        self.entity_cache = {}
        self.relation_cache = {}
        self.subgraph_cache = {}

    def initialize(self):
        """初始化图RAG检索系统"""
        logger.info("初始化图RAG检索系统...")

        try:
            self.driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password)
            )
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Neo4j连接成功")
        except Exception as e:
            logger.error(f"Neo4j连接失败: {e}")
            return

        self._build_graph_index()

    def _build_graph_index(self):
        """构建图索引以加速查询"""
        logger.info("构建图结构索引...")

        try:
            with self.driver.session() as session:
                entity_query = """
                MATCH (n)
                WHERE n.nodeId IS NOT NULL
                WITH n, COUNT { (n)--() } as degree
                RETURN labels(n) as node_labels, n.nodeId as node_id,
                       COALESCE(n.name, n.typeName, n.difficulty, toString(n.level), n.caliber) as name, degree
                ORDER BY degree DESC
                LIMIT 1000
                """

                result = session.run(entity_query)
                for record in result:
                    node_id = record["node_id"]
                    self.entity_cache[node_id] = {
                        "labels": record["node_labels"],
                        "name": record["name"],
                        "degree": record["degree"]
                    }

                relation_query = """
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(r) as frequency
                ORDER BY frequency DESC
                """

                result = session.run(relation_query)
                for record in result:
                    rel_type = record["rel_type"]
                    self.relation_cache[rel_type] = record["frequency"]

                logger.info(f"索引构建完成: {len(self.entity_cache)}个实体, {len(self.relation_cache)}个关系类型")

        except Exception as e:
            logger.error(f"构建图索引失败: {e}")

    def understand_graph_query(self, query: str) -> GraphQuery:
        """理解查询的图结构意图"""
        prompt = f"""
        作为图数据库专家，分析以下查询的图结构意图。

        已知图中大致有以下节点和关系：
        - 节点类型：
          - Map / Area / KeyCard（地图、区域、房卡）
          - Equipment / EquipmentType（装备及其类型）
          - Firearm / FirearmType（枪械及其类型）
          - Attachment / AttachmentType（配件及其类型）
          - Ammunition（弹药）
          - Collectible / CollectibleType（收集品及其类型）
          - Level / Difficulty（等级与难度）
        - 主要关系：
          - (Map)-[:HAS_AREA]->(Area)
          - (Area)-[:HAS_KEY_CARD]->(KeyCard)
          - (Equipment)-[:OF_EQ_TYPE]->(EquipmentType)
          - (Firearm)-[:OF_FIRE_TYPE]->(FirearmType)
          - (Firearm)-[:CAN_ATTACH]->(Attachment)
          - (Firearm)-[:USES_AMMO]->(Ammunition)
          - (Attachment)-[:OF_ATT_TYPE]->(AttachmentType)
          - (Collectible)-[:OF_COL_TYPE]->(CollectibleType)
          - (KeyCard|Equipment|Collectible)-[:HAS_LEVEL]->(Level)

        查询：{query}

        请识别：
        1. query_type：entity_relation, multi_hop, subgraph, path_finding, clustering
        2. source_entities：图中可能有对应节点的实体名称列表
        3. target_entities：需要限制的路径终点
        4. relation_types：优先考虑的关系类型
        5. max_depth：图遍历深度（1-3）
        6. constraints：可选的属性级约束

        返回JSON：
        {{
          "query_type": "multi_hop",
          "source_entities": ["..."],
          "target_entities": ["..."],
          "relation_types": ["..."],
          "max_depth": 2,
          "constraints": {{}}
        }}
        """

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )

            result = json.loads(response.choices[0].message.content.strip())

            return GraphQuery(
                query_type=QueryType(result.get("query_type", "subgraph")),
                source_entities=result.get("source_entities", []),
                target_entities=result.get("target_entities", []),
                relation_types=result.get("relation_types", []),
                max_depth=result.get("max_depth", 2),
                max_nodes=50
            )

        except Exception as e:
            logger.error(f"查询意图理解失败: {e}")
            return GraphQuery(
                query_type=QueryType.SUBGRAPH,
                source_entities=[query],
                max_depth=2
            )

    def multi_hop_traversal(self, graph_query: GraphQuery) -> List[GraphPath]:
        """多跳图遍历"""
        logger.info(f"执行多跳遍历: {graph_query.source_entities} -> {graph_query.target_entities}")

        paths = []

        if not self.driver:
            logger.error("Neo4j连接未建立")
            return paths

        try:
            with self.driver.session() as session:
                source_entities = graph_query.source_entities
                target_keywords = graph_query.target_entities or []
                max_depth = graph_query.max_depth

                if graph_query.query_type in (QueryType.MULTI_HOP, QueryType.ENTITY_RELATION):
                    target_filter_clause = ""
                    if target_keywords:
                        target_filter_clause = """
                    AND ANY(kw IN $target_keywords WHERE
                        (target.name IS NOT NULL AND (toString(target.name) CONTAINS kw OR kw CONTAINS toString(target.name)))
                    )"""

                    cypher_query = f"""
                    UNWIND $source_entities as source_name
                    MATCH (source)
                    WHERE source.name CONTAINS source_name OR source.nodeId = source_name

                    MATCH path = (source)-[*1..{max_depth}]-(target)
                    WHERE NOT source = target{target_filter_clause}

                    WITH path, source, target,
                         length(path) as path_len,
                         relationships(path) as rels,
                         nodes(path) as path_nodes

                    WITH path, source, target, path_len, rels, path_nodes,
                         (1.0 / path_len) +
                         (REDUCE(s = 0.0, n IN path_nodes | s + COUNT {{ (n)--() }}) / 10.0 / size(path_nodes)) +
                         (CASE WHEN ANY(r IN rels WHERE type(r) IN $relation_types) THEN 0.3 ELSE 0.0 END) as relevance

                    ORDER BY relevance DESC
                    LIMIT 20

                    RETURN path, source, target, path_len, rels, path_nodes, relevance
                    """

                    params = {
                        "source_entities": source_entities,
                        "relation_types": graph_query.relation_types or []
                    }
                    if target_keywords:
                        params["target_keywords"] = target_keywords

                    result = session.run(cypher_query, params)

                    for record in result:
                        path_data = self._parse_neo4j_path(record)
                        if path_data:
                            paths.append(path_data)

        except Exception as e:
            logger.error(f"多跳遍历失败: {e}")

        logger.info(f"多跳遍历完成，找到 {len(paths)} 条路径")
        return paths

    def extract_knowledge_subgraph(self, graph_query: GraphQuery) -> KnowledgeSubgraph:
        """提取知识子图"""
        logger.info(f"提取知识子图: {graph_query.source_entities}")

        if not self.driver:
            logger.error("Neo4j连接未建立")
            return self._fallback_subgraph_extraction(graph_query)

        try:
            with self.driver.session() as session:
                cypher_query = f"""
                UNWIND $source_entities as entity_name
                MATCH (source)
                WHERE source.name CONTAINS entity_name
                   OR source.nodeId = entity_name

                MATCH (source)-[r*1..{graph_query.max_depth}]-(neighbor)
                WITH source, collect(DISTINCT neighbor) as neighbors,
                     collect(DISTINCT r) as relationships
                WHERE size(neighbors) <= $max_nodes

                WITH source, neighbors, relationships,
                     size(neighbors) as node_count,
                     size(relationships) as rel_count

                RETURN
                    source,
                    neighbors[0..{graph_query.max_nodes}] as nodes,
                    relationships[0..{graph_query.max_nodes}] as rels,
                    {{
                        node_count: node_count,
                        relationship_count: rel_count,
                        density: CASE WHEN node_count > 1 THEN toFloat(rel_count) / (node_count * (node_count - 1) / 2) ELSE 0.0 END
                    }} as metrics
                """

                result = session.run(cypher_query, {
                    "source_entities": graph_query.source_entities,
                    "max_nodes": graph_query.max_nodes
                })

                record = result.single()
                if record:
                    return self._build_knowledge_subgraph(record)

        except Exception as e:
            logger.error(f"子图提取失败: {e}")

        return self._fallback_subgraph_extraction(graph_query)

    def graph_structure_reasoning(self, subgraph: KnowledgeSubgraph, query: str) -> List[str]:
        """基于图结构的推理"""
        reasoning_chains = []

        try:
            reasoning_patterns = self._identify_reasoning_patterns(subgraph)

            for pattern in reasoning_patterns:
                chain = self._build_reasoning_chain(pattern, subgraph)
                if chain:
                    reasoning_chains.append(chain)

            validated_chains = self._validate_reasoning_chains(reasoning_chains, query)

            logger.info(f"图结构推理完成，生成 {len(validated_chains)} 条推理链")
            return validated_chains

        except Exception as e:
            logger.error(f"图结构推理失败: {e}")
            return []

    def graph_rag_search(self, query: str, top_k: int = 5) -> List[Document]:
        """图RAG主搜索接口"""
        logger.info(f"开始图RAG检索: {query}")

        if not self.driver:
            logger.warning("Neo4j连接未建立，返回空结果")
            return []

        graph_query = self.understand_graph_query(query)
        logger.info(f"查询类型: {graph_query.query_type.value}")

        results = []

        try:
            if graph_query.query_type in [QueryType.MULTI_HOP, QueryType.PATH_FINDING]:
                paths = self.multi_hop_traversal(graph_query)
                results.extend(self._paths_to_documents(paths, query))

            elif graph_query.query_type in [QueryType.SUBGRAPH, QueryType.CLUSTERING]:
                subgraph = self.extract_knowledge_subgraph(graph_query)
                reasoning_chains = self.graph_structure_reasoning(subgraph, query)
                results.extend(self._subgraph_to_documents(subgraph, reasoning_chains, query))

            elif graph_query.query_type == QueryType.ENTITY_RELATION:
                paths = self.multi_hop_traversal(graph_query)
                results.extend(self._paths_to_documents(paths, query))

            results = self._rank_by_graph_relevance(results, query)

            logger.info(f"图RAG检索完成，返回 {len(results[:top_k])} 个结果")
            return results[:top_k]

        except Exception as e:
            logger.error(f"图RAG检索失败: {e}")
            return []

    def _parse_neo4j_path(self, record) -> Optional[GraphPath]:
        """解析Neo4j路径记录"""
        try:
            path_nodes = []
            for node in record["path_nodes"]:
                path_nodes.append({
                    "id": node.get("nodeId", ""),
                    "name": node.get("name", ""),
                    "labels": list(node.labels),
                    "properties": dict(node)
                })

            relationships = []
            for rel in record["rels"]:
                relationships.append({
                    "type": rel.type,
                    "properties": dict(rel.items()) if hasattr(rel, "items") else dict(rel)
                })

            return GraphPath(
                nodes=path_nodes,
                relationships=relationships,
                path_length=record["path_len"],
                relevance_score=record["relevance"],
                path_type="multi_hop"
            )

        except Exception as e:
            logger.error(f"路径解析失败: {e}")
            return None

    def _build_knowledge_subgraph(self, record) -> KnowledgeSubgraph:
        """构建知识子图对象"""
        try:
            def node_to_dict(node):
                data = dict(node.items()) if hasattr(node, "items") else dict(node)
                data["labels"] = list(node.labels) if hasattr(node, "labels") else []
                data.setdefault("nodeId", node.get("nodeId") if hasattr(node, "get") else data.get("nodeId"))
                data.setdefault("name", node.get("name") if hasattr(node, "get") else data.get("name"))
                return data

            def rel_to_dict(rel):
                data = dict(rel.items()) if hasattr(rel, "items") else dict(rel)
                data["type"] = rel.type if hasattr(rel, "type") else data.get("type")
                return data

            central_nodes = [node_to_dict(record["source"])]
            connected_nodes = [node_to_dict(node) for node in record["nodes"]]

            relationships = []
            rels = record["rels"] or []
            for rel_item in rels:
                if isinstance(rel_item, list):
                    for rel in rel_item:
                        relationships.append(rel_to_dict(rel))
                else:
                    relationships.append(rel_to_dict(rel_item))

            return KnowledgeSubgraph(
                central_nodes=central_nodes,
                connected_nodes=connected_nodes,
                relationships=relationships,
                graph_metrics=record["metrics"],
                reasoning_chains=[]
            )
        except Exception as e:
            logger.error(f"构建知识子图失败: {e}")
            return KnowledgeSubgraph(
                central_nodes=[],
                connected_nodes=[],
                relationships=[],
                graph_metrics={},
                reasoning_chains=[]
            )

    def _paths_to_documents(self, paths: List[GraphPath], query: str) -> List[Document]:
        """将图路径转换为Document对象"""
        documents = []

        for path in paths:
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
                    "entity_name": path.nodes[0].get("name", "图结构结果") if path.nodes else "图结构结果"
                }
            )
            documents.append(doc)

        return documents

    def _subgraph_to_documents(self, subgraph: KnowledgeSubgraph,
                              reasoning_chains: List[str], query: str) -> List[Document]:
        """将知识子图转换为Document对象"""
        documents = []

        subgraph_desc = self._build_subgraph_description(subgraph)

        doc = Document(
            page_content=subgraph_desc,
            metadata={
                "search_type": "knowledge_subgraph",
                "node_count": len(subgraph.connected_nodes),
                "relationship_count": len(subgraph.relationships),
                "graph_density": subgraph.graph_metrics.get("density", 0.0),
                "reasoning_chains": reasoning_chains,
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

        return f"关于 {', '.join(central_names)} 的知识网络，包含 {node_count} 个相关概念和 {rel_count} 个关系。"

    def _rank_by_graph_relevance(self, documents: List[Document], query: str) -> List[Document]:
        """基于图结构相关性排序"""
        return sorted(documents,
                     key=lambda x: x.metadata.get("relevance_score", 0.0),
                     reverse=True)

    def _identify_reasoning_patterns(self, subgraph: KnowledgeSubgraph) -> List[str]:
        """识别推理模式"""
        return ["因果关系", "组成关系", "相似关系"]

    def _build_reasoning_chain(self, pattern: str, subgraph: KnowledgeSubgraph) -> Optional[str]:
        """构建推理链"""
        return f"基于{pattern}的推理链"

    def _validate_reasoning_chains(self, chains: List[str], query: str) -> List[str]:
        """验证推理链"""
        return chains[:3]

    def _fallback_subgraph_extraction(self, graph_query: GraphQuery) -> KnowledgeSubgraph:
        """降级子图提取"""
        return KnowledgeSubgraph(
            central_nodes=[],
            connected_nodes=[],
            relationships=[],
            graph_metrics={},
            reasoning_chains=[]
        )

    def close(self):
        """关闭资源连接"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
            logger.info("图RAG检索系统已关闭")
