"""Auto-split from graph_rag_retrieval.py."""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

from rag_modules.graph_retrieval.types import (
    GraphPath,
    GraphQuery,
    KnowledgeSubgraph,
    QueryType,
)

logger = logging.getLogger(__name__)

GRAPH_INDEX_ENTITY_LIMIT = 1000
SOURCE_CANDIDATE_DEFAULT_PER_ENTITY = 3
SOURCE_MATCH_CONTAINS_MIN_LEN = 3
MULTI_HOP_RELATION_BONUS = 0.3
MULTI_HOP_NODE_DEGREE_DIVISOR = 10.0
MULTI_HOP_RESULT_LIMIT = 20
ENTITY_RELATION_LIMIT = 40
ENTITY_RELATION_PREFERRED_BONUS = 0.3
ENTITY_RELATION_BASE_SCORE = 0.8
ENTITY_RELATION_DEGREE_DIVISOR = 50.0
SHORTEST_PATH_BASE_SCORE = 1.2
SHORTEST_PATH_PREFERRED_BONUS = 0.3
SHORTEST_PATH_LIMIT = 20

class GraphTraversalMixin:
    def _build_graph_index(self):
        """构建图索引以加速查询"""
        logger.info("构建图结构索引...")
        
        try:
            with self.driver.session(database=self.database) as session:
                # 构建实体索引 - 修复Neo4j语法兼容性问题
                entity_query = """
                MATCH (n)
                WHERE n.nodeId IS NOT NULL
                WITH n, COUNT { (n)--() } as degree
                RETURN labels(n) as node_labels, n.nodeId as node_id, 
                       COALESCE(n.name, n.typeName, n.class, n.difficulty, toString(n.level), n.caliber) as name, degree
                ORDER BY degree DESC
                LIMIT $entity_limit
                """
                
                result = session.run(entity_query, {"entity_limit": GRAPH_INDEX_ENTITY_LIMIT})
                for record in result:
                    node_id = record["node_id"]
                    self.entity_cache[node_id] = {
                        "labels": record["node_labels"],
                        "name": record["name"],
                        "degree": record["degree"]
                    }
                
                # 构建关系类型索引
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

    def _resolve_source_node_ids(
        self,
        source_entities: List[str],
        session,
        max_candidates_per_entity: int = SOURCE_CANDIDATE_DEFAULT_PER_ENTITY,
    ) -> List[str]:
        """将自然语言实体解析成 nodeId，优先高精度匹配。"""
        if not source_entities:
            return []

        query = """
        UNWIND $source_entities AS source_name
        WITH trim(source_name) AS source_name
        WHERE source_name <> ""
        MATCH (n)
        WHERE n.nodeId IS NOT NULL
        WITH source_name, n,
             CASE
               WHEN toLower(toString(n.nodeId)) = toLower(source_name)
                 OR toLower(COALESCE(toString(n.name), "")) = toLower(source_name)
                 OR toLower(COALESCE(toString(n.typeName), "")) = toLower(source_name)
                 OR toLower(COALESCE(toString(n.class), "")) = toLower(source_name)
               THEN 1.0
               WHEN any(alias IN COALESCE(n.aliases, []) WHERE toLower(toString(alias)) = toLower(source_name))
               THEN 0.95
               WHEN toLower(COALESCE(toString(n.name), "")) STARTS WITH toLower(source_name)
                 OR toLower(COALESCE(toString(n.typeName), "")) STARTS WITH toLower(source_name)
                 OR toLower(COALESCE(toString(n.class), "")) STARTS WITH toLower(source_name)
               THEN 0.80
               WHEN size(source_name) >= $contains_min_len AND (
                    toLower(COALESCE(toString(n.name), "")) CONTAINS toLower(source_name)
                 OR toLower(COALESCE(toString(n.typeName), "")) CONTAINS toLower(source_name)
                 OR toLower(COALESCE(toString(n.class), "")) CONTAINS toLower(source_name)
               )
               THEN 0.60
               ELSE 0.0
             END AS match_score
        WHERE match_score > 0.0
        RETURN n.nodeId AS node_id, max(match_score) AS best_score
        ORDER BY best_score DESC
        LIMIT $global_limit
        """
        result = session.run(
            query,
            {
                "source_entities": source_entities,
                "max_candidates_per_entity": max_candidates_per_entity,
                "global_limit": max_candidates_per_entity * max(1, len(source_entities)),
                "contains_min_len": SOURCE_MATCH_CONTAINS_MIN_LEN,
            },
        )
        return [record["node_id"] for record in result if record.get("node_id")]

    def multi_hop_traversal(self, graph_query: GraphQuery) -> List[GraphPath]:
        """
        多跳图遍历：这是图RAG的核心优势
        通过图结构发现隐含的知识关联
        """
        logger.info(f"执行多跳遍历: {graph_query.source_entities} -> {graph_query.target_entities}")
        
        paths = []
        
        if not self.driver:
            logger.error("Neo4j连接未建立")
            return paths
            
        try:
            with self.driver.session(database=self.database) as session:
                # 构建多跳遍历查询
                source_entities = graph_query.source_entities
                target_keywords = graph_query.target_entities or []
                max_depth = self._normalize_max_depth(graph_query.max_depth)
                allowed_relation_types, preferred_relation_types = self._resolve_relation_filters(
                    graph_query.relation_types
                )
                source_node_ids = self._resolve_source_node_ids(source_entities, session=session)
                if not source_node_ids:
                    logger.info("未解析到有效源实体节点，跳过多跳遍历")
                    return paths
                
                # 根据查询类型选择不同的遍历策略
                if graph_query.query_type in (QueryType.MULTI_HOP, QueryType.ENTITY_RELATION):
                    # 根据是否有目标关键词动态拼接过滤条件
                    target_filter_clause = ""
                    if target_keywords:
                        target_filter_clause = """
                    AND ANY(kw IN $target_keywords WHERE
                        toLower(COALESCE(toString(target.nodeId), "")) = toLower(kw)
                        OR toLower(COALESCE(toString(target.name), "")) = toLower(kw)
                        OR toLower(COALESCE(toString(target.name), "")) STARTS WITH toLower(kw)
                        OR (size(kw) >= 3 AND toLower(COALESCE(toString(target.name), "")) CONTAINS toLower(kw))
                    )"""
                    
                    cypher_query = f"""
                    // 多跳推理查询
                    UNWIND $source_node_ids as source_node_id
                    MATCH (source {{nodeId: source_node_id}})
                    
                    // 执行多跳遍历
                    MATCH path = (source)-[rels*1..{max_depth}]-(target)
                    WHERE NOT source = target
                    AND ALL(rel IN rels WHERE type(rel) IN $allowed_relation_types)
                    {target_filter_clause}
                    
                    // 计算路径相关性
                    WITH path, source, target,
                         length(path) as path_len,
                         rels,
                         nodes(path) as path_nodes
                    
                    // 路径评分：短路径 + 高度数节点 + 关系类型匹配
                    WITH path, source, target, path_len, rels, path_nodes,
                         (1.0 / path_len) + 
                         (REDUCE(s = 0.0, n IN path_nodes | s + COUNT {{ (n)--() }}) / $node_degree_divisor / size(path_nodes)) +
                         (CASE WHEN ANY(r IN rels WHERE type(r) IN $preferred_relation_types) THEN $relation_bonus ELSE 0.0 END) as relevance
                    
                    ORDER BY relevance DESC
                    LIMIT $result_limit
                    
                    RETURN path, source, target, path_len, rels, path_nodes, relevance
                    """
                    
                    params = {
                        "source_node_ids": source_node_ids,
                        "allowed_relation_types": allowed_relation_types,
                        "preferred_relation_types": preferred_relation_types,
                        "node_degree_divisor": MULTI_HOP_NODE_DEGREE_DIVISOR,
                        "relation_bonus": MULTI_HOP_RELATION_BONUS,
                        "result_limit": MULTI_HOP_RESULT_LIMIT,
                    }
                    if target_keywords:
                        params["target_keywords"] = target_keywords
                    
                    result = session.run(cypher_query, params)
                    
                    for record in result:
                        path_data = self._parse_neo4j_path(record)
                        if path_data:
                            paths.append(path_data)
                
                elif graph_query.query_type == QueryType.PATH_FINDING:
                    # 最短路径查找
                    paths.extend(self._find_shortest_paths(graph_query, session))
                    
        except Exception as e:
            logger.error(f"多跳遍历失败: {e}")
            
        logger.info(f"多跳遍历完成，找到 {len(paths)} 条路径")
        return paths

    def extract_knowledge_subgraph(self, graph_query: GraphQuery) -> KnowledgeSubgraph:
        """
        提取知识子图：获取实体相关的完整知识网络
        这体现了图RAG的整体性思维
        """
        logger.info(f"提取知识子图: {graph_query.source_entities}")
        
        if not self.driver:
            logger.error("Neo4j连接未建立")
            return self._fallback_subgraph_extraction(graph_query)
        
        try:
            with self.driver.session(database=self.database) as session:
                max_depth = self._normalize_max_depth(graph_query.max_depth)
                allowed_relation_types, _ = self._resolve_relation_filters(graph_query.relation_types)
                source_node_ids = self._resolve_source_node_ids(graph_query.source_entities, session=session)
                if not source_node_ids:
                    logger.info("未解析到有效源实体节点，降级为空子图")
                    return self._fallback_subgraph_extraction(graph_query)
                # 简化的子图提取（不依赖APOC）
                cypher_query = f"""
                // 找到源实体
                UNWIND $source_node_ids as source_node_id
                MATCH (source {{nodeId: source_node_id}})

                // 获取指定深度的邻居（按所有源实体合并为单条记录，避免 single() 多记录异常）
                OPTIONAL MATCH (source)-[r*1..{max_depth}]-(neighbor)
                WHERE ALL(rel IN r WHERE type(rel) IN $allowed_relation_types)
                WITH
                    collect(DISTINCT source) AS sources,
                    [node IN collect(DISTINCT neighbor) WHERE node IS NOT NULL] AS neighbors,
                    [path_rels IN collect(r) WHERE path_rels IS NOT NULL] AS rel_paths

                WITH
                    sources,
                    neighbors[0..$max_nodes] AS nodes,
                    rel_paths[0..$max_nodes] AS rels

                WITH
                    sources,
                    nodes,
                    rels,
                    size(nodes) as node_count,
                    size(rels) as rel_count

                RETURN
                    sources,
                    nodes,
                    rels,
                    {{
                        node_count: node_count,
                        relationship_count: rel_count,
                        density: CASE WHEN node_count > 1 THEN toFloat(rel_count) / (node_count * (node_count - 1) / 2) ELSE 0.0 END
                    }} as metrics
                """
                
                result = session.run(cypher_query, {
                    "source_node_ids": source_node_ids,
                    "max_nodes": graph_query.max_nodes,
                    "allowed_relation_types": allowed_relation_types,
                })
                
                record = result.single()
                if record:
                    return self._build_knowledge_subgraph(record)
                    
        except Exception as e:
            logger.error(f"子图提取失败: {e}")
            
        # 降级方案：简单邻居查询
        return self._fallback_subgraph_extraction(graph_query)

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

            source_record = record.get("source")
            source_list = record.get("sources")
            if isinstance(source_list, list):
                central_nodes = [node_to_dict(node) for node in source_list if node is not None]
            elif source_record is not None:
                central_nodes = [node_to_dict(source_record)]
            else:
                central_nodes = []
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

    def _find_entity_relations(self, graph_query: GraphQuery, session) -> List[GraphPath]:
        """查找实体间关系"""
        paths: List[GraphPath] = []
        source_node_ids = self._resolve_source_node_ids(graph_query.source_entities, session=session)
        if not source_node_ids:
            return paths

        target_node_ids = self._resolve_source_node_ids(graph_query.target_entities or [], session=session)
        allowed_relation_types, preferred_relation_types = self._resolve_relation_filters(graph_query.relation_types)

        target_filter_clause = ""
        params: Dict[str, Any] = {
            "source_node_ids": source_node_ids,
            "allowed_relation_types": allowed_relation_types,
            "preferred_relation_types": preferred_relation_types,
            "limit": ENTITY_RELATION_LIMIT,
            "preferred_bonus": ENTITY_RELATION_PREFERRED_BONUS,
            "base_score": ENTITY_RELATION_BASE_SCORE,
            "degree_divisor": ENTITY_RELATION_DEGREE_DIVISOR,
        }
        if target_node_ids:
            target_filter_clause = "AND target.nodeId IN $target_node_ids"
            params["target_node_ids"] = target_node_ids

        cypher_query = f"""
        UNWIND $source_node_ids AS source_node_id
        MATCH (source {{nodeId: source_node_id}})-[rel]-(target)
        WHERE type(rel) IN $allowed_relation_types
        {target_filter_clause}
        WITH source, target, rel,
             (CASE WHEN type(rel) IN $preferred_relation_types THEN $preferred_bonus ELSE 0.0 END) +
             ($base_score + toFloat(COUNT {{ (target)--() }}) / $degree_divisor) AS relevance
        RETURN source, target,
               [source, target] AS path_nodes,
               [rel] AS rels,
               1 AS path_len,
               relevance
        ORDER BY relevance DESC
        LIMIT $limit
        """

        try:
            result = session.run(cypher_query, params)
            for record in result:
                path = self._parse_neo4j_path(record)
                if path:
                    path.path_type = "entity_relation"
                    paths.append(path)
        except Exception as e:
            logger.error(f"实体关系查询失败: {e}")

        return paths

    def _find_shortest_paths(self, graph_query: GraphQuery, session) -> List[GraphPath]:
        """查找最短路径"""
        paths: List[GraphPath] = []
        source_node_ids = self._resolve_source_node_ids(graph_query.source_entities, session=session)
        target_node_ids = self._resolve_source_node_ids(graph_query.target_entities or [], session=session)
        if not source_node_ids or not target_node_ids:
            return paths

        max_depth = self._normalize_max_depth(graph_query.max_depth)
        allowed_relation_types, preferred_relation_types = self._resolve_relation_filters(graph_query.relation_types)
        if not allowed_relation_types:
            return paths

        cypher_query = f"""
        UNWIND $source_node_ids AS source_node_id
        UNWIND $target_node_ids AS target_node_id
        MATCH (source {{nodeId: source_node_id}})
        MATCH (target {{nodeId: target_node_id}})
        WHERE source.nodeId <> target.nodeId
        MATCH path = shortestPath((source)-[*..{max_depth}]-(target))
        WHERE path IS NOT NULL
        WITH source, target, nodes(path) AS path_nodes, relationships(path) AS rels, length(path) AS path_len
        WHERE ALL(rel IN rels WHERE type(rel) IN $allowed_relation_types)
        WITH source, target, path_nodes, rels, path_len,
             ($base_score / toFloat(path_len)) +
             (CASE WHEN ANY(rel IN rels WHERE type(rel) IN $preferred_relation_types) THEN $preferred_bonus ELSE 0.0 END) AS relevance
        RETURN source, target, path_nodes, rels, path_len, relevance
        ORDER BY relevance DESC
        LIMIT $limit
        """

        try:
            result = session.run(
                cypher_query,
                {
                    "source_node_ids": source_node_ids,
                    "target_node_ids": target_node_ids,
                    "allowed_relation_types": allowed_relation_types,
                    "preferred_relation_types": preferred_relation_types,
                    "base_score": SHORTEST_PATH_BASE_SCORE,
                    "preferred_bonus": SHORTEST_PATH_PREFERRED_BONUS,
                    "limit": SHORTEST_PATH_LIMIT,
                },
            )
            for record in result:
                path = self._parse_neo4j_path(record)
                if path:
                    path.path_type = "shortest_path"
                    paths.append(path)
        except Exception as e:
            logger.error(f"最短路径查询失败: {e}")

        return paths

    def _fallback_subgraph_extraction(self, graph_query: GraphQuery) -> KnowledgeSubgraph:
        """降级子图提取"""
        return KnowledgeSubgraph(
            central_nodes=[],
            connected_nodes=[],
            relationships=[],
            graph_metrics={},
            reasoning_chains=[]
        )
