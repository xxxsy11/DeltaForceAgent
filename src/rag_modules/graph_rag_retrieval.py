"""
真正的图RAG检索模块
基于图结构的知识推理和检索，而非简单的关键词匹配
"""

import json
import logging
import re
from collections import Counter, defaultdict, deque
from typing import List, Dict, Tuple, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum

from langchain_core.documents import Document
from neo4j import GraphDatabase
from .llm_utils import invoke_llm_text

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """查询类型枚举"""
    ENTITY_RELATION = "entity_relation"  # 实体关系查询：A和B有什么关系？
    MULTI_HOP = "multi_hop"  # 多跳查询：A通过什么连接到C？
    SUBGRAPH = "subgraph"  # 子图查询：A相关的所有信息
    PATH_FINDING = "path_finding"  # 路径查找：从A到B的最佳路径
    CLUSTERING = "clustering"  # 聚类查询：和A相似的都有什么？

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
    """
    真正的图RAG检索系统
    核心特点：
    1. 查询意图理解：识别图查询模式
    2. 多跳图遍历：深度关系探索
    3. 子图提取：相关知识网络
    4. 图结构推理：基于拓扑的推理
    5. 动态查询规划：自适应遍历策略
    """
    
    def __init__(self, config, llm_client):
        self.config = config
        self.llm_client = llm_client
        self.driver = None
        self.database = getattr(config, "neo4j_database", "neo4j")
        
        # 图结构缓存
        self.entity_cache = {}
        self.relation_cache = {}
        self.subgraph_cache = {}
        self.default_relation_whitelist = {
            "HAS_AREA",
            "HAS_KEY_CARD",
            "HAS_DIFFICULTY",
            "HAS_LEVEL",
            "HAS_SKILL",
            "OF_CLA_TYPE",
            "OF_EQ_TYPE",
            "OF_COL_TYPE",
            "OF_FIRE_TYPE",
            "OF_ATT_TYPE",
            "OF_AMMO_TYPE",
            "USES_AMMO",
            "CAN_ATTACH",
        }

    def _normalize_max_depth(self, max_depth: int) -> int:
        configured_depth = getattr(self.config, "max_graph_depth", 2)
        return max(1, min(int(max_depth or 1), int(configured_depth)))

    def _expand_relation_types_for_query(self, relation_types: Optional[List[str]]) -> List[str]:
        if not relation_types:
            return []
        expanded: List[str] = []
        seen = set()
        for relation_type in relation_types:
            if relation_type is None:
                continue
            rel = str(relation_type).strip()
            if not rel or rel in seen:
                continue
            seen.add(rel)
            expanded.append(rel)
        return expanded

    def _resolve_relation_filters(self, relation_types: Optional[List[str]]) -> Tuple[List[str], List[str]]:
        """
        解析关系过滤策略：
        - allowed_relation_types: 用于Cypher路径白名单过滤
        - preferred_relation_types: 用于路径打分加权
        """
        preferred = self._expand_relation_types_for_query(relation_types)
        if preferred:
            allowed = list(set(preferred))
        else:
            allowed = list(self.default_relation_whitelist)
        return allowed, preferred
        
    def initialize(self):
        """初始化图RAG检索系统"""
        logger.info("初始化图RAG检索系统...")
        
        # 连接Neo4j
        try:
            self.driver = GraphDatabase.driver(
                self.config.neo4j_uri, 
                auth=(self.config.neo4j_user, self.config.neo4j_password)
            )
            # 测试连接
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            logger.info("Neo4j连接成功")
        except Exception as e:
            logger.error(f"Neo4j连接失败: {e}")
            return
        
        # 预热：构建实体和关系索引
        self._build_graph_index()
        
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

    def _extract_json_payload(self, content: str) -> Dict[str, Any]:
        raw = (content or "").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
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

    def _normalize_string_list(self, value: Any, max_size: int = 8) -> List[str]:
        if not isinstance(value, list):
            return []
        normalized: List[str] = []
        seen = set()
        for item in value:
            text = str(item).strip() if item is not None else ""
            if not text:
                continue
            lowered = text.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            normalized.append(text)
            if len(normalized) >= max_size:
                break
        return normalized

    def _sanitize_graph_query_payload(self, payload: Dict[str, Any], fallback_query: str) -> GraphQuery:
        query_type_raw = str(payload.get("query_type", QueryType.SUBGRAPH.value)).strip()
        try:
            query_type = QueryType(query_type_raw)
        except ValueError:
            query_type = QueryType.SUBGRAPH

        source_entities = self._normalize_string_list(payload.get("source_entities"), max_size=8)
        if not source_entities:
            source_entities = [fallback_query.strip()] if fallback_query.strip() else []

        target_entities = self._normalize_string_list(payload.get("target_entities"), max_size=8)

        preferred_relations = self._expand_relation_types_for_query(payload.get("relation_types"))
        allowed_relation_pool = set(self.default_relation_whitelist) | set(self.relation_cache.keys())
        relation_types = [rel for rel in preferred_relations if rel in allowed_relation_pool]

        constraints = payload.get("constraints")
        if not isinstance(constraints, dict):
            constraints = {}

        return GraphQuery(
            query_type=query_type,
            source_entities=source_entities,
            target_entities=target_entities,
            relation_types=relation_types,
            max_depth=self._normalize_max_depth(payload.get("max_depth", 2)),
            max_nodes=50,
            constraints=constraints,
        )
    
    def understand_graph_query(self, query: str) -> GraphQuery:
        """
        理解查询的图结构意图
        这是图RAG的核心：从自然语言到图查询的转换
        """
        prompt = f"""
        你是三角洲行动图数据库查询规划器。只输出 JSON，不要输出任何解释文字。

        问题：{query}

        图中主要关系类型：
        ["HAS_AREA","HAS_KEY_CARD","HAS_DIFFICULTY","HAS_LEVEL","HAS_SKILL",
         "OF_CLA_TYPE","OF_EQ_TYPE","OF_COL_TYPE","OF_FIRE_TYPE","OF_ATT_TYPE","OF_AMMO_TYPE",
         "USES_AMMO","CAN_ATTACH"]

        query_type 只能是：
        - "entity_relation"
        - "multi_hop"
        - "subgraph"
        - "path_finding"
        - "clustering"

        输出 JSON 结构必须是：
        {{
          "query_type": "subgraph",
          "source_entities": ["实体名1"],
          "target_entities": [],
          "relation_types": ["HAS_AREA"],
          "max_depth": 2,
          "constraints": {{}}
        }}

        规则：
        1. source_entities 只放在图里最可能存在的具体实体名。
        2. relation_types 只允许使用上面的关系类型；不确定就返回空数组。
        3. max_depth 只允许 1~3 的整数。
        4. constraints 仅保存属性筛选（如等级、口径），无则返回空对象。
        """
        
        try:
            llm_text = invoke_llm_text(
                llm_client=self.llm_client,
                prompt=prompt,
                model=self.config.llm_model,
                temperature=0.0,
                max_tokens=1000,
            )
            payload = self._extract_json_payload(llm_text)
            if not payload:
                raise ValueError("LLM 未返回可解析 JSON")

            return self._sanitize_graph_query_payload(payload, fallback_query=query)
            
        except Exception as e:
            logger.error(f"查询意图理解失败: {e}")
            # 降级方案：默认子图查询
            return GraphQuery(
                query_type=QueryType.SUBGRAPH,
                source_entities=[query.strip()] if query.strip() else [],
                max_depth=2
            )

    def _resolve_source_node_ids(self, source_entities: List[str], session, max_candidates_per_entity: int = 3) -> List[str]:
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
               THEN 1.0
               WHEN any(alias IN COALESCE(n.aliases, []) WHERE toLower(toString(alias)) = toLower(source_name))
               THEN 0.95
               WHEN toLower(COALESCE(toString(n.name), "")) STARTS WITH toLower(source_name)
                 OR toLower(COALESCE(toString(n.typeName), "")) STARTS WITH toLower(source_name)
               THEN 0.80
               WHEN size(source_name) >= 3 AND (
                    toLower(COALESCE(toString(n.name), "")) CONTAINS toLower(source_name)
                 OR toLower(COALESCE(toString(n.typeName), "")) CONTAINS toLower(source_name)
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
                         (REDUCE(s = 0.0, n IN path_nodes | s + COUNT {{ (n)--() }}) / 10.0 / size(path_nodes)) +
                         (CASE WHEN ANY(r IN rels WHERE type(r) IN $preferred_relation_types) THEN 0.3 ELSE 0.0 END) as relevance
                    
                    ORDER BY relevance DESC
                    LIMIT 20
                    
                    RETURN path, source, target, path_len, rels, path_nodes, relevance
                    """
                    
                    params = {
                        "source_node_ids": source_node_ids,
                        "allowed_relation_types": allowed_relation_types,
                        "preferred_relation_types": preferred_relation_types,
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
    
    def adaptive_query_planning(self, query: str) -> List[GraphQuery]:
        """
        自适应查询规划：根据查询复杂度动态调整策略
        """
        # 分析查询复杂度
        complexity_score = self._analyze_query_complexity(query)
        
        query_plans = []
        
        if complexity_score < 0.3:
            # 简单查询：直接邻居查询
            plan = GraphQuery(
                query_type=QueryType.ENTITY_RELATION,
                source_entities=[query],
                max_depth=1,
                max_nodes=20
            )
            query_plans.append(plan)
            
        elif complexity_score < 0.7:
            # 中等复杂度：多跳查询
            plan = GraphQuery(
                query_type=QueryType.MULTI_HOP,
                source_entities=[query],
                max_depth=2,
                max_nodes=50
            )
            query_plans.append(plan)
            
        else:
            # 复杂查询：子图提取 + 推理
            plan1 = GraphQuery(
                query_type=QueryType.SUBGRAPH,
                source_entities=[query],
                max_depth=3,
                max_nodes=100
            )
            plan2 = GraphQuery(
                query_type=QueryType.MULTI_HOP,
                source_entities=[query],
                max_depth=3,
                max_nodes=50
            )
            query_plans.extend([plan1, plan2])
            
        return query_plans
    
    def graph_rag_search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        图RAG主搜索接口：整合所有图RAG能力
        """
        logger.info(f"开始图RAG检索: {query}")
        
        if not self.driver:
            logger.warning("Neo4j连接未建立，返回空结果")
            return []
        
        # 1. 查询意图理解
        graph_query = self.understand_graph_query(query)
        logger.info(f"查询类型: {graph_query.query_type.value}")
        
        results = []
        
        try:
            # 2. 根据查询类型执行不同策略
            if graph_query.query_type == QueryType.MULTI_HOP:
                # 多跳遍历
                paths = self.multi_hop_traversal(graph_query)
                results.extend(self._paths_to_documents(paths, query))

            elif graph_query.query_type == QueryType.PATH_FINDING:
                # 最短路径查找
                with self.driver.session(database=self.database) as session:
                    paths = self._find_shortest_paths(graph_query, session)
                results.extend(self._paths_to_documents(paths, query))
                
            elif graph_query.query_type in [QueryType.SUBGRAPH, QueryType.CLUSTERING]:
                # 子图提取 / 聚类查询：都视为“围绕核心实体的局部知识网络”
                subgraph = self.extract_knowledge_subgraph(graph_query)
                
                # 图结构推理
                reasoning_chains = self.graph_structure_reasoning(subgraph, query)
                
                results.extend(self._subgraph_to_documents(subgraph, reasoning_chains, query))
                
            elif graph_query.query_type == QueryType.ENTITY_RELATION:
                # 实体关系查询（优先一跳关系）
                with self.driver.session(database=self.database) as session:
                    paths = self._find_entity_relations(graph_query, session)
                if not paths:
                    # 兜底到少量跳遍历
                    paths = self.multi_hop_traversal(graph_query)
                results.extend(self._paths_to_documents(paths, query))
            
            # 3. 图结构相关性排序
            results = self._rank_by_graph_relevance(results, query)
            
            logger.info(f"图RAG检索完成，返回 {len(results[:top_k])} 个结果")
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"图RAG检索失败: {e}")
            return []
    
    # ========== 辅助方法 ==========
    
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
        
        return f"关于 {', '.join(central_names)} 的知识网络，包含 {node_count} 个相关概念和 {rel_count} 个关系。"
    
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
            if len(normalized) < 2 or normalized in seen:
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
        if rel_count >= node_count or density > 0.25:
            patterns.append("dense_association")

        if subgraph.central_nodes and subgraph.connected_nodes:
            patterns.append("hub_coverage")

        if relation_counter and "relation_summary" not in patterns:
            patterns.append("relation_summary")

        return patterns[:5]
    
    def _build_reasoning_chain(self, pattern: str, subgraph: KnowledgeSubgraph) -> Optional[str]:
        """基于图统计构建可读推理链。"""
        central_names = [node.get("name", "未知实体") for node in subgraph.central_nodes if node.get("name")]
        central_desc = "、".join(central_names[:2]) if central_names else "核心实体"
        relation_counter = self._relation_type_counter(subgraph)
        top_relations = relation_counter.most_common(3)
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
            connected_names = [node.get("name", "") for node in subgraph.connected_nodes[:5] if node.get("name")]
            if not connected_names:
                return None
            return f"中心扩散推理：{central_desc} 的相关节点包括 {'、'.join(connected_names[:3])}，可作为证据覆盖范围。"

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
            score += min(0.6, term_hits * 0.2)
            if any(marker.lower() in lowered for marker in relation_markers):
                score += 0.25
            if "推理" in chain:
                score += 0.05
            scored.append((score, chain))

        scored.sort(key=lambda item: item[0], reverse=True)
        validated = [chain for _, chain in scored[:5]]
        return validated
    
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
            "limit": 40,
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
             (CASE WHEN type(rel) IN $preferred_relation_types THEN 0.3 ELSE 0.0 END) +
             (0.8 + toFloat(COUNT {{ (target)--() }}) / 50.0) AS relevance
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
             (1.2 / toFloat(path_len)) +
             (CASE WHEN ANY(rel IN rels WHERE type(rel) IN $preferred_relation_types) THEN 0.3 ELSE 0.0 END) AS relevance
        RETURN source, target, path_nodes, rels, path_len, relevance
        ORDER BY relevance DESC
        LIMIT 20
        """

        try:
            result = session.run(
                cypher_query,
                {
                    "source_node_ids": source_node_ids,
                    "target_node_ids": target_node_ids,
                    "allowed_relation_types": allowed_relation_types,
                    "preferred_relation_types": preferred_relation_types,
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
    
    def close(self):
        """关闭资源连接"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
            logger.info("图RAG检索系统已关闭") 
