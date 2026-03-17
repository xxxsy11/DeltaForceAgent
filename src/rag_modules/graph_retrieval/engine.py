"""Graph retrieval engine composition."""

from __future__ import annotations

import logging
from typing import List

from langchain_core.documents import Document
from neo4j import GraphDatabase

from rag_modules.graph_retrieval.query_mixin import GraphQueryMixin
from rag_modules.graph_retrieval.reasoning_mixin import GraphReasoningMixin
from rag_modules.graph_retrieval.traversal_mixin import GraphTraversalMixin
from rag_modules.graph_retrieval.types import GraphQuery, QueryType

logger = logging.getLogger(__name__)


class GraphRAGRetrieval(GraphQueryMixin, GraphTraversalMixin, GraphReasoningMixin):
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

    def close(self):
        """关闭资源连接"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
            logger.info("图RAG检索系统已关闭")
