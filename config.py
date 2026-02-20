"""
基于图数据库的RAG系统配置文件（Delta Force）
"""

import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class GraphRAGConfig:
    """基于图数据库的RAG系统配置类"""

    # 运行模式
    # build: 离线建库
    # serve: 在线问答（只加载已有索引）
    # rebuild: 删除后重建索引
    # agent: LangGraph 多Agent入口（当前接入RAG工具）
    run_mode: str = os.getenv("RAG_RUN_MODE", "agent")

    # Neo4j数据库配置（可通过环境变量覆盖）
    neo4j_uri: str = os.getenv("NEO4J_URI", "neo4j://58.199.146.145:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "delta_agent")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")

    # Milvus配置（可通过环境变量覆盖）
    milvus_host: str = os.getenv("MILVUS_HOST", "localhost")
    milvus_port: int = int(os.getenv("MILVUS_PORT", "19530"))
    milvus_collection_name: str = os.getenv("MILVUS_COLLECTION", "deltaforce_knowledge")
    milvus_dimension: int = 512  # BGE-small-zh-v1.5的向量维度

    # 模型配置
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    llm_model: str = os.getenv("LLM_MODEL", "kimi-k2-0711-preview")

    # 检索配置（LightRAG Round-robin策略）
    top_k: int = 5
    hybrid_dual_weight: float = 0.55
    hybrid_vector_weight: float = 0.45
    rrf_k: int = 60
    entity_contains_min_len: int = 3

    # 生成配置
    temperature: float = 0.1
    max_tokens: int = 2048

    # 图数据处理配置
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_graph_depth: int = 2  # 图遍历最大深度
    enable_llm_relation_keys: bool = False

    def __post_init__(self):
        """初始化后的处理"""
        valid_modes = {"build", "serve", "rebuild", "agent"}
        if self.run_mode not in valid_modes:
            raise ValueError(f"run_mode 必须是 {valid_modes} 之一，当前值: {self.run_mode}")
        if self.hybrid_dual_weight < 0 or self.hybrid_vector_weight < 0:
            raise ValueError("hybrid_dual_weight 和 hybrid_vector_weight 必须 >= 0")
        if self.hybrid_dual_weight + self.hybrid_vector_weight == 0:
            raise ValueError("hybrid_dual_weight + hybrid_vector_weight 不能同时为 0")
        if self.rrf_k <= 0:
            raise ValueError("rrf_k 必须 > 0")
        if self.entity_contains_min_len < 1:
            raise ValueError("entity_contains_min_len 必须 >= 1")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GraphRAGConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'run_mode': self.run_mode,
            'neo4j_uri': self.neo4j_uri,
            'neo4j_user': self.neo4j_user,
            'neo4j_password': self.neo4j_password,
            'neo4j_database': self.neo4j_database,
            'milvus_host': self.milvus_host,
            'milvus_port': self.milvus_port,
            'milvus_collection_name': self.milvus_collection_name,
            'milvus_dimension': self.milvus_dimension,
            'embedding_model': self.embedding_model,
            'llm_model': self.llm_model,
            'top_k': self.top_k,
            'hybrid_dual_weight': self.hybrid_dual_weight,
            'hybrid_vector_weight': self.hybrid_vector_weight,
            'rrf_k': self.rrf_k,
            'entity_contains_min_len': self.entity_contains_min_len,

            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'max_graph_depth': self.max_graph_depth
        }

# 默认配置实例
DEFAULT_CONFIG = GraphRAGConfig() 
