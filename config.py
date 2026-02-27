"""DeltaForce Agent 全局配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class GraphRAGConfig:
    """系统配置（非敏感项在此，密钥放 .env）。"""

    # 运行模式
    run_mode: str = "agent"  # build / serve / rebuild / agent

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_database: str = "neo4j"
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection_name: str = "deltaforce_knowledge"
    milvus_dimension: int = 512

    # 模型
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_model: str = "kimi-k2-0711-preview"
    agent_intent_model: str = "kimi-k2-0711-preview"
    agent_planner_model: str = "kimi-k2-0711-preview"
    agent_specialist_model: str = "kimi-k2-0711-preview"
    agent_summary_model: str = "kimi-k2-0711-preview"
    agent_memory_model: str = "kimi-k2-0711-preview"

    # 检索
    top_k: int = 5
    hybrid_dual_weight: float = 0.55
    hybrid_vector_weight: float = 0.45
    rrf_k: int = 60
    entity_contains_min_len: int = 3

    # 生成
    temperature: float = 0.1
    max_tokens: int = 2048

    # 图处理
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_graph_depth: int = 2
    enable_llm_relation_keys: bool = False

    # 短期记忆（内存）
    memory_enabled: bool = True
    memory_recent_raw_limit: int = 10
    memory_pending_turns_trigger: int = 4
    memory_pending_tokens_trigger: int = 500
    memory_summary_max_tokens: int = 400
    memory_rebase_every_n_merges: int = 5
    memory_include_pending_in_prompt: bool = True
    memory_drop_failed_tool_messages: bool = True

    # 规划
    task_planning_enabled: bool = True

    # 长期记忆（PostgreSQL + pgvector）
    memory_persistent_enabled: bool = True
    memory_persistent_dsn: str = os.getenv("MEMORY_PERSISTENT_DSN", "")
    memory_persistent_vector_dim: int = 512
    memory_persistent_recall_top_k: int = 6
    memory_persistent_vector_top_k: int = 20
    memory_persistent_bm25_top_k: int = 20
    memory_persistent_bm25_candidate_limit: int = 200
    memory_persistent_rrf_k: int = 60
    memory_persistent_trigger_threshold: int = 2
    memory_persistent_market_ttl_hours: int = 24

    # 本地可视化记忆镜像
    memory_local_observer_enabled: bool = True
    memory_local_observer_dir: str = "/data/DeltaForce_Agent/data/memory/readable"

    # 市场数据后端（开源仓库仅保留抽象）
    df_market_backend_module: str = os.getenv("DF_MARKET_BACKEND_MODULE", "")
    df_market_backend_class: str = os.getenv("DF_MARKET_BACKEND_CLASS", "MarketDataBackendImpl")
    df_market_latest_price_operation: str = os.getenv("DF_MARKET_LATEST_PRICE_OPERATION", "latest_price")
    df_market_history_price_operation: str = os.getenv("DF_MARKET_HISTORY_PRICE_OPERATION", "history_price")
    df_market_object_lookup_operation: str = os.getenv("DF_MARKET_OBJECT_LOOKUP_OPERATION", "object_lookup")
    df_market_place_profit_rank_operation: str = os.getenv("DF_MARKET_PLACE_PROFIT_RANK_OPERATION", "place_profit_rank")
    df_market_place_profit_history_operation: str = os.getenv("DF_MARKET_PLACE_PROFIT_HISTORY_OPERATION", "place_profit_history")
    df_market_object_lookup_limit: int = int(os.getenv("DF_MARKET_OBJECT_LOOKUP_LIMIT", "3000"))

    def __post_init__(self):
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
        if self.memory_recent_raw_limit < 1:
            raise ValueError("memory_recent_raw_limit 必须 >= 1")
        if self.memory_pending_turns_trigger < 1:
            raise ValueError("memory_pending_turns_trigger 必须 >= 1")
        if self.memory_pending_tokens_trigger < 1:
            raise ValueError("memory_pending_tokens_trigger 必须 >= 1")
        if self.memory_summary_max_tokens < 64:
            raise ValueError("memory_summary_max_tokens 建议 >= 64")
        if self.memory_persistent_vector_dim < 32:
            raise ValueError("memory_persistent_vector_dim 必须 >= 32")
        if self.memory_persistent_recall_top_k < 1:
            raise ValueError("memory_persistent_recall_top_k 必须 >= 1")
        if self.memory_persistent_rrf_k < 1:
            raise ValueError("memory_persistent_rrf_k 必须 >= 1")
        if not self.neo4j_password:
            raise ValueError("缺少 NEO4J_PASSWORD，请在 .env 中配置")
        if self.memory_persistent_enabled and not self.memory_persistent_dsn:
            raise ValueError("缺少 MEMORY_PERSISTENT_DSN，请在 .env 中配置")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GraphRAGConfig":
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_mode": self.run_mode,
            "neo4j_uri": self.neo4j_uri,
            "neo4j_user": self.neo4j_user,
            "neo4j_password": self.neo4j_password,
            "neo4j_database": self.neo4j_database,
            "milvus_host": self.milvus_host,
            "milvus_port": self.milvus_port,
            "milvus_collection_name": self.milvus_collection_name,
            "milvus_dimension": self.milvus_dimension,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "agent_intent_model": self.agent_intent_model,
            "agent_planner_model": self.agent_planner_model,
            "agent_specialist_model": self.agent_specialist_model,
            "agent_summary_model": self.agent_summary_model,
            "agent_memory_model": self.agent_memory_model,
            "top_k": self.top_k,
            "hybrid_dual_weight": self.hybrid_dual_weight,
            "hybrid_vector_weight": self.hybrid_vector_weight,
            "rrf_k": self.rrf_k,
            "entity_contains_min_len": self.entity_contains_min_len,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "max_graph_depth": self.max_graph_depth,
            "enable_llm_relation_keys": self.enable_llm_relation_keys,
            "memory_enabled": self.memory_enabled,
            "memory_recent_raw_limit": self.memory_recent_raw_limit,
            "memory_pending_turns_trigger": self.memory_pending_turns_trigger,
            "memory_pending_tokens_trigger": self.memory_pending_tokens_trigger,
            "memory_summary_max_tokens": self.memory_summary_max_tokens,
            "memory_rebase_every_n_merges": self.memory_rebase_every_n_merges,
            "memory_include_pending_in_prompt": self.memory_include_pending_in_prompt,
            "memory_drop_failed_tool_messages": self.memory_drop_failed_tool_messages,
            "task_planning_enabled": self.task_planning_enabled,
            "memory_persistent_enabled": self.memory_persistent_enabled,
            "memory_persistent_dsn": self.memory_persistent_dsn,
            "memory_persistent_vector_dim": self.memory_persistent_vector_dim,
            "memory_persistent_recall_top_k": self.memory_persistent_recall_top_k,
            "memory_persistent_vector_top_k": self.memory_persistent_vector_top_k,
            "memory_persistent_bm25_top_k": self.memory_persistent_bm25_top_k,
            "memory_persistent_bm25_candidate_limit": self.memory_persistent_bm25_candidate_limit,
            "memory_persistent_rrf_k": self.memory_persistent_rrf_k,
            "memory_persistent_trigger_threshold": self.memory_persistent_trigger_threshold,
            "memory_persistent_market_ttl_hours": self.memory_persistent_market_ttl_hours,
            "memory_local_observer_enabled": self.memory_local_observer_enabled,
            "memory_local_observer_dir": self.memory_local_observer_dir,
            "df_market_backend_module": self.df_market_backend_module,
            "df_market_backend_class": self.df_market_backend_class,
            "df_market_latest_price_operation": self.df_market_latest_price_operation,
            "df_market_history_price_operation": self.df_market_history_price_operation,
            "df_market_object_lookup_operation": self.df_market_object_lookup_operation,
            "df_market_place_profit_rank_operation": self.df_market_place_profit_rank_operation,
            "df_market_place_profit_history_operation": self.df_market_place_profit_history_operation,
            "df_market_object_lookup_limit": self.df_market_object_lookup_limit,
        }


DEFAULT_CONFIG = GraphRAGConfig()
