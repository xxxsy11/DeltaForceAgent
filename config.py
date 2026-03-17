"""项目级运行配置。"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_MEMORY_OBSERVER_DIR = PROJECT_ROOT / "data" / "memory" / "readable"


def _auto_local_device() -> str:
    try:
        import torch  # 按需导入，避免启动时强依赖 torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


@dataclass
class GraphRAGConfig:
    """系统配置。非敏感项保留在代码中，密钥通过环境变量注入。"""

    # 运行模式：build / serve / rebuild / agent
    run_mode: str = "agent"

    # Neo4j
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")
    # 密码通过环境变量注入
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")

    # Milvus
    milvus_host: str = os.getenv("MILVUS_HOST", "127.0.0.1")
    milvus_port: int = int(os.getenv("MILVUS_PORT", "19530"))
    milvus_collection_name: str = "deltaforce_knowledge"
    milvus_dimension: int = 512  # BGE-small-zh-v1.5的向量维度

    # 模型
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_model: str = "kimi-k2-0711-preview"
    # 默认由本地 Qwen 处理意图与规划，分析与总结仍可使用线上模型
    agent_intent_model: str = os.getenv("AGENT_INTENT_MODEL", "models/Qwen3-8B")
    agent_planner_model: str = os.getenv("AGENT_PLANNER_MODEL", "models/Qwen3-8B")
    agent_specialist_model: str = "kimi-k2-0711-preview"
    agent_summary_model: str = "kimi-k2-0711-preview"
    agent_memory_model: str = "kimi-k2-0711-preview"
    # 本地模型推理配置
    agent_local_enabled: bool = os.getenv("AGENT_LOCAL_ENABLED", "1").strip().lower() not in {"0", "false", "off", "no"}
    agent_local_device: str = os.getenv("AGENT_LOCAL_DEVICE", _auto_local_device()).strip()
    agent_local_max_new_tokens: int = int(os.getenv("AGENT_LOCAL_MAX_NEW_TOKENS", "384"))
    agent_local_no_think: bool = os.getenv("AGENT_LOCAL_NO_THINK", "1").strip().lower() not in {"0", "false", "off", "no"}
    agent_stage_trace_enabled: bool = os.getenv("AGENT_STAGE_TRACE_ENABLED", "1").strip().lower() not in {"0", "false", "off", "no"}
    # LangSmith / LangGraph 可观测
    langsmith_enabled: bool = os.getenv("LANGSMITH_ENABLED", "0").strip().lower() not in {"0", "false", "off", "no"}
    langsmith_project: str = os.getenv("LANGSMITH_PROJECT", "delta-agent").strip()
    langsmith_endpoint: str = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com").strip()
    langsmith_api_key: str = os.getenv("LANGSMITH_API_KEY", "").strip()
    langsmith_tracing_v2: bool = os.getenv("LANGSMITH_TRACING_V2", "1").strip().lower() not in {"0", "false", "off", "no"}
    langsmith_run_prefix: str = os.getenv("LANGSMITH_RUN_PREFIX", "delta-agent").strip()
    langsmith_graph_name: str = os.getenv("LANGSMITH_GRAPH_NAME", "delta-agent-graph").strip()
    langsmith_tags: str = os.getenv("LANGSMITH_TAGS", "delta-agent,langgraph").strip()
    agent_intent_adapter_path: str = os.getenv(
        "AGENT_INTENT_ADAPTER_PATH",
        str(DEFAULT_OUTPUTS_DIR / "intent_sft" / "qwen3_8b_lora"),
    )
    agent_tool_selection_adapter_path: str = os.getenv(
        "AGENT_TOOL_SELECTION_ADAPTER_PATH",
        str(DEFAULT_OUTPUTS_DIR / "tool_selection_sft" / "qwen3_8b_lora"),
    )
    agent_planning_adapter_path: str = os.getenv(
        "AGENT_PLANNING_ADAPTER_PATH",
        str(DEFAULT_OUTPUTS_DIR / "planning_sft" / "qwen3_8b_lora"),
    )
    # Self-Improving / Agentic RL 轨迹采集配置
    self_improve_enabled: bool = os.getenv("SELF_IMPROVE_ENABLED", "1").strip().lower() not in {"0", "false", "off", "no"}
    self_improve_output_dir: str = os.getenv("SELF_IMPROVE_OUTPUT_DIR", "data/self_improve/raw_trajectories")
    self_improve_collect_only_with_tools: bool = os.getenv("SELF_IMPROVE_COLLECT_ONLY_WITH_TOOLS", "1").strip().lower() not in {"0", "false", "off", "no"}
    self_improve_reward_tool_match: float = float(os.getenv("SELF_IMPROVE_REWARD_TOOL_MATCH", "1.2"))
    self_improve_reward_args_ok: float = float(os.getenv("SELF_IMPROVE_REWARD_ARGS_OK", "1.0"))
    self_improve_reward_exec_success: float = float(os.getenv("SELF_IMPROVE_REWARD_EXEC_SUCCESS", "1.5"))
    self_improve_reward_quality_pass: float = float(os.getenv("SELF_IMPROVE_REWARD_QUALITY_PASS", "1.0"))
    self_improve_reward_plan_trigger_correct: float = float(os.getenv("SELF_IMPROVE_REWARD_PLAN_TRIGGER_CORRECT", "1.5"))
    self_improve_penalty_overplan: float = float(os.getenv("SELF_IMPROVE_PENALTY_OVERPLAN", "0.8"))
    self_improve_penalty_underplan: float = float(os.getenv("SELF_IMPROVE_PENALTY_UNDERPLAN", "1.8"))
    self_improve_reward_plan_coverage: float = float(os.getenv("SELF_IMPROVE_REWARD_PLAN_COVERAGE", "2.0"))
    self_improve_reward_dependency_consistency: float = float(
        os.getenv("SELF_IMPROVE_REWARD_DEPENDENCY_CONSISTENCY", "1.5")
    )
    self_improve_reward_plan_exec_alignment: float = float(os.getenv("SELF_IMPROVE_REWARD_PLAN_EXEC_ALIGNMENT", "1.5"))
    self_improve_penalty_redundancy: float = float(os.getenv("SELF_IMPROVE_PENALTY_REDUNDANCY", "1.2"))
    self_improve_reward_efficiency: float = float(os.getenv("SELF_IMPROVE_REWARD_EFFICIENCY", "0.8"))
    self_improve_reward_terminal_success: float = float(os.getenv("SELF_IMPROVE_REWARD_TERMINAL_SUCCESS", "4.0"))
    self_improve_reward_terminal_partial: float = float(os.getenv("SELF_IMPROVE_REWARD_TERMINAL_PARTIAL", "1.5"))
    self_improve_penalty_terminal_fail: float = float(os.getenv("SELF_IMPROVE_PENALTY_TERMINAL_FAIL", "2.0"))
    self_improve_reward_recovery_success: float = float(os.getenv("SELF_IMPROVE_REWARD_RECOVERY_SUCCESS", "1.2"))
    self_improve_penalty_blind_retry: float = float(os.getenv("SELF_IMPROVE_PENALTY_BLIND_RETRY", "1.2"))
    self_improve_reasonable_max_steps: int = int(os.getenv("SELF_IMPROVE_REASONABLE_MAX_STEPS", "3"))
    self_improve_penalty_retry: float = float(os.getenv("SELF_IMPROVE_PENALTY_RETRY", "0.7"))
    self_improve_penalty_budget_exhausted: float = float(os.getenv("SELF_IMPROVE_PENALTY_BUDGET_EXHAUSTED", "1.5"))
    self_improve_penalty_latency_over_s: float = float(os.getenv("SELF_IMPROVE_PENALTY_LATENCY_OVER_S", "0.3"))
    self_improve_latency_budget_s: float = float(os.getenv("SELF_IMPROVE_LATENCY_BUDGET_S", "8.0"))
    # 自进化混合奖励（规则 + LLM Judge）
    self_improve_reward_rule_weight: float = float(os.getenv("SELF_IMPROVE_REWARD_RULE_WEIGHT", "1.0"))
    self_improve_reward_llm_weight: float = float(os.getenv("SELF_IMPROVE_REWARD_LLM_WEIGHT", "0.8"))
    self_improve_llm_judge_enabled: bool = os.getenv("SELF_IMPROVE_LLM_JUDGE_ENABLED", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }
    self_improve_llm_judge_model: str = os.getenv("SELF_IMPROVE_LLM_JUDGE_MODEL", "kimi-k2-0711-preview").strip()
    self_improve_llm_judge_timeout_seconds: int = int(os.getenv("SELF_IMPROVE_LLM_JUDGE_TIMEOUT_SECONDS", "30"))
    self_improve_llm_judge_max_tokens: int = int(os.getenv("SELF_IMPROVE_LLM_JUDGE_MAX_TOKENS", "600"))
    self_improve_llm_judge_temperature: float = float(os.getenv("SELF_IMPROVE_LLM_JUDGE_TEMPERATURE", "0.0"))
    self_improve_llm_weight_overall: float = float(os.getenv("SELF_IMPROVE_LLM_WEIGHT_OVERALL", "0.6"))
    self_improve_llm_weight_planning_quality: float = float(
        os.getenv("SELF_IMPROVE_LLM_WEIGHT_PLANNING_QUALITY", "1.0")
    )
    self_improve_llm_weight_dependency_consistency: float = float(
        os.getenv("SELF_IMPROVE_LLM_WEIGHT_DEPENDENCY_CONSISTENCY", "0.8")
    )
    self_improve_llm_weight_argument_quality: float = float(
        os.getenv("SELF_IMPROVE_LLM_WEIGHT_ARGUMENT_QUALITY", "0.8")
    )
    self_improve_llm_weight_execution_consistency: float = float(
        os.getenv("SELF_IMPROVE_LLM_WEIGHT_EXECUTION_CONSISTENCY", "0.9")
    )
    self_improve_llm_weight_result_quality: float = float(os.getenv("SELF_IMPROVE_LLM_WEIGHT_RESULT_QUALITY", "1.0"))
    self_improve_llm_hard_adjustment_cap: float = float(os.getenv("SELF_IMPROVE_LLM_HARD_ADJUSTMENT_CAP", "2.0"))

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

    # 内存会话记忆（内存态）
    memory_enabled: bool = True
    memory_recent_raw_limit: int = 10
    memory_pending_turns_trigger: int = 4
    memory_pending_tokens_trigger: int = 500
    memory_summary_max_tokens: int = 400
    memory_rebase_every_n_merges: int = 5
    memory_include_pending_in_prompt: bool = True
    memory_drop_failed_tool_messages: bool = True
    # 二级任务规划（TaskPlanning）是否启用
    task_planning_enabled: bool = True
    # 持久化长期记忆（PostgreSQL + pgvector）
    memory_persistent_enabled: bool = os.getenv("MEMORY_PERSISTENT_ENABLED", "1").strip().lower() not in {"0", "false", "off", "no"}
    # PostgreSQL DSN（敏感）
    memory_persistent_dsn: str = os.getenv("MEMORY_PERSISTENT_DSN", "")
    memory_persistent_vector_dim: int = 512
    memory_persistent_recall_top_k: int = 6
    memory_persistent_vector_top_k: int = 20
    memory_persistent_bm25_top_k: int = 20
    memory_persistent_bm25_candidate_limit: int = 200
    memory_persistent_rrf_k: int = 60
    memory_persistent_trigger_threshold: int = 2
    memory_persistent_market_ttl_hours: int = 24
    memory_persistent_connect_timeout_seconds: int = 10
    # 长期记忆可视化镜像（本地自然语言）
    memory_local_observer_enabled: bool = True
    memory_local_observer_dir: str = str(DEFAULT_MEMORY_OBSERVER_DIR)

    # 质量审查与重试预算
    answer_min_chars: int = int(os.getenv("ANSWER_MIN_CHARS", "12"))
    retry_max_total: int = int(os.getenv("RETRY_MAX_TOTAL", "3"))
    retry_max_intent_recognition: int = int(os.getenv("RETRY_MAX_INTENT_RECOGNITION", "1"))
    retry_max_tool_selection_review: int = int(os.getenv("RETRY_MAX_TOOL_SELECTION_REVIEW", "1"))
    retry_max_task_planning: int = int(os.getenv("RETRY_MAX_TASK_PLANNING", "1"))
    retry_max_execution: int = int(os.getenv("RETRY_MAX_EXECUTION", "1"))
    retry_max_specialist_analysis: int = int(os.getenv("RETRY_MAX_SPECIALIST_ANALYSIS", "1"))
    retry_max_summary: int = int(os.getenv("RETRY_MAX_SUMMARY", "1"))
    execution_retry_max: int = int(os.getenv("EXECUTION_RETRY_MAX", "1"))
    execution_max_concurrency: int = int(os.getenv("EXECUTION_MAX_CONCURRENCY", "4"))
    replan_retry_max: int = int(os.getenv("REPLAN_RETRY_MAX", "1"))

    # 三角洲开放 API（价格查询）
    df_api_base_url: str = os.getenv("DF_API_BASE_URL", "https://df-api.shallow.ink").strip()
    df_api_token: str = os.getenv("DF_API_TOKEN", "")
    # 允许配置多个候选路径，逗号分隔；按顺序重试
    df_api_latest_price_paths: str = "/df/object/price/latest"
    df_api_history_price_paths: str = "/df/object/price/history/v2"
    df_api_object_lookup_paths: str = "/df/object/price/latest/v3"
    df_api_place_profit_rank_paths: str = "/df/place/profitRank/v1"
    df_api_place_profit_history_paths: str = "/df/place/profitHistory"
    df_api_object_lookup_limit: int = 3000
    df_api_timeout_seconds: int = 15
    sell_fee_rate: float = float(os.getenv("SELL_FEE_RATE", "0.13"))

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
        if self.memory_persistent_connect_timeout_seconds < 1:
            raise ValueError("memory_persistent_connect_timeout_seconds 必须 >= 1")
        if self.task_planning_enabled not in {True, False}:
            raise ValueError("task_planning_enabled 必须是布尔值")
        if self.agent_local_max_new_tokens < 32:
            raise ValueError("agent_local_max_new_tokens 必须 >= 32")
        if self.self_improve_latency_budget_s <= 0:
            raise ValueError("self_improve_latency_budget_s 必须 > 0")
        if self.self_improve_reasonable_max_steps < 1:
            raise ValueError("self_improve_reasonable_max_steps 必须 >= 1")
        if self.self_improve_reward_rule_weight < 0:
            raise ValueError("self_improve_reward_rule_weight 必须 >= 0")
        if self.self_improve_reward_llm_weight < 0:
            raise ValueError("self_improve_reward_llm_weight 必须 >= 0")
        if self.self_improve_llm_judge_timeout_seconds < 1:
            raise ValueError("self_improve_llm_judge_timeout_seconds 必须 >= 1")
        if self.self_improve_llm_judge_max_tokens < 64:
            raise ValueError("self_improve_llm_judge_max_tokens 必须 >= 64")
        if self.self_improve_llm_hard_adjustment_cap <= 0:
            raise ValueError("self_improve_llm_hard_adjustment_cap 必须 > 0")
        if self.retry_max_total < 0:
            raise ValueError("retry_max_total 必须 >= 0")
        if self.answer_min_chars < 1:
            raise ValueError("answer_min_chars 必须 >= 1")
        if self.execution_max_concurrency < 1:
            raise ValueError("execution_max_concurrency 必须 >= 1")
        if not (0.0 <= float(self.sell_fee_rate) < 1.0):
            raise ValueError("sell_fee_rate 必须在 [0, 1) 区间")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GraphRAGConfig":
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
            'agent_intent_model': self.agent_intent_model,
            'agent_planner_model': self.agent_planner_model,
            'agent_specialist_model': self.agent_specialist_model,
            'agent_summary_model': self.agent_summary_model,
            'agent_memory_model': self.agent_memory_model,
            'agent_local_enabled': self.agent_local_enabled,
            'agent_local_device': self.agent_local_device,
            'agent_local_max_new_tokens': self.agent_local_max_new_tokens,
            'agent_local_no_think': self.agent_local_no_think,
            'agent_stage_trace_enabled': self.agent_stage_trace_enabled,
            'langsmith_enabled': self.langsmith_enabled,
            'langsmith_project': self.langsmith_project,
            'langsmith_endpoint': self.langsmith_endpoint,
            'langsmith_api_key': self.langsmith_api_key,
            'langsmith_tracing_v2': self.langsmith_tracing_v2,
            'langsmith_run_prefix': self.langsmith_run_prefix,
            'langsmith_graph_name': self.langsmith_graph_name,
            'langsmith_tags': self.langsmith_tags,
            'agent_intent_adapter_path': self.agent_intent_adapter_path,
            'agent_tool_selection_adapter_path': self.agent_tool_selection_adapter_path,
            'agent_planning_adapter_path': self.agent_planning_adapter_path,
            'self_improve_enabled': self.self_improve_enabled,
            'self_improve_output_dir': self.self_improve_output_dir,
            'self_improve_collect_only_with_tools': self.self_improve_collect_only_with_tools,
            'self_improve_reward_tool_match': self.self_improve_reward_tool_match,
            'self_improve_reward_args_ok': self.self_improve_reward_args_ok,
            'self_improve_reward_exec_success': self.self_improve_reward_exec_success,
            'self_improve_reward_quality_pass': self.self_improve_reward_quality_pass,
            'self_improve_reward_plan_trigger_correct': self.self_improve_reward_plan_trigger_correct,
            'self_improve_penalty_overplan': self.self_improve_penalty_overplan,
            'self_improve_penalty_underplan': self.self_improve_penalty_underplan,
            'self_improve_reward_plan_coverage': self.self_improve_reward_plan_coverage,
            'self_improve_reward_dependency_consistency': self.self_improve_reward_dependency_consistency,
            'self_improve_reward_plan_exec_alignment': self.self_improve_reward_plan_exec_alignment,
            'self_improve_penalty_redundancy': self.self_improve_penalty_redundancy,
            'self_improve_reward_efficiency': self.self_improve_reward_efficiency,
            'self_improve_reward_terminal_success': self.self_improve_reward_terminal_success,
            'self_improve_reward_terminal_partial': self.self_improve_reward_terminal_partial,
            'self_improve_penalty_terminal_fail': self.self_improve_penalty_terminal_fail,
            'self_improve_reward_recovery_success': self.self_improve_reward_recovery_success,
            'self_improve_penalty_blind_retry': self.self_improve_penalty_blind_retry,
            'self_improve_reasonable_max_steps': self.self_improve_reasonable_max_steps,
            'self_improve_penalty_retry': self.self_improve_penalty_retry,
            'self_improve_penalty_budget_exhausted': self.self_improve_penalty_budget_exhausted,
            'self_improve_penalty_latency_over_s': self.self_improve_penalty_latency_over_s,
            'self_improve_latency_budget_s': self.self_improve_latency_budget_s,
            'self_improve_reward_rule_weight': self.self_improve_reward_rule_weight,
            'self_improve_reward_llm_weight': self.self_improve_reward_llm_weight,
            'self_improve_llm_judge_enabled': self.self_improve_llm_judge_enabled,
            'self_improve_llm_judge_model': self.self_improve_llm_judge_model,
            'self_improve_llm_judge_timeout_seconds': self.self_improve_llm_judge_timeout_seconds,
            'self_improve_llm_judge_max_tokens': self.self_improve_llm_judge_max_tokens,
            'self_improve_llm_judge_temperature': self.self_improve_llm_judge_temperature,
            'self_improve_llm_weight_overall': self.self_improve_llm_weight_overall,
            'self_improve_llm_weight_planning_quality': self.self_improve_llm_weight_planning_quality,
            'self_improve_llm_weight_dependency_consistency': self.self_improve_llm_weight_dependency_consistency,
            'self_improve_llm_weight_argument_quality': self.self_improve_llm_weight_argument_quality,
            'self_improve_llm_weight_execution_consistency': self.self_improve_llm_weight_execution_consistency,
            'self_improve_llm_weight_result_quality': self.self_improve_llm_weight_result_quality,
            'self_improve_llm_hard_adjustment_cap': self.self_improve_llm_hard_adjustment_cap,
            'top_k': self.top_k,
            'hybrid_dual_weight': self.hybrid_dual_weight,
            'hybrid_vector_weight': self.hybrid_vector_weight,
            'rrf_k': self.rrf_k,
            'entity_contains_min_len': self.entity_contains_min_len,

            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'max_graph_depth': self.max_graph_depth,
            'memory_enabled': self.memory_enabled,
            'memory_recent_raw_limit': self.memory_recent_raw_limit,
            'memory_pending_turns_trigger': self.memory_pending_turns_trigger,
            'memory_pending_tokens_trigger': self.memory_pending_tokens_trigger,
            'memory_summary_max_tokens': self.memory_summary_max_tokens,
            'memory_rebase_every_n_merges': self.memory_rebase_every_n_merges,
            'memory_include_pending_in_prompt': self.memory_include_pending_in_prompt,
            'memory_drop_failed_tool_messages': self.memory_drop_failed_tool_messages,
            'task_planning_enabled': self.task_planning_enabled,
            'memory_persistent_enabled': self.memory_persistent_enabled,
            'memory_persistent_dsn': self.memory_persistent_dsn,
            'memory_persistent_vector_dim': self.memory_persistent_vector_dim,
            'memory_persistent_recall_top_k': self.memory_persistent_recall_top_k,
            'memory_persistent_vector_top_k': self.memory_persistent_vector_top_k,
            'memory_persistent_bm25_top_k': self.memory_persistent_bm25_top_k,
            'memory_persistent_bm25_candidate_limit': self.memory_persistent_bm25_candidate_limit,
            'memory_persistent_rrf_k': self.memory_persistent_rrf_k,
            'memory_persistent_trigger_threshold': self.memory_persistent_trigger_threshold,
            'memory_persistent_market_ttl_hours': self.memory_persistent_market_ttl_hours,
            'memory_persistent_connect_timeout_seconds': self.memory_persistent_connect_timeout_seconds,
            'memory_local_observer_enabled': self.memory_local_observer_enabled,
            'memory_local_observer_dir': self.memory_local_observer_dir,
            'answer_min_chars': self.answer_min_chars,
            'retry_max_total': self.retry_max_total,
            'retry_max_intent_recognition': self.retry_max_intent_recognition,
            'retry_max_tool_selection_review': self.retry_max_tool_selection_review,
            'retry_max_task_planning': self.retry_max_task_planning,
            'retry_max_execution': self.retry_max_execution,
            'retry_max_specialist_analysis': self.retry_max_specialist_analysis,
            'retry_max_summary': self.retry_max_summary,
            'execution_retry_max': self.execution_retry_max,
            'execution_max_concurrency': self.execution_max_concurrency,
            'replan_retry_max': self.replan_retry_max,

            'df_api_base_url': self.df_api_base_url,
            'df_api_token': self.df_api_token,
            'df_api_latest_price_paths': self.df_api_latest_price_paths,
            'df_api_history_price_paths': self.df_api_history_price_paths,
            'df_api_object_lookup_paths': self.df_api_object_lookup_paths,
            'df_api_place_profit_rank_paths': self.df_api_place_profit_rank_paths,
            'df_api_place_profit_history_paths': self.df_api_place_profit_history_paths,
            'df_api_object_lookup_limit': self.df_api_object_lookup_limit,
            'df_api_timeout_seconds': self.df_api_timeout_seconds,
            'sell_fee_rate': self.sell_fee_rate,
        }


def validate_runtime_config(config: GraphRAGConfig) -> None:
    """运行时必需项校验。仅在真正启动系统时调用。"""
    if not config.neo4j_password:
        raise ValueError("缺少 NEO4J_PASSWORD，请在 .env 中配置")
    if config.memory_persistent_enabled and not config.memory_persistent_dsn:
        raise ValueError("缺少 MEMORY_PERSISTENT_DSN，请在 .env 中配置")
    if config.langsmith_enabled and not config.langsmith_api_key:
        raise ValueError("启用了 LANGSMITH_ENABLED，但缺少 LANGSMITH_API_KEY")


# 默认配置快照（允许在导入阶段创建；运行时必填项由 validate_runtime_config 校验）
DEFAULT_CONFIG = GraphRAGConfig()
