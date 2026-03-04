# DeltaForce_Agent 技术架构文档

## 1. 项目定位
`DeltaForce_Agent` 是面向《三角洲行动》场景的多智能体问答与分析系统。系统目标不是单轮问答，而是提供可持续对话、可检索记忆、可审查回退、可评测复现的工程化 Agent 能力。

核心能力：
- 主 Agent + 子 Agent 编排执行
- 知识库 RAG 与长期记忆 RAG 双检索
- 结构化工具调用（价格、历史、建议、制造利润、对比、稳定性、综合回答）
- 质量审查与分阶段重试
- 本地模型（Qwen3-8B + LoRA）与在线模型混合部署
- 全链路可观测（阶段耗时、路由、重试轨迹）

---

## 2. 总体架构

### 2.1 编排引擎
- 编排实现：`src/agents/graph.py`
- 运行入口：`main.py`、`src/agents/runner.py`
- 状态载体：`src/agents/state.py`

系统采用 LangGraph 进行状态机编排，所有节点通过统一 `AgentState` 读写中间状态，避免隐式共享变量。

### 2.2 主流程（A2A）
1. `main_orchestrator`：初始化本轮元信息  
2. `persistent_memory_recall`：长期记忆门控与召回  
3. `intent_recognition`：意图识别、实体解析、工具初选、技能命中  
4. `task_planning`（按需）：复杂问题二级规划  
5. `execution`：工具执行  
6. `tool_output_validator`：工具输出质量校验  
7. `specialist_analysis`（按需）：复杂问题专业分析  
8. `summary`：结果整合输出  
9. `answer_reviewer`：最终回答审查  
10. `retry_router`（按需）：选择回退阶段  
11. `memory_compression`：短期记忆压缩与事实提取  
12. `persistent_memory_write`：长期记忆持久化

---

## 3. 关键节点职责

### 3.1 意图识别层
- 文件：`src/agents/intent_recognition.py`
- 规则分析器：`src/agents/intent_analyzer.py`
- Prompt：`src/prompts/intent_recognition_prompt.txt`

处理逻辑：
- 规则先验（高置信度问题先收敛）
- LLM 结构化输出（JSON + 字段校验）
- 结合会话记忆做省略/代词补全
- 与 Skills 策略层协同生成工具链

工程增强点：
- 对制造利润相关问法（制造台、特勤处制造、技术中心、工作台、制药台、防具台）提升规则优先级，降低误路由到对比类工具。

### 3.2 任务规划层
- 文件：`src/agents/task_planning.py`
- 规划器：`src/agents/tool_planner.py`

只在复杂问题触发。负责把“意图 + 主体”映射为可执行工具链与参数，减少执行层的歧义失败。

### 3.3 工具复核层
- 文件：`src/agents/tool_selection_review.py`

当工具执行异常且判定为“工具/参数层问题”时触发，进行工具重选，避免无意义全链路重跑。

### 3.4 执行与整合层
- 执行：`src/agents/execution_agent.py`
- 总结：`src/agents/summary_agent.py`
- 专业分析：`src/agents/specialist_analysis.py`

执行层输出结构化 `tool_results`，总结层按事实优先原则合并回答，专业分析层仅在复杂任务触发。

### 3.5 质量审查与重试层
- 工具校验：`src/agents/tool_output_validator.py`
- 回答审查：`src/agents/answer_reviewer.py`
- 回退路由：`src/agents/retry_router.py`

机制：
- 先校验工具结果，再校验最终回答
- 按错误类型回退到最小必要阶段
- 分阶段重试预算控制，超限后降级返回

---

## 4. Skills 策略层
- 定义目录：`src/skills/definitions/`
- 选择与组装：`src/skills/registry.py`

作用：
- 将“问题类别 -> 工具链”配置化
- 降低纯 prompt 规划波动
- 提高高频任务的一致性与可解释性

---

## 5. 工具与服务层

### 5.1 工具注册
- 文件：`src/tools/registry.py`

已注册工具：
- `rag_knowledge_search`
- `df_market_latest_price`
- `df_market_history_price`
- `df_market_price_advice`
- `df_place_profit_rank`
- `df_multi_item_compare`
- `df_profit_stability`
- `df_answer_composer`

### 5.2 价格工具实现
- 入口：`src/tools/df_price_tools.py`
- 服务层：`src/services/df_price_service.py`

说明：
- 实时市场能力通过服务抽象封装，工具层统一做参数解析、约束和结果格式化。
- 开源仓库保留工具调用链与工程接口，不披露实时数据接入细节。

### 5.3 RAG 服务封装
- 文件：`src/services/rag_service.py`
- 核心模块：`src/rag_modules/*`

统一暴露 `query_async()`，上层节点无需感知检索策略细节。

---

## 6. 双 RAG 架构

### 6.1 知识库 RAG（GraphRAG）
依托 Neo4j + Milvus + BM25 混合检索，支持图结构推理与传统文本检索融合。

关键模块：
- `src/rag_modules/intelligent_query_router.py`
- `src/rag_modules/hybrid_retrieval.py`
- `src/rag_modules/graph_rag_retrieval.py`
- `src/rag_modules/rag_system.py`

### 6.2 长期记忆 RAG（Memory RAG）
- 召回节点：`src/memory/persistent_memory_recall_node.py`
- 存储实现：`src/memory/persistent_memory_store.py`

召回策略：
1. 门控打分  
2. 向量召回  
3. 关键词召回  
4. RRF 融合  
5. 拼接到 `memory_context`

---

## 7. 记忆系统设计

### 7.1 短期记忆（会话内）
- 管理器：`src/memory/session_memory.py`
- 压缩器：`src/memory/memory_compression_agent.py`

关键状态：
- `recent_raw`：最近原文窗口
- `pending_buffer`：待压缩缓冲
- `rolling_summary`：滚动摘要
- `memory_context`：运行时上下文

### 7.2 长期记忆（会话间）
- 写入节点：`src/memory/persistent_memory_write_node.py`
- 存储模块：`src/memory/persistent_memory_store.py`

主要存储实体：
- 完整对话记录（turn）
- 阶段摘要（summary）
- 结构化事实（fact）

支持 `user_id + session_id` 隔离。

---

## 8. 异步执行模型

### 8.1 已异步化范围
- 主流程节点 `run()` 全部采用异步调用
- 本地模型运行时提供 `ainvoke`（`src/agents/local_qwen_runtime.py`）
- 规划、总结、专业分析、记忆压缩路径改为异步调用
- 交互入口使用 `graph.ainvoke(...)`

### 8.2 当前保留的同步封装点
以下模块内部仍为同步实现，但已在异步层做线程封装：
- `src/services/rag_service.py`
- `src/memory/persistent_memory_store.py`

价格服务 `src/services/df_price_service.py` 当前保持同步 HTTP 实现（按业务约束保留）。

---

## 9. 可观测性与交互输出

### 9.1 阶段级追踪
- 文件：`src/agents/graph.py`
- 输出模式：中文阶段日志（开始/完成/耗时/关键结果）

示例：
- 正在意图识别 -> 完成（用时、识别意图、选中工具）
- 正在任务规划 -> 完成（计划工具数）
- 正在工具执行 -> 完成（成功/失败数量）

### 9.2 路由耗时
- 文件：`src/agents/runner.py`
- 输出：输入到工具选择完成耗时 + 选中工具

### 9.3 日志噪声控制
- 文件：`main.py`
- 对 `httpx/openai/transformers/sentence_transformers` 等日志进行降噪，突出业务流程信息。

---

## 10. 本地模型与 LoRA-SFT

### 10.1 本地推理运行时
- 文件：`src/agents/local_qwen_runtime.py`

能力：
- Qwen 基座单例加载
- 多 Adapter 按需切换
- no-think 推理模式
- 进程退出自动释放缓存

### 10.2 三模块 LoRA
默认适配路径：
- Intent：`outputs/intent_sft/qwen3_8b_lora`
- Tool Selection：`outputs/tool_selection_sft/qwen3_8b_lora`
- Planning：`outputs/planning_sft/qwen3_8b_lora`

---

## 11. 模型与评测资产

本仓库提供三类与模型优化相关的资产：
- 三模块 LoRA-SFT 的离线评测报告：`docs/SFT_EVAL_REPORT.md`
- 系统级评测样本：`data/benchmarks/system_eval_cases.json`
- 系统级自动化评测脚本：`data/scripts/system_full_metrics_suite.py`

说明：
- 训练代码与完整训练数据流水线不在当前开源仓库中维护；仓库保留可复现的评测入口与报告结果。
- 如需训练侧实现，可在此评测接口基础上接入外部训练工程。

---

## 12. Benchmark 与系统验证

核心脚本：
- `data/scripts/system_full_metrics_suite.py`
- `data/scripts/data_pipeline.py`

评测覆盖：
- 意图识别准确率
- 工具命中率
- 复杂任务规划触发与执行率
- 双 RAG 召回链路
- 重试链路与预算耗尽
- 阶段时延与端到端时延

---

## 13. 配置治理

### 13.1 `config.py`
存放可版本化配置：
- 模型与 LoRA 路径
- 记忆阈值与召回参数
- 重试预算
- 工具参数
- 阶段追踪开关

### 13.2 `.env`
存放敏感配置：
- API Key
- 数据库连接信息

---

## 14. 关键代码路径索引
- 编排：`src/agents/graph.py`
- 运行入口：`src/agents/runner.py`
- 状态定义：`src/agents/state.py`
- 意图识别：`src/agents/intent_recognition.py`
- 规则分析器：`src/agents/intent_analyzer.py`
- 工具规划：`src/agents/tool_planner.py`
- 任务规划：`src/agents/task_planning.py`
- 执行：`src/agents/execution_agent.py`
- 工具校验：`src/agents/tool_output_validator.py`
- 回答审查：`src/agents/answer_reviewer.py`
- 重试路由：`src/agents/retry_router.py`
- 本地模型运行时：`src/agents/local_qwen_runtime.py`
- 工具注册：`src/tools/registry.py`
- 价格工具：`src/tools/df_price_tools.py`
- 价格服务：`src/services/df_price_service.py`
- RAG 服务：`src/services/rag_service.py`
- 记忆模块：`src/memory/`
