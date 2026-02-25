# Delta Agent Project Structure

## 目录说明

- `main.py`
  - Multi-Agent 统一启动入口（LangGraph）
- `config.py`
  - 全局配置与运行模式
- `src/rag_modules/rag_system.py`
  - RAG 系统运行时主逻辑（build/serve/rebuild 调试入口）
- `src/rag_modules/`
  - 现有 RAG 核心模块（Neo4j/Milvus/Router/GraphRAG/Generation）。
- `src/services/`
  - 业务服务层。`RAGService` 将整套 RAG 封装成一个可调用能力。
  - `df_price_service.py`：市场价格服务（支持 objectName -> id 自动解析）。
  - `market_data_backend.py`：市场数据后端抽象与私有扩展加载器。
- `src/tools/`
  - 通用工具层（被 Agent 调用）：
  - `rag_knowledge_tool.py`：RAG 查询工具
  - `df_price_tools.py`：价格与利润分析工具
  - `registry.py`：统一工具注册中心
- `src/agents/`
  - 多 Agent 编排层（主Agent + 子Agent）：
  - `main_orchestrator.py`：主调度 Agent（A2A 起点）
  - `intent_recognition.py`：意图识别与简单/复杂边界判定
  - `task_planning.py`：复杂问题任务规划（可替换专业模型）
  - `execution_agent.py`：工具执行与结构化分析报告生成
  - `specialist_analysis.py`：专业分析子 Agent（按需触发）
  - `summary_agent.py`：最终总结输出
  - `intent_analyzer.py`：规则意图分析器（轻量兜底）
  - `tool_planner.py`：LLM 规划与总结能力
  - `graph.py`：LangGraph A2A 工作流定义
  - `runner.py`：交互与单次调用入口
- `data/`
  - 业务原始数据与 Neo4j/Milvus 数据目录。
- `data/scripts/`
  - `data_pipeline.py`：统一数据管道脚本（统计、CSV导出、Cypher导出、Neo4j导入、清库、重建）。

## 运行说明

- 离线建库：
  - `RAG_RUN_MODE=build`
- 在线问答：
  - `RAG_RUN_MODE=serve`
- Agent 模式（LangGraph）：
  - `RAG_RUN_MODE=agent`

推荐直接使用：
- `python main.py`
- `python data/scripts/data_pipeline.py --help`

RAG 子系统调试：
- `RAG_RUN_MODE=build PYTHONPATH=src python -m rag_modules.rag_system`
- `RAG_RUN_MODE=serve PYTHONPATH=src python -m rag_modules.rag_system`
- `RAG_RUN_MODE=rebuild PYTHONPATH=src python -m rag_modules.rag_system`

## Agent 调度逻辑

- LangGraph 流程：
  - `main_orchestrator -> intent_recognition`
  - 简单问题：`intent_recognition -> execution -> summary`
  - 复杂问题：`intent_recognition -> task_planning -> execution -> (specialist_analysis) -> summary`
- A2A 通信通过 `AgentState.agent_messages` 实现。
- `execution_agent` 统一产出 `analysis_report`，`summary_agent` 负责最终面向用户输出。
- 后续新增工具时：
  - 在 `src/tools/` 增加工具文件
  - 在 `src/tools/registry.py` 注册
  - 在 `intent_analyzer.py` 补充规则映射（可选）
