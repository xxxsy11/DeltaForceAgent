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
- `src/tools/`
  - 通用工具层（被 Agent 调用）：
  - `rag_knowledge_tool.py`：RAG 查询工具
  - `registry.py`：统一工具注册中心
- `src/agents/`
  - 多 Agent 编排层：
  - `intent_analyzer.py`：意图分析（先判定该用哪个工具）
  - `graph.py`：LangGraph 工作流（intent -> tool_exec -> responder）
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

- 用户问题先进入 `intent_analyzer.py`
- 若判定为资料查询/知识分析：调用 `rag_knowledge_search`
- 后续新增工具时，只需：
  - 在 `src/tools/` 增加工具文件
  - 在 `src/tools/registry.py` 注册
  - 在 `intent_analyzer.py` 添加映射规则
