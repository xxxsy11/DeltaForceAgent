# Conversation Benchmark Schema

`system_conversation_benchmark_100.json` 使用 conversation-level 结构。

## 顶层
- `meta`: 基本信息
- `cases`: 100 条完整系统样例

## case 结构
- `case_id`: 样例ID
- `user_id`: 样例用户ID（已隔离）
- `sessions`: 会话序列，默认包含两段：
  - `main`: 主会话，覆盖意图识别、工具调用、知识库RAG、短期记忆
  - `reentry`: 重入会话，覆盖长期记忆召回

## turn 结构
- `query`: 用户输入
- `expected_tool`
- `expected_skill`
- `expected_intents`
- `expected_entities`
- `expected_tool_query_contains`
- `answer_keywords`
- `expect_memory_resolution`
- `expect_persistent_recall`

## 设计原则
- 每条 case 都是完整端到端链路，不再是单轮拆分。
- 每条 case 使用独立 `user_id`，避免记忆冲突。
- 每条 case 都包含跨会话重入，用于长期记忆验证。
