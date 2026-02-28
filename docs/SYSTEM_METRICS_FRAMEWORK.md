# 全项目评测指标框架

## 1. 路由与工具层
- `intent_accuracy`
  - 定义：预测意图是否命中标注意图集合。
  - 计算：`命中条数 / 总样本数`。
  - 说明：使用 `selected_tool -> canonical intent` 映射做归一化，避免 LLM 自由文本意图描述导致误判。
- `tool_accuracy`
  - 定义：`selected_tool` 是否等于样本标注工具。
- `skill_accuracy`
  - 定义：skills 开启时，`selected_skill` 是否等于标注技能；skills 关闭时，要求空技能。

## 2. 主体解析层
- `entity_resolution_accuracy`
  - 定义：期望主体是否在以下任一载体被正确解析：`understanding_entities / tool_query / final_answer`。
  - 计算：`主体解析正确样本数 / 总样本数`。

## 3. 回答质量层（弱监督）
- `answer_keyword_coverage`
  - 定义：回答覆盖预期关键词的比例。
  - 计算：`命中关键词数 / 关键词总数`，再对样本求平均。
- `single_turn_success_rate`
  - 定义：单轮回答未命中失败标记（查询失败/工具调用失败/5xx 等）的比例。

## 4. 多轮与记忆层
- `multi_turn_success_rate`
  - 定义：多轮场景中，末轮同时满足工具正确、技能正确、`tool_query` 包含期望主体、且回答成功。
- `multi_turn_keyword_coverage`
  - 定义：多轮末轮回答关键词覆盖率平均值。

## 5. 用户隔离层
- `user_isolation_success`
  - 定义：同 session 不同 user_id 重入时，各自回答仅命中各自上下文目标对象。

## 6. 性能层
- `avg_latency_sec`
  - 定义：单轮样本平均耗时（秒）。
- `p95_latency_sec`
  - 定义：单轮样本 P95 时延（秒）。
- `avg_attempts`
  - 定义：每条样本平均重试次数（用于观察外部 API 波动影响）。

## 7. 综合评分
- `overall_score`
  - 线性加权：
  - `0.18*intent_accuracy`
  - `+0.20*tool_accuracy`
  - `+0.12*skill_accuracy`
  - `+0.12*entity_resolution_accuracy`
  - `+0.12*answer_keyword_coverage`
  - `+0.10*single_turn_success_rate`
  - `+0.10*multi_turn_success_rate`
  - `+0.04*multi_turn_keyword_coverage`
  - `+0.02*user_isolation_success`

## 8. 当前基准集
- 文件：`data/benchmarks/system_eval_cases.json`
- 覆盖：
  - 单轮 10 条（知识问答、实时价格、历史价格、建议、组合回答、对比、利润稳定性、制造利润榜）
  - 多轮 2 组（代词价格、代词对比）
  - user 隔离 1 组（A/B 双用户重入）
