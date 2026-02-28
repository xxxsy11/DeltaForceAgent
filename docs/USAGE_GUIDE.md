# Delta Agent 使用指南（功能与工具映射）

## 1. 功能总览

| 功能 | 工具名 | 典型提问 |
|---|---|---|
| 三角洲知识问答（地图/武器/配件/干员/关系） | `rag_knowledge_search` | `介绍一下非洲之心` |
| 查询物品最新价格 | `df_market_latest_price` | `非洲之心现在什么价格` |
| 查询物品历史价格 | `df_market_history_price` | `查询一下非洲之心的历史价格` |
| 买卖建议（结合当前价+历史区间） | `df_market_price_advice` | `非洲之心现在建议买吗` |
| 特勤处制造利润榜（四大分组 Top1/Top3） | `df_place_profit_rank` | `制造什么子弹利润最高` |
| 多物品横向对比（性价比排序） | `df_multi_item_compare` | `非洲之心和海洋之泪对比一下` |
| 制造利润稳定性分析（波动/回撤/趋势） | `df_profit_stability` | `分析碳纤维散射箭矢利润稳定性` |
| 综合回答（资料+价格+建议+制造利润） | `df_answer_composer` | `介绍一下非洲之心并告诉我现在价格和是否建议买` |

## 2. 各功能详细说明（含实例）

### 2.1 知识问答（`rag_knowledge_search`）
- 作用：回答三角洲知识库问题（RAG）。
- 底层：`RAGService -> AdvancedGraphRAGSystem`（自动选择图检索/混合检索）。
- 输入示例：`介绍一下非洲之心`
- 输出示例（节选）：`非洲之心是六级工艺藏品，可交易，关联工艺藏品类别与六级关系。`

### 2.2 最新价格查询（`df_market_latest_price`）
- 作用：查询单物品当前价格。
- 底层 API：`/df/object/price/latest`，名称解析使用 `/df/object/price/latest/v3`。
- 输入示例：`非洲之心现在什么价格`
- 输出示例：`非洲之心 的最新价格为 12818829。 更新时间：2026-02-25 07:40:51。`

### 2.3 历史价格查询（`df_market_history_price`）
- 作用：查询历史价格区间统计。
- 底层 API：`/df/object/price/history/v2`。
- 输入示例：`查询一下非洲之心的历史价格`
- 输出示例（节选）：`历史价格查询成功，共 4950 条记录。最新样本价 ... 最早样本价 ... 区间最低 ... 区间最高 ...`

### 2.4 买卖建议分析（`df_market_price_advice`）
- 作用：判断贵/便宜、是否建议买卖、可赚可亏空间。
- 内部组合调用：`df_market_latest_price + df_market_history_price`。
- 输入示例：`非洲之心现在建议买吗`
- 输出示例（节选）：`当前区间位置 37.3%（中位）；买入建议小仓位分批；卖出建议结合持仓成本。`

### 2.5 特勤处利润榜（`df_place_profit_rank`）
- 作用：返回制造利润 Top1/Top3。
- 底层 API：`/df/place/profitRank/v1`。
- 分组范围：技术中心（枪械/配件）、工作台（子弹）、制药台（药品/针剂/维修工具包）、防具台（头盔/护甲/胸挂/背包）。
- 输入示例：`制造什么子弹利润最高`
- 输出示例：

```text
特勤处制造净利润榜（指定分组，Top1，排序维度: totalprofit）

工作台（子弹）
1. 9x39mm BP｜净利润 387,779｜小时利润 48,472.38｜采样时间 2026-02-25 06:30:00
```

### 2.6 多物品横向对比（`df_multi_item_compare`）
- 作用：多物品横向比较，输出相对优先级。
- 内部组合调用：对每个物品执行 `df_market_latest_price + df_market_history_price`。
- 输入示例：`非洲之心和海洋之泪对比一下`
- 输出示例（节选）：`返回两者现价、区间位置、上行空间、下行风险，并给出结论：当前更优为非洲之心。`

### 2.7 利润稳定性分析（`df_profit_stability`）
- 作用：分析某制造品利润“稳不稳”。
- 底层 API：`/df/place/profitHistory`。
- 指标：样本数、平均利润、标准差、CV、正利润占比、最大回撤、趋势、稳定性评级。
- 输入示例：`分析碳纤维散射箭矢利润稳定性`
- 输出示例（节选）：`稳定性评级：低；建议谨慎生产。`

### 2.8 综合回答（`df_answer_composer`）
- 作用：自动拼接资料、价格、建议等结果，输出一段完整回答。
- 内部组合调用：`rag_knowledge_search + df_market_latest_price + df_market_price_advice`，命中制造类问题时可追加 `df_place_profit_rank`。
- 输入示例：`介绍一下非洲之心并告诉我现在价格和是否建议买`
- 输出示例（节选）：`【资料介绍】+【实时价格】+【买卖建议】` 三段合并输出。

### 2.9 运行流程实例（主Agent + 子Agent）
- 简单问题示例：`非洲之心现在什么价格`
- 流程结果：`flow_type=simple`，`used_tools=['df_market_latest_price']`，由 `SummaryAgent` 直接返回工具结果。
- 复杂问题示例：`非洲之心和海洋之泪对比一下`
- 流程结果：`flow_type=complex`，`used_tools=['df_multi_item_compare']`，并触发 `SpecialistAnalysisAgent` 补充专业洞察。
