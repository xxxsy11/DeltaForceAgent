# 系统全量功能与指标测试报告

## 1. 测试元信息
- 生成时间(UTC): `2026-03-01T03:13:45.692657+00:00`
- 主用户: `full-user-main`
- 主会话: `full-main-20260301_030710`
- 断言: `PASS=10, FAIL=0, TOTAL=10`

## 2. 核心指标
### core
- `intent_accuracy`: `1.0`
- `tool_accuracy`: `1.0`
- `skill_accuracy`: `1.0`
- `entity_resolution_accuracy`: `1.0`
- `answer_keyword_coverage`: `0.9667`
- `single_turn_success_rate`: `1.0`
- `multi_turn_success_rate`: `1.0`
- `multi_turn_keyword_coverage`: `0.75`
- `user_isolation_success`: `True`
- `overall_score`: `0.8467`

### kb_rag
- `kb_rag_case_count`: `2`
- `kb_rag_route_accuracy`: `1.0`
- `kb_rag_entity_recall_proxy`: `1.0`
- `kb_rag_answer_keyword_coverage`: `1.0`
- `kb_rag_success_rate`: `1.0`
- `kb_rag_avg_latency_sec`: `15.19`

### ltm_rag
- `ltm_case_count`: `5`
- `ltm_gate_trigger_rate`: `1.0`
- `ltm_recall_hit_rate`: `0.2`
- `ltm_recall_hit_rate_when_triggered`: `0.2`
- `ltm_entity_resolution_accuracy`: `1.0`
- `ltm_success_rate`: `1.0`
- `ltm_avg_recall_hits`: `1.2`

### engineering
- `turn_count`: `30`
- `avg_latency_sec`: `12.91`
- `p50_latency_sec`: `11.67`
- `p95_latency_sec`: `22.91`
- `avg_attempts`: `1.0`
- `validator_reject_rate`: `0.0333`
- `reviewer_reject_rate`: `0.0`
- `retry_invocation_rate`: `0.0333`
- `retry_budget_exhausted_rate`: `0.0`
- `quality_gate_pass_rate`: `1.0`
- `persistent_write_block_rate`: `0.0`
- `tool_stage_success_rate`: `0.9667`
- `tool_stage_fail_rate`: `0.0333`
- `compression_trigger_rate`: `0.0333`

## 3. 单轮样本结果
- `st_01` | tool=`rag_knowledge_search` | skill=`knowledge_profile` | intent_ok=True | tool_ok=True | entity_ok=True | answer_fail=False
- `st_02` | tool=`df_market_latest_price` | skill=`market_latest_price` | intent_ok=True | tool_ok=True | entity_ok=True | answer_fail=False
- `st_03` | tool=`df_market_history_price` | skill=`market_history_price` | intent_ok=True | tool_ok=True | entity_ok=True | answer_fail=False
- `st_04` | tool=`df_market_price_advice` | skill=`market_price_advice` | intent_ok=True | tool_ok=True | entity_ok=True | answer_fail=False
- `st_05` | tool=`df_answer_composer` | skill=`answer_composer` | intent_ok=True | tool_ok=True | entity_ok=True | answer_fail=False
- `st_06` | tool=`df_multi_item_compare` | skill=`market_multi_item_compare` | intent_ok=True | tool_ok=True | entity_ok=True | answer_fail=False
- `st_07` | tool=`df_profit_stability` | skill=`profit_stability` | intent_ok=True | tool_ok=True | entity_ok=True | answer_fail=False
- `st_08` | tool=`df_place_profit_rank` | skill=`place_profit_rank` | intent_ok=True | tool_ok=True | entity_ok=True | answer_fail=False
- `st_09` | tool=`df_place_profit_rank` | skill=`place_profit_rank` | intent_ok=True | tool_ok=True | entity_ok=True | answer_fail=False
- `st_10` | tool=`rag_knowledge_search` | skill=`knowledge_profile` | intent_ok=True | tool_ok=True | entity_ok=True | answer_fail=False

## 4. 多轮样本结果
- `mt_01_pronoun_price` | target_tool=`df_market_latest_price` | tool_query_ok=True | answer_fail=False
- `mt_02_pronoun_compare` | target_tool=`df_multi_item_compare` | tool_query_ok=True | answer_fail=False

## 5. 集成功能覆盖（全工具）
- R1 | query=`介绍一下非洲之心` | tool=`rag_knowledge_search` | gate=0 | recall_hits=0 | retries=0
  - output: `结论   非洲之心是《三角洲行动》中唯一一枚“六级工艺藏品”级别的可交易钻石类收藏品，也是目前资料库中等级最高的工艺藏品。  关键信息   1. 基本属性      • 名称：非洲之心      • 类型：Collectible（工艺藏品）      • 等级：六级（最高档）      • 描述：世界上最大的钻石，璀璨夺目，象征永恒的爱。      • 重量：0.62      • 尺寸：1×1 单位格      • 可交易：是    ...`
- R2 | query=`它现在什么价格` | tool=`df_market_latest_price` | gate=2 | recall_hits=0 | retries=0
  - output: `非洲之心 的最新价格为 13412037。 更新时间：2026-03-01 03:10:34。`
- R3 | query=`它的历史价格` | tool=`df_market_history_price` | gate=5 | recall_hits=0 | retries=0
  - output: `非洲之心 历史价格查询成功，共 4924 条记录。最新样本价：13412037.0（2026-03-01 11:00:00），最早样本价：12972936.0（2025-11-01 11:30:00），区间最低：12439045.0，区间最高：13458000.0。`
- R4 | query=`它现在建议买吗` | tool=`df_market_price_advice` | gate=4 | recall_hits=0 | retries=0
  - output: `结论   不建议现在买入，应等待回调。  关键依据   1. 当前价 13,412,037 已处历史区间 95.5% 高位，距区间最高仅 45,963 上行空间，而回撤至区间最低需承担 972,992 风险。   2. 工具明确提示“当前处于历史偏高区，谨慎追高，建议等待回调再买”。   3. 卖出手续费按 13% 估算，进一步压缩短线盈利空间。  不确定性   无。`
- R5 | query=`再介绍一下海洋之泪并告诉我现在价格` | tool=`df_answer_composer` | gate=0 | recall_hits=0 | retries=0
  - output: `结论   海洋之泪是六级工艺藏品级可交易收藏品，当前价 31,721,258。  关键依据   1. 基本属性：六级 Collectible，重量 0.05，尺寸 1×1，唯一数量 1。   2. 外观：天然珍珠，通体镭射光泽。   3. 实时价格：31,721,258（2026-03-01 03:11:00）。  不确定性   无。`
- R6 | query=`对比一下这两个物品` | tool=`df_multi_item_compare` | gate=4 | recall_hits=0 | retries=0
  - output: `结论   当前仅非洲之心有完整区间数据，其现价 13,412,037 已处 95.5% 高位，短线性价比不佳；海洋之泪因工具返回 400 错误，暂无区间信息，无法直接对比。  关键依据   1. 非洲之心：区间 12,439,045–13,458,000，现价距区间顶仅 45,963 上行，回撤风险 972,992。   2. 工具提示“当前性价比相对更优的是非洲之心”，但区间位置 95.5% 仍属高位。   3. 卖出统一按 13% ...`
- R7 | query=`分析碳纤维散射箭矢利润稳定性` | tool=`df_profit_stability` | gate=2 | recall_hits=0 | retries=0
  - output: `结论   碳纤维散射箭矢利润稳定性低，不建议批量生产。  关键依据   1. 167 条样本中，正利润仅占 44.3%，平均利润 -5,193，波动系数 14.73。   2. 最大回撤 259.6%，风险极高；最新利润 60,393 但趋势变化 +148.2%，短线波动剧烈。   3. 工具建议“利润波动较大或负利润占比偏高，建议谨慎生产”，并假设卖出手续费 13%。  不确定性   无。`
- R8 | query=`特勤处制造什么子弹利润最高` | tool=`df_place_profit_rank` | gate=0 | recall_hits=0 | retries=0
  - output: `特勤处制造净利润榜（指定分组，Top1，排序维度: totalprofit）  工作台（子弹） 1. 9x39mm BP｜净利润 374,241｜小时利润 46,780.13｜采样时间 2026-03-01 11:00:00`
- R9 | query=`特勤处四大分组利润top3` | tool=`df_place_profit_rank` | gate=0 | recall_hits=0 | retries=0
  - output: `特勤处制造净利润榜（全部分组，Top3，排序维度: totalprofit）  技术中心（枪械/配件） 1. M14射手步枪｜净利润 99,736｜小时利润 3,324.53｜采样时间 2026-03-01 11:00:00 2. SR-3M紧凑突击步枪｜净利润 49,975｜小时利润 1,041.15｜采样时间 2026-03-01 11:00:00 3. AUG突击步枪｜净利润 47,416｜小时利润 1,975.67｜采样时间 2...`

## 6. 跨会话重入
- query: `刚才两个物品里，哪个更适合买入，简短回答`
- tool: `df_multi_item_compare` / tool_query=`非洲之心、海洋之泪 对比`
- gate=`2` recall_hits=`6` retries=`0`
- output: `结论：非洲之心更适合买入。   关键依据：   1. 区间位置95.5%，已接近顶部，但下行风险仅972,992，远小于海洋之泪的9,716,359。   2. 工具结论明确“当前性价比相对更优的是非洲之心”。   3. 海洋之泪现价31,721,258，区间位置65.5%，下行空间更大。    不确定性：非洲之心需等待回调至12.5M附近再分批建仓，当前不建议追高。`

## 7. 故障注入与重试链路
- query: `非洲之心现在什么价格`
- tool: `df_market_latest_price` / tool_query=`objectName=非洲之心`
- retry_count_total=`1` validator_reject=`True` reviewer_reject=`False`
- injected: `1`
- output: `非洲之心 的最新价格为 13412037。 更新时间：2026-03-01 03:13:45。`

## 8. user_id 隔离
- user-A reentry query: `它现在价格是多少`
- user-A output: `非洲之心 的最新价格为 13412037。 更新时间：2026-03-01 03:13:23。`
- user-B reentry query: `它现在价格是多少`
- user-B output: `QBZ95-1突击步枪 的最新价格为 38102。 更新时间：2026-03-01 03:13:36。`
- isolation_ok: `True`

## 9. 持久化计数
- `{'user_id': 'full-user-main', 'session_id': 'full-main-20260301_030710', 'chat_turns': 30, 'memory_summaries': 2, 'memory_facts': 10}`
- `{'user_id': 'user-A', 'session_id': 'full-iso-20260301_030710', 'chat_turns': 6, 'memory_summaries': 0, 'memory_facts': 0}`
- `{'user_id': 'user-B', 'session_id': 'full-iso-20260301_030710', 'chat_turns': 6, 'memory_summaries': 0, 'memory_facts': 0}`

## 10. 断言
- `tool_coverage_complete`: PASS (severity=normal)
- `single_turn_core_quality`: PASS (severity=normal)
- `kb_rag_quality`: PASS (severity=normal)
- `ltm_rag_resolution`: PASS (severity=normal)
- `review_retry_budget_safe`: PASS (severity=normal)
- `multi_turn_reentry_no_compare_failure`: PASS (severity=normal)
- `user_isolation_effective`: PASS (severity=normal)
- `persistent_memory_written`: PASS (severity=normal)
- `fault_injection_retry_chain`: PASS (severity=normal)
- `retry_mechanism_observed`: PASS (severity=warning)