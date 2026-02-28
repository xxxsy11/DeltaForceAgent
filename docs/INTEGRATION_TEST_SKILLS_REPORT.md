# Skills 集成测试报告

## 1. 总览
- 报告时间(UTC): `2026-02-27T15:52:10.788284+00:00`
- 主用户: `it-user-0001`
- 主会话ID: `integration-skills-main-20260227_154756`
- 重入会话ID: `integration-skills-main-20260227_154756`
- 断言统计: `PASS=8, FAIL=0, TOTAL=8`

## 2. 主会话明细
### Round 1
- 输入: `介绍一下非洲之心`
- 技能: `knowledge_profile`
- 工具: `rag_knowledge_search`
- 计划来源: `skill_planning`
- 记忆门控: `score=0, gate_triggered=False, hits=0`
- 短期记忆状态: `{'user_id': 'it-user-0001', 'session_id': 'integration-skills-main-20260227_154756', 'recent_raw_turns': 2, 'pending_turns': 0, 'pending_tokens': 0, 'rolling_summary_chars': 0, 'merge_count': 0, 'updated_at_utc': '2026-02-27T15:48:10.193119+00:00'}`
- 输出:
```text
结论  
非洲之心是《三角洲行动》中唯一一枚六级“工艺藏品”类可交易收藏品，也是目前检索到的最高等级钻石藏品。

关键属性  
- 名称：非洲之心  
- 类型：Collectible（工艺藏品）  
- 等级：六级（检索范围内最高）  
- 描述：世界上最大的钻石，璀璨夺目，象征永恒的爱  
- 可交易：是  
- 数量：1  
- 重量：0.62  
- 尺寸：1×1 单位  

关联关系  
- 属于工艺藏品类别（OF_COL_TYPE → 工艺藏品）  
- 被归类为六级藏品（HAS_LEVEL → 六级）
```

### Round 2
- 输入: `它现在什么价格`
- 技能: `market_latest_price`
- 工具: `df_market_latest_price`
- 计划来源: `skill_planning`
- 记忆门控: `score=2, gate_triggered=True, hits=0`
- 短期记忆状态: `{'user_id': 'it-user-0001', 'session_id': 'integration-skills-main-20260227_154756', 'recent_raw_turns': 4, 'pending_turns': 0, 'pending_tokens': 0, 'rolling_summary_chars': 0, 'merge_count': 0, 'updated_at_utc': '2026-02-27T15:48:13.250334+00:00'}`
- 输出:
```text
非洲之心 的最新价格为 12544659。 更新时间：2026-02-27 15:48:13。
```

### Round 3
- 输入: `它的历史价格`
- 技能: `market_history_price`
- 工具: `df_market_history_price`
- 计划来源: `skill_planning`
- 记忆门控: `score=5, gate_triggered=True, hits=0`
- 短期记忆状态: `{'user_id': 'it-user-0001', 'session_id': 'integration-skills-main-20260227_154756', 'recent_raw_turns': 6, 'pending_turns': 0, 'pending_tokens': 0, 'rolling_summary_chars': 0, 'merge_count': 0, 'updated_at_utc': '2026-02-27T15:48:15.924399+00:00'}`
- 输出:
```text
非洲之心 历史价格查询成功，共 4925 条记录。最新样本价：12544659.0（2026-02-27 23:30:00），最早样本价：12738765.0（2025-10-31 00:00:00），区间最低：12439045.0，区间最高：13458000.0。
```

### Round 4
- 输入: `它现在建议买吗`
- 技能: `market_price_advice`
- 工具: `df_market_price_advice`
- 计划来源: `skill_planning`
- 记忆门控: `score=4, gate_triggered=True, hits=0`
- 短期记忆状态: `{'user_id': 'it-user-0001', 'session_id': 'integration-skills-main-20260227_154756', 'recent_raw_turns': 8, 'pending_turns': 0, 'pending_tokens': 0, 'rolling_summary_chars': 0, 'merge_count': 0, 'updated_at_utc': '2026-02-27T15:48:26.562645+00:00'}`
- 输出:
```text
结论  
建议买入：当前价格 12,544,659 处于历史区间 10.4% 低位，可分批布局。

关键依据  
1. 价格位置：12,544,659 位于 12,439,045–13,458,000 区间的 10.4% 低位。  
2. 上行空间：若回到区间最高，理论收益 913,341；下行风险仅 105,614。  
3. 工具明确给出“可考虑分批布局”的买入建议。

不确定性  
卖出统一按 13% 手续费估算净收益。
```

### Round 5
- 输入: `再介绍一下海洋之泪并告诉我现在价格`
- 技能: `answer_composer`
- 工具: `df_answer_composer`
- 计划来源: `skill_planning`
- 记忆门控: `score=0, gate_triggered=False, hits=0`
- 短期记忆状态: `{'user_id': 'it-user-0001', 'session_id': 'integration-skills-main-20260227_154756', 'recent_raw_turns': 10, 'pending_turns': 0, 'pending_tokens': 0, 'rolling_summary_chars': 0, 'merge_count': 0, 'updated_at_utc': '2026-02-27T15:48:47.947444+00:00'}`
- 输出:
```text
结论  
海洋之泪是六级工艺藏品，1×1格天然珍珠，通体镭射光泽；当前价格 30,030,287（2026-02-27 15:48:39）。

关键依据  
1. 属性：六级 Collectible，重量 0.05，描述“浑然天成，珠圆玉润”。  
2. 价格：30,030,287，处于历史区间 22,004,899–36,829,220 的 54.1% 中位。  
3. 策略：建议小仓位分批，勿重仓；卖出视持仓成本而定。

不确定性  
卖出统一按 13% 手续费估算净收益。
```

### Round 6
- 输入: `对比一下这两个物品`
- 技能: `market_multi_item_compare`
- 工具: `df_multi_item_compare`
- 计划来源: `fallback_task_planning`
- 记忆门控: `score=4, gate_triggered=True, hits=0`
- 短期记忆状态: `{'user_id': 'it-user-0001', 'session_id': 'integration-skills-main-20260227_154756', 'recent_raw_turns': 10, 'pending_turns': 2, 'pending_tokens': 134, 'rolling_summary_chars': 0, 'merge_count': 0, 'updated_at_utc': '2026-02-27T15:49:02.192455+00:00'}`
- 输出:
```text
结论  
当前性价比更优的是非洲之心。

关键依据  
1. 价格位置：非洲之心现价 12,544,659 处于历史区间 10.4% 低位，海洋之泪现价 30,030,287 处于 54.1% 中位。  
2. 风险收益：非洲之心上行空间 913,341、下行风险 105,614；海洋之泪上行 6,798,933、下行 8,025,388。  
3. 工具结论：直接判定“当前性价比相对更优的是非洲之心”。

不确定性  
卖出统一按 13% 手续费估算净收益。
```

### Round 7
- 输入: `分析碳纤维散射箭矢利润稳定性`
- 技能: `profit_stability`
- 工具: `df_profit_stability`
- 计划来源: `skill_planning`
- 记忆门控: `score=2, gate_triggered=True, hits=0`
- 短期记忆状态: `{'user_id': 'it-user-0001', 'session_id': 'integration-skills-main-20260227_154756', 'recent_raw_turns': 10, 'pending_turns': 0, 'pending_tokens': 0, 'rolling_summary_chars': 147, 'merge_count': 1, 'updated_at_utc': '2026-02-27T15:49:24.393682+00:00'}`
- 输出:
```text
结论  
碳纤维散射箭矢利润稳定性评级为“低”，当前不建议批量生产。

关键依据  
1. 样本 99 条，正利润占比仅 30.3%，平均利润 -25,836.26，波动系数 CV 3.33。  
2. 最大回撤 259.6%，最新利润 -30,715，连续亏损风险高。  
3. 工具直接给出“建议谨慎生产”。

不确定性  
卖出统一按 13% 手续费估算净收益。
```

### Round 8
- 输入: `特勤处制造什么子弹利润最高`
- 技能: `place_profit_rank`
- 工具: `df_place_profit_rank`
- 计划来源: `skill_planning`
- 记忆门控: `score=0, gate_triggered=False, hits=0`
- 短期记忆状态: `{'user_id': 'it-user-0001', 'session_id': 'integration-skills-main-20260227_154756', 'recent_raw_turns': 10, 'pending_turns': 2, 'pending_tokens': 69, 'rolling_summary_chars': 147, 'merge_count': 1, 'updated_at_utc': '2026-02-27T15:49:27.292434+00:00'}`
- 输出:
```text
特勤处制造净利润榜（指定分组，Top1，排序维度: totalprofit）

工作台（子弹）
1. 9x39mm BP｜净利润 352,446｜小时利润 44,055.75｜采样时间 2026-02-27 23:30:00
```

### Round 9
- 输入: `特勤处四大分组利润top3`
- 技能: `place_profit_rank`
- 工具: `df_place_profit_rank`
- 计划来源: `skill_planning`
- 记忆门控: `score=0, gate_triggered=False, hits=0`
- 短期记忆状态: `{'user_id': 'it-user-0001', 'session_id': 'integration-skills-main-20260227_154756', 'recent_raw_turns': 10, 'pending_turns': 0, 'pending_tokens': 0, 'rolling_summary_chars': 290, 'merge_count': 2, 'updated_at_utc': '2026-02-27T15:49:46.911368+00:00'}`
- 输出:
```text
特勤处制造净利润榜（全部分组，Top3，排序维度: totalprofit）

技术中心（枪械/配件）
1. M14射手步枪｜净利润 80,616｜小时利润 2,687.20｜采样时间 2026-02-27 23:30:00
2. AUG突击步枪｜净利润 61,537｜小时利润 2,564.04｜采样时间 2026-02-27 23:30:00
3. SR-3M紧凑突击步枪｜净利润 59,978｜小时利润 1,249.54｜采样时间 2026-02-27 23:30:00

工作台（子弹）
1. 9x39mm BP｜净利润 352,446｜小时利润 44,055.75｜采样时间 2026-02-27 23:30:00
2. 5.45x39mm BS｜净利润 335,681｜小时利润 41,960.13｜采样时间 2026-02-27 23:30:00
3. 7.62*39mm AP SUB｜净利润 334,194｜小时利润 41,774.25｜采样时间 2026-02-27 23:30:00

制药台（药品/针剂/维修工具包）
1. 精密护甲维修包｜净利润 66,037｜小时利润 8,254.63｜采样时间 2026-02-27 23:30:00
2. 高级头盔维修组合｜净利润 54,991｜小时利润 6,873.88｜采样时间 2026-02-27 23:30:00
3. 战地医疗箱｜净利润 52,544｜小时利润 6,568｜采样时间 2026-02-27 23:30:00

防具台（头盔/护甲/胸挂/背包）
1. 精英防弹背心｜净利润 240,580｜小时利润 30,072.50｜采样时间 2026-02-27 23:30:00
2. 重型突击背心｜净利润 219,688｜小时利润 27,461｜采样时间 2026-02-27 23:30:00
3. FS复合防弹衣｜净利润 213,326｜小时利润 26,665.75｜采样时间 2026-02-27 23:30:00
```

### Round 10
- 输入: `介绍一下非洲之心以及告诉我它现在什么价格`
- 技能: `answer_composer`
- 工具: `df_answer_composer`
- 计划来源: `skill_planning`
- 记忆门控: `score=2, gate_triggered=True, hits=6`
- 短期记忆状态: `{'user_id': 'it-user-0001', 'session_id': 'integration-skills-main-20260227_154756', 'recent_raw_turns': 10, 'pending_turns': 2, 'pending_tokens': 128, 'rolling_summary_chars': 290, 'merge_count': 2, 'updated_at_utc': '2026-02-27T15:50:14.609945+00:00'}`
- 输出:
```text
结论  
非洲之心是《三角洲行动》唯一六级可交易钻石藏品，当前报价 12,544,659（2026-02-27 15:50:04）。

关键依据  
1. 属性：工艺藏品，六级，重量 0.62，尺寸 1×1，仅 1 枚。  
2. 价格：12,544,659 处于历史区间 12,439,045–13,458,000 的 10.4% 低位。  
3. 策略：工具建议低位分批买入，理论上行 913,341，下行风险 105,614，卖出按 13% 手续费估算。

不确定性  
报价可能已变动，需实时确认。
```

## 3. 会话归档与跨会话重入
- 归档前: `{'user_id': 'it-user-0001', 'session_id': 'integration-skills-main-20260227_154756', 'recent_raw_turns': 10, 'pending_turns': 2, 'pending_tokens': 128, 'rolling_summary_chars': 290, 'merge_count': 2, 'updated_at_utc': '2026-02-27T15:50:14.609945+00:00'}`
- 归档后: `{'user_id': 'it-user-0001', 'session_id': 'integration-skills-main-20260227_154756', 'recent_raw_turns': 0, 'pending_turns': 0, 'pending_tokens': 0, 'rolling_summary_chars': 342, 'merge_count': 3, 'updated_at_utc': '2026-02-27T15:50:21.314806+00:00'}`
- 重入问题: `刚才两个物品里，哪个更适合买入，简短回答`
- 重入技能: `market_multi_item_compare`
- 重入工具: `df_multi_item_compare`
- 重入门控: `score=2, gate_triggered=True, hits=6`
```text
结论：非洲之心更适合买入。  
关键依据：  
1. 现价12,544,659处于历史区间10.4%低位，安全边际高。  
2. 上行空间913,341，下行风险仅105,614，盈亏比优。  
3. 对比海洋之泪54.1%中位价，性价比更优。  

不确定性：卖出手续费按13%估算。
```

## 4. user_id 隔离验证
- user-A 预热输入: `介绍一下非洲之心`
- user-A 重入输入: `它现在价格是多少`
- user-A 输出摘要: `非洲之心 的最新价格为 12544659。 更新时间：2026-02-27 15:51:55。`
- user-B 预热输入: `介绍一下QBZ95-1突击步枪`
- user-B 重入输入: `它现在价格是多少`
- user-B 输出摘要: `QBZ95-1突击步枪 的最新价格为 38047。 更新时间：2026-02-27 15:52:10。`

## 5. 持久化记录计数
- `{'user_id': 'it-user-0001', 'session_id': 'integration-skills-main-20260227_154756', 'chat_turns': 33, 'memory_summaries': 3, 'memory_facts': 14}`
- `{'user_id': 'user-A', 'session_id': 'integration-skills-iso-20260227_154756', 'chat_turns': 6, 'memory_summaries': 0, 'memory_facts': 0}`
- `{'user_id': 'user-B', 'session_id': 'integration-skills-iso-20260227_154756', 'chat_turns': 6, 'memory_summaries': 0, 'memory_facts': 0}`

## 6. 断言结果
- `all_tools_covered_in_main_session`: PASS
- `all_skills_covered_in_main_session`: PASS
- `skill_tool_consistency`: PASS
- `main_session_no_failure_answer`: PASS
- `compression_or_finalize_flush_happened`: PASS
- `reentry_compare_resolved_entities`: PASS
- `user_isolation_effective`: PASS
- `persistent_db_records_written`: PASS