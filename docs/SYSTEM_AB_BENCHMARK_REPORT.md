# 全项目 A/B 基准测试报告

## 1. 测试范围
- A 组：skills 开启（AGENT_SKILLS_ENABLED=1）
- B 组：skills 关闭（AGENT_SKILLS_ENABLED=0）
- 覆盖：意图识别、工具选择、实体解析、回答关键词覆盖、多轮代词/对比、user 隔离、持久化写入、时延。

## 2. 核心指标对比
| 指标 | A(skills on) | B(skills off) | Δ(A-B) |
|---|---:|---:|---:|
| intent_accuracy | 1.0 | 1.0 | 0.0 |
| tool_accuracy | 1.0 | 1.0 | 0.0 |
| skill_accuracy | 1.0 | 1.0 | 0.0 |
| entity_resolution_accuracy | 1.0 | 1.0 | 0.0 |
| answer_keyword_coverage | 0.9167 | 0.9167 | 0.0 |
| single_turn_success_rate | 1.0 | 1.0 | 0.0 |
| multi_turn_success_rate | 1.0 | 1.0 | 0.0 |
| multi_turn_keyword_coverage | 0.75 | 0.75 | 0.0 |
| overall_score | 0.98 | 0.98 | 0.0 |
| avg_latency_sec | 8.53 | 7.75 | 0.78 |
| p95_latency_sec | 15.73 | 12.52 | 3.21 |

## 3. 结论
- user 隔离：A=True, B=True
- 综合分：A=0.98，B=0.98，差值=0.0

## 4. 结果文件
- 详细 JSON：`docs/SYSTEM_AB_BENCHMARK_RESULT.json`
- 详细报告：`docs/SYSTEM_AB_BENCHMARK_REPORT.md`
- 简版报告：`docs/SYSTEM_AB_BENCHMARK_REPORT_BRIEF.md`