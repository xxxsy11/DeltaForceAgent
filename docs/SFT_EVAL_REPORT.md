# SFT 三模块离线评测报告
- 对比顺序: `kimi-k2-0711 -> Base Qwen3-8B -> Qwen3-8B LoRA`

> 说明：该评测是“离线结构化输出”任务，不是端到端系统回答质量评测。

## 意图识别(Intent)
| 指标 | 指标含义 | Kimi | Base | LoRA | \| | Δ(LoRA-Kimi) | Δ(LoRA-Base) |
|---|---|---:|---:|---:|---:|---:|---:|
| `json_parse_rate` ↑ | JSON 解析成功率，衡量结构化输出稳定性 | 0.9800 | 0.0000 | **1.0000** | \| | +0.0200 | +1.0000 |
| `intent_accuracy` ↑ | 意图分类是否与标注一致 | 0.0200 | 0.0000 | **1.0000** | \| | +0.9800 | +1.0000 |
| `flow_type_accuracy` ↑ | 单工具/复杂流程类型判断准确率 | 0.0000 | 0.0000 | **0.9800** | \| | +0.9800 | +0.9800 |
| `planning_flag_accuracy` ↑ | 是否需要规划的判断准确率 | 0.6200 | 0.3400 | **0.6800** | \| | +0.0600 | +0.3400 |
| `tool_exact_match` ↑ | 工具集合与标注完全一致的比例 | 0.0200 | 0.0000 | **1.0000** | \| | +0.9800 | +1.0000 |

## 工具选择(Tool Selection)
| 指标 | 指标含义 | Kimi | Base | LoRA | \| | Δ(LoRA-Kimi) | Δ(LoRA-Base) |
|---|---|---:|---:|---:|---:|---:|---:|
| `json_parse_rate` ↑ | JSON 解析成功率，衡量结构化输出稳定性 | **1.0000** | 0.0000 | **1.0000** | \| | +0.0000 | +1.0000 |
| `planning_flag_accuracy` ↑ | 是否需要规划的判断准确率 | 0.5800 | 0.4000 | **1.0000** | \| | +0.4200 | +0.6000 |
| `tool_exact_match` ↑ | 工具集合与标注完全一致的比例 | 0.2600 | 0.0000 | **1.0000** | \| | +0.7400 | +1.0000 |
| `tool_f1` ↑ | 工具选择的综合精确率/召回率指标 | 0.6179 | 0.0000 | **1.0000** | \| | +0.3821 | +1.0000 |
| `tool_query_coverage` ↑ | 目标工具查询参数覆盖率 | 0.0000 | 0.0000 | **1.0000** | \| | +1.0000 | +1.0000 |
| `tool_query_exact_match` ↑ | 工具查询参数完全匹配率 | 0.0000 | 0.0000 | **1.0000** | \| | +1.0000 | +1.0000 |
| `confidence_mae` ↓ | 置信度误差（平均绝对误差，越低越好） | 0.0561 | 0.3771 | **0.0421** | \| | -0.0140 | -0.3350 |

## 复杂任务规划(Planning)
| 指标 | 指标含义 | Kimi | Base | LoRA | \| | Δ(LoRA-Kimi) | Δ(LoRA-Base) |
|---|---|---:|---:|---:|---:|---:|---:|
| `json_parse_rate` ↑ | JSON 解析成功率，衡量结构化输出稳定性 | 0.6200 | 0.0000 | **0.9800** | \| | +0.3600 | +0.9800 |
| `intent_accuracy` ↑ | 意图分类是否与标注一致 | 0.0000 | 0.0000 | **0.9800** | \| | +0.9800 | +0.9800 |
| `planning_flag_accuracy` ↑ | 是否需要规划的判断准确率 | 0.6200 | 0.0000 | **0.9800** | \| | +0.3600 | +0.9800 |
| `plan_chain_exact_match` ↑ | 规划工具链顺序完全匹配率 | 0.0000 | 0.0000 | **0.8200** | \| | +0.8200 | +0.8200 |
| `plan_set_match` ↑ | 规划工具集合匹配率（忽略顺序） | 0.0000 | 0.0000 | **0.8200** | \| | +0.8200 | +0.8200 |
| `plan_query_coverage` ↑ | 规划步骤查询参数覆盖率 | 0.4800 | 0.0000 | **0.9200** | \| | +0.4400 | +0.9200 |
| `confidence_mae` ↓ | 置信度误差（平均绝对误差，越低越好） | 0.1657 | 0.3396 | **0.0530** | \| | -0.1127 | -0.2865 |

## Tool-Planning GRPO 补充评测（当前最终版本）
- 训练数据：`data/dataset/agentic_rl/tool_planning_synth_1000/train.jsonl`
- 评测数据：`data/dataset/agentic_rl/tool_planning_synth_1000/test.jsonl`
- 样本数：100
- 训练脚本：`training/tool_planning_rl/train_grpo.py`
- 评测脚本：`training/tool_planning_rl/eval.py`
- 训练配置：`training/tool_planning_rl/configs/grpo_train.json`
- 产物目录：`outputs/tool_planning_rl/qwen3_8b_grpo_lora_synth_optimized_zero3`
- 说明：该版本为当前保留的最终 GRPO LoRA 结果，已完成完整训练与离线评测。

| 指标 | 指标含义 | Base Qwen3-8B | GRPO LoRA |
|---|---|---:|---:|
| `json_parse_rate` ↑ | JSON 解析成功率，衡量结构化输出稳定性 | 0.1300 | 0.1300 |
| `selected_tool_accuracy` ↑ | 选择的主工具是否与标注一致 | 0.0000 | 0.0000 |
| `planning_flag_accuracy` ↑ | 是否需要规划的判断准确率 | 0.7300 | 0.7300 |
| `plan_chain_exact_match` ↑ | 规划工具链顺序完全匹配率 | 0.0000 | 0.0000 |
| `plan_set_match` ↑ | 规划工具集合匹配率（忽略顺序） | 0.0000 | 0.0000 |
| `tool_query_coverage` ↑ | 规划步骤查询参数覆盖率 | 0.0000 | 0.0000 |
| `underplan_rate` ↓ | 规划不足比例（越低越好） | 1.0000 | 0.9800 |
| `overplan_rate` ↓ | 过度规划比例（越低越好） | 0.0000 | 0.0000 |
| `avg_pred_steps` | 平均预测步骤数 | 0.0000 | 0.0200 |
| `avg_gold_steps` | 标注平均步骤数 | 1.2700 | 1.2700 |

### 本轮训练指标
- 指标文件：`outputs/tool_planning_rl/qwen3_8b_grpo_lora_synth_optimized_zero3/train_metrics.json`

| 指标 | 数值 |
|---|---:|
| `train_runtime` | 124814.1461 |
| `train_loss` | 0.0077 |
| `eval_eval_loss` | 0.0002 |

### 结果解读
- 本轮 GRPO 训练已跑通，最终 LoRA 权重与完整评测结果已产出。
- 但从当前离线指标看，GRPO 尚未带来核心规划指标的实质提升；主要改善仅体现在 `underplan_rate` 略有下降。
- 现阶段的主要瓶颈不是训练链路未打通，而是输出 schema 约束、动作空间闭集和奖励信号对“正确工具链”学习的牵引仍然不足。
- 因此该结果更适合作为 Agentic RL 工程闭环的第一版验证，而不是最终有效策略模型。