# Self-Improving Agents（Tool-Planning Agentic RL）设计与实现

## 1. 目标与范围

本方案在现有多智能体系统中新增一条“自进化闭环”，先聚焦 **Tool-Planning** 策略层：

- 提升工具首选命中率（Top1）
- 提升工具参数可执行率
- 提升执行成功率
- 降低重试率与超时率

当前版本不直接对 `intent_recognition` 做 RL，避免同时优化多个策略带来的稳定性风险。

---

## 2. 在线闭环（已实现）

### 2.1 新增采集节点

- 文件：`src/agents/self_improving_data_agent.py`
- 图编排接入：`src/agents/graph.py`
- 链路位置：`persistent_memory_write -> self_improving_data -> END`

该节点会将每轮对话转换成可训练样本并写入：

- `data/self_improve/raw_trajectories/tool_planning_trajectory_YYYYMMDD.jsonl`

### 2.2 采集样本结构（核心字段）

- `state`：query、memory_context、intent、flow_type、requires_task_planning、retry_count_used
- `action`：selected_tool、tool_calls、requires_task_planning、plan_source
- `outcome`：tool_results、validation_result、review_result、quality_gate_passed、retry_count_total、stage_timings_ms
- `reward`：`total + components`

---

## 3. 奖励函数设计（已落地，Plan-centric + LLM Judge）

奖励在 `SelfImprovingDataAgent` 中按分层组件计算并写入轨迹，采用混合方案：

`总分 = 规则分权重 * 规则分 + LLM分权重 * LLM评审分`

其中：
- 规则分：保证可解释、可复现、可控（硬约束）。
- LLM评审分：补充主观质量判断（计划合理性、步骤依赖、结果可用性）。

LLM Judge 提示词：
- `src/prompts/self_improve_llm_judge_prompt.txt`

### 3.1 规划层
1. `plan_trigger`：复杂任务该规划时是否触发；简单任务是否过度规划。  
2. `plan_coverage`：计划步骤对目标步骤的覆盖度。  
3. `plan_order`：计划步骤与执行步骤的顺序一致性。  
4. `plan_exec_alignment`：计划工具与实际执行工具的一致性。  
5. `redundancy_penalty`：无效冗余步骤惩罚。  
6. `efficiency`：在成功前提下，步数越短奖励越高。  

### 3.2 执行层
1. `tool_match`：意图与工具语义匹配程度。  
2. `args_ok`：参数完整率。  
3. `exec_success`：执行成功率。  
4. `quality_pass`：质量门通过奖励。  

### 3.3 终局层
1. `terminal`：`success / partial / fail` 三档奖励。  

### 3.4 恢复层
1. `recovery`：失败后重规划并恢复成功给正奖励；盲目重试给负奖励。  

### 3.5 成本惩罚
1. `retry_penalty`：重试次数惩罚。  
2. `budget_exhausted_penalty`：预算耗尽惩罚。  
3. `latency_penalty`：超时惩罚。  

默认权重全部由 `config.py` 控制，可通过环境变量覆盖（`SELF_IMPROVE_*`）。

### 3.6 LLM Judge 评分维度
1. `planning_quality`：规划触发与链路完整性。  
2. `dependency_consistency`：相邻步骤依赖连续性。  
3. `argument_quality`：工具参数是否可执行。  
4. `execution_consistency`：执行结果与规划一致性。  
5. `result_quality`：最终结果对用户目标的对齐程度。  

LLM 输出结构化 JSON，系统将 0~10 分映射到中心化分值，并按配置权重合成 `llm_total`。

---

## 4. 数据构建流程（已实现）

### 4.1 构建 RL 数据集

脚本：
- `data/scripts/self_improve/build_tool_planning_rl_dataset.py`

作用：
1. 读取原始轨迹
2. 过滤低 reward 样本
3. 指纹去重（同类样本保留高 reward）
4. 切分 train/dev/test

默认输出：

- `data/dataset/agentic_rl/tool_planning/train.jsonl`
- `data/dataset/agentic_rl/tool_planning/dev.jsonl`
- `data/dataset/agentic_rl/tool_planning/test.jsonl`
- `data/dataset/agentic_rl/tool_planning/build_summary.json`

### 4.2 轨迹质量统计

脚本：
- `data/scripts/self_improve/summarize_trajectory_rewards.py`

输出：
- `data/self_improve/reports/trajectory_reward_summary.json`

---

## 5. RL 训练流程（已实现）

### 5.1 GRPO 训练入口

- 训练脚本：`training/tool_planning_rl/train_grpo.py`
- 配置文件：`training/tool_planning_rl/configs/grpo_train.json`
- 启动脚本：`training/tool_planning_rl/run_grpo_train.sh`

训练形态：
- Base 模型：Qwen3-8B
- 训练方式：GRPO + LoRA
- 奖励：训练脚本中的 `reward_fn`（结构化动作匹配 + 先验 reward）

### 5.2 Dry-run 验证

将配置中的 `dry_run=true`，可仅验证：
- 数据加载
- prompt 构建
- 奖励函数计算

---

## 6. 独立“自进化数据 Agent”设计说明

是的，当前已经实现一个独立 Agent：

- 名称：`SelfImprovingDataAgent`
- 职责：只做轨迹抽取、奖励计算、样本落盘
- 与业务节点解耦，不改变原有回答语义

这样可以保证：
1. 主链路稳定
2. 训练数据持续积累
3. 训练迭代与线上推理可独立发布

---

## 7. 推荐执行顺序

1. 开启系统跑在线流量，累积轨迹  
2. 使用 Self-Improve Manager 自动做“触发判断 -> 数据构建 -> 训练”  
3. 用固定 benchmark 对比（Base vs SFT vs RL）  

### 7.1 自进化触发策略（当前版本）

Manager 脚本：
- `data/scripts/self_improve/self_improve_manager.py`

默认触发门槛：
- 有效样本数 `>= 50`
- 距上次训练 `>= 2 天`
- 且触发至少一个退化规则（如 tool_match 下滑、quality 下滑、blind_retry 激增）

训练保底：
- 构建后样本数低于 `min_train_records`（默认 200）则不启动训练。

---

## 8. 关键命令

```bash
# 1) 一键跑自进化管理（自动判断是否触发训练）
bash data/scripts/self_improve/run_self_improve_manager.sh

# 2) 直接调用 manager（可加 dry-run）
python data/scripts/self_improve/self_improve_manager.py --llm-filter-enabled --dry-run

# 3) 一键跑完整自进化闭环（已改为 manager）
bash data/scripts/self_improve/run_self_improve_cycle.sh
```
