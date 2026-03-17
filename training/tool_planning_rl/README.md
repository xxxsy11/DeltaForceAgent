# Tool-Planning Agentic RL（GRPO）

本目录用于对 `tool_planning` 策略进行 Agentic RL 训练，目标是提升：
- 工具选择命中率
- 参数可执行率
- 执行成功率
- 降低重试与时延

## 1. 数据来源

先运行在线系统采集轨迹（`src/agents/self_improving_data_agent.py` 会自动写入）：

- 原始轨迹：`data/self_improve/raw_trajectories/tool_planning_trajectory_*.jsonl`

再构建 RL 数据集：

```bash
python data/scripts/self_improve/build_tool_planning_rl_dataset.py
```

输出目录：
- `data/dataset/agentic_rl/tool_planning/train.jsonl`
- `data/dataset/agentic_rl/tool_planning/dev.jsonl`
- `data/dataset/agentic_rl/tool_planning/test.jsonl`

## 2. 训练（GRPO + LoRA）

```bash
bash training/tool_planning_rl/run_grpo_train.sh
```

默认配置文件：
- `training/tool_planning_rl/configs/grpo_train.json`

注意：
- 当前仓库内默认只保留 `grpo_train.json` 这一个 GRPO 训练配置。
- 默认训练集已切到 `data/dataset/agentic_rl/tool_planning_synth_1000` 这份优化后的合成数据。
- 配置里启用了 DeepSpeed Zero-3，请通过 `run_grpo_train.sh`、`deepspeed` 或 `torchrun` 启动，不要直接执行 `python train_grpo.py`。

可通过环境变量切换断点续训：

```bash
RESUME_FROM_CHECKPOINT=auto \
bash training/tool_planning_rl/run_grpo_train.sh
```

奖励函数说明：
- 训练时按“规划触发、计划覆盖、步骤顺序、计划执行一致性、终局成功、恢复能力、成本惩罚”综合打分，并叠加在线先验 reward。

## 3. 快速检查（不启动训练）

将配置中的 `"dry_run": true`，然后执行：

```bash
bash training/tool_planning_rl/run_grpo_train.sh
```

或者在确认不使用 DeepSpeed 配置时：

```bash
python training/tool_planning_rl/train_grpo.py \
  --train_config training/tool_planning_rl/configs/grpo_train.json
```

此模式只验证数据加载与奖励函数逻辑。
