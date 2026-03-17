# Tool Selection SFT（LoRA）

## 目录
- `train.py`：训练脚本
- `eval.py`：离线评估脚本
- `run_train.sh`：双卡启动脚本
- `configs/train.json`：训练配置

## 数据
- 训练集：`data/dataset/sft/tool_selection/train.jsonl`
- 验证集：`data/dataset/sft/tool_selection/dev.jsonl`
- 测试集：`data/dataset/sft/tool_selection/test.jsonl`

## 训练
```bash
cd /path/to/DeltaForce_Agent
bash training/tool_selection_sft/run_train.sh
```

单进程（不走 accelerate）：

```bash
cd /path/to/DeltaForce_Agent
TOOL_SELECTION_SFT_LAUNCH_MODE=single bash training/tool_selection_sft/run_train.sh
```

自定义配置：

```bash
cd /path/to/DeltaForce_Agent
TRAIN_CONFIG=training/tool_selection_sft/configs/train.json bash training/tool_selection_sft/run_train.sh
```

## 输出
- Adapter：`outputs/tool_selection_sft/qwen3_8b_lora`
