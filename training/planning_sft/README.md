# Planning SFT（LoRA）

## 目录
- `train.py`：训练脚本
- `run_train.sh`：训练启动脚本（默认分布式，可切单机）
- `configs/train.json`：训练配置

## 数据
- 训练集：`data/dataset/sft/planning/train.jsonl`
- 验证集：`data/dataset/sft/planning/dev.jsonl`
- 测试集：`data/dataset/sft/planning/test.jsonl`

## 训练
分布式（默认）：
```bash
cd /path/to/DeltaForce_Agent
bash training/planning_sft/run_train.sh
```

单进程：
```bash
cd /path/to/DeltaForce_Agent
PLANNING_SFT_LAUNCH_MODE=single bash training/planning_sft/run_train.sh
```

自定义配置：

```bash
cd /path/to/DeltaForce_Agent
TRAIN_CONFIG=training/planning_sft/configs/train.json bash training/planning_sft/run_train.sh
```

## 输出
- Adapter：`outputs/planning_sft/qwen3_8b_lora`
