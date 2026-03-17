# Intent SFT（LoRA）

## 目录
- `train.py`：训练脚本
- `eval.py`：离线评估脚本
- `run_train.sh`：双卡启动脚本（Accelerate + DeepSpeed ZeRO2）
- `run_eval_test.sh`：测试集评估脚本
- `merge_lora.py`：LoRA 合并脚本
- `configs/train.json`：训练配置

## 数据
- 训练集：`data/dataset/sft/intent/train.jsonl`
- 验证集：`data/dataset/sft/intent/dev.jsonl`
- 测试集：`data/dataset/sft/intent/test.jsonl`

## 训练
```bash
cd /path/to/DeltaForce_Agent
bash training/intent_sft/run_train.sh
```

单进程（不走 accelerate）：

```bash
cd /path/to/DeltaForce_Agent
INTENT_SFT_LAUNCH_MODE=single bash training/intent_sft/run_train.sh
```

自定义配置：

```bash
cd /path/to/DeltaForce_Agent
TRAIN_CONFIG=training/intent_sft/configs/train.json bash training/intent_sft/run_train.sh
```

## 评估
```bash
cd /path/to/DeltaForce_Agent
bash training/intent_sft/run_eval_test.sh
```

## 输出
- Adapter：`outputs/intent_sft/qwen3_8b_lora`
- 评估结果：`outputs/intent_sft/eval`
