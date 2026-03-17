#!/usr/bin/env python3
"""Intent SFT training entrypoint (Transformers + PEFT + TRL + Accelerate/DeepSpeed)."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer


@dataclass
class TrainConfig:
    base_model_path: str = "models/Qwen3-8B"
    train_file: str = "data/dataset/final/intent/train.jsonl"
    eval_file: str = "data/dataset/final/intent/dev.jsonl"
    output_dir: str = "outputs/intent_sft/qwen3_8b_lora"
    max_seq_length: int = 1024
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    save_total_limit: int = 3
    seed: int = 42
    bf16: bool = False
    fp16: bool = True
    gradient_checkpointing: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    use_qlora: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True
    deepspeed_config: str = "training/common/configs/deepspeed_zero2.json"
    report_to: List[str] = field(default_factory=lambda: ["wandb"])
    run_name: str = "deltaforceagent-intent-sft"
    logging_dir: str = "outputs/intent_sft/logs"
    wandb_project: str = "deltaforceagent-sft"
    wandb_entity: str = ""
    wandb_tags: List[str] = field(default_factory=lambda: ["intent", "sft"])
    wandb_mode: str = "online"
    system_prompt: str = (
        "你是DeltaAgent的意图识别与工具规划子Agent。"
        "你必须根据用户问题、记忆上下文和可用工具输出严格JSON，不要输出任何额外文本。"
    )


def load_config(path: str) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cfg = TrainConfig()
    for key, value in raw.items():
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown config key: {key}")
        setattr(cfg, key, value)
    return cfg


def normalize_label(label: Dict[str, Any]) -> Dict[str, Any]:
    confidence = label.get("confidence", 0.5)
    try:
        confidence_value = float(confidence)
    except Exception:
        confidence_value = 0.5
    confidence_value = max(0.0, min(1.0, confidence_value))
    return {
        "intent": str(label.get("intent", "general_chat")),
        "entities": [str(x) for x in (label.get("entities") or [])][:5],
        "flow_type": str(label.get("flow_type", "single")),
        "requires_task_planning": bool(label.get("requires_task_planning", False)),
        "selected_tools": [str(x) for x in (label.get("selected_tools") or [])][:4],
        "confidence": confidence_value,
    }


def build_messages(example: Dict[str, Any], system_prompt: str) -> List[Dict[str, str]]:
    query = str(example.get("user_query", "")).strip()
    memory_context = str(example.get("memory_context", "")).strip()
    tools = example.get("available_tools") or []
    tools_text = ", ".join(str(t) for t in tools)

    user_content = (
        "请完成意图识别与工具规划。\n"
        "要求：仅输出JSON对象，字段必须为 intent/entities/flow_type/requires_task_planning/selected_tools/confidence。\n"
        f"用户问题: {query}\n"
        f"记忆上下文: {memory_context if memory_context else '<EMPTY>'}\n"
        f"可用工具: {tools_text}\n"
    )
    target = json.dumps(normalize_label(example.get("label") or {}), ensure_ascii=False)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": target},
    ]


def build_text_dataset(cfg: TrainConfig, tokenizer):
    data_files = {"train": cfg.train_file}
    has_eval = bool(cfg.eval_file and Path(cfg.eval_file).exists())
    if has_eval:
        data_files["eval"] = cfg.eval_file

    dataset = load_dataset("json", data_files=data_files)

    def _fmt(example: Dict[str, Any]) -> Dict[str, str]:
        messages = build_messages(example, cfg.system_prompt)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    train_ds = dataset["train"].map(_fmt, remove_columns=dataset["train"].column_names)
    eval_ds = None
    if has_eval:
        eval_ds = dataset["eval"].map(_fmt, remove_columns=dataset["eval"].column_names)
    return train_ds, eval_ds


def dtype_from_name(name: str):
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(str(name).lower(), torch.float16)


def create_model_and_tokenizer(cfg: TrainConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if cfg.bf16 else torch.float16,
    }

    if cfg.use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=dtype_from_name(cfg.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        )
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(cfg.base_model_path, **model_kwargs)

    if cfg.use_qlora:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


def create_sft_config(cfg: TrainConfig, *, has_eval: bool) -> SFTConfig:
    kwargs: Dict[str, Any] = {
        "output_dir": cfg.output_dir,
        "num_train_epochs": cfg.num_train_epochs,
        "learning_rate": cfg.learning_rate,
        "lr_scheduler_type": cfg.lr_scheduler_type,
        "warmup_ratio": cfg.warmup_ratio,
        "weight_decay": cfg.weight_decay,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "logging_steps": cfg.logging_steps,
        "save_steps": cfg.save_steps,
        "eval_steps": cfg.eval_steps,
        "save_total_limit": cfg.save_total_limit,
        "seed": cfg.seed,
        "bf16": cfg.bf16,
        "fp16": cfg.fp16,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "report_to": cfg.report_to,
        "run_name": cfg.run_name,
        "logging_dir": cfg.logging_dir,
        "dataset_text_field": "text",
    }

    if cfg.deepspeed_config and Path(cfg.deepspeed_config).exists():
        kwargs["deepspeed"] = cfg.deepspeed_config

    sig = inspect.signature(SFTConfig.__init__).parameters
    if "max_seq_length" in sig:
        kwargs["max_seq_length"] = cfg.max_seq_length
    elif "max_length" in sig:
        kwargs["max_length"] = cfg.max_seq_length
    if "evaluation_strategy" in sig:
        kwargs["evaluation_strategy"] = "steps" if has_eval else "no"
    elif "eval_strategy" in sig:
        kwargs["eval_strategy"] = "steps" if has_eval else "no"

    filtered = {k: v for k, v in kwargs.items() if k in sig}
    return SFTConfig(**filtered)


def setup_wandb_env(cfg: TrainConfig):
    targets = [str(x).strip().lower() for x in (cfg.report_to or [])]
    if "wandb" not in targets:
        return
    if cfg.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)
    if cfg.wandb_entity:
        os.environ.setdefault("WANDB_ENTITY", cfg.wandb_entity)
    if cfg.wandb_tags:
        os.environ.setdefault("WANDB_TAGS", ",".join(str(x) for x in cfg.wandb_tags))
    if cfg.wandb_mode:
        os.environ.setdefault("WANDB_MODE", cfg.wandb_mode)
    if cfg.run_name:
        os.environ.setdefault("WANDB_NAME", cfg.run_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", required=True, help="Path to json config")
    args = parser.parse_args()

    cfg = load_config(args.train_config)
    os.makedirs(cfg.output_dir, exist_ok=True)
    if cfg.logging_dir:
        os.makedirs(cfg.logging_dir, exist_ok=True)
    setup_wandb_env(cfg)
    set_seed(cfg.seed)

    model, tokenizer = create_model_and_tokenizer(cfg)
    train_ds, eval_ds = build_text_dataset(cfg, tokenizer)
    sft_args = create_sft_config(cfg, has_eval=eval_ds is not None)

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    train_result = trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    metrics = dict(train_result.metrics)
    if eval_ds is not None:
        eval_metrics = trainer.evaluate()
        metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})

    with open(Path(cfg.output_dir) / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(Path(cfg.output_dir) / "resolved_train_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, ensure_ascii=False, indent=2)

    print("Training complete.")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
