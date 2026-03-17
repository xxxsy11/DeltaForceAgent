#!/usr/bin/env python3
"""Tool-Planning Agentic RL（GRPO）训练入口。"""

from __future__ import annotations

import argparse
from collections import Counter
import inspect
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

try:
    from trl import GRPOConfig, GRPOTrainer
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"当前环境未安装 GRPO 所需 trl 版本: {exc}")


@dataclass
class TrainConfig:
    base_model_path: str = "models/Qwen3-8B"
    train_file: str = "data/dataset/agentic_rl/tool_planning/train.jsonl"
    eval_file: str = "data/dataset/agentic_rl/tool_planning/dev.jsonl"
    output_dir: str = "outputs/tool_planning_rl/qwen3_8b_grpo_lora"
    max_prompt_length: int = 1024
    max_completion_length: int = 256
    num_train_epochs: int = 1
    max_steps: int = -1
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_generations: int = 4
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    seed: int = 42
    bf16: bool = False
    fp16: bool = True
    gradient_checkpointing: bool = True
    deepspeed: str = ""
    beta: float = 0.04
    use_vllm: bool = False
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.85
    generation_batch_size: int = 1
    ds3_gather_for_generation: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
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
    report_to: List[str] = field(default_factory=lambda: ["wandb"])
    run_name: str = "deltaforceagent-tool-planning-grpo"
    logging_dir: str = "outputs/tool_planning_rl/logs"
    wandb_project: str = "deltaforceagent-rl"
    wandb_entity: str = ""
    wandb_tags: List[str] = field(default_factory=lambda: ["tool-planning", "agentic-rl", "grpo", "qwen3-8b"])
    wandb_mode: str = "online"
    reward_json_parse: float = 0.2
    reward_plan_trigger_correct: float = 1.5
    penalty_overplan: float = 0.8
    penalty_underplan: float = 1.8
    reward_plan_coverage: float = 2.0
    reward_step_order: float = 1.5
    reward_dependency_consistency: float = 1.5
    reward_plan_exec_alignment: float = 1.5
    penalty_redundancy: float = 1.2
    reward_tool_match: float = 1.0
    reward_args_valid: float = 1.0
    reward_terminal_success: float = 4.0
    reward_terminal_partial: float = 1.5
    penalty_terminal_fail: float = 2.0
    reward_recovery_success: float = 1.2
    penalty_blind_retry: float = 1.2
    reward_prior_scale: float = 0.3
    penalty_invalid_tool_name: float = 2.0
    penalty_mapped_tool_name: float = 0.8
    reward_all_tools_valid: float = 0.8
    reward_selected_in_calls: float = 0.3
    dry_run: bool = False


AVAILABLE_TOOLS: List[str] = [
    "rag_knowledge_search",
    "df_market_latest_price",
    "df_market_history_price",
    "df_market_price_advice",
    "df_place_profit_rank",
    "df_multi_item_compare",
    "df_profit_stability",
    "df_answer_composer",
]

SYSTEM_PROMPT = (
    "你是 DeltaAgent 的 Tool-Planning 策略模型。"
    "你必须只输出 JSON 对象，字段必须包含 selected_tool/tool_calls/requires_task_planning。"
    "tool_calls 是数组，每个元素必须包含 tool_name/tool_query。"
    f"可用工具严格限定为: {', '.join(AVAILABLE_TOOLS)}。"
    "禁止输出任何未注册工具名。"
)

REWARD_WEIGHTS: Dict[str, float] = {
    "json_parse": 0.2,
    "plan_trigger_correct": 1.5,
    "penalty_overplan": 0.8,
    "penalty_underplan": 1.8,
    "plan_coverage": 2.0,
    "step_order": 1.5,
    "dependency_consistency": 1.5,
    "plan_exec_alignment": 1.5,
    "penalty_redundancy": 1.2,
    "tool_match": 1.0,
    "args_valid": 1.0,
    "terminal_success": 4.0,
    "terminal_partial": 1.5,
    "penalty_terminal_fail": 2.0,
    "recovery_success": 1.2,
    "penalty_blind_retry": 1.2,
    "prior_scale": 0.3,
    "penalty_invalid_tool_name": 2.0,
    "penalty_mapped_tool_name": 0.8,
    "reward_all_tools_valid": 0.8,
    "reward_selected_in_calls": 0.3,
}


def load_config(path: str) -> TrainConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg = TrainConfig()
    for k, v in raw.items():
        if not hasattr(cfg, k):
            raise ValueError(f"Unknown config key: {k}")
        setattr(cfg, k, v)
    return cfg


def _safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    payload = str(text or "").strip()
    if not payload:
        return None
    try:
        obj = json.loads(payload)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = payload.find("{")
    end = payload.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(payload[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


def _extract_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        if "content" in completion:
            return str(completion.get("content", "") or "")
        if "text" in completion:
            return str(completion.get("text", "") or "")
    if isinstance(completion, list):
        chunks: List[str] = []
        for item in completion:
            if isinstance(item, dict):
                chunks.append(str(item.get("content", "") or item.get("text", "") or ""))
            else:
                chunks.append(str(item))
        return "".join(chunks)
    return str(completion or "")


def _format_prompt(prompt_obj: Dict[str, Any]) -> str:
    user_query = str(prompt_obj.get("user_query", "") or "")
    memory_context = str(prompt_obj.get("memory_context", "") or "")
    intent = str(prompt_obj.get("intent", "") or "")
    flow_type = str(prompt_obj.get("flow_type", "") or "")
    requires_planning = bool(prompt_obj.get("requires_task_planning", False))
    retry_count = int(prompt_obj.get("retry_count_used", 0) or 0)
    available_tools = prompt_obj.get("available_tools", AVAILABLE_TOOLS)
    if not isinstance(available_tools, list) or not available_tools:
        available_tools = AVAILABLE_TOOLS
    tools_text = ", ".join(str(x) for x in available_tools)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "请根据以下上下文输出策略 JSON：\n"
        f"- user_query: {user_query}\n"
        f"- memory_context: {memory_context}\n"
        f"- intent: {intent}\n"
        f"- flow_type: {flow_type}\n"
        f"- requires_task_planning_hint: {requires_planning}\n"
        f"- retry_count_used: {retry_count}\n"
        f"- available_tools: {tools_text}\n"
        "硬约束：selected_tool 与 tool_calls[*].tool_name 必须来自 available_tools；"
        "不能输出别名、泛化词（如 search_internet/tool_1 等）。\n"
    )


def build_dataset(cfg: TrainConfig):
    data_files = {"train": cfg.train_file}
    has_eval = bool(cfg.eval_file and Path(cfg.eval_file).exists())
    if has_eval:
        data_files["eval"] = cfg.eval_file
    ds = load_dataset("json", data_files=data_files)

    def _map_row(row: Dict[str, Any]) -> Dict[str, Any]:
        prompt_obj = row.get("prompt", {}) or {}
        response_obj = row.get("response", {}) or {}
        reward_obj = row.get("reward", {}) or {}
        outcome_obj = row.get("outcome", {}) or {}
        prompt_text = _format_prompt(prompt_obj)
        return {
            "prompt": prompt_text,
            "available_tools": [str(x) for x in (prompt_obj.get("available_tools") or AVAILABLE_TOOLS)],
            "should_plan": bool(outcome_obj.get("should_plan", False)),
            "expected_steps": int(outcome_obj.get("expected_steps", 1) or 1),
            "retry_count_total": int(outcome_obj.get("retry_count_total", 0) or 0),
            "terminal_status": str(outcome_obj.get("terminal_status", "") or "fail"),
            "target_selected_tool": str(response_obj.get("selected_tool", "") or ""),
            "target_requires_task_planning": bool(response_obj.get("requires_task_planning", False)),
            "target_tool_calls": response_obj.get("tool_calls", []) or [],
            "reward_prior": float(reward_obj.get("total", 0.0) or 0.0),
        }

    train_ds = ds["train"].map(_map_row, remove_columns=ds["train"].column_names)
    eval_ds = ds["eval"].map(_map_row, remove_columns=ds["eval"].column_names) if has_eval else None
    return train_ds, eval_ds


def _extract_tool_names(calls: Any) -> List[str]:
    if not isinstance(calls, list):
        return []
    out: List[str] = []
    for item in calls:
        if not isinstance(item, dict):
            continue
        name = str(item.get("tool_name", "") or "").strip()
        if name:
            out.append(name)
    return out


def _canonicalize_tool_name(name: str, query: str, available_tools: List[str]) -> tuple[str, bool]:
    raw = str(name or "").strip()
    if not raw:
        return "", False
    if raw in available_tools:
        return raw, False

    text = f"{raw} {query}".lower()
    mapped = ""
    if ("对比" in raw) or ("compare" in text):
        mapped = "df_multi_item_compare"
    elif ("稳定" in raw) or ("回撤" in text) or ("波动" in text) or ("stability" in text):
        mapped = "df_profit_stability"
    elif ("利润" in raw) or ("制造" in text) or ("top" in text) or ("工作台" in text) or ("特勤处" in text):
        mapped = "df_place_profit_rank"
    elif ("建议" in raw) or ("买卖" in text) or ("贵了" in text) or ("便宜" in text) or ("advice" in text):
        mapped = "df_market_price_advice"
    elif ("历史" in raw) or ("走势" in text) or ("区间" in text) or ("history" in text):
        mapped = "df_market_history_price"
    elif ("综合" in raw) or ("compose" in text) or ("answer" in text):
        mapped = "df_answer_composer"
    elif ("价格" in raw) or ("price" in text) or ("current" in text) or ("最新" in text) or ("现在" in text):
        mapped = "df_market_latest_price"
    elif ("知识" in raw) or ("search" in text) or ("介绍" in text) or ("是什么" in text):
        mapped = "rag_knowledge_search"

    if mapped and mapped in available_tools:
        return mapped, True
    return "", False


def _normalize_predicted_action(parsed: Dict[str, Any], available_tools: List[str]) -> tuple[Dict[str, Any], int, int]:
    pred_requires = bool(parsed.get("requires_task_planning", False))
    raw_selected = str(parsed.get("selected_tool", "") or "").strip()
    calls_raw = parsed.get("tool_calls", []) or []
    if not isinstance(calls_raw, list):
        calls_raw = []

    normalized_calls: List[Dict[str, str]] = []
    invalid_count = 0
    mapped_count = 0
    for item in calls_raw:
        if not isinstance(item, dict):
            invalid_count += 1
            continue
        raw_name = str(item.get("tool_name", "") or "").strip()
        raw_query = str(item.get("tool_query", "") or "").strip()
        canon_name, mapped = _canonicalize_tool_name(raw_name, raw_query, available_tools)
        if not canon_name:
            invalid_count += 1
            continue
        if mapped:
            mapped_count += 1
        normalized_calls.append({"tool_name": canon_name, "tool_query": raw_query})

    selected_query = str(normalized_calls[0].get("tool_query", "") or "") if normalized_calls else ""
    canon_selected, selected_mapped = _canonicalize_tool_name(raw_selected, selected_query, available_tools)
    if selected_mapped:
        mapped_count += 1
    if not canon_selected and normalized_calls:
        canon_selected = normalized_calls[0]["tool_name"]
    if raw_selected and not canon_selected:
        invalid_count += 1

    normalized = {
        "selected_tool": canon_selected,
        "requires_task_planning": pred_requires,
        "tool_calls": normalized_calls,
    }
    return normalized, invalid_count, mapped_count


def _valid_call_ratio(calls: Any) -> float:
    if not isinstance(calls, list) or not calls:
        return 0.0
    valid = 0
    for item in calls:
        if not isinstance(item, dict):
            continue
        name = str(item.get("tool_name", "") or "").strip()
        query = str(item.get("tool_query", "") or "").strip()
        if name and query:
            valid += 1
    return valid / max(1, len(calls))


def _order_ratio(pred_names: List[str], gold_names: List[str]) -> float:
    if not gold_names:
        return 0.0
    n = min(len(pred_names), len(gold_names))
    if n <= 0:
        return 0.0
    hit = 0
    for idx in range(n):
        if pred_names[idx] == gold_names[idx]:
            hit += 1
    return hit / max(1, len(gold_names))


def _alignment_ratio(pred_names: List[str], gold_names: List[str]) -> float:
    if not gold_names:
        return 0.0
    pred_set = set(pred_names)
    overlap = 0
    for name in gold_names:
        if name in pred_set:
            overlap += 1
    return overlap / max(1, len(gold_names))


def _clean_chars(text: str) -> List[str]:
    out: List[str] = []
    for ch in str(text or ""):
        if ch.isalnum() or ("\u4e00" <= ch <= "\u9fff"):
            out.append(ch)
    return out


def _dependency_consistency_proxy(calls: Any) -> float:
    if not isinstance(calls, list) or len(calls) <= 1:
        return 1.0
    overlaps: List[float] = []
    for i in range(len(calls) - 1):
        left = calls[i] if isinstance(calls[i], dict) else {}
        right = calls[i + 1] if isinstance(calls[i + 1], dict) else {}
        q1 = set(_clean_chars(str(left.get("tool_query", "") or "")))
        q2 = set(_clean_chars(str(right.get("tool_query", "") or "")))
        if not q1 or not q2:
            overlaps.append(0.0)
            continue
        jaccard = len(q1 & q2) / max(1, len(q1 | q2))
        overlaps.append(jaccard)
    if not overlaps:
        return 0.0
    return sum(overlaps) / len(overlaps)


def reward_fn(
    completions: List[Any],
    available_tools: List[List[str]],
    should_plan: List[bool],
    expected_steps: List[int],
    retry_count_total: List[int],
    terminal_status: List[str],
    target_selected_tool: List[str],
    target_requires_task_planning: List[bool],
    target_tool_calls: List[List[Dict[str, Any]]],
    reward_prior: List[float],
    **_: Any,
) -> List[float]:
    scores: List[float] = []
    for idx, completion in enumerate(completions):
        text = _extract_completion_text(completion)
        parsed = _safe_json_parse(text)

        score = 0.0
        if parsed is None:
            scores.append(-2.0)
            continue
        score += REWARD_WEIGHTS["json_parse"]  # JSON 可解析

        tools = available_tools[idx] if idx < len(available_tools) else AVAILABLE_TOOLS
        tools = [str(x) for x in tools if str(x).strip()]
        if not tools:
            tools = AVAILABLE_TOOLS
        normalized_pred, invalid_count, mapped_count = _normalize_predicted_action(parsed=parsed, available_tools=tools)

        pred_requires = bool(normalized_pred.get("requires_task_planning", False))
        should_plan_flag = bool(should_plan[idx] if idx < len(should_plan) else False)
        if should_plan_flag and pred_requires:
            score += REWARD_WEIGHTS["plan_trigger_correct"]
        elif should_plan_flag and not pred_requires:
            score -= REWARD_WEIGHTS["penalty_underplan"]
        elif (not should_plan_flag) and pred_requires:
            score -= REWARD_WEIGHTS["penalty_overplan"]
        else:
            score += REWARD_WEIGHTS["plan_trigger_correct"] * 0.5

        pred_tool = str(normalized_pred.get("selected_tool", "") or "").strip()
        gold_tool = str((target_selected_tool[idx] if idx < len(target_selected_tool) else "") or "").strip()
        if pred_tool and pred_tool == gold_tool:
            score += REWARD_WEIGHTS["tool_match"]
        else:
            score -= 0.8

        gold_requires = bool(target_requires_task_planning[idx] if idx < len(target_requires_task_planning) else False)
        if pred_requires == gold_requires:
            score += 0.5
        else:
            score -= 0.3

        pred_calls = normalized_pred.get("tool_calls", []) or []
        if isinstance(pred_calls, list) and pred_calls:
            ratio = _valid_call_ratio(pred_calls)
            score += REWARD_WEIGHTS["args_valid"] * ratio
            if pred_tool and any(str((x or {}).get("tool_name", "") or "").strip() == pred_tool for x in pred_calls if isinstance(x, dict)):
                score += REWARD_WEIGHTS["reward_selected_in_calls"]
        else:
            score -= 0.6

        if invalid_count > 0:
            score -= REWARD_WEIGHTS["penalty_invalid_tool_name"] * min(invalid_count, 3)
        else:
            score += REWARD_WEIGHTS["reward_all_tools_valid"]
        if mapped_count > 0:
            score -= REWARD_WEIGHTS["penalty_mapped_tool_name"] * min(mapped_count, 2)

        gold_calls = target_tool_calls[idx] if idx < len(target_tool_calls) else []
        gold_names = _extract_tool_names(gold_calls)
        pred_names = _extract_tool_names(pred_calls)
        expected_n = int(expected_steps[idx] if idx < len(expected_steps) else max(1, len(gold_names)))
        coverage = min(1.0, (len(pred_names) / max(1, expected_n)))
        score += REWARD_WEIGHTS["plan_coverage"] * coverage
        score += REWARD_WEIGHTS["step_order"] * _order_ratio(pred_names=pred_names, gold_names=gold_names)
        score += REWARD_WEIGHTS["plan_exec_alignment"] * _alignment_ratio(pred_names=pred_names, gold_names=gold_names)
        score += REWARD_WEIGHTS["dependency_consistency"] * _dependency_consistency_proxy(pred_calls)

        redundancy = max(0, len(pred_names) - max(1, expected_n))
        score -= REWARD_WEIGHTS["penalty_redundancy"] * redundancy

        terminal = str(terminal_status[idx] if idx < len(terminal_status) else "fail").strip().lower()
        if terminal == "success":
            score += REWARD_WEIGHTS["terminal_success"]
        elif terminal == "partial":
            score += REWARD_WEIGHTS["terminal_partial"]
        else:
            score -= REWARD_WEIGHTS["penalty_terminal_fail"]

        retry_n = int(retry_count_total[idx] if idx < len(retry_count_total) else 0)
        if retry_n > 0 and terminal == "success":
            score += REWARD_WEIGHTS["recovery_success"]
        elif retry_n > 0 and terminal != "success":
            score -= REWARD_WEIGHTS["penalty_blind_retry"] * min(retry_n, 3)

        prior = float(reward_prior[idx] if idx < len(reward_prior) else 0.0)
        score += REWARD_WEIGHTS["prior_scale"] * math.tanh(prior / 3.0)
        scores.append(round(score, 6))
    return scores


def setup_wandb_env(cfg: TrainConfig) -> None:
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


def create_grpo_args(cfg: TrainConfig, *, has_eval: bool) -> GRPOConfig:
    gradient_checkpointing = bool(cfg.gradient_checkpointing)
    ds_path = str(cfg.deepspeed or "").strip()
    is_zero3 = "zero3" in ds_path.lower()
    if is_zero3 and gradient_checkpointing:
        gradient_checkpointing = False
    kwargs: Dict[str, Any] = {
        "output_dir": cfg.output_dir,
        "num_train_epochs": cfg.num_train_epochs,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "warmup_ratio": cfg.warmup_ratio,
        "lr_scheduler_type": cfg.lr_scheduler_type,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "num_generations": cfg.num_generations,
        "logging_steps": cfg.logging_steps,
        "save_steps": cfg.save_steps,
        "eval_steps": cfg.eval_steps,
        "save_total_limit": cfg.save_total_limit,
        "seed": cfg.seed,
        "bf16": cfg.bf16,
        "fp16": cfg.fp16,
        "gradient_checkpointing": gradient_checkpointing,
        "max_prompt_length": cfg.max_prompt_length,
        "max_completion_length": cfg.max_completion_length,
        "beta": cfg.beta,
        "use_vllm": cfg.use_vllm,
        "vllm_tensor_parallel_size": cfg.vllm_tensor_parallel_size,
        "vllm_gpu_memory_utilization": cfg.vllm_gpu_memory_utilization,
        "generation_batch_size": cfg.generation_batch_size,
        "ds3_gather_for_generation": cfg.ds3_gather_for_generation,
        "report_to": cfg.report_to,
        "run_name": cfg.run_name,
        "logging_dir": cfg.logging_dir,
    }
    if ds_path:
        kwargs["deepspeed"] = ds_path
        # Zero-3 must not pass `device_map`, while Zero-2 benefits from pinning to one visible rank
        # to avoid `device_map='auto'` distributed checks in accelerate.
        if is_zero3:
            kwargs["model_init_kwargs"] = {"device_map": None}
        else:
            kwargs["model_init_kwargs"] = {"device_map": {"": 0}}
    if int(cfg.max_steps) > 0:
        kwargs["max_steps"] = int(cfg.max_steps)
    sig = inspect.signature(GRPOConfig.__init__).parameters
    if gradient_checkpointing and "gradient_checkpointing_kwargs" in sig:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if "evaluation_strategy" in sig:
        kwargs["evaluation_strategy"] = "steps" if has_eval else "no"
    elif "eval_strategy" in sig:
        kwargs["eval_strategy"] = "steps" if has_eval else "no"
    filtered = {k: v for k, v in kwargs.items() if k in sig}
    return GRPOConfig(**filtered)


def _load_model_for_deepspeed(cfg: TrainConfig):
    """Load model instance directly to avoid `device_map` conflicts with some trl/transformers combos."""
    model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    try:
        import torch

        if cfg.bf16:
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif cfg.fp16:
            model_kwargs["torch_dtype"] = torch.float16
    except Exception:
        pass
    return AutoModelForCausalLM.from_pretrained(cfg.base_model_path, **model_kwargs)


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    base = Path(output_dir)
    if not base.exists():
        return None
    latest_step = -1
    latest_path: Optional[Path] = None
    for item in base.glob("checkpoint-*"):
        if not item.is_dir():
            continue
        match = re.match(r"checkpoint-(\d+)$", item.name)
        if not match:
            continue
        step = int(match.group(1))
        if step > latest_step:
            latest_step = step
            latest_path = item
    return str(latest_path) if latest_path else None


def _detect_world_size(cli_local_rank: int) -> int:
    env_world = str(os.environ.get("WORLD_SIZE", "") or "").strip()
    if env_world.isdigit():
        return max(1, int(env_world))
    env_local = str(os.environ.get("LOCAL_RANK", "") or "").strip()
    if env_local.lstrip("-").isdigit() and int(env_local) >= 0:
        return 1
    if cli_local_rank >= 0:
        return 1
    return 1


def _ensure_valid_launch(cfg: TrainConfig, cli_local_rank: int) -> None:
    if not str(cfg.deepspeed or "").strip():
        return
    env_local = str(os.environ.get("LOCAL_RANK", "") or "").strip()
    env_world = str(os.environ.get("WORLD_SIZE", "") or "").strip()
    launched = cli_local_rank >= 0 or (env_local.lstrip("-").isdigit() and int(env_local) >= 0) or (
        env_world.isdigit() and int(env_world) > 1
    )
    if not launched:
        raise SystemExit(
            "检测到 deepspeed 配置。请使用 `bash training/tool_planning_rl/run_grpo_train.sh` "
            "或 `deepspeed/torchrun` 启动，而不是直接 `python train_grpo.py`。"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", required=True, help="Path to GRPO train config JSON")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed launchers (deepspeed/torchrun).")
    parser.add_argument(
        "--resume_from_checkpoint",
        default="auto",
        help="auto|none|<checkpoint_path>; default auto resumes from latest checkpoint under output_dir.",
    )
    args = parser.parse_args()

    cfg = load_config(args.train_config)
    _ensure_valid_launch(cfg, args.local_rank)
    if str(cfg.deepspeed or "").strip() and "zero3" in str(cfg.deepspeed).lower() and cfg.gradient_checkpointing:
        print(
            "[train][warn] DeepSpeed Zero-3 with gradient_checkpointing is unstable in the current stack. "
            "Force disabling gradient_checkpointing for this run."
        )
        cfg.gradient_checkpointing = False
    REWARD_WEIGHTS.update(
        {
            "json_parse": float(cfg.reward_json_parse),
            "plan_trigger_correct": float(cfg.reward_plan_trigger_correct),
            "penalty_overplan": float(cfg.penalty_overplan),
            "penalty_underplan": float(cfg.penalty_underplan),
            "plan_coverage": float(cfg.reward_plan_coverage),
            "step_order": float(cfg.reward_step_order),
            "dependency_consistency": float(cfg.reward_dependency_consistency),
            "plan_exec_alignment": float(cfg.reward_plan_exec_alignment),
            "penalty_redundancy": float(cfg.penalty_redundancy),
            "tool_match": float(cfg.reward_tool_match),
            "args_valid": float(cfg.reward_args_valid),
            "terminal_success": float(cfg.reward_terminal_success),
            "terminal_partial": float(cfg.reward_terminal_partial),
            "penalty_terminal_fail": float(cfg.penalty_terminal_fail),
            "recovery_success": float(cfg.reward_recovery_success),
            "penalty_blind_retry": float(cfg.penalty_blind_retry),
            "prior_scale": float(cfg.reward_prior_scale),
            "penalty_invalid_tool_name": float(cfg.penalty_invalid_tool_name),
            "penalty_mapped_tool_name": float(cfg.penalty_mapped_tool_name),
            "reward_all_tools_valid": float(cfg.reward_all_tools_valid),
            "reward_selected_in_calls": float(cfg.reward_selected_in_calls),
        }
    )
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    if cfg.logging_dir:
        Path(cfg.logging_dir).mkdir(parents=True, exist_ok=True)

    setup_wandb_env(cfg)
    set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    train_ds, eval_ds = build_dataset(cfg)
    train_rows = len(train_ds)
    eval_rows = len(eval_ds) if eval_ds is not None else 0
    world_size = _detect_world_size(args.local_rank)
    global_batch = max(1, cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps * world_size)
    print(
        f"[train] dataset: train={train_rows}, eval={eval_rows}, world_size={world_size}, "
        f"global_batch={global_batch}, num_generations={cfg.num_generations}"
    )
    if train_rows < global_batch:
        print(
            "[train][warn] train rows smaller than global batch. "
            "This run will produce very few optimizer steps."
        )
    tool_counter = Counter(str(train_ds[i]["target_selected_tool"]) for i in range(train_rows))
    if tool_counter:
        top_tool, top_n = tool_counter.most_common(1)[0]
        top_ratio = top_n / max(1, train_rows)
        print(f"[train] dominant tool in train set: {top_tool} ({top_n}/{train_rows}, {top_ratio:.1%})")
        if top_ratio >= 0.6:
            print(
                "[train][warn] train set is highly imbalanced. "
                "GRPO may overfit to a single tool policy."
            )
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    grpo_args = create_grpo_args(cfg, has_eval=eval_ds is not None)

    if cfg.dry_run:
        rows = [train_ds[i] for i in range(min(2, len(train_ds)))]
        fake = [
            json.dumps(
                {
                    "selected_tool": row["target_selected_tool"],
                    "tool_calls": row["target_tool_calls"],
                    "requires_task_planning": row["target_requires_task_planning"],
                },
                ensure_ascii=False,
            )
            for row in rows
        ]
        rewards = reward_fn(
            completions=fake,
            available_tools=[row["available_tools"] for row in rows],
            should_plan=[row["should_plan"] for row in rows],
            expected_steps=[row["expected_steps"] for row in rows],
            retry_count_total=[row["retry_count_total"] for row in rows],
            terminal_status=[row["terminal_status"] for row in rows],
            target_selected_tool=[row["target_selected_tool"] for row in rows],
            target_requires_task_planning=[row["target_requires_task_planning"] for row in rows],
            target_tool_calls=[row["target_tool_calls"] for row in rows],
            reward_prior=[row["reward_prior"] for row in rows],
        )
        print("Dry-run reward:", rewards)
        return

    model_or_path: Any = cfg.base_model_path
    try:
        trainer = GRPOTrainer(
            model=model_or_path,
            args=grpo_args,
            reward_funcs=[reward_fn],
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            peft_config=lora_cfg,
            processing_class=tokenizer,
        )
    except ModuleNotFoundError as exc:
        if str(cfg.deepspeed or "").strip() and getattr(exc, "name", "") == "mpi4py":
            raise SystemExit(
                "DeepSpeed Zero-3 初始化触发了 mpi4py 检查。请通过 "
                "`bash training/tool_planning_rl/run_grpo_train.sh` 或 `deepspeed/torchrun` 启动。"
            ) from exc
        raise
    except ValueError as exc:
        err = str(exc)
        if str(cfg.deepspeed or "").strip() and "DeepSpeed Zero-3 is not compatible with passing a `device_map`" in err:
            print("[train] detected Zero-3/device_map conflict, fallback to preloaded model instance.")
            model_or_path = _load_model_for_deepspeed(cfg)
            trainer = GRPOTrainer(
                model=model_or_path,
                args=grpo_args,
                reward_funcs=[reward_fn],
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                peft_config=lora_cfg,
                processing_class=tokenizer,
            )
        else:
            raise
    resume_from: Optional[str] = None
    resume_arg = str(args.resume_from_checkpoint or "").strip().lower()
    if resume_arg == "auto":
        resume_from = find_latest_checkpoint(cfg.output_dir)
    elif resume_arg and resume_arg != "none":
        resume_from = str(args.resume_from_checkpoint).strip()

    if resume_from:
        print(f"[train] resume from checkpoint: {resume_from}")
        result = trainer.train(resume_from_checkpoint=resume_from)
    else:
        print("[train] start from scratch")
        result = trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    metrics = dict(result.metrics)
    if eval_ds is not None and hasattr(trainer, "evaluate"):
        eval_metrics = trainer.evaluate()
        metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})

    Path(cfg.output_dir, "train_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(cfg.output_dir, "resolved_train_config.json").write_text(
        json.dumps(cfg.__dict__, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
