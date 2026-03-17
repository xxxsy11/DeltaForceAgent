#!/usr/bin/env python3
"""Offline eval for planning SFT adapter."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def normalize_label(label: Dict[str, Any], available_tools: List[str]) -> Dict[str, Any]:
    task_plan_raw = label.get("task_plan") or []
    task_plan: List[Dict[str, str]] = []
    for step in task_plan_raw:
        if not isinstance(step, dict):
            continue
        tool_name = str(step.get("tool_name", "")).strip()
        tool_query = str(step.get("tool_query", "")).strip()
        if not tool_name or tool_name not in available_tools:
            continue
        if not tool_query:
            continue
        task_plan.append({"tool_name": tool_name, "tool_query": tool_query})

    try:
        confidence = float(label.get("confidence", 0.5))
    except Exception:
        confidence = 0.5

    return {
        "intent": str(label.get("intent", "unknown_planning_intent")).strip() or "unknown_planning_intent",
        "requires_task_planning": bool(label.get("requires_task_planning", False)),
        "task_plan": task_plan,
        "confidence": max(0.0, min(1.0, confidence)),
    }


def build_prompt(sample: Dict[str, Any], system_prompt: str) -> List[Dict[str, str]]:
    query = str(sample.get("user_query", "")).strip()
    memory_context = str(sample.get("memory_context", "")).strip()
    tools = ", ".join(str(x) for x in (sample.get("available_tools") or []))
    user_content = (
        "请完成复杂任务规划。\n"
        "要求：仅输出JSON对象，字段必须为 intent/reason/requires_task_planning/confidence/task_plan。\n"
        "task_plan必须是有序数组，元素为 {tool_name,tool_query}。\n"
        f"用户问题: {query}\n"
        f"记忆上下文: {memory_context if memory_context else '<EMPTY>'}\n"
        f"可用工具: {tools}\n"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def extract_json(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _tool_seq(plan: List[Dict[str, str]]) -> List[str]:
    return [str(x.get("tool_name", "")).strip() for x in plan if str(x.get("tool_name", "")).strip()]


def compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    total = max(len(rows), 1)
    parse_ok = 0
    intent_ok = 0
    planning_flag_ok = 0
    plan_chain_exact = 0
    plan_set_match = 0
    plan_query_coverage = 0
    confidence_mae_sum = 0.0

    for row in rows:
        pred = row["pred_norm"]
        gold = row["gold_norm"]

        if row["parsed_ok"]:
            parse_ok += 1
        if pred["intent"] == gold["intent"]:
            intent_ok += 1
        if pred["requires_task_planning"] == gold["requires_task_planning"]:
            planning_flag_ok += 1

        pred_chain = _tool_seq(pred["task_plan"])
        gold_chain = _tool_seq(gold["task_plan"])
        if pred_chain == gold_chain:
            plan_chain_exact += 1
        if set(pred_chain) == set(gold_chain):
            plan_set_match += 1

        # All gold plan tools should have non-empty tool_query in pred.
        pred_query_map = {str(x.get("tool_name", "")).strip(): str(x.get("tool_query", "")).strip() for x in pred["task_plan"]}
        if all(pred_query_map.get(t, "") for t in gold_chain):
            plan_query_coverage += 1

        confidence_mae_sum += abs(float(pred["confidence"]) - float(gold["confidence"]))

    return {
        "total": len(rows),
        "json_parse_rate": parse_ok / total,
        "intent_accuracy": intent_ok / total,
        "planning_flag_accuracy": planning_flag_ok / total,
        "plan_chain_exact_match": plan_chain_exact / total,
        "plan_set_match": plan_set_match / total,
        "plan_query_coverage": plan_query_coverage / total,
        "confidence_mae": confidence_mae_sum / total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--report_file", required=True)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0", help="Model device, e.g. cuda:0 or cpu")
    parser.add_argument(
        "--system_prompt",
        default=(
            "你是DeltaAgent的复杂任务规划子Agent。"
            "你必须根据用户问题、记忆上下文和可用工具输出严格JSON，"
            "字段必须是 intent/reason/requires_task_planning/confidence/task_plan。"
            "task_plan为有序数组，元素结构为{tool_name,tool_query}。"
            "不要输出任何额外文本。"
        ),
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cpu = str(args.device).lower() == "cpu"
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32 if use_cpu else torch.float16,
    )
    base_model.to(args.device)
    model = base_model
    adapter_path = str(args.adapter_path or "").strip()
    if adapter_path:
        model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    samples = load_jsonl(args.test_file)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    results: List[Dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        messages = build_prompt(sample, args.system_prompt)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        gen_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        pred_raw = extract_json(text)
        available_tools = [str(x) for x in (sample.get("available_tools") or [])]
        pred_norm = normalize_label(pred_raw, available_tools)
        gold_norm = normalize_label(sample.get("label") or {}, available_tools)

        results.append(
            {
                "idx": idx,
                "id": sample.get("id", f"sample-{idx}"),
                "user_query": sample.get("user_query", ""),
                "gold": sample.get("label") or {},
                "pred_raw": pred_raw,
                "pred_norm": pred_norm,
                "gold_norm": gold_norm,
                "parsed_ok": bool(pred_raw),
            }
        )

    metrics = compute_metrics(results)

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "base_model_path": args.base_model_path,
        "adapter_path": adapter_path,
        "metrics": metrics,
    }
    with open(args.report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
