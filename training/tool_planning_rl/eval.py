#!/usr/bin/env python3
"""Offline eval for Tool-Planning GRPO/LoRA adapters."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "你是DeltaAgent的Tool-Planning策略模型。"
    "你必须只输出JSON对象，字段必须包含 selected_tool/tool_calls/requires_task_planning。"
    "tool_calls是数组，每个元素必须包含tool_name/tool_query。"
)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_prompt(prompt_obj: Dict[str, Any]) -> str:
    user_query = str(prompt_obj.get("user_query", "") or "")
    memory_context = str(prompt_obj.get("memory_context", "") or "")
    intent = str(prompt_obj.get("intent", "") or "")
    flow_type = str(prompt_obj.get("flow_type", "") or "")
    requires_planning = bool(prompt_obj.get("requires_task_planning", False))
    retry_count = int(prompt_obj.get("retry_count_used", 0) or 0)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "请根据以下上下文输出策略JSON：\n"
        f"- user_query: {user_query}\n"
        f"- memory_context: {memory_context}\n"
        f"- intent: {intent}\n"
        f"- flow_type: {flow_type}\n"
        f"- requires_task_planning_hint: {requires_planning}\n"
        f"- retry_count_used: {retry_count}\n"
    )


def extract_json(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return {}
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def normalize_tool_calls(value: Any) -> List[Dict[str, str]]:
    if not isinstance(value, list):
        return []
    out: List[Dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        tool_name = str(item.get("tool_name", "") or "").strip()
        tool_query = str(item.get("tool_query", "") or "").strip()
        if not tool_name:
            continue
        out.append({"tool_name": tool_name, "tool_query": tool_query})
    return out


def normalize_prediction(obj: Dict[str, Any]) -> Dict[str, Any]:
    calls = normalize_tool_calls(obj.get("tool_calls", []))
    selected = str(obj.get("selected_tool", "") or "").strip()
    if not selected and calls:
        selected = calls[0]["tool_name"]
    return {
        "selected_tool": selected,
        "requires_task_planning": bool(obj.get("requires_task_planning", False)),
        "tool_calls": calls,
    }


def chain_of(calls: List[Dict[str, str]]) -> List[str]:
    return [str(x.get("tool_name", "")).strip() for x in calls if str(x.get("tool_name", "")).strip()]


def query_coverage(pred_calls: List[Dict[str, str]], gold_calls: List[Dict[str, str]]) -> float:
    if not gold_calls:
        return 1.0
    pred_map: Dict[str, List[str]] = {}
    for item in pred_calls:
        name = str(item.get("tool_name", "")).strip()
        query = str(item.get("tool_query", "")).strip()
        pred_map.setdefault(name, []).append(query)
    hit = 0
    for g in gold_calls:
        name = str(g.get("tool_name", "")).strip()
        found = any(str(q).strip() for q in pred_map.get(name, []))
        if found:
            hit += 1
    return hit / max(1, len(gold_calls))


def compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    total = max(len(rows), 1)
    parse_ok = 0
    selected_ok = 0
    planning_ok = 0
    chain_exact = 0
    set_match = 0
    qcov_sum = 0.0
    underplan = 0
    overplan = 0
    pred_steps_sum = 0
    gold_steps_sum = 0

    for row in rows:
        pred = row["pred_norm"]
        gold = row["gold_norm"]
        if row["parsed_ok"]:
            parse_ok += 1
        if pred["selected_tool"] == gold["selected_tool"]:
            selected_ok += 1
        if pred["requires_task_planning"] == gold["requires_task_planning"]:
            planning_ok += 1

        pred_chain = chain_of(pred["tool_calls"])
        gold_chain = chain_of(gold["tool_calls"])
        if pred_chain == gold_chain:
            chain_exact += 1
        if set(pred_chain) == set(gold_chain):
            set_match += 1
        qcov_sum += query_coverage(pred["tool_calls"], gold["tool_calls"])

        pred_steps = len(pred_chain)
        gold_steps = len(gold_chain)
        pred_steps_sum += pred_steps
        gold_steps_sum += gold_steps
        if pred_steps < gold_steps:
            underplan += 1
        elif pred_steps > gold_steps:
            overplan += 1

    return {
        "total": len(rows),
        "json_parse_rate": parse_ok / total,
        "selected_tool_accuracy": selected_ok / total,
        "planning_flag_accuracy": planning_ok / total,
        "plan_chain_exact_match": chain_exact / total,
        "plan_set_match": set_match / total,
        "tool_query_coverage": qcov_sum / total,
        "underplan_rate": underplan / total,
        "overplan_rate": overplan / total,
        "avg_pred_steps": pred_steps_sum / total,
        "avg_gold_steps": gold_steps_sum / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--report_file", required=True)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
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

    rows: List[Dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        prompt = build_prompt(sample.get("prompt", {}) or {})
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
        pred_norm = normalize_prediction(pred_raw)
        gold_resp = sample.get("response", {}) or {}
        gold_norm = {
            "selected_tool": str(gold_resp.get("selected_tool", "") or "").strip(),
            "requires_task_planning": bool(gold_resp.get("requires_task_planning", False)),
            "tool_calls": normalize_tool_calls(gold_resp.get("tool_calls", [])),
        }
        rows.append(
            {
                "idx": idx,
                "sample_id": sample.get("sample_id", f"sample-{idx}"),
                "parsed_ok": bool(pred_raw),
                "pred_raw": pred_raw,
                "pred_norm": pred_norm,
                "gold_norm": gold_norm,
            }
        )

    metrics = compute_metrics(rows)
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "base_model_path": args.base_model_path,
        "adapter_path": adapter_path,
        "test_file": args.test_file,
        "metrics": metrics,
    }
    with open(args.report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
