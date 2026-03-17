#!/usr/bin/env python3
"""Evaluate Kimi on intent/tool-selection/planning SFT test sets."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[2]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_json(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return {}
    try:
        parsed = json.loads(m.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def call_kimi(client: OpenAI, model: str, messages: List[Dict[str, str]], max_tokens: int, retries: int = 4) -> str:
    for i in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0,
                max_tokens=max_tokens,
                timeout=30,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            if i == retries - 1:
                return ""
            time.sleep(1.0 * (i + 1))
    return ""


def normalize_intent(label: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "intent": str(label.get("intent", "general_chat")),
        "entities": [str(x) for x in (label.get("entities") or [])],
        "flow_type": str(label.get("flow_type", "single")),
        "requires_task_planning": bool(label.get("requires_task_planning", False)),
        "selected_tools": [str(x) for x in (label.get("selected_tools") or [])],
    }


def intent_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    total = max(1, len(rows))
    parse_ok = sum(1 for r in rows if r["parsed_ok"])
    intent_ok = sum(1 for r in rows if r["pred_norm"]["intent"] == r["gold_norm"]["intent"])
    flow_ok = sum(1 for r in rows if r["pred_norm"]["flow_type"] == r["gold_norm"]["flow_type"])
    planning_ok = sum(
        1
        for r in rows
        if r["pred_norm"]["requires_task_planning"] == r["gold_norm"]["requires_task_planning"]
    )
    tool_exact = sum(
        1
        for r in rows
        if sorted(r["pred_norm"]["selected_tools"]) == sorted(r["gold_norm"]["selected_tools"])
    )
    return {
        "total": len(rows),
        "json_parse_rate": parse_ok / total,
        "intent_accuracy": intent_ok / total,
        "flow_type_accuracy": flow_ok / total,
        "planning_flag_accuracy": planning_ok / total,
        "tool_exact_match": tool_exact / total,
    }


def normalize_tool(label: Dict[str, Any], available_tools: List[str]) -> Dict[str, Any]:
    selected_tools = [str(x) for x in (label.get("selected_tools") or []) if str(x) in available_tools]
    tq_raw_any = label.get("tool_queries") or {}
    tq_raw = tq_raw_any if isinstance(tq_raw_any, dict) else {}
    tq: Dict[str, str] = {}
    for t in selected_tools:
        v = str(tq_raw.get(t, "")).strip()
        if v:
            tq[t] = v
    try:
        conf = float(label.get("confidence", 0.5))
    except Exception:
        conf = 0.5
    return {
        "selected_tools": selected_tools,
        "tool_queries": tq,
        "requires_task_planning": bool(label.get("requires_task_planning", False)),
        "confidence": max(0.0, min(1.0, conf)),
    }


def tool_f1(pred: List[str], gold: List[str]) -> float:
    ps, gs = set(pred), set(gold)
    if not ps and not gs:
        return 1.0
    if not ps or not gs:
        return 0.0
    tp = len(ps & gs)
    p = tp / len(ps)
    r = tp / len(gs)
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def tool_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    total = max(1, len(rows))
    parse_ok = sum(1 for r in rows if r["parsed_ok"])
    planning_ok = sum(
        1
        for r in rows
        if r["pred_norm"]["requires_task_planning"] == r["gold_norm"]["requires_task_planning"]
    )
    tool_exact = sum(
        1
        for r in rows
        if sorted(r["pred_norm"]["selected_tools"]) == sorted(r["gold_norm"]["selected_tools"])
    )
    f1_sum = 0.0
    q_cov = 0
    q_exact = 0
    mae = 0.0
    for r in rows:
        pred = r["pred_norm"]
        gold = r["gold_norm"]
        f1_sum += tool_f1(pred["selected_tools"], gold["selected_tools"])
        mae += abs(pred["confidence"] - gold["confidence"])
        gold_tools = gold["selected_tools"]
        if all(pred["tool_queries"].get(t, "").strip() for t in gold_tools):
            q_cov += 1
        ok = sorted(pred["selected_tools"]) == sorted(gold["selected_tools"])
        if ok:
            for t in gold_tools:
                if pred["tool_queries"].get(t, "").strip() != gold["tool_queries"].get(t, "").strip():
                    ok = False
                    break
        if ok:
            q_exact += 1
    return {
        "total": len(rows),
        "json_parse_rate": parse_ok / total,
        "planning_flag_accuracy": planning_ok / total,
        "tool_exact_match": tool_exact / total,
        "tool_f1": f1_sum / total,
        "tool_query_coverage": q_cov / total,
        "tool_query_exact_match": q_exact / total,
        "confidence_mae": mae / total,
    }


def normalize_plan(label: Dict[str, Any], available_tools: List[str]) -> Dict[str, Any]:
    task_plan_raw = label.get("task_plan") or []
    task_plan: List[Dict[str, str]] = []
    for s in task_plan_raw:
        if not isinstance(s, dict):
            continue
        tn = str(s.get("tool_name", "")).strip()
        tq = str(s.get("tool_query", "")).strip()
        if tn and tq and tn in available_tools:
            task_plan.append({"tool_name": tn, "tool_query": tq})
    try:
        conf = float(label.get("confidence", 0.5))
    except Exception:
        conf = 0.5
    return {
        "intent": str(label.get("intent", "unknown_planning_intent")).strip() or "unknown_planning_intent",
        "requires_task_planning": bool(label.get("requires_task_planning", False)),
        "task_plan": task_plan,
        "confidence": max(0.0, min(1.0, conf)),
    }


def _plan_seq(plan: List[Dict[str, str]]) -> List[str]:
    return [str(x.get("tool_name", "")).strip() for x in plan if str(x.get("tool_name", "")).strip()]


def planning_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    total = max(1, len(rows))
    parse_ok = sum(1 for r in rows if r["parsed_ok"])
    intent_ok = sum(1 for r in rows if r["pred_norm"]["intent"] == r["gold_norm"]["intent"])
    planning_ok = sum(
        1
        for r in rows
        if r["pred_norm"]["requires_task_planning"] == r["gold_norm"]["requires_task_planning"]
    )
    chain_exact = 0
    set_match = 0
    q_cov = 0
    mae = 0.0
    for r in rows:
        pred = r["pred_norm"]
        gold = r["gold_norm"]
        pchain = _plan_seq(pred["task_plan"])
        gchain = _plan_seq(gold["task_plan"])
        if pchain == gchain:
            chain_exact += 1
        if set(pchain) == set(gchain):
            set_match += 1
        pmap = {str(x.get("tool_name", "")).strip(): str(x.get("tool_query", "")).strip() for x in pred["task_plan"]}
        if all(pmap.get(t, "") for t in gchain):
            q_cov += 1
        mae += abs(pred["confidence"] - gold["confidence"])
    return {
        "total": len(rows),
        "json_parse_rate": parse_ok / total,
        "intent_accuracy": intent_ok / total,
        "planning_flag_accuracy": planning_ok / total,
        "plan_chain_exact_match": chain_exact / total,
        "plan_set_match": set_match / total,
        "plan_query_coverage": q_cov / total,
        "confidence_mae": mae / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="kimi-k2-0711-preview")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    api_key = os.getenv("MOONSHOT_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("MOONSHOT_API_KEY not set")
    client = OpenAI(api_key=api_key, base_url="https://api.moonshot.cn/v1")

    out_dir = ROOT / "outputs/sft_module_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # intent
    intent_rows: List[Dict[str, Any]] = []
    for i, s in enumerate(load_jsonl(ROOT / "data/dataset/sft/intent/test.jsonl")[: args.max_samples]):
        if i % 10 == 0:
            print(f"[intent] {i}/{args.max_samples}")
        messages = [
            {
                "role": "system",
                "content": "你是DeltaAgent的意图识别与工具规划子Agent。你必须根据用户问题、记忆上下文和可用工具输出严格JSON，不要输出任何额外文本。",
            },
            {
                "role": "user",
                "content": (
                    "请完成意图识别与工具规划。\n"
                    "要求：仅输出JSON对象，字段必须为 intent/entities/flow_type/requires_task_planning/selected_tools/confidence。\n"
                    f"用户问题: {str(s.get('user_query','')).strip()}\n"
                    f"记忆上下文: {str(s.get('memory_context','')).strip() or '<EMPTY>'}\n"
                    f"可用工具: {', '.join(str(x) for x in (s.get('available_tools') or []))}\n"
                ),
            },
        ]
        text = call_kimi(client, args.model, messages, args.max_tokens)
        pred_raw = extract_json(text)
        intent_rows.append(
            {
                "idx": i,
                "id": s.get("id", f"intent-{i}"),
                "parsed_ok": bool(pred_raw),
                "pred_norm": normalize_intent(pred_raw),
                "gold_norm": normalize_intent(s.get("label") or {}),
            }
        )
    intent_report = {"model": args.model, "metrics": intent_metrics(intent_rows)}
    (out_dir / "intent_kimi_report.json").write_text(json.dumps(intent_report, ensure_ascii=False, indent=2), encoding="utf-8")

    # tool selection
    tool_rows: List[Dict[str, Any]] = []
    for i, s in enumerate(load_jsonl(ROOT / "data/dataset/sft/tool_selection/test.jsonl")[: args.max_samples]):
        if i % 10 == 0:
            print(f"[tool_selection] {i}/{args.max_samples}")
        avail = [str(x) for x in (s.get("available_tools") or [])]
        messages = [
            {
                "role": "system",
                "content": "你是DeltaAgent的工具选择子Agent。你必须根据用户问题、记忆上下文和可用工具输出严格JSON，字段必须是 selected_tools/tool_queries/requires_task_planning/confidence。不要输出任何额外文本。",
            },
            {
                "role": "user",
                "content": (
                    "请完成工具选择与参数构造。\n"
                    "要求：仅输出JSON对象，字段必须为 selected_tools/tool_queries/requires_task_planning/confidence。\n"
                    f"用户问题: {str(s.get('user_query','')).strip()}\n"
                    f"记忆上下文: {str(s.get('memory_context','')).strip() or '<EMPTY>'}\n"
                    f"可用工具: {', '.join(avail)}\n"
                ),
            },
        ]
        text = call_kimi(client, args.model, messages, args.max_tokens)
        pred_raw = extract_json(text)
        tool_rows.append(
            {
                "idx": i,
                "id": s.get("id", f"tool-{i}"),
                "parsed_ok": bool(pred_raw),
                "pred_norm": normalize_tool(pred_raw, avail),
                "gold_norm": normalize_tool(s.get("label") or {}, avail),
            }
        )
    tool_report = {"model": args.model, "metrics": tool_metrics(tool_rows)}
    (out_dir / "tool_kimi_report.json").write_text(json.dumps(tool_report, ensure_ascii=False, indent=2), encoding="utf-8")

    # planning
    plan_rows: List[Dict[str, Any]] = []
    for i, s in enumerate(load_jsonl(ROOT / "data/dataset/sft/planning/test.jsonl")[: args.max_samples]):
        if i % 10 == 0:
            print(f"[planning] {i}/{args.max_samples}")
        avail = [str(x) for x in (s.get("available_tools") or [])]
        messages = [
            {
                "role": "system",
                "content": "你是DeltaAgent的复杂任务规划子Agent。你必须根据用户问题、记忆上下文和可用工具输出严格JSON，字段必须是 intent/reason/requires_task_planning/confidence/task_plan。task_plan为有序数组，元素结构为{tool_name,tool_query}。不要输出任何额外文本。",
            },
            {
                "role": "user",
                "content": (
                    "请完成复杂任务规划。\n"
                    "要求：仅输出JSON对象，字段必须为 intent/reason/requires_task_planning/confidence/task_plan。\n"
                    "task_plan必须是有序数组，元素为 {tool_name,tool_query}。\n"
                    f"用户问题: {str(s.get('user_query','')).strip()}\n"
                    f"记忆上下文: {str(s.get('memory_context','')).strip() or '<EMPTY>'}\n"
                    f"可用工具: {', '.join(avail)}\n"
                ),
            },
        ]
        text = call_kimi(client, args.model, messages, args.max_tokens)
        pred_raw = extract_json(text)
        plan_rows.append(
            {
                "idx": i,
                "id": s.get("id", f"plan-{i}"),
                "parsed_ok": bool(pred_raw),
                "pred_norm": normalize_plan(pred_raw, avail),
                "gold_norm": normalize_plan(s.get("label") or {}, avail),
            }
        )
    planning_report = {"model": args.model, "metrics": planning_metrics(plan_rows)}
    (out_dir / "planning_kimi_report.json").write_text(
        json.dumps(planning_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "max_samples": args.max_samples,
                "reports": {
                    "intent": str(out_dir / "intent_kimi_report.json"),
                    "tool_selection": str(out_dir / "tool_kimi_report.json"),
                    "planning": str(out_dir / "planning_kimi_report.json"),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
