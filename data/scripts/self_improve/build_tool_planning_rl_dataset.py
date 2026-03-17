#!/usr/bin/env python3
"""从 self-improving 轨迹构建 Tool-Planning Agentic RL 数据集。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from rag_modules.llm_utils import extract_text_content  # noqa: E402
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "self_improve" / "raw_trajectories"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "dataset" / "agentic_rl" / "tool_planning"
DEFAULT_FILTER_PROMPT_PATH = PROJECT_ROOT / "src" / "prompts" / "self_improve_data_filter_prompt.txt"

DEV_RATIO_DEFAULT = 0.1
TEST_RATIO_DEFAULT = 0.1
REWARD_MIN_DEFAULT = -10.0
TOP_K_RESULTS_DEFAULT = 3
HASH_SLICE = 16
LLM_FILTER_MAX_JSON_CHARS = 2000


@dataclass
class BuildConfig:
    input_dir: Path
    output_dir: Path
    reward_min: float
    dev_ratio: float
    test_ratio: float
    seed: int
    llm_filter_enabled: bool
    llm_filter_model: str
    llm_filter_timeout_seconds: int
    llm_filter_max_tokens: int
    llm_filter_temperature: float
    llm_filter_prompt_path: Path
    llm_filter_drop_confidence_threshold: float
    llm_filter_hard_case_min_confidence: float
    llm_filter_hard_case_max_ratio: float


def parse_args() -> BuildConfig:
    parser = argparse.ArgumentParser(description="构建 Tool-Planning Agentic RL 数据集")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="轨迹 JSONL 目录")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录")
    parser.add_argument("--reward-min", type=float, default=REWARD_MIN_DEFAULT, help="最低 reward 过滤阈值")
    parser.add_argument("--dev-ratio", type=float, default=DEV_RATIO_DEFAULT, help="验证集占比")
    parser.add_argument("--test-ratio", type=float, default=TEST_RATIO_DEFAULT, help="测试集占比")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--llm-filter-enabled", action="store_true", help="启用 LLM 样本筛选")
    parser.add_argument("--llm-filter-model", default="kimi-k2-0711-preview", help="LLM 筛选模型")
    parser.add_argument("--llm-filter-timeout-seconds", type=int, default=20, help="LLM 筛选超时秒数")
    parser.add_argument("--llm-filter-max-tokens", type=int, default=220, help="LLM 筛选返回 token 上限")
    parser.add_argument("--llm-filter-temperature", type=float, default=0.0, help="LLM 筛选温度")
    parser.add_argument(
        "--llm-filter-prompt-path",
        default=str(DEFAULT_FILTER_PROMPT_PATH),
        help="LLM 样本筛选 prompt 模板路径",
    )
    parser.add_argument(
        "--llm-filter-drop-confidence-threshold",
        type=float,
        default=0.8,
        help="decision=drop 时最小置信度阈值，低于阈值将保留样本",
    )
    parser.add_argument(
        "--llm-filter-hard-case-min-confidence",
        type=float,
        default=0.85,
        help="decision=hard_case 时最低置信度阈值，低于阈值降级为 keep",
    )
    parser.add_argument(
        "--llm-filter-hard-case-max-ratio",
        type=float,
        default=0.4,
        help="hard_case 在保留样本中的最大占比，超出会自动降级一部分为 keep",
    )
    args = parser.parse_args()
    return BuildConfig(
        input_dir=Path(args.input_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        reward_min=float(args.reward_min),
        dev_ratio=float(args.dev_ratio),
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
        llm_filter_enabled=bool(args.llm_filter_enabled),
        llm_filter_model=str(args.llm_filter_model),
        llm_filter_timeout_seconds=int(args.llm_filter_timeout_seconds),
        llm_filter_max_tokens=int(args.llm_filter_max_tokens),
        llm_filter_temperature=float(args.llm_filter_temperature),
        llm_filter_prompt_path=Path(args.llm_filter_prompt_path).resolve(),
        llm_filter_drop_confidence_threshold=float(args.llm_filter_drop_confidence_threshold),
        llm_filter_hard_case_min_confidence=float(args.llm_filter_hard_case_min_confidence),
        llm_filter_hard_case_max_ratio=float(args.llm_filter_hard_case_max_ratio),
    )


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except Exception:
                continue
            if isinstance(item, dict):
                yield item


class LLMRecordFilter:
    DEFAULT_PROMPT = """
你是数据清洗评审器。请判断这条 Tool-Planning 训练样本是否保留。
目标：剔除明显错误/不可学样本，保留可学样本与困难样本。

只输出 JSON：
{
  "decision": "keep|drop|hard_case",
  "confidence": 0.0,
  "reason": ""
}

规则：
1) tool_calls 为空、selected_tool 为空、结构严重缺失 -> drop
2) 样本存在明显错误但可用于困难训练（如复杂任务欠规划、盲重试）-> hard_case
3) 其余可学习样本 -> keep
4) confidence 取值 0~1

样本：
{record_json}
""".strip()

    def __init__(self, cfg: BuildConfig):
        self.enabled = bool(cfg.llm_filter_enabled)
        self.prompt_template = self._load_prompt(cfg.llm_filter_prompt_path)
        api_key = os.getenv("MOONSHOT_API_KEY", "").strip()
        if self.enabled and api_key:
            self.client = ChatOpenAI(
                model=cfg.llm_filter_model,
                temperature=cfg.llm_filter_temperature,
                max_tokens=cfg.llm_filter_max_tokens,
                api_key=api_key,
                base_url="https://api.moonshot.cn/v1",
                timeout=cfg.llm_filter_timeout_seconds,
            )
        else:
            self.client = None
            self.enabled = False
        self.drop_threshold = max(0.0, min(1.0, float(cfg.llm_filter_drop_confidence_threshold)))

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        raw = str(text or "").strip()
        if len(raw) > LLM_FILTER_MAX_JSON_CHARS:
            raw = raw[:LLM_FILTER_MAX_JSON_CHARS]
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

    def _load_prompt(self, path: Path) -> str:
        try:
            text = path.read_text(encoding="utf-8").strip()
            return text or self.DEFAULT_PROMPT
        except Exception:
            return self.DEFAULT_PROMPT

    def _heuristic_decision(self, record: Dict[str, Any]) -> Dict[str, Any]:
        response = record.get("response", {}) or {}
        selected_tool = str(response.get("selected_tool", "") or "").strip()
        tool_calls = response.get("tool_calls", []) or []
        outcome = record.get("outcome", {}) or {}
        failure_tags = outcome.get("failure_tags", []) or []

        if (not selected_tool) or (not isinstance(tool_calls, list)) or (len(tool_calls) == 0):
            return {"decision": "drop", "confidence": 0.95, "reason": "empty_action"}
        if any(str(tag) in {"underplanning", "blind_retry"} for tag in failure_tags):
            return {"decision": "hard_case", "confidence": 0.8, "reason": "valuable_hard_case"}
        return {"decision": "keep", "confidence": 0.9, "reason": "learnable_sample"}

    def decide(self, record: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled or self.client is None:
            return self._heuristic_decision(record)
        prompt = self.prompt_template.replace("{record_json}", json.dumps(record, ensure_ascii=False))
        try:
            resp = self.client.invoke(prompt)
            payload = self._extract_json(extract_text_content(getattr(resp, "content", resp)))
        except Exception:
            return self._heuristic_decision(record)

        decision = str(payload.get("decision", "keep") or "keep").strip().lower()
        if decision not in {"keep", "drop", "hard_case"}:
            decision = "keep"
        confidence = payload.get("confidence", 0.7)
        try:
            conf = float(confidence)
        except Exception:
            conf = 0.7
        conf = max(0.0, min(1.0, conf))
        reason = str(payload.get("reason", "") or "")
        return {"decision": decision, "confidence": conf, "reason": reason}


def load_rows(input_dir: Path) -> List[Dict[str, Any]]:
    files = sorted(input_dir.glob("tool_planning_trajectory_*.jsonl"))
    rows: List[Dict[str, Any]] = []
    for file in files:
        for row in iter_jsonl(file):
            rows.append(row)
    return rows


def _tool_results_brief(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    brief = []
    for item in results[:TOP_K_RESULTS_DEFAULT]:
        if not isinstance(item, dict):
            continue
        brief.append(
            {
                "tool_name": str(item.get("tool_name", "") or ""),
                "ok": bool(item.get("ok", False)),
                "error_type": str(item.get("error_type", "") or ""),
                "error_code": str(item.get("error_code", "") or ""),
            }
        )
    return brief


def to_rl_record(row: Dict[str, Any]) -> Dict[str, Any]:
    state = row.get("state", {}) or {}
    action = row.get("action", {}) or {}
    outcome = row.get("outcome", {}) or {}
    reward = row.get("reward", {}) or {}
    tool_calls = action.get("tool_calls", []) or []
    selected_tool = str(action.get("selected_tool", "") or "")
    reward_total = float(reward.get("total", 0.0) or 0.0)

    prompt_payload = {
        "task": "tool_planning_policy",
        "user_query": str(state.get("user_query", "") or ""),
        "memory_context": str(state.get("memory_context", "") or ""),
        "intent": str(state.get("intent", "") or ""),
        "flow_type": str(state.get("flow_type", "") or ""),
        "requires_task_planning": bool(state.get("requires_task_planning", False)),
        "retry_count_used": int(state.get("retry_count_used", 0) or 0),
    }
    response_payload = {
        "selected_tool": selected_tool,
        "tool_calls": tool_calls,
        "requires_task_planning": bool(action.get("requires_task_planning", False)),
        "plan_source": str(action.get("plan_source", "") or ""),
    }

    fingerprint_src = json.dumps(
        {
            "query": prompt_payload.get("user_query", ""),
            "intent": prompt_payload.get("intent", ""),
            "selected_tool": selected_tool,
            "tool_calls": tool_calls,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    fingerprint = hashlib.sha1(fingerprint_src.encode("utf-8")).hexdigest()[:HASH_SLICE]

    return {
        "sample_id": str(row.get("sample_id", "") or ""),
        "episode_id": str(row.get("episode_id", "") or ""),
        "created_at_utc": str(row.get("created_at_utc", "") or ""),
        "fingerprint": fingerprint,
        "prompt": prompt_payload,
        "response": response_payload,
        "reward": {
            "total": reward_total,
            "components": reward.get("components", {}) or {},
        },
        "outcome": {
            "quality_gate_passed": bool(outcome.get("quality_gate_passed", False)),
            "retry_count_total": int(outcome.get("retry_count_total", 0) or 0),
            "retry_budget_exhausted": bool(outcome.get("retry_budget_exhausted", False)),
            "terminal_status": str(outcome.get("terminal_status", "") or ""),
            "should_plan": bool(outcome.get("should_plan", False)),
            "expected_steps": int(outcome.get("expected_steps", 1) or 1),
            "planned_steps": int(outcome.get("planned_steps", 0) or 0),
            "executed_steps": int(outcome.get("executed_steps", 0) or 0),
            "failure_tags": outcome.get("failure_tags", []) or [],
            "tool_results_brief": _tool_results_brief(outcome.get("tool_results", []) or []),
        },
    }


def deduplicate(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keep: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        fp = str(rec.get("fingerprint", "") or "")
        if not fp:
            continue
        prev = keep.get(fp)
        if prev is None:
            keep[fp] = rec
            continue
        prev_reward = float((prev.get("reward", {}) or {}).get("total", 0.0) or 0.0)
        now_reward = float((rec.get("reward", {}) or {}).get("total", 0.0) or 0.0)
        if now_reward > prev_reward:
            keep[fp] = rec
    return list(keep.values())


def _has_hard_case_signal(rec: Dict[str, Any]) -> bool:
    outcome = rec.get("outcome", {}) or {}
    tags = [str(x) for x in (outcome.get("failure_tags", []) or [])]
    if any(tag in {"underplanning", "blind_retry", "redundant_steps", "weak_dependency"} for tag in tags):
        return True
    terminal = str(outcome.get("terminal_status", "") or "").strip().lower()
    if terminal in {"partial", "fail"}:
        return True
    reward = rec.get("reward", {}) or {}
    total = float(reward.get("total", 0.0) or 0.0)
    if total < 5.0:
        return True
    return False


def rebalance_hard_cases(records: List[Dict[str, Any]], max_ratio: float) -> Tuple[List[Dict[str, Any]], int]:
    if not records:
        return records, 0
    max_ratio = max(0.0, min(1.0, float(max_ratio)))
    hard_idxs = [i for i, r in enumerate(records) if str((r.get("filter", {}) or {}).get("decision", "")) == "hard_case"]
    max_allowed = int(round(len(records) * max_ratio))
    if len(hard_idxs) <= max_allowed:
        return records, 0
    # 优先降级“低置信 hard_case”
    ranked = sorted(
        hard_idxs,
        key=lambda i: float(((records[i].get("filter", {}) or {}).get("confidence", 0.0) or 0.0)),
    )
    need_demote = len(hard_idxs) - max_allowed
    demoted = 0
    for idx in ranked[:need_demote]:
        f = records[idx].get("filter", {}) or {}
        f["decision_raw"] = "hard_case"
        f["decision"] = "keep"
        f["reason"] = (str(f.get("reason", "") or "") + "|demote_by_ratio").strip("|")
        records[idx]["filter"] = f
        demoted += 1
    return records, demoted


def split_records(
    records: List[Dict[str, Any]],
    dev_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    total = len(shuffled)
    test_n = int(total * test_ratio)
    dev_n = int(total * dev_ratio)
    train_n = max(0, total - dev_n - test_n)

    train = shuffled[:train_n]
    dev = shuffled[train_n : train_n + dev_n]
    test = shuffled[train_n + dev_n :]
    return train, dev, test


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    cfg = parse_args()
    if not cfg.input_dir.exists():
        raise SystemExit(f"输入目录不存在: {cfg.input_dir}")

    rows = load_rows(cfg.input_dir)
    records = [to_rl_record(row) for row in rows]
    records = [rec for rec in records if float((rec.get("reward", {}) or {}).get("total", 0.0) or 0.0) >= cfg.reward_min]
    records = deduplicate(records)

    filter_agent = LLMRecordFilter(cfg)
    filtered: List[Dict[str, Any]] = []
    llm_filter_stats = {
        "enabled": bool(filter_agent.enabled),
        "raw_keep": 0,
        "raw_drop": 0,
        "raw_hard_case": 0,
        "kept": 0,
        "dropped": 0,
        "hard_case": 0,
        "drop_recovered_low_conf": 0,
        "hard_case_recovered_low_conf": 0,
        "hard_case_recovered_no_signal": 0,
        "hard_case_demoted_by_ratio": 0,
    }
    for rec in records:
        decision = filter_agent.decide(rec)
        d = str(decision.get("decision", "keep") or "keep")
        c = float(decision.get("confidence", 0.0) or 0.0)
        rec["filter"] = {
            "decision": d,
            "confidence": c,
            "reason": str(decision.get("reason", "") or ""),
        }
        if d == "drop":
            llm_filter_stats["raw_drop"] += 1
        elif d == "hard_case":
            llm_filter_stats["raw_hard_case"] += 1
        else:
            llm_filter_stats["raw_keep"] += 1

        if d == "drop":
            if c >= cfg.llm_filter_drop_confidence_threshold:
                llm_filter_stats["dropped"] += 1
                continue
            llm_filter_stats["drop_recovered_low_conf"] += 1
            llm_filter_stats["kept"] += 1
            filtered.append(rec)
            continue
        if d == "hard_case":
            if c < cfg.llm_filter_hard_case_min_confidence:
                rec["filter"]["decision_raw"] = "hard_case"
                rec["filter"]["decision"] = "keep"
                rec["filter"]["reason"] = (str(rec["filter"].get("reason", "") or "") + "|demote_low_conf").strip("|")
                llm_filter_stats["hard_case_recovered_low_conf"] += 1
                llm_filter_stats["kept"] += 1
                filtered.append(rec)
                continue
            if not _has_hard_case_signal(rec):
                rec["filter"]["decision_raw"] = "hard_case"
                rec["filter"]["decision"] = "keep"
                rec["filter"]["reason"] = (str(rec["filter"].get("reason", "") or "") + "|demote_no_signal").strip("|")
                llm_filter_stats["hard_case_recovered_no_signal"] += 1
                llm_filter_stats["kept"] += 1
                filtered.append(rec)
                continue
            llm_filter_stats["hard_case"] += 1
            llm_filter_stats["kept"] += 1
            filtered.append(rec)
            continue
        llm_filter_stats["kept"] += 1
        filtered.append(rec)

    filtered, demoted_by_ratio = rebalance_hard_cases(
        records=filtered,
        max_ratio=cfg.llm_filter_hard_case_max_ratio,
    )
    llm_filter_stats["hard_case_demoted_by_ratio"] = demoted_by_ratio
    llm_filter_stats["hard_case"] = sum(
        1 for rec in filtered if str((rec.get("filter", {}) or {}).get("decision", "")) == "hard_case"
    )
    llm_filter_stats["kept"] = len(filtered)

    records = filtered
    train, dev, test = split_records(
        records=records,
        dev_ratio=cfg.dev_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.seed,
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(cfg.output_dir / "train.jsonl", train)
    write_jsonl(cfg.output_dir / "dev.jsonl", dev)
    write_jsonl(cfg.output_dir / "test.jsonl", test)

    summary = {
        "input_dir": str(cfg.input_dir),
        "output_dir": str(cfg.output_dir),
        "rows_raw": len(rows),
        "rows_after_filter": len(records),
        "split": {
            "train": len(train),
            "dev": len(dev),
            "test": len(test),
        },
        "reward_min": cfg.reward_min,
        "seed": cfg.seed,
        "llm_filter": llm_filter_stats,
        "llm_filter_thresholds": {
            "drop_confidence": cfg.llm_filter_drop_confidence_threshold,
            "hard_case_min_confidence": cfg.llm_filter_hard_case_min_confidence,
            "hard_case_max_ratio": cfg.llm_filter_hard_case_max_ratio,
        },
    }
    (cfg.output_dir / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
