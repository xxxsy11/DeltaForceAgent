#!/usr/bin/env python3
"""生成 Tool-Planning GRPO 训练用的合成数据集（JSONL）。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "dataset" / "agentic_rl" / "tool_planning_synth_1000"

TRAIN_RATIO = 0.8
DEV_RATIO = 0.1
TEST_RATIO = 0.1

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


@dataclass
class SynthConfig:
    output_dir: Path
    num_samples: int
    seed: int


def parse_args() -> SynthConfig:
    parser = argparse.ArgumentParser(description="生成 Tool-Planning GRPO 合成数据")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录")
    parser.add_argument("--num-samples", type=int, default=1000, help="总样本数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    return SynthConfig(
        output_dir=Path(args.output_dir).resolve(),
        num_samples=max(100, int(args.num_samples)),
        seed=int(args.seed),
    )


def _load_names() -> Dict[str, List[str]]:
    neo4j_dir = PROJECT_ROOT / "data" / "neo4j"
    files = {
        "collection": neo4j_dir / "collection.json",
        "firearms": neo4j_dir / "firearms.json",
        "equipment": neo4j_dir / "equipment.json",
        "ammo": neo4j_dir / "ammo.json",
        "operator": neo4j_dir / "operator.json",
        "map": neo4j_dir / "map.json",
    }
    out: Dict[str, List[str]] = {}
    for key, path in files.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        names: List[str] = []
        for node in data.get("nodes", []):
            props = node.get("props", {}) or {}
            labels = node.get("labels", []) or []
            if key == "collection" and "Collectible" not in labels:
                continue
            if key == "firearms" and "Firearm" not in labels:
                continue
            if key == "equipment" and "Equipment" not in labels:
                continue
            if key == "ammo" and "Ammo" not in labels:
                continue
            if key == "operator" and "Operator" not in labels:
                continue
            if key == "map" and "Map" not in labels:
                continue
            name = props.get("name") or props.get("objectName") or props.get("displayName")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
        out[key] = list(dict.fromkeys(names))
    return out


def _pick_unique(rng: random.Random, pool: List[str], n: int) -> List[str]:
    if not pool:
        return []
    if n <= 1:
        return [rng.choice(pool)]
    n = min(n, len(pool))
    return rng.sample(pool, n)


def _tool_result_brief(calls: List[Dict[str, str]], terminal: str) -> List[Dict[str, Any]]:
    ok = terminal == "success"
    first = calls[0]["tool_name"] if calls else ""
    return [
        {
            "tool_name": first,
            "ok": ok,
            "error_type": "" if ok else "tool_failed",
            "error_code": "" if ok else "synthetic_partial",
        }
    ]


def _fingerprint(query: str, calls: List[Dict[str, str]]) -> str:
    payload = {"query": query, "calls": calls}
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _pick_available_tools(
    rng: random.Random,
    required: List[str],
    *,
    extra_min: int = 2,
    extra_max: int = 4,
) -> List[str]:
    required_tools = [tool for tool in required if tool in AVAILABLE_TOOLS]
    if rng.random() < 0.35:
        return list(AVAILABLE_TOOLS)
    pool = [tool for tool in AVAILABLE_TOOLS if tool not in required_tools]
    extra_n = min(len(pool), rng.randint(extra_min, extra_max))
    distractors = rng.sample(pool, k=extra_n) if extra_n > 0 else []
    final = list(dict.fromkeys(required_tools + distractors))
    rng.shuffle(final)
    return final


def _memory_last_user(query: str) -> str:
    return f"[最近对话]\n- 用户: {query}"


def _memory_retry(previous_query: str, current_query: str) -> str:
    return (
        "[最近对话]\n"
        f"- 用户: {previous_query}\n"
        "- 助手: 工具执行失败，未返回稳定结果\n"
        f"- 用户: {current_query}"
    )


def _reward_total(
    rng: random.Random,
    terminal: str,
    should_plan: bool,
    tool_calls_n: int,
    retry_count_total: int,
    filter_decision: str,
) -> Tuple[float, Dict[str, float]]:
    base = 10.2 if terminal == "success" else 5.8
    if should_plan:
        base += 0.9
    if tool_calls_n > 1:
        base += 0.7
    if tool_calls_n > 2:
        base += 0.4
    if retry_count_total > 0:
        base -= 0.8
    if filter_decision == "hard_case":
        base -= 0.4
    jitter = rng.uniform(-0.9, 0.9)
    total = round(base + jitter, 6)
    components = {
        "synthetic": 1.0,
        "terminal": 1.0 if terminal == "success" else -0.6,
        "planning": 1.0 if should_plan else 0.2,
        "tool_count": float(tool_calls_n),
        "retry_penalty": float(-retry_count_total * 0.8),
        "hard_case_adjust": -0.4 if filter_decision == "hard_case" else 0.0,
    }
    return total, components


def _build_record(
    *,
    idx: int,
    user_id: str,
    session_id: str,
    created_at: datetime,
    query: str,
    intent: str,
    flow_type: str,
    should_plan: bool,
    tool_calls: List[Dict[str, str]],
    available_tools: List[str],
    plan_source: str,
    memory_context: str,
    terminal_status: str,
    retry_count_total: int,
    filter_decision: str,
    rng: random.Random,
) -> Dict[str, Any]:
    selected_tool = tool_calls[0]["tool_name"] if tool_calls else ""
    expected_steps = max(1, len(tool_calls))
    reward_total, reward_components = _reward_total(
        rng=rng,
        terminal=terminal_status,
        should_plan=should_plan,
        tool_calls_n=len(tool_calls),
        retry_count_total=retry_count_total,
        filter_decision=filter_decision,
    )
    failure_tags: List[str] = []
    if should_plan and len(tool_calls) <= 1:
        failure_tags.append("underplanning")
    if retry_count_total > 0 and terminal_status != "success":
        failure_tags.append("blind_retry")
    rec = {
        "sample_id": f"synthetic-{idx:06d}",
        "episode_id": session_id,
        "created_at_utc": created_at.isoformat(),
        "fingerprint": _fingerprint(query, tool_calls),
        "prompt": {
            "task": "tool_planning_policy",
            "user_query": query,
            "memory_context": memory_context,
            "intent": intent,
            "flow_type": flow_type,
            "requires_task_planning": should_plan,
            "retry_count_used": retry_count_total,
            "available_tools": available_tools,
        },
        "response": {
            "selected_tool": selected_tool,
            "tool_calls": tool_calls,
            "requires_task_planning": should_plan,
            "plan_source": plan_source,
        },
        "reward": {
            "total": reward_total,
            "components": reward_components,
        },
        "outcome": {
            "quality_gate_passed": True,
            "retry_count_total": retry_count_total,
            "retry_budget_exhausted": False,
            "terminal_status": terminal_status,
            "should_plan": should_plan,
            "expected_steps": expected_steps,
            "planned_steps": len(tool_calls),
            "executed_steps": len(tool_calls),
            "failure_tags": failure_tags,
            "tool_results_brief": _tool_result_brief(tool_calls, terminal_status),
        },
        "filter": {
            "decision": filter_decision,
            "confidence": 0.9 if filter_decision == "keep" else 0.85,
            "reason": "synthetic_generation",
        },
    }
    # 明确保留 user/session 信息方便追踪
    rec["prompt"]["user_id"] = user_id
    rec["prompt"]["session_id"] = session_id
    return rec


def _category_plan(total: int) -> List[Tuple[str, int]]:
    # 单工具与多步样本更均衡，增加上下文/重试类型，降低模板化偏差。
    ratios = [
        ("knowledge", 0.13),
        ("latest_price", 0.09),
        ("history_price", 0.07),
        ("price_advice", 0.10),
        ("place_profit_rank", 0.08),
        ("compare_single", 0.08),
        ("profit_stability", 0.06),
        ("answer_composer", 0.04),
        ("multi_compare_advice", 0.08),
        ("multi_profile_latest", 0.07),
        ("multi_profile_advice", 0.05),
        ("multi_rank_advice", 0.04),
        ("multi_profile_history_advice", 0.04),
        ("retry_latest", 0.04),
        ("followup_compare_advice", 0.03),
    ]
    counts: List[Tuple[str, int]] = []
    used = 0
    for i, (name, ratio) in enumerate(ratios):
        if i == len(ratios) - 1:
            n = total - used
        else:
            n = int(round(total * ratio))
            used += n
        counts.append((name, n))
    # 修正 rounding
    s = sum(n for _, n in counts)
    if s != total:
        diff = total - s
        last_name, last_n = counts[-1]
        counts[-1] = (last_name, last_n + diff)
    return counts


def generate_records(cfg: SynthConfig) -> List[Dict[str, Any]]:
    rng = random.Random(cfg.seed)
    names = _load_names()
    all_items = names["collection"] + names["firearms"] + names["equipment"] + names["ammo"]
    maps = names["map"] or ["零号大坝", "航天基地"]
    stations = ["技术中心", "工作台", "制药台", "防具台", "特勤处制造"]

    categories = _category_plan(cfg.num_samples)
    records: List[Dict[str, Any]] = []
    now = datetime.now(timezone.utc)
    idx = 1

    def mk_user_session(i: int) -> Tuple[str, str]:
        user = f"user_{(i % 120) + 1}"
        sess = f"{user}-s{(i % 8) + 1}"
        return user, sess

    for cat, count in categories:
        for _ in range(count):
            user_id, session_id = mk_user_session(idx)
            created_at = now - timedelta(minutes=(cfg.num_samples - idx))

            terminal = "success" if rng.random() < 0.88 else "partial"
            retry_count = 0 if terminal == "success" else (1 if rng.random() < 0.5 else 0)
            filter_decision = "hard_case" if rng.random() < 0.12 else "keep"

            if cat == "knowledge":
                item = rng.choice(all_items)
                query = rng.choice(
                    [
                        f"介绍一下{item}",
                        f"{item}是什么",
                        f"说说{item}的关键特点",
                        f"{item}有什么作用",
                    ]
                )
                intent = f"介绍{item}"
                flow_type = "simple"
                should_plan = False
                calls = [{"tool_name": "rag_knowledge_search", "tool_query": query}]
                available_tools = _pick_available_tools(rng, ["rag_knowledge_search"])
                plan_source = "skill_planning"
                memory = _memory_last_user(query)

            elif cat == "latest_price":
                item = rng.choice(all_items)
                query = rng.choice([f"{item}现在多少钱", f"{item}最新价格", f"查下{item}当前价格"])
                intent = f"查询{item}最新价格"
                flow_type = "simple"
                should_plan = False
                calls = [{"tool_name": "df_market_latest_price", "tool_query": f"objectName={item}"}]
                available_tools = _pick_available_tools(rng, ["df_market_latest_price"])
                plan_source = "skill_planning"
                memory = _memory_last_user(query)

            elif cat == "history_price":
                item = rng.choice(all_items)
                query = rng.choice([f"{item}最近一周历史价格", f"{item}历史区间", f"看下{item}走势"])
                intent = f"查询{item}历史价格"
                flow_type = "simple"
                should_plan = False
                calls = [{"tool_name": "df_market_history_price", "tool_query": f"objectName={item}"}]
                available_tools = _pick_available_tools(rng, ["df_market_history_price"])
                plan_source = "skill_planning"
                memory = _memory_last_user(query)

            elif cat == "price_advice":
                item = rng.choice(all_items)
                query = rng.choice([f"{item}现在适合买入吗", f"{item}现在建议买还是卖", f"{item}贵了还是便宜了"])
                intent = f"评估{item}买卖建议"
                flow_type = "complex"
                should_plan = False
                calls = [{"tool_name": "df_market_price_advice", "tool_query": f"objectName={item}"}]
                available_tools = _pick_available_tools(rng, ["df_market_price_advice"])
                plan_source = "skill_planning"
                memory = _memory_last_user(query)

            elif cat == "place_profit_rank":
                station = rng.choice(stations)
                query = rng.choice([f"{station}现在制造什么利润最高", f"{station}利润Top3", f"{station}制造推荐"])
                intent = f"查询{station}利润榜"
                flow_type = "simple"
                should_plan = False
                calls = [{"tool_name": "df_place_profit_rank", "tool_query": query}]
                available_tools = _pick_available_tools(rng, ["df_place_profit_rank"])
                plan_source = "skill_planning"
                memory = _memory_last_user(query)

            elif cat == "compare_single":
                e1, e2 = _pick_unique(rng, all_items, 2)
                query = rng.choice(
                    [
                        f"{e1}和{e2}对比",
                        f"{e1}、{e2}哪个好",
                        f"比较一下{e1}和{e2}",
                    ]
                )
                intent = f"对比{e1}与{e2}"
                flow_type = "complex"
                should_plan = True
                calls = [{"tool_name": "df_multi_item_compare", "tool_query": f"{e1}、{e2} 对比"}]
                available_tools = _pick_available_tools(rng, ["df_multi_item_compare"])
                plan_source = "fallback_task_planning"
                memory = _memory_last_user(query)

            elif cat == "profit_stability":
                item = rng.choice(all_items)
                query = rng.choice([f"{item}利润稳定性怎么样", f"{item}波动和回撤如何", f"{item}稳不稳"])
                intent = f"分析{item}稳定性"
                flow_type = "complex"
                should_plan = False
                calls = [{"tool_name": "df_profit_stability", "tool_query": f"objectName={item}"}]
                available_tools = _pick_available_tools(rng, ["df_profit_stability"])
                plan_source = "skill_planning"
                memory = _memory_last_user(query)

            elif cat == "answer_composer":
                item = rng.choice(all_items)
                query = rng.choice(
                    [
                        f"把{item}的资料和价格整理成一句结论",
                        f"把{item}的资料与买卖建议合成一个简短回答",
                        f"{item}相关信息请直接整理成最终答复",
                    ]
                )
                intent = f"综合回答{item}"
                flow_type = "complex"
                should_plan = False
                calls = [{"tool_name": "df_answer_composer", "tool_query": query}]
                available_tools = _pick_available_tools(
                    rng,
                    ["df_answer_composer"],
                    extra_min=3,
                    extra_max=5,
                )
                plan_source = "skill_planning"
                memory = _memory_last_user(query)

            elif cat == "multi_compare_advice":
                e1, e2 = _pick_unique(rng, all_items, 2)
                query = rng.choice(
                    [
                        "刚才对比的那两个里，哪个更适合买入，简短回答",
                        "基于刚才的两个物品，对比并给买入建议",
                        "对比后再给出买卖建议",
                    ]
                )
                intent = "对比后给买卖建议"
                flow_type = "complex"
                should_plan = True
                calls = [
                    {"tool_name": "df_multi_item_compare", "tool_query": f"{e1}、{e2} 对比"},
                    {"tool_name": "df_market_price_advice", "tool_query": f"objectName={e1}"},
                ]
                available_tools = _pick_available_tools(rng, ["df_multi_item_compare", "df_market_price_advice"])
                plan_source = "fallback_task_planning"
                memory = f"[最近对话]\n- 用户: 对比 {e1} 和 {e2}\n- 助手: 已给出对比结论"

            elif cat == "multi_profile_latest":
                item = rng.choice(all_items)
                query = rng.choice([f"介绍一下{item}并告诉我最新价格", f"先讲{item}再查实时价格", f"{item}资料和价格一起给我"])
                intent = f"资料+最新价:{item}"
                flow_type = "complex"
                should_plan = True
                calls = [
                    {"tool_name": "rag_knowledge_search", "tool_query": f"介绍一下{item}"},
                    {"tool_name": "df_market_latest_price", "tool_query": f"objectName={item}"},
                ]
                available_tools = _pick_available_tools(rng, ["rag_knowledge_search", "df_market_latest_price"])
                plan_source = "fallback_task_planning"
                memory = _memory_last_user(query)

            elif cat == "multi_profile_advice":
                item = rng.choice(all_items)
                query = rng.choice([f"介绍下{item}，并判断现在是否值得买", f"先讲{item}再给买卖建议", f"{item}资料和建议都要"])
                intent = f"资料+建议:{item}"
                flow_type = "complex"
                should_plan = True
                calls = [
                    {"tool_name": "rag_knowledge_search", "tool_query": f"介绍一下{item}"},
                    {"tool_name": "df_market_price_advice", "tool_query": f"objectName={item}"},
                ]
                available_tools = _pick_available_tools(rng, ["rag_knowledge_search", "df_market_price_advice"])
                plan_source = "fallback_task_planning"
                memory = _memory_last_user(query)

            elif cat == "multi_rank_advice":
                station = rng.choice(stations)
                item = rng.choice(all_items)
                query = rng.choice([f"{station}利润榜第一是什么，并给我买卖建议", f"先看{station}利润排名，再分析{item}是否适合买入"])
                intent = f"利润榜+建议:{station}"
                flow_type = "complex"
                should_plan = True
                calls = [
                    {"tool_name": "df_place_profit_rank", "tool_query": f"{station} 利润Top3"},
                    {"tool_name": "df_market_price_advice", "tool_query": f"objectName={item}"},
                ]
                available_tools = _pick_available_tools(rng, ["df_place_profit_rank", "df_market_price_advice"])
                plan_source = "fallback_task_planning"
                memory = f"[最近对话]\n- 用户: 查询{station}利润排行"

            elif cat == "multi_profile_history_advice":
                item = rng.choice(all_items)
                query = rng.choice(
                    [
                        f"先介绍{item}，再看走势，最后告诉我是否适合买",
                        f"{item}的资料、历史价格和买卖建议都给我",
                        f"讲清楚{item}，看下走势，再判断值不值得买",
                    ]
                )
                intent = f"资料+走势+建议:{item}"
                flow_type = "complex"
                should_plan = True
                calls = [
                    {"tool_name": "rag_knowledge_search", "tool_query": f"介绍一下{item}"},
                    {"tool_name": "df_market_history_price", "tool_query": f"objectName={item}"},
                    {"tool_name": "df_market_price_advice", "tool_query": f"objectName={item}"},
                ]
                available_tools = _pick_available_tools(
                    rng,
                    ["rag_knowledge_search", "df_market_history_price", "df_market_price_advice"],
                )
                plan_source = "fallback_task_planning"
                memory = _memory_last_user(query)

            elif cat == "retry_latest":
                item = rng.choice(all_items)
                previous_query = rng.choice([f"查下{item}价格", f"{item}当前价格是多少"])
                query = rng.choice([f"刚才没查到，再查一次{item}最新价格", f"重新查一下{item}现在多少钱"])
                intent = f"重试查询{item}最新价格"
                flow_type = "complex"
                should_plan = False
                terminal = "success" if rng.random() < 0.75 else "partial"
                retry_count = 1
                calls = [{"tool_name": "df_market_latest_price", "tool_query": f"objectName={item}"}]
                available_tools = _pick_available_tools(rng, ["df_market_latest_price"])
                plan_source = "retry_router"
                memory = _memory_retry(previous_query, query)

            elif cat == "followup_compare_advice":
                e1, e2 = _pick_unique(rng, all_items, 2)
                chosen = rng.choice([e1, e2])
                query = rng.choice(
                    [
                        f"按刚才的对比结果，只看{chosen}的话现在值得买吗",
                        f"基于刚才对比，给{chosen}一个买卖建议",
                        f"不重新对比了，直接判断{chosen}现在是否适合买入",
                    ]
                )
                intent = f"基于历史对比给{chosen}建议"
                flow_type = "complex"
                should_plan = False
                calls = [{"tool_name": "df_market_price_advice", "tool_query": f"objectName={chosen}"}]
                available_tools = _pick_available_tools(
                    rng,
                    ["df_market_price_advice", "df_multi_item_compare"],
                )
                plan_source = "skill_planning"
                memory = (
                    "[最近对话]\n"
                    f"- 用户: 对比 {e1} 和 {e2}\n"
                    "- 助手: 已给出对比结论\n"
                    f"- 用户: {query}"
                )

            else:
                # 兜底
                item = rng.choice(maps)
                query = f"介绍一下{item}"
                intent = f"介绍{item}"
                flow_type = "simple"
                should_plan = False
                calls = [{"tool_name": "rag_knowledge_search", "tool_query": query}]
                available_tools = _pick_available_tools(rng, ["rag_knowledge_search"])
                plan_source = "skill_planning"
                memory = _memory_last_user(query)

            record = _build_record(
                idx=idx,
                user_id=user_id,
                session_id=session_id,
                created_at=created_at,
                query=query,
                intent=intent,
                flow_type=flow_type,
                should_plan=should_plan,
                tool_calls=calls,
                available_tools=available_tools,
                plan_source=plan_source,
                memory_context=memory,
                terminal_status=terminal,
                retry_count_total=retry_count,
                filter_decision=filter_decision,
                rng=rng,
            )
            records.append(record)
            idx += 1
    return records


def split_records(records: List[Dict[str, Any]], seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in records:
        prompt = row.get("prompt", {}) or {}
        key = str(prompt.get("user_query", "") or row.get("fingerprint", ""))
        groups.setdefault(key, []).append(row)
    grouped_rows = list(groups.values())
    rng.shuffle(grouped_rows)
    rows = [row for group in grouped_rows for row in group]
    n = len(rows)
    n_train = int(n * TRAIN_RATIO)
    n_dev = int(n * DEV_RATIO)
    train = rows[:n_train]
    dev = rows[n_train : n_train + n_dev]
    test = rows[n_train + n_dev :]
    return train, dev, test


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    cfg = parse_args()
    records = generate_records(cfg)
    train, dev, test = split_records(records, cfg.seed)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(cfg.output_dir / "train.jsonl", train)
    write_jsonl(cfg.output_dir / "dev.jsonl", dev)
    write_jsonl(cfg.output_dir / "test.jsonl", test)

    def _count_multi(rows: List[Dict[str, Any]]) -> int:
        return sum(1 for r in rows if len(((r.get("response", {}) or {}).get("tool_calls", []) or [])) > 1)

    summary = {
        "output_dir": str(cfg.output_dir),
        "num_samples": len(records),
        "split": {"train": len(train), "dev": len(dev), "test": len(test)},
        "should_plan_count": sum(1 for r in records if bool((r.get("prompt", {}) or {}).get("requires_task_planning", False))),
        "multi_tool_count": _count_multi(records),
        "three_tool_count": sum(
            1 for r in records if len(((r.get("response", {}) or {}).get("tool_calls", []) or [])) >= 3
        ),
        "seed": cfg.seed,
    }
    (cfg.output_dir / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
