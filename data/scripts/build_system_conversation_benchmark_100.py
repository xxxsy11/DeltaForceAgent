#!/usr/bin/env python3
"""构建高丰富度的 100 条 conversation-level benchmark。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


VALID_TOOLS = [
    "rag_knowledge_search",
    "df_market_latest_price",
    "df_market_history_price",
    "df_market_price_advice",
    "df_place_profit_rank",
    "df_multi_item_compare",
    "df_profit_stability",
    "df_answer_composer",
]


def _load_pools(root: Path) -> Dict[str, List[str]]:
    pools: Dict[str, List[str]] = {
        "Collectible": [],
        "Firearm": [],
        "Attachment": [],
        "Ammo": [],
        "Operator": [],
        "Map": [],
        "Equipment": [],
    }
    for fp in sorted((root / "data/neo4j").glob("*.json")):
        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        nodes = payload.get("nodes")
        if not isinstance(nodes, list):
            continue

        for node in nodes:
            if not isinstance(node, dict):
                continue
            labels = node.get("labels") or []
            props = node.get("props") or {}
            name = props.get("name") or props.get("idname") or props.get("typeName")
            if not isinstance(name, str):
                continue
            v = name.strip()
            if not v:
                continue
            if len(v) > 32:
                continue
            if v in {"一级", "二级", "三级", "四级", "五级", "六级"}:
                continue
            if any(x in v for x in ["普通技能", "终极技能", "被动技能"]):
                continue
            for lab in labels:
                if lab in pools:
                    pools[lab].append(v)

    for key, vals in pools.items():
        seen = set()
        uniq = []
        for x in vals:
            if x in seen:
                continue
            seen.add(x)
            uniq.append(x)
        pools[key] = uniq
    return pools


def _pick(pool: List[str], idx: int, fallback: str) -> str:
    if not pool:
        return fallback
    return pool[idx % len(pool)]


def _turn(
    query: str,
    expected_tool: str,
    expected_skill: str,
    expected_intents: List[str],
    expected_entities: List[str],
    expected_tool_query_contains: List[str],
    answer_keywords: List[str],
    complexity: str,
    expect_memory_resolution: bool = False,
    expect_persistent_recall: bool = False,
    expect_requires_task_planning: bool = False,
    expected_tool_chain: List[str] | None = None,
) -> Dict:
    chain = expected_tool_chain or []
    return {
        "query": query,
        "complexity": complexity,
        "expected_intents": expected_intents,
        "expected_tool": expected_tool,
        "expected_tool_candidates": [expected_tool],
        "expected_skill": expected_skill,
        "expected_entities": expected_entities,
        "expected_tool_query_contains": expected_tool_query_contains,
        "answer_keywords": answer_keywords,
        "expect_memory_resolution": expect_memory_resolution,
        "expect_persistent_recall": expect_persistent_recall,
        "expect_requires_task_planning": expect_requires_task_planning,
        "expected_tool_chain": chain,
    }


def build_cases(root: Path) -> List[Dict]:
    pools = _load_pools(root)

    market_pool = (pools["Collectible"][:80] + pools["Firearm"][:50] + pools["Attachment"][:50])
    if len(market_pool) < 20:
        market_pool += ["非洲之心", "海洋之泪", "主战坦克模型", "QBZ95-1突击步枪", "腾龙突击步枪"]

    ammo_pool = pools["Ammo"][:60] or ["5.56x45mm M855A1", "9x39mm SP6", "碳纤维散射箭矢"]
    knowledge_pool = (pools["Operator"][:14] + pools["Map"][:5] + pools["Equipment"][:40])
    if len(knowledge_pool) < 10:
        knowledge_pool += ["金卢娜", "长弓溪谷", "H70 精英头盔"]

    place_q = [
        "特勤处制造什么子弹利润最高",
        "特勤处四大分组利润top3",
        "工作台现在什么子弹收益最高",
        "技术中心枪械配件利润top3",
        "防具台制造什么利润高",
        "制药台什么最赚钱",
        "特勤处制造利润榜top1",
        "给我制造台利润排行",
    ]

    intro_tpl = ["介绍一下{x}", "给我讲讲{x}", "{x}是什么", "说说{x}的关键特点"]
    latest_tpl = ["它现在什么价格", "它当前价格是多少", "那它最新报价呢", "它现在值多少钱"]
    history_tpl = ["它的历史价格", "再查下它历史价格", "给我看它历史价格区间", "它过去价格走势如何"]
    advice_tpl = ["那现在建议买吗", "它现在贵了还是便宜了", "它现在适合卖出吗", "给我它的买卖建议"]
    composer_tpl = [
        "再介绍一下{x}并告诉我现在价格",
        "给我{x}资料和实时价格",
        "分析{x}并给出当前报价",
        "说说{x}，再告诉我最新价格",
    ]
    cmp_tpl = ["对比一下{a}和{b}", "比较{a}和{b}", "{a}与{b}哪个更值得买", "分析对比{a}、{b}"]
    cmp_ref_tpl = ["再对比一下这两个，给买入建议", "这两个现在谁更适合买入", "这两者再做个简短对比"]
    add3_tpl = ["再加上{c}，三个一起对比", "把{c}也加进来，三者对比", "把{c}纳入后重排优先级"]
    stable_tpl = ["分析{x}利润稳定性", "{x}利润波动怎么样", "给我{x}稳定性结论"]
    know2_tpl = ["顺便介绍一下{x}", "再讲一下{x}", "补充说明{x}"]

    complex_chain_q_tpl = [
        "请先介绍{a}，再给出{a}和{b}对比，并结合最新价格给出买入建议",
        "先查询{a}当前价格，再比较{a}和{b}，最后一句话给建议",
        "围绕{a}和{b}做完整分析：资料、价格、对比、建议",
    ]

    reentry_cmp_tpl = [
        "刚才对比的那两个里，哪个更适合买入，简短回答",
        "上一个会话里那两个物品，当前优先买哪个",
        "延续刚才比较，这两者现在谁更优",
    ]
    reentry_first_tpl = [
        "那第一个物品现在价格呢",
        "刚才第一个现在多少钱",
        "第一个当前报价是多少",
    ]
    reentry_advice_tpl = [
        "再给出一句买卖建议",
        "基于刚才结论再补一句建议",
        "最后给一条操作建议",
    ]

    cases: List[Dict] = []
    for i in range(100):
        a = _pick(market_pool, i * 5, "非洲之心")
        b = _pick(market_pool, i * 5 + 1, "海洋之泪")
        c = _pick(market_pool, i * 5 + 2, "主战坦克模型")
        if b == a:
            b = _pick(market_pool, i * 5 + 7, "海洋之泪")
        if c in {a, b}:
            c = _pick(market_pool, i * 5 + 13, "主战坦克模型")

        ammo = _pick(ammo_pool, i, "5.56x45mm M855A1")
        kx = _pick(knowledge_pool, i, "金卢娜")

        turns_s1 = [
            _turn(
                query=intro_tpl[i % len(intro_tpl)].format(x=a),
                expected_tool="rag_knowledge_search",
                expected_skill="knowledge_profile",
                expected_intents=["knowledge_query", "general_query"],
                expected_entities=[a],
                expected_tool_query_contains=[a],
                answer_keywords=[a],
                complexity="simple",
            ),
            _turn(
                query=latest_tpl[i % len(latest_tpl)],
                expected_tool="df_market_latest_price",
                expected_skill="market_latest_price",
                expected_intents=["market_price_latest_query"],
                expected_entities=[a],
                expected_tool_query_contains=[a],
                answer_keywords=["价格"],
                complexity="simple",
                expect_memory_resolution=True,
            ),
            _turn(
                query=history_tpl[i % len(history_tpl)],
                expected_tool="df_market_history_price",
                expected_skill="market_history_price",
                expected_intents=["market_price_history_query"],
                expected_entities=[a],
                expected_tool_query_contains=[a],
                answer_keywords=["历史价格", "区间"],
                complexity="simple",
                expect_memory_resolution=True,
            ),
            _turn(
                query=advice_tpl[i % len(advice_tpl)],
                expected_tool="df_market_price_advice",
                expected_skill="market_price_advice",
                expected_intents=["market_price_advice_query"],
                expected_entities=[a],
                expected_tool_query_contains=[a],
                answer_keywords=["建议"],
                complexity="complex",
                expect_memory_resolution=True,
            ),
            _turn(
                query=composer_tpl[i % len(composer_tpl)].format(x=b),
                expected_tool="df_answer_composer",
                expected_skill="answer_composer",
                expected_intents=["answer_composer_query"],
                expected_entities=[b],
                expected_tool_query_contains=[b],
                answer_keywords=[b, "价格"],
                complexity="complex",
            ),
            _turn(
                query=cmp_tpl[i % len(cmp_tpl)].format(a=a, b=b),
                expected_tool="df_multi_item_compare",
                expected_skill="market_multi_item_compare",
                expected_intents=["market_compare_query"],
                expected_entities=[a, b],
                expected_tool_query_contains=[a, b],
                answer_keywords=["对比"],
                complexity="complex",
                expect_requires_task_planning=True,
                expected_tool_chain=["df_multi_item_compare"],
            ),
            _turn(
                query=cmp_ref_tpl[i % len(cmp_ref_tpl)],
                expected_tool="df_multi_item_compare",
                expected_skill="market_multi_item_compare",
                expected_intents=["market_compare_query"],
                expected_entities=[a, b],
                expected_tool_query_contains=[a, b],
                answer_keywords=["对比", "买入"],
                complexity="complex",
                expect_memory_resolution=True,
                expect_requires_task_planning=True,
                expected_tool_chain=["df_multi_item_compare"],
            ),
            _turn(
                query=add3_tpl[i % len(add3_tpl)].format(c=c),
                expected_tool="df_multi_item_compare",
                expected_skill="market_multi_item_compare",
                expected_intents=["market_compare_query"],
                expected_entities=[a, b, c],
                expected_tool_query_contains=[a, b],
                answer_keywords=["对比"],
                complexity="complex",
                expect_requires_task_planning=True,
                expected_tool_chain=["df_multi_item_compare"],
            ),
            _turn(
                query=stable_tpl[i % len(stable_tpl)].format(x=ammo),
                expected_tool="df_profit_stability",
                expected_skill="profit_stability",
                expected_intents=["profit_stability_query"],
                expected_entities=[ammo],
                expected_tool_query_contains=[ammo],
                answer_keywords=["稳定", "利润"],
                complexity="complex",
            ),
            _turn(
                query=place_q[i % len(place_q)],
                expected_tool="df_place_profit_rank",
                expected_skill="place_profit_rank",
                expected_intents=["place_profit_query"],
                expected_entities=[],
                expected_tool_query_contains=[],
                answer_keywords=["利润"],
                complexity="simple",
            ),
            _turn(
                query=know2_tpl[i % len(know2_tpl)].format(x=kx),
                expected_tool="rag_knowledge_search",
                expected_skill="knowledge_profile",
                expected_intents=["knowledge_query", "general_query"],
                expected_entities=[kx],
                expected_tool_query_contains=[kx],
                answer_keywords=[kx],
                complexity="simple",
            ),
            _turn(
                query=complex_chain_q_tpl[i % len(complex_chain_q_tpl)].format(a=a, b=b),
                expected_tool="df_answer_composer",
                expected_skill="answer_composer",
                expected_intents=["answer_composer_query", "market_compare_query", "market_price_advice_query"],
                expected_entities=[a, b],
                expected_tool_query_contains=[a, b],
                answer_keywords=["建议", "对比"],
                complexity="complex",
                expect_requires_task_planning=True,
                expected_tool_chain=[
                    "rag_knowledge_search",
                    "df_market_latest_price",
                    "df_multi_item_compare",
                    "df_market_price_advice",
                ],
            ),
        ]

        turns_s2 = [
            _turn(
                query=reentry_cmp_tpl[i % len(reentry_cmp_tpl)],
                expected_tool="df_multi_item_compare",
                expected_skill="market_multi_item_compare",
                expected_intents=["market_compare_query"],
                expected_entities=[a, b],
                expected_tool_query_contains=[a, b],
                answer_keywords=["买入", "对比"],
                complexity="complex",
                expect_memory_resolution=True,
                expect_persistent_recall=True,
                expect_requires_task_planning=True,
                expected_tool_chain=["df_multi_item_compare"],
            ),
            _turn(
                query=reentry_first_tpl[i % len(reentry_first_tpl)],
                expected_tool="df_market_latest_price",
                expected_skill="market_latest_price",
                expected_intents=["market_price_latest_query"],
                expected_entities=[a],
                expected_tool_query_contains=[a],
                answer_keywords=["价格"],
                complexity="simple",
                expect_memory_resolution=True,
                expect_persistent_recall=True,
            ),
            _turn(
                query=reentry_advice_tpl[i % len(reentry_advice_tpl)],
                expected_tool="df_market_price_advice",
                expected_skill="market_price_advice",
                expected_intents=["market_price_advice_query"],
                expected_entities=[a],
                expected_tool_query_contains=[a],
                answer_keywords=["建议"],
                complexity="complex",
                expect_memory_resolution=True,
                expect_persistent_recall=True,
            ),
        ]

        cases.append(
            {
                "case_id": f"case_{i+1:03d}",
                "user_id": f"user_{i+1}",
                "sessions": [
                    {"session_id": f"case_{i+1:03d}_s1", "turns": turns_s1},
                    {"session_id": f"case_{i+1:03d}_s2", "turns": turns_s2},
                ],
            }
        )

    return cases


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root / "data/benchmarks/system_conversation_benchmark_100.json"
    payload = {"cases": build_cases(root)}

    # 轻量合法性检查
    for case in payload["cases"]:
        assert case["user_id"].startswith("user_")
        for session in case["sessions"]:
            for turn in session["turns"]:
                assert turn["expected_tool"] in VALID_TOOLS
                for t in turn.get("expected_tool_chain", []) or []:
                    assert t in VALID_TOOLS
                assert turn["complexity"] in {"simple", "complex"}

    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote: {out}")
    print(f"[ok] cases: {len(payload['cases'])}")


if __name__ == "__main__":
    main()
