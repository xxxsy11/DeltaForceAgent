#!/usr/bin/env python3
"""
将知识图谱JSON数据转换为CSV格式，用于Neo4j导入

使用方法:
  python json_to_csv.py
"""

import json
import csv
import os
from collections import OrderedDict
from pathlib import Path


LIST_DELIM = os.getenv("LIST_DELIM", "|")


def normalize_labels(labels):
    """规范化标签"""
    return ";".join(labels)


def normalize_list(value):
    """规范化列表"""
    return LIST_DELIM.join(str(v) for v in value)


def normalize_props(props):
    """规范化属性"""
    normalized = dict(props)
    if "effect" in normalized and "effects" not in normalized:
        normalized["effects"] = normalized.pop("effect")
    for k, v in list(normalized.items()):
        if isinstance(v, list):
            normalized[k] = normalize_list(v)
    return normalized


def collect_node_fields(nodes):
    """收集所有节点字段"""
    fields = OrderedDict()
    fields["nodeId"] = True
    fields[":LABEL"] = True
    for n in nodes:
        props = normalize_props(n.get("props", {}))
        for k in props.keys():
            fields[k] = True
    return list(fields.keys())


def collect_rel_fields(rels):
    """收集所有关系字段"""
    fields = OrderedDict()
    fields[":START_ID"] = True
    fields[":END_ID"] = True
    fields[":TYPE"] = True
    for r in rels:
        props = normalize_props(r.get("props", {}))
        for k in props.keys():
            fields[k] = True
    return list(fields.keys())


def main():
    """主函数"""
    base = Path(__file__).resolve().parent.parent
    src = base / "data" / "neo4j" / "data.json"
    out_nodes = base / "data" / "neo4j" / "nodes.csv"
    out_rels = base / "data" / "neo4j" / "relationships.csv"

    with src.open("r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    rels = data.get("relationships", [])

    node_fields = collect_node_fields(nodes)
    rel_fields = collect_rel_fields(rels)

    with out_nodes.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=node_fields)
        w.writeheader()
        for n in nodes:
            row = {
                "nodeId": n["id"],
                ":LABEL": normalize_labels(n.get("labels", []))
            }
            row.update(normalize_props(n.get("props", {})))
            w.writerow(row)

    with out_rels.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rel_fields)
        w.writeheader()
        for r in rels:
            row = {
                ":START_ID": r["from"],
                ":END_ID": r["to"],
                ":TYPE": r["type"]
            }
            row.update(normalize_props(r.get("props", {})))
            w.writerow(row)

    print("CSV文件已生成:")
    print(f"  {out_nodes}")
    print(f"  {out_rels}")


if __name__ == "__main__":
    main()
