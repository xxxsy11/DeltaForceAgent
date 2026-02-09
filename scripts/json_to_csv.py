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
from typing import List, Dict


LIST_DELIM = os.getenv("LIST_DELIM", "|")

# JSON 数据目录
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "neo4j"

# JSON 文件列表
JSON_FILES = [
    "map.json",
    "operator.json",
    "ammo.json",
    "equipment.json",
    "firearms.json",
    "attachments.json",
    "collection.json",
]


def load_json_files() -> tuple[List[Dict], List[Dict]]:
    """读取所有JSON文件"""
    all_nodes = []
    all_rels = []

    for fname in JSON_FILES:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            print(f"  跳过: {fname}")
            continue

        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        nodes = data.get('nodes', [])
        rels = data.get('relationships', [])

        all_nodes.extend(nodes)
        all_rels.extend(rels)

        print(f"  {fname}: {len(nodes)} 节点, {len(rels)} 关系")

    return all_nodes, all_rels


def normalize_labels(labels):
    return ";".join(labels)


def normalize_list(value):
    return LIST_DELIM.join(str(v) for v in value)


def normalize_props(props):
    normalized = dict(props)
    if "effect" in normalized and "effects" not in normalized:
        normalized["effects"] = normalized.pop("effect")
    for k, v in list(normalized.items()):
        if isinstance(v, list):
            normalized[k] = normalize_list(v)
        elif isinstance(v, dict):
            normalized[k] = json.dumps(v, ensure_ascii=False)
    return normalized


def collect_node_fields(nodes):
    fields = OrderedDict()
    fields["nodeId"] = True
    fields[":LABEL"] = True
    for n in nodes:
        props = normalize_props(n.get("props", {}))
        for k in props.keys():
            fields[k] = True
    return list(fields.keys())


def collect_rel_fields(rels):
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
    print("读取数据...")
    nodes, rels = load_json_files()

    if not nodes:
        print("错误: 没有找到任何节点数据")
        return

    print(f"\n总计: {len(nodes)} 节点, {len(rels)} 关系")

    out_dir = DATA_DIR / "csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_nodes = out_dir / "nodes.csv"
    out_rels = out_dir / "relationships.csv"

    node_fields = collect_node_fields(nodes)
    rel_fields = collect_rel_fields(rels)

    with out_nodes.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=node_fields, extrasaction='ignore')
        w.writeheader()
        for n in nodes:
            row = {
                "nodeId": n["id"],
                ":LABEL": normalize_labels(n.get("labels", []))
            }
            row.update(normalize_props(n.get("props", {})))
            w.writerow(row)

    with out_rels.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rel_fields, extrasaction='ignore')
        w.writeheader()
        for r in rels:
            row = {
                ":START_ID": r["from"],
                ":END_ID": r["to"],
                ":TYPE": r["type"]
            }
            row.update(normalize_props(r.get("props", {})))
            w.writerow(row)

    print("\nCSV文件已生成:")
    print(f"  {out_nodes}")
    print(f"  {out_rels}")


if __name__ == "__main__":
    main()
