#!/usr/bin/env python3
"""
Delta Agent 数据处理统一脚本。

整合能力：
- 统计 JSON 图数据
- JSON -> CSV（neo4j-admin/import 可用）
- JSON -> Cypher 导入素材
- 直接导入 Neo4j
- 清空 Neo4j
- 重建 Neo4j（清空 + 导入）
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "neo4j"
DEFAULT_CSV_DIR = DEFAULT_DATA_DIR / "csv"
DEFAULT_CYPHER_DIR = DEFAULT_DATA_DIR / "cypher"
DEFAULT_JSON_FILES = [
    "map.json",
    "operator.json",
    "ammo.json",
    "equipment.json",
    "firearms.json",
    "attachments.json",
    "collection.json",
]

LOCAL_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
REMOTE_URI = os.getenv("NEO4J_REMOTE_URI", "bolt://58.199.146.145:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "all-in-rag")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

SAFE_TOKEN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_json_files(value: str | None) -> List[str]:
    if not value:
        return list(DEFAULT_JSON_FILES)
    return [x.strip() for x in value.split(",") if x.strip()]


def safe_token(value: str, fallback: str) -> str:
    if value and SAFE_TOKEN_RE.fullmatch(value):
        return value
    return fallback


def normalize_props(props: Dict[str, Any], list_delim: str = "|") -> Dict[str, Any]:
    normalized = dict(props or {})
    if "effect" in normalized and "effects" not in normalized:
        normalized["effects"] = normalized.pop("effect")
    for key, value in list(normalized.items()):
        if isinstance(value, list):
            normalized[key] = list_delim.join(str(v) for v in value)
        elif isinstance(value, dict):
            normalized[key] = json.dumps(value, ensure_ascii=False)
    return normalized


def load_json_graph(data_dir: Path, json_files: Iterable[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    all_nodes: List[Dict[str, Any]] = []
    all_relationships: List[Dict[str, Any]] = []

    print(f"📖 数据目录: {data_dir}")
    for name in json_files:
        file_path = data_dir / name
        if not file_path.exists():
            print(f"⚠️  缺失文件，已跳过: {file_path}")
            continue
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        nodes = data.get("nodes", [])
        relationships = data.get("relationships", [])
        all_nodes.extend(nodes)
        all_relationships.extend(relationships)
        print(f"  - {name}: 节点 {len(nodes)}，关系 {len(relationships)}")

    return all_nodes, all_relationships


def show_stats(nodes: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> None:
    print("\n📊 图数据统计")
    print(f"  节点总数: {len(nodes)}")
    print(f"  关系总数: {len(relationships)}")

    label_counter: Dict[str, int] = defaultdict(int)
    rel_counter: Dict[str, int] = defaultdict(int)

    for node in nodes:
        for label in node.get("labels", []):
            label_counter[label] += 1
    for rel in relationships:
        rel_counter[rel.get("type", "RELATED_TO")] += 1

    if label_counter:
        print("  节点类型 Top10:")
        for label, count in sorted(label_counter.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"    - {label}: {count}")
    if rel_counter:
        print("  关系类型 Top10:")
        for rel_type, count in sorted(rel_counter.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"    - {rel_type}: {count}")


def collect_node_fields(nodes: List[Dict[str, Any]]) -> List[str]:
    fields: "OrderedDict[str, bool]" = OrderedDict()
    fields["nodeId"] = True
    fields[":LABEL"] = True
    for node in nodes:
        props = normalize_props(node.get("props", {}))
        for key in props.keys():
            fields[key] = True
    return list(fields.keys())


def collect_rel_fields(rels: List[Dict[str, Any]]) -> List[str]:
    fields: "OrderedDict[str, bool]" = OrderedDict()
    fields[":START_ID"] = True
    fields[":END_ID"] = True
    fields[":TYPE"] = True
    for rel in rels:
        props = normalize_props(rel.get("props", {}))
        for key in props.keys():
            fields[key] = True
    return list(fields.keys())


def export_csv(nodes: List[Dict[str, Any]], rels: List[Dict[str, Any]], output_dir: Path, list_delim: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    nodes_csv = output_dir / "nodes.csv"
    rels_csv = output_dir / "relationships.csv"

    node_fields = collect_node_fields(nodes)
    rel_fields = collect_rel_fields(rels)

    with nodes_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=node_fields, extrasaction="ignore")
        writer.writeheader()
        for node in nodes:
            row = {
                "nodeId": node.get("id"),
                ":LABEL": ";".join(node.get("labels", [])),
            }
            row.update(normalize_props(node.get("props", {}), list_delim=list_delim))
            writer.writerow(row)

    with rels_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rel_fields, extrasaction="ignore")
        writer.writeheader()
        for rel in rels:
            row = {
                ":START_ID": rel.get("from"),
                ":END_ID": rel.get("to"),
                ":TYPE": rel.get("type", "RELATED_TO"),
            }
            row.update(normalize_props(rel.get("props", {}), list_delim=list_delim))
            writer.writerow(row)

    print(f"\n✅ CSV 导出完成: {output_dir}")
    print(f"  - {nodes_csv}")
    print(f"  - {rels_csv}")


def export_cypher_assets(
    nodes: List[Dict[str, Any]],
    rels: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    nodes_csv = output_dir / "nodes.csv"
    rels_csv = output_dir / "relationships.csv"
    cypher_script = output_dir / "delta_import.cypher"

    with nodes_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["nodeId", "labels", "name"],
        )
        writer.writeheader()
        for node in nodes:
            props = node.get("props", {})
            writer.writerow(
                {
                    "nodeId": node.get("id"),
                    "labels": node.get("labels", ["Node"])[0],
                    "name": props.get("name", node.get("id")),
                }
            )

    with rels_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["relationshipId", "relationshipType", "startNodeId", "endNodeId"],
        )
        writer.writeheader()
        for idx, rel in enumerate(rels):
            writer.writerow(
                {
                    "relationshipId": f"rel_{idx:06d}",
                    "relationshipType": rel.get("type", "RELATED_TO"),
                    "startNodeId": rel.get("from"),
                    "endNodeId": rel.get("to"),
                }
            )

    cypher_content = """// Delta Agent Neo4j Import Script
CREATE CONSTRAINT node_id_unique IF NOT EXISTS FOR (n:Node) REQUIRE n.nodeId IS UNIQUE;
CREATE FULLTEXT INDEX node_fulltext_search IF NOT EXISTS FOR (n:Node) ON EACH [n.name];

LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
CALL {
  WITH row
  CALL apoc.create.node([row.labels, 'Node'], {nodeId: row.nodeId, name: row.name}) YIELD node
  RETURN node
} IN TRANSACTIONS OF 1000 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///relationships.csv' AS row
CALL {
  WITH row
  MATCH (source:Node {nodeId: row.startNodeId})
  MATCH (target:Node {nodeId: row.endNodeId})
  CALL apoc.create.relationship(
    source,
    row.relationshipType,
    {relationshipId: row.relationshipId},
    target
  ) YIELD rel
  RETURN rel
} IN TRANSACTIONS OF 1000 ROWS;
"""
    cypher_script.write_text(cypher_content, encoding="utf-8")

    print(f"\n✅ Cypher 导入素材导出完成: {output_dir}")
    print(f"  - {nodes_csv}")
    print(f"  - {rels_csv}")
    print(f"  - {cypher_script}")


def connect_neo4j(uri: str, user: str, password: str):
    try:
        from neo4j import GraphDatabase  # type: ignore
    except ImportError as e:
        raise RuntimeError("缺少 neo4j 依赖，请先安装 neo4j Python 包。") from e
    return GraphDatabase.driver(uri, auth=(user, password))


def create_constraints(session) -> None:
    statements = [
        "CREATE CONSTRAINT node_id_unique IF NOT EXISTS FOR (n:Node) REQUIRE n.nodeId IS UNIQUE",
        "CREATE INDEX map_name_index IF NOT EXISTS FOR (m:Map) ON (m.name)",
        "CREATE INDEX area_name_index IF NOT EXISTS FOR (a:Area) ON (a.name)",
        "CREATE INDEX equipment_name_index IF NOT EXISTS FOR (e:Equipment) ON (e.name)",
        "CREATE INDEX firearm_name_index IF NOT EXISTS FOR (f:Firearm) ON (f.name)",
        "CREATE INDEX attachment_name_index IF NOT EXISTS FOR (a:Attachment) ON (a.name)",
        "CREATE INDEX ammo_name_index IF NOT EXISTS FOR (a:Ammo) ON (a.name)",
        "CREATE INDEX collectible_name_index IF NOT EXISTS FOR (c:Collectible) ON (c.name)",
        "CREATE INDEX operator_name_index IF NOT EXISTS FOR (o:Operator) ON (o.name)",
        "CREATE FULLTEXT INDEX node_fulltext_search IF NOT EXISTS FOR (n:Node) ON EACH [n.name]",
    ]
    for statement in statements:
        session.run(statement).consume()


def prepare_node_props(node: Dict[str, Any]) -> Dict[str, Any]:
    props = dict(node.get("props", {}))
    display_name = (
        props.get("name")
        or props.get("typeName")
        or props.get("difficulty")
        or props.get("level")
        or props.get("colorName")
        or props.get("conceptType")
        or props.get("category")
        or node.get("id")
    )
    for key, value in list(props.items()):
        if isinstance(value, dict):
            props[key] = json.dumps(value, ensure_ascii=False)
    props["name"] = str(display_name)
    return props


def import_nodes(session, nodes: List[Dict[str, Any]], batch_size: int = 300) -> None:
    grouped_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        label = safe_token(node.get("labels", ["Node"])[0], "Node")
        grouped_rows[label].append(
            {
                "nodeId": node.get("id"),
                "props": prepare_node_props(node),
            }
        )

    for label, rows in grouped_rows.items():
        query = f"""
        UNWIND $rows AS row
        MERGE (n:Node:{label} {{nodeId: row.nodeId}})
        SET n += row.props
        """
        for idx in range(0, len(rows), batch_size):
            session.run(query, rows=rows[idx : idx + batch_size]).consume()


def import_relationships(session, relationships: List[Dict[str, Any]], batch_size: int = 500) -> None:
    grouped_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for idx, rel in enumerate(relationships):
        rel_type = safe_token(rel.get("type", "RELATED_TO"), "RELATED_TO")
        grouped_rows[rel_type].append(
            {
                "sourceId": rel.get("from"),
                "targetId": rel.get("to"),
                "props": {
                    "relationshipId": f"rel_{idx:06d}",
                    **normalize_props(rel.get("props", {})),
                },
            }
        )

    for rel_type, rows in grouped_rows.items():
        query = f"""
        UNWIND $rows AS row
        MATCH (source:Node {{nodeId: row.sourceId}})
        MATCH (target:Node {{nodeId: row.targetId}})
        MERGE (source)-[r:{rel_type}]->(target)
        SET r += row.props
        """
        for idx in range(0, len(rows), batch_size):
            session.run(query, rows=rows[idx : idx + batch_size]).consume()


def db_stats(session) -> Tuple[int, int]:
    node_count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    return int(node_count), int(rel_count)


def wipe_neo4j(session, execute: bool, dry_run: bool) -> None:
    node_count, rel_count = db_stats(session)
    print(f"📊 当前数据库: 节点 {node_count}, 关系 {rel_count}")
    if node_count == 0 and rel_count == 0:
        print("✅ 数据库已为空")
        return

    if dry_run:
        print("🧪 dry-run 模式：未执行删除。")
        return

    if not execute:
        confirm = input("确认删除全部数据？输入 'yes' 继续: ")
        if confirm.strip().lower() != "yes":
            print("❌ 已取消")
            return

    summary = session.run("MATCH (n) DETACH DELETE n").consume()
    print(
        "✅ 清空完成: "
        f"删除节点 {summary.counters.nodes_deleted}, "
        f"删除关系 {summary.counters.relationships_deleted}"
    )


def resolve_uri(uri: str | None, remote: bool) -> str:
    if uri:
        return uri
    return REMOTE_URI if remote else LOCAL_URI


def run_import_neo4j(
    *,
    uri: str,
    user: str,
    password: str,
    database: str,
    nodes: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    batch_size: int,
    create_index: bool,
) -> None:
    driver = connect_neo4j(uri, user, password)
    try:
        with driver.session(database=database) as session:
            if create_index:
                print("🔧 创建约束和索引...")
                create_constraints(session)
            print(f"📦 导入节点: {len(nodes)}")
            import_nodes(session, nodes, batch_size=batch_size)
            print(f"🔗 导入关系: {len(relationships)}")
            import_relationships(session, relationships, batch_size=batch_size)
            node_count, rel_count = db_stats(session)
            print(f"✅ 导入完成，当前数据库: 节点 {node_count}, 关系 {rel_count}")
    finally:
        driver.close()


def run_wipe_neo4j(
    *,
    uri: str,
    user: str,
    password: str,
    database: str,
    execute: bool,
    dry_run: bool,
) -> None:
    driver = connect_neo4j(uri, user, password)
    try:
        with driver.session(database=database) as session:
            wipe_neo4j(session, execute=execute, dry_run=dry_run)
    finally:
        driver.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Delta Agent 数据处理统一脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="JSON 数据目录")
    parser.add_argument("--json-files", default=",".join(DEFAULT_JSON_FILES), help="JSON 文件列表，逗号分隔")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("stats", help="统计 JSON 图数据")

    export_csv_parser = subparsers.add_parser("export-csv", help="导出 csv/nodes.csv + relationships.csv")
    export_csv_parser.add_argument("--output-dir", default=str(DEFAULT_CSV_DIR), help="CSV 输出目录")
    export_csv_parser.add_argument("--list-delim", default=os.getenv("LIST_DELIM", "|"), help="列表字段连接符")

    export_cypher_parser = subparsers.add_parser("export-cypher", help="导出 cypher 导入素材")
    export_cypher_parser.add_argument("--output-dir", default=str(DEFAULT_CYPHER_DIR), help="Cypher 输出目录")

    import_parser = subparsers.add_parser("import-neo4j", help="将 JSON 直接导入 Neo4j")
    import_parser.add_argument("--uri", default=None, help="Neo4j URI")
    import_parser.add_argument("--remote", action="store_true", help="使用 NEO4J_REMOTE_URI")
    import_parser.add_argument("--user", default=NEO4J_USER, help="Neo4j 用户")
    import_parser.add_argument("--password", default=NEO4J_PASSWORD, help="Neo4j 密码")
    import_parser.add_argument("--database", default=NEO4J_DATABASE, help="Neo4j 数据库名")
    import_parser.add_argument("--batch-size", type=int, default=300, help="导入批次大小")
    import_parser.add_argument("--skip-index", action="store_true", help="跳过约束与索引创建")

    wipe_parser = subparsers.add_parser("wipe-neo4j", help="清空 Neo4j")
    wipe_parser.add_argument("--uri", default=None, help="Neo4j URI")
    wipe_parser.add_argument("--remote", action="store_true", help="使用 NEO4J_REMOTE_URI")
    wipe_parser.add_argument("--user", default=NEO4J_USER, help="Neo4j 用户")
    wipe_parser.add_argument("--password", default=NEO4J_PASSWORD, help="Neo4j 密码")
    wipe_parser.add_argument("--database", default=NEO4J_DATABASE, help="Neo4j 数据库名")
    wipe_parser.add_argument("--execute", action="store_true", help="跳过确认直接删除")
    wipe_parser.add_argument("--dry-run", action="store_true", help="仅统计，不删除")

    rebuild_parser = subparsers.add_parser("rebuild-neo4j", help="清空并重建 Neo4j")
    rebuild_parser.add_argument("--uri", default=None, help="Neo4j URI")
    rebuild_parser.add_argument("--remote", action="store_true", help="使用 NEO4J_REMOTE_URI")
    rebuild_parser.add_argument("--user", default=NEO4J_USER, help="Neo4j 用户")
    rebuild_parser.add_argument("--password", default=NEO4J_PASSWORD, help="Neo4j 密码")
    rebuild_parser.add_argument("--database", default=NEO4J_DATABASE, help="Neo4j 数据库名")
    rebuild_parser.add_argument("--batch-size", type=int, default=300, help="导入批次大小")
    rebuild_parser.add_argument("--execute", action="store_true", help="跳过确认直接删除")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    json_files = parse_json_files(args.json_files)

    nodes: List[Dict[str, Any]] = []
    relationships: List[Dict[str, Any]] = []

    if args.command in {"stats", "export-csv", "export-cypher", "import-neo4j", "rebuild-neo4j"}:
        nodes, relationships = load_json_graph(data_dir, json_files)
        if not nodes:
            print("❌ 未加载到任何节点数据，请检查 data-dir 或 json-files。")
            sys.exit(1)

    if args.command == "stats":
        show_stats(nodes, relationships)
        return

    if args.command == "export-csv":
        export_csv(nodes, relationships, Path(args.output_dir).resolve(), args.list_delim)
        return

    if args.command == "export-cypher":
        export_cypher_assets(nodes, relationships, Path(args.output_dir).resolve())
        return

    if args.command == "import-neo4j":
        run_import_neo4j(
            uri=resolve_uri(args.uri, args.remote),
            user=args.user,
            password=args.password,
            database=args.database,
            nodes=nodes,
            relationships=relationships,
            batch_size=args.batch_size,
            create_index=not args.skip_index,
        )
        return

    if args.command == "wipe-neo4j":
        run_wipe_neo4j(
            uri=resolve_uri(args.uri, args.remote),
            user=args.user,
            password=args.password,
            database=args.database,
            execute=args.execute,
            dry_run=args.dry_run,
        )
        return

    if args.command == "rebuild-neo4j":
        uri = resolve_uri(args.uri, args.remote)
        run_wipe_neo4j(
            uri=uri,
            user=args.user,
            password=args.password,
            database=args.database,
            execute=args.execute,
            dry_run=False,
        )
        run_import_neo4j(
            uri=uri,
            user=args.user,
            password=args.password,
            database=args.database,
            nodes=nodes,
            relationships=relationships,
            batch_size=args.batch_size,
            create_index=True,
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
