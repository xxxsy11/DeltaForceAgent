#!/usr/bin/env python3
"""
从JSON导入知识图谱到Neo4j

使用方法:
  python import_to_neo4j.py               # 导入到本地Neo4j
  python import_to_neo4j.py --uri bolt://your-uri:7687  # 指定URI
"""

from neo4j import GraphDatabase
import json
import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict


# Neo4j 连接配置
DEFAULT_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

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
            print(f"  跳过: {fname} (不存在)")
            continue

        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        nodes = data.get('nodes', [])
        rels = data.get('relationships', [])

        all_nodes.extend(nodes)
        all_rels.extend(rels)

        print(f"  {fname}: {len(nodes)} 节点, {len(rels)} 关系")

    return all_nodes, all_rels


def create_constraints(driver):
    """创建约束和索引"""
    print("\n创建约束和索引...")

    constraints = [
        "CREATE CONSTRAINT node_id_unique IF NOT EXISTS FOR (n:Node) REQUIRE n.nodeId IS UNIQUE",
        "CREATE INDEX map_name_index IF NOT EXISTS FOR (m:Map) ON (m.name)",
        "CREATE INDEX area_name_index IF NOT EXISTS FOR (a:Area) ON (a.name)",
        "CREATE INDEX equipment_name_index IF NOT EXISTS FOR (e:Equipment) ON (e.name)",
        "CREATE INDEX firearm_name_index IF NOT EXISTS FOR (f:Firearm) ON (f.name)",
        "CREATE INDEX attachment_name_index IF NOT EXISTS FOR (a:Attachment) ON (a.name)",
        "CREATE INDEX ammo_name_index IF NOT EXISTS FOR (a:Ammo) ON (a.name)",
        "CREATE INDEX collectible_name_index IF NOT EXISTS FOR (c:Collectible) ON (c.name)",
        "CREATE INDEX operator_name_index IF NOT EXISTS FOR (o:Operator) ON (o.name)",
    ]

    with driver.session() as session:
        for c in constraints:
            try:
                session.run(c)
            except Exception as e:
                if "already exists" not in str(e):
                    print(f"  警告: {e}")


def import_nodes(driver, nodes: List[Dict]):
    """导入节点"""
    print(f"\n导入 {len(nodes)} 个节点...")

    batch_size = 100
    name_fields = ['name', 'typeName', 'difficulty', 'level', 'colorName']

    with driver.session() as session:
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]

            for node in batch:
                node_id = node['id']
                labels = node.get('labels', [])
                props = node.get('props', {})

                primary_label = labels[0] if labels else "Node"

                # 找一个合适的显示名称
                display_name = node_id
                for field in name_fields:
                    if field in props and props[field]:
                        display_name = str(props[field])
                        break

                # 构建属性
                node_props = {"nodeId": node_id, "name": display_name}

                for key, value in props.items():
                    if value is not None:
                        if isinstance(value, (list, dict)):
                            node_props[key] = json.dumps(value, ensure_ascii=False)
                        else:
                            node_props[key] = value

                # 创建节点
                labels_str = ":".join([primary_label, "Node"])
                props_str = ", ".join([f"n.{k} = ${k}" for k in node_props.keys()])

                cypher = f"MERGE (n:{labels_str} {{nodeId: $nodeId}}) SET {props_str}"

                try:
                    session.run(cypher, **node_props)
                except Exception as e:
                    print(f"    失败: {node_id} - {e}")

            if (i + batch_size) % 500 == 0 or i + batch_size >= len(nodes):
                print(f"  进度: {min(i + batch_size, len(nodes))}/{len(nodes)}")

    print("  完成")


def import_relationships(driver, relationships: List[Dict]):
    """导入关系"""
    if not relationships:
        print("\n没有关系需要导入")
        return

    print(f"\n导入 {len(relationships)} 条关系...")

    batch_size = 100

    with driver.session() as session:
        for i in range(0, len(relationships), batch_size):
            batch = relationships[i:i + batch_size]

            for rel in batch:
                rel_type = rel.get('type', 'RELATED_TO')
                source_id = rel.get('from')
                target_id = rel.get('to')

                cypher = f"""
                    MATCH (source:Node {{nodeId: $sourceId}})
                    MATCH (target:Node {{nodeId: $targetId}})
                    MERGE (source)-[r:{rel_type}]->(target)
                """

                try:
                    session.run(cypher, sourceId=source_id, targetId=target_id)
                except Exception as e:
                    pass  # 忽略关系导入失败

            if (i + batch_size) % 500 == 0 or i + batch_size >= len(relationships):
                print(f"  进度: {min(i + batch_size, len(relationships))}/{len(relationships)}")

    print("  完成")


def show_stats(driver):
    """显示统计信息"""
    print("\n" + "="*50)
    print("数据库统计")
    print("="*50)

    with driver.session() as session:
        # 节点统计
        result = session.run("""
            MATCH (n)
            WITH labels(n) as labels, count(n) as count
            UNWIND labels as label
            RETURN label as type, sum(count) as total
            ORDER BY total DESC
        """)
        print("\n节点类型:")
        for r in result:
            if r['type'] != 'Node':
                print(f"  {r['type']:20s}: {r['total']:5,}")

        # 关系统计
        result = session.run("""
            MATCH ()-[r]->()
            RETURN type(r) as type, count(r) as total
            ORDER BY total DESC
        """)
        print("\n关系类型:")
        has_rel = False
        for r in result:
            has_rel = True
            print(f"  {r['type']:20s}: {r['total']:5,}")
        if not has_rel:
            print("  (无)")

    print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="从JSON导入知识图谱到Neo4j")
    parser.add_argument("--uri", type=str, default=DEFAULT_URI,
                       help="Neo4j URI")
    parser.add_argument("--data-dir", type=str,
                       help="JSON数据目录")

    args = parser.parse_args()

    # 确定数据目录
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR

    if not data_dir.exists():
        print(f"错误: 数据目录不存在: {data_dir}")
        sys.exit(1)

    # 读取数据
    print(f"读取数据: {data_dir}")
    nodes, relationships = load_json_files()

    if not nodes:
        print("错误: 没有找到任何节点数据")
        sys.exit(1)

    print(f"\n总计: {len(nodes)} 节点, {len(relationships)} 关系")

    # 连接并导入
    print(f"\n连接到: {args.uri}")
    driver = GraphDatabase.driver(args.uri, auth=(USER, PASSWORD))

    try:
        create_constraints(driver)
        import_nodes(driver, nodes)
        import_relationships(driver, relationships)
        show_stats(driver)
        print("导入完成!")
    finally:
        driver.close()
        print("连接已关闭\n")


if __name__ == "__main__":
    main()
