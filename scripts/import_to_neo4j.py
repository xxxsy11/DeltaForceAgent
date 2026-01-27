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
from typing import Dict, List, Any


# Neo4j 连接配置
DEFAULT_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


def create_constraints(driver):
    """创建约束和索引"""
    print("创建约束和索引...")

    constraints = [
        "CREATE CONSTRAINT node_id_unique IF NOT EXISTS FOR (n:Node) REQUIRE n.nodeId IS UNIQUE",
        "CREATE INDEX map_name_index IF NOT EXISTS FOR (m:Map) ON (m.name)",
        "CREATE INDEX area_name_index IF NOT EXISTS FOR (a:Area) ON (a.name)",
        "CREATE INDEX equipment_name_index IF NOT EXISTS FOR (e:Equipment) ON (e.name)",
        "CREATE INDEX firearm_name_index IF NOT EXISTS FOR (f:Firearm) ON (f.name)",
    ]

    with driver.session() as session:
        for constraint in constraints:
            try:
                session.run(constraint)
                print(f"  创建: {constraint[:60]}...")
            except Exception as e:
                if "already exists" in str(e) or "equivalent" in str(e):
                    print(f"  已存在: {constraint[:60]}...")
                else:
                    print(f"  警告: {constraint[:60]}... {e}")


def import_nodes(driver, nodes: List[Dict]):
    """导入节点"""
    print(f"\n导入 {len(nodes)} 个节点...")

    batch_size = 100
    name_field_candidates = ['name', 'typeName', 'difficulty', 'level',
                              'colorName', 'conceptType', 'category']

    with driver.session() as session:
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]

            for node in batch:
                node_id = node['id']
                labels = node.get('labels', [])
                props = node.get('props', {})

                primary_label = labels[0] if labels else "Node"

                display_name = None
                for field in name_field_candidates:
                    if field in props and props[field]:
                        display_name = str(props[field])
                        break

                if not display_name:
                    display_name = node_id

                node_props = {
                    "nodeId": node_id,
                    "name": display_name,
                }

                for key, value in props.items():
                    if value is not None:
                        if isinstance(value, list):
                            node_props[key] = json.dumps(value, ensure_ascii=False)
                        else:
                            node_props[key] = value

                labels_str = ":".join([primary_label, "Node"])
                props_str = ", ".join([f"n.{k} = ${k}" for k in node_props.keys()])

                cypher = f"""
                    MERGE (n:{labels_str} {{nodeId: $nodeId}})
                    SET {props_str}
                """

                try:
                    session.run(cypher, **node_props)
                except Exception as e:
                    print(f"    节点 {node_id} 导入失败: {e}")

            if (i + batch_size) % 500 == 0:
                print(f"  进度: {i + batch_size}/{len(nodes)}")

    print(f"  节点导入完成")


def import_relationships(driver, relationships: List[Dict]):
    """导入关系"""
    print(f"\n导入 {len(relationships)} 条关系...")

    batch_size = 100
    imported = 0
    failed = 0

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
                    imported += 1
                except Exception as e:
                    failed += 1

            if (i + batch_size) % 500 == 0:
                print(f"  进度: {i + batch_size}/{len(relationships)}")

    print(f"  关系导入完成: 成功 {imported}, 失败 {failed}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="从JSON导入知识图谱到Neo4j",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--uri", type=str, default=DEFAULT_URI,
                       help="Neo4j URI (默认: bolt://localhost:7687)")
    parser.add_argument("--file", type=str,
                       help="JSON文件路径 (默认: data/neo4j/data.json)")

    args = parser.parse_args()

    # 确定JSON文件路径
    if args.file:
        json_path = Path(args.file)
    else:
        base = Path(__file__).resolve().parent.parent
        json_path = base / "data" / "neo4j" / "data.json"

    if not json_path.exists():
        print(f"错误: 文件不存在: {json_path}")
        sys.exit(1)

    # 读取JSON数据
    print(f"读取数据文件: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    relationships = data.get("relationships", [])

    print(f"  节点: {len(nodes)}")
    print(f"  关系: {len(relationships)}")

    # 连接Neo4j
    print(f"\n连接到Neo4j: {args.uri}")
    driver = GraphDatabase.driver(args.uri, auth=(USER, PASSWORD))

    try:
        # 创建约束
        create_constraints(driver)

        # 导入节点
        import_nodes(driver, nodes)

        # 导入关系
        import_relationships(driver, relationships)

        print("\n导入完成!")

    finally:
        driver.close()
        print("连接已关闭")


if __name__ == "__main__":
    main()
