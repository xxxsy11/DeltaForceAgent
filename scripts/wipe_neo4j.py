#!/usr/bin/env python3
"""
清理Neo4j数据库中的所有数据

使用方法:
  python wipe_neo4j.py                    # 使用默认配置
  python wipe_neo4j.py --uri bolt://localhost:7687
  python wipe_neo4j.py --dry-run          # 试运行模式
"""

from neo4j import GraphDatabase
import argparse
import os


DEFAULT_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


def wipe_database(uri: str, user: str, password: str, dry_run: bool = False):
    print(f"\n连接到: {uri}")

    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        with driver.session(database="neo4j") as session:
            # 统计现有数据
            result = session.run("MATCH (n) RETURN count(n) as c")
            node_count = result.single()["c"]

            result = session.run("MATCH ()-[r]->() RETURN count(r) as c")
            rel_count = result.single()["c"]

            print(f"当前数据: {node_count:,} 节点, {rel_count:,} 关系")

            if node_count == 0 and rel_count == 0:
                print("\n数据库已经是空的")
                return

            # 按类型统计
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

            if dry_run:
                print("\n试运行模式，未删除数据")
                return

            # 确认
            confirm = input(f"\n确认删除所有数据? (yes/no): ")
            if confirm.lower() != 'yes':
                print("已取消")
                return

            # 删除
            print("删除中...")
            result = session.run("MATCH (n) DETACH DELETE n")
            summary = result.consume()
            print(f"已删除: {summary.counters.nodes_deleted:,} 节点, {summary.counters.relationships_deleted:,} 关系")

    finally:
        driver.close()
        print("\n完成\n")


def main():
    parser = argparse.ArgumentParser(description="清理Neo4j数据库")
    parser.add_argument("--uri", type=str, default=DEFAULT_URI,
                       help="Neo4j URI")
    parser.add_argument("--dry-run", action="store_true",
                       help="试运行模式")

    args = parser.parse_args()
    wipe_database(args.uri, USER, PASSWORD, args.dry_run)


if __name__ == "__main__":
    main()
