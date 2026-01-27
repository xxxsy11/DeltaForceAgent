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


# Neo4j 连接配置
DEFAULT_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


def wipe_database(uri: str, user: str, password: str, dry_run: bool = False):
    """清理数据库中的所有数据"""
    print(f"\n{'='*60}")
    print(f"连接到 Neo4j: {uri}")
    print(f"{'='*60}\n")

    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        with driver.session(database="neo4j") as session:
            # 统计现有数据
            print("清理前的数据统计:")
            result = session.run("MATCH (n) RETURN count(n) as nodeCount")
            node_count = result.single()["nodeCount"]
            print(f"  节点数: {node_count:,}")

            result = session.run("MATCH ()-[r]->() RETURN count(r) as relCount")
            rel_count = result.single()["relCount"]
            print(f"  关系数: {rel_count:,}")

            if node_count == 0 and rel_count == 0:
                print("\n数据库已经是空的，无需清理")
                return

            # 按节点类型统计
            print("\n节点类型分布:")
            result = session.run("""
                MATCH (n)
                WITH labels(n) as labels, count(n) as count
                UNWIND labels as label
                RETURN label as nodeType, sum(count) as totalCount
                ORDER BY totalCount DESC
            """)
            for record in result:
                print(f"  {record['nodeType']:25s}: {record['totalCount']:6,} 个")

            if dry_run:
                print("\n这是试运行模式，实际未删除数据")
                return

            # 确认删除
            print(f"\n即将删除所有 {node_count:,} 个节点和 {rel_count:,} 条关系")
            confirm = input("确认删除? (输入 'yes' 继续): ")

            if confirm.lower() != 'yes':
                print("操作已取消")
                return

            # 执行删除
            print("\n正在删除数据...")
            result = session.run("MATCH (n) DETACH DELETE n")
            summary = result.consume()
            print(f"  已删除 {summary.counters.nodes_deleted:,} 个节点")
            print(f"  已删除 {summary.counters.relationships_deleted:,} 条关系")

            # 验证删除结果
            result = session.run("MATCH (n) RETURN count(n) as c")
            c = result.single()["c"]
            result = session.run("MATCH ()-[r]->() RETURN count(r) as c")
            r = result.single()["c"]

            print(f"\n清理完成!")
            print(f"  剩余节点: {c}")
            print(f"  剩余关系: {r}")

    finally:
        driver.close()
        print(f"\n{'='*60}")
        print("连接已关闭")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="清理 Neo4j 数据库中的所有数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--uri", type=str, default=DEFAULT_URI,
                       help="Neo4j URI (默认: bolt://localhost:7687)")
    parser.add_argument("--dry-run", action="store_true",
                       help="试运行模式，不实际删除数据")

    args = parser.parse_args()

    wipe_database(args.uri, USER, PASSWORD, args.dry_run)


if __name__ == "__main__":
    main()
