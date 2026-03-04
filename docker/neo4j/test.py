import os

from neo4j import GraphDatabase


uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
user = os.getenv("NEO4J_USER", "neo4j")
password = os.getenv("NEO4J_PASSWORD", "")
database = os.getenv("NEO4J_DATABASE", "neo4j")

if not password:
    raise SystemExit("请先设置 NEO4J_PASSWORD")

driver = GraphDatabase.driver(uri, auth=(user, password))
with driver.session(database=database) as session:
    print(session.run("RETURN 1 AS ok").single()["ok"])
driver.close()
