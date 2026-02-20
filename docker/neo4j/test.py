from neo4j import GraphDatabase
driver = GraphDatabase.driver("bolt://58.199.146.145:7687", auth=("neo4j","delta_agent"))
with driver.session(database="neo4j") as s:
    print(s.run("RETURN 1 AS ok").single()["ok"])
driver.close()