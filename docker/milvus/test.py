import os

from pymilvus import MilvusClient, connections, utility


milvus_uri = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")

client = MilvusClient(milvus_uri)
print("Milvus 连接成功！")

collections = client.list_collections()
print(f"现有集合: {collections}")

connections.connect("default", uri=milvus_uri)
print(f"使用 utility 列出集合: {utility.list_collections()}")
