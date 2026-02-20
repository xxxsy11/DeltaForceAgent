from pymilvus import MilvusClient

# 连接到远程 Milvus
client = MilvusClient("http://58.199.146.145:19530")

# 测试连接 - 使用 MilvusClient 的方法
print("Milvus 连接成功！")

# 使用 MilvusClient 的方法列出集合
collections = client.list_collections()
print(f"现有集合: {collections}")

# 或者测试健康检查
from pymilvus import utility
# 使用 utility 需要先连接
from pymilvus import connections
connections.connect("default", uri="http://58.199.146.145:19530")
print(f"使用 utility 列出集合: {utility.list_collections()}")
