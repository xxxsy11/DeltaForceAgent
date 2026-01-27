"""
图数据库数据准备模块
从Neo4j读取图数据并转换为可检索的文档格式
"""

import logging
import json
from typing import List, Dict, Any
from dataclasses import dataclass

from neo4j import GraphDatabase
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """图节点数据结构"""
    node_id: str
    labels: List[str]
    name: str
    properties: Dict[str, Any]


@dataclass
class GraphRelation:
    """图关系数据结构"""
    start_node_id: str
    end_node_id: str
    relation_type: str
    properties: Dict[str, Any]


class GraphDataPreparationModule:
    """图数据库数据准备模块"""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None

        self.nodes: List[GraphNode] = []
        self.relationships: List[GraphRelation] = []
        self.documents: List[Document] = []
        self.chunks: List[Document] = []

        self._connect()

    def _connect(self):
        """建立Neo4j连接"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                database=self.database
            )
            logger.info(f"已连接到Neo4j数据库: {self.uri}")

            with self.driver.session() as session:
                session.run("RETURN 1")
                logger.info("Neo4j连接测试成功")

        except Exception as e:
            logger.error(f"连接Neo4j失败: {e}")
            raise

    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j连接已关闭")

    def _display_name(self, props: Dict[str, Any]) -> str:
        """选择显示名称"""
        for key in ("name", "typeName", "difficulty", "caliber", "colorName"):
            value = props.get(key)
            if value:
                return str(value)
        if props.get("level") is not None:
            return str(props.get("level"))
        return ""

    def _primary_label(self, labels: List[str]) -> str:
        """选择主标签"""
        for label in labels:
            if label != "Node":
                return label
        return labels[0] if labels else "Node"

    def load_graph_data(self) -> Dict[str, Any]:
        """从Neo4j加载图数据"""
        logger.info("正在从Neo4j加载图数据...")

        with self.driver.session() as session:
            node_query = """
            MATCH (n)
            WHERE n.nodeId IS NOT NULL
            RETURN n.nodeId as nodeId, labels(n) as labels,
                   properties(n) as properties
            """
            result = session.run(node_query)
            self.nodes = []
            for record in result:
                props = dict(record["properties"])
                name = self._display_name(props) or record["nodeId"]
                node = GraphNode(
                    node_id=record["nodeId"],
                    labels=record["labels"],
                    name=name,
                    properties=props
                )
                self.nodes.append(node)

            rel_query = """
            MATCH (s)-[r]->(t)
            WHERE s.nodeId IS NOT NULL AND t.nodeId IS NOT NULL
            RETURN s.nodeId as source_id, t.nodeId as target_id, type(r) as rel_type,
                   properties(r) as properties
            """
            rel_result = session.run(rel_query)
            self.relationships = []
            for record in rel_result:
                self.relationships.append(GraphRelation(
                    start_node_id=record["source_id"],
                    end_node_id=record["target_id"],
                    relation_type=record["rel_type"],
                    properties=dict(record["properties"]) if record["properties"] else {}
                ))

        logger.info(f"加载节点 {len(self.nodes)} 个，关系 {len(self.relationships)} 条")
        return {
            "nodes": len(self.nodes),
            "relationships": len(self.relationships)
        }

    def _format_properties(self, props: Dict[str, Any]) -> List[str]:
        """格式化属性为可读文本"""
        lines = []
        for k, v in props.items():
            if k in ("nodeId",):
                continue
            if v is None or v == "":
                continue
            if isinstance(v, list):
                v_str = "、".join([str(x) for x in v])
            elif isinstance(v, dict):
                v_str = json.dumps(v, ensure_ascii=False)
            else:
                v_str = str(v)
            lines.append(f"{k}: {v_str}")
        return lines

    def _fetch_neighbors(self, node_id: str, limit: int = 15) -> List[str]:
        """获取邻居信息"""
        neighbors = []
        with self.driver.session() as session:
            query = """
            MATCH (n {nodeId: $node_id})-[r]-(m)
            WITH n, r, m,
                 CASE WHEN startNode(r) = n THEN '->' ELSE '<-' END as dir
            RETURN type(r) as rel_type, dir as direction,
                   labels(m) as labels,
                   COALESCE(m.name, m.typeName, m.difficulty, toString(m.level), m.caliber) as name
            LIMIT $limit
            """
            result = session.run(query, {"node_id": node_id, "limit": limit})
            for record in result:
                name = record["name"] or "(无名)"
                label = record["labels"][0] if record["labels"] else "Node"
                neighbors.append(f"{record['direction']}[{record['rel_type']}] {label}:{name}")
        return neighbors

    def build_entity_documents(self) -> List[Document]:
        """构建实体文档"""
        logger.info("正在构建实体文档...")
        documents: List[Document] = []

        for node in self.nodes:
            primary_label = self._primary_label(node.labels)
            content_parts = [f"# {node.name}", f"类型: {primary_label}"]

            prop_lines = self._format_properties(node.properties)
            if prop_lines:
                content_parts.append("\n## 属性")
                content_parts.extend(prop_lines)

            neighbor_lines = self._fetch_neighbors(node.node_id)
            if neighbor_lines:
                content_parts.append("\n## 关联关系")
                content_parts.extend(neighbor_lines)

            full_content = "\n".join(content_parts)

            doc = Document(
                page_content=full_content,
                metadata={
                    "node_id": node.node_id,
                    "entity_name": node.name,
                    "node_type": primary_label,
                    "labels": node.labels,
                    "doc_type": "entity",
                    "content_length": len(full_content)
                }
            )
            documents.append(doc)

        self.documents = documents
        logger.info(f"成功构建 {len(documents)} 个实体文档")
        return documents

    def chunk_documents(self, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
        """对文档进行分块处理"""
        logger.info(f"正在进行文档分块，块大小: {chunk_size}, 重叠: {chunk_overlap}")

        if not self.documents:
            raise ValueError("请先构建文档")

        chunks: List[Document] = []
        chunk_id = 0

        for doc in self.documents:
            content = doc.page_content

            if len(content) <= chunk_size:
                chunk = Document(
                    page_content=content,
                    metadata={
                        **doc.metadata,
                        "chunk_id": f"{doc.metadata['node_id']}_chunk_{chunk_id}",
                        "parent_id": doc.metadata["node_id"],
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "chunk_size": len(content),
                        "doc_type": "chunk"
                    }
                )
                chunks.append(chunk)
                chunk_id += 1
                continue

            sections = content.split('\n## ')
            if len(sections) <= 1:
                total_chunks = (len(content) - 1) // (chunk_size - chunk_overlap) + 1
                for i in range(total_chunks):
                    start = i * (chunk_size - chunk_overlap)
                    end = min(start + chunk_size, len(content))
                    chunk_content = content[start:end]
                    chunk = Document(
                        page_content=chunk_content,
                        metadata={
                            **doc.metadata,
                            "chunk_id": f"{doc.metadata['node_id']}_chunk_{chunk_id}",
                            "parent_id": doc.metadata["node_id"],
                            "chunk_index": i,
                            "total_chunks": total_chunks,
                            "chunk_size": len(chunk_content),
                            "doc_type": "chunk"
                        }
                    )
                    chunks.append(chunk)
                    chunk_id += 1
            else:
                total_chunks = len(sections)
                for i, section in enumerate(sections):
                    if i == 0:
                        chunk_content = section
                    else:
                        chunk_content = f"## {section}"
                    chunk = Document(
                        page_content=chunk_content,
                        metadata={
                            **doc.metadata,
                            "chunk_id": f"{doc.metadata['node_id']}_chunk_{chunk_id}",
                            "parent_id": doc.metadata["node_id"],
                            "chunk_index": i,
                            "total_chunks": total_chunks,
                            "chunk_size": len(chunk_content),
                            "doc_type": "chunk",
                            "section_title": section.split('\n')[0] if i > 0 else "主标题"
                        }
                    )
                    chunks.append(chunk)
                    chunk_id += 1

        self.chunks = chunks
        logger.info(f"文档分块完成，共生成 {len(chunks)} 个块")
        return chunks

    def get_statistics(self) -> Dict[str, Any]:
        """获取数据统计信息"""
        stats = {
            "total_nodes": len(self.nodes),
            "total_relationships": len(self.relationships),
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks)
        }

        if self.nodes:
            label_counts: Dict[str, int] = {}
            for node in self.nodes:
                label = self._primary_label(node.labels)
                label_counts[label] = label_counts.get(label, 0) + 1
            stats["label_counts"] = label_counts

        if self.documents:
            stats["avg_content_length"] = sum(
                doc.metadata.get("content_length", 0) for doc in self.documents
            ) / max(1, len(self.documents))

        if self.chunks:
            stats["avg_chunk_size"] = sum(
                chunk.metadata.get("chunk_size", 0) for chunk in self.chunks
            ) / max(1, len(self.chunks))

        return stats

    def __del__(self):
        self.close()
