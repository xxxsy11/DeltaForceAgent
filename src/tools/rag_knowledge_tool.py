"""
RAG 知识查询工具
"""

from langchain_core.tools import tool

from services import RAGService


def build_rag_knowledge_tool(rag_service: RAGService):
    @tool("rag_knowledge_search")
    def rag_knowledge_search(query: str) -> str:
        """用于三角洲知识库的资料查询、知识问答、关系分析。"""
        result = rag_service.query(question=query, explain_routing=False)
        return str(result.get("answer", "")).strip()

    return rag_knowledge_search
