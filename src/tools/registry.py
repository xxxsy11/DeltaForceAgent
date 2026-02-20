"""
工具注册中心
"""

from typing import Dict, Optional

from services import RAGService
from tools.rag_knowledge_tool import build_rag_knowledge_tool


class ToolRegistry:
    def __init__(self, rag_service: Optional[RAGService] = None):
        self.rag_service = rag_service or RAGService()
        self.tools: Dict[str, object] = {
            "rag_knowledge_search": build_rag_knowledge_tool(self.rag_service),
        }

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self.tools

    def invoke(self, tool_name: str, query: str) -> str:
        if not self.has_tool(tool_name):
            return f"未找到工具: {tool_name}"
        try:
            return self.tools[tool_name].invoke({"query": query})
        except Exception as exc:
            return f"工具调用失败({tool_name}): {exc}"

    def close(self):
        self.rag_service.close()
