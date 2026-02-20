"""
RAG 业务服务：将现有 RAG 系统封装成可复用能力。
"""

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

from config import DEFAULT_CONFIG, GraphRAGConfig

if TYPE_CHECKING:
    from rag_modules.rag_system import AdvancedGraphRAGSystem

logger = logging.getLogger(__name__)


class RAGService:
    """
    RAG 服务：
    - 启动并持有 RAG 系统实例
    - 对外暴露统一 query 方法
    """

    def __init__(self, config: Optional[GraphRAGConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.system: Optional[Any] = None
        self.ready = False

    def startup(self):
        """按在线服务模式启动，不触发离线重建。"""
        if self.ready and self.system:
            return

        from rag_modules.rag_system import AdvancedGraphRAGSystem
        self.system = AdvancedGraphRAGSystem(config=self.config)
        self.system.initialize_system(enable_qa_modules=True)
        self.system.load_knowledge_base_for_serving()
        self.ready = True
        logger.info("RAG Service 启动完成")

    def query(self, question: str, explain_routing: bool = False) -> Dict[str, Any]:
        if not question or not question.strip():
            return {"answer": "问题为空，请输入有效问题。", "analysis": None}

        self.startup()
        answer, analysis = self.system.ask_question_with_routing(
            question=question.strip(),
            stream=False,
            explain_routing=explain_routing,
        )
        return {"answer": answer, "analysis": analysis}

    def close(self):
        if self.system:
            self.system._cleanup()
        self.ready = False
        logger.info("RAG Service 已关闭")
