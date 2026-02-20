"""
Multi-Agent 运行入口
"""

from typing import Optional

from config import DEFAULT_CONFIG, GraphRAGConfig
from services import RAGService
from tools import ToolRegistry
from agents.graph import build_multi_agent_graph


def _build_initial_state(query: str):
    return {
        "user_query": query,
        "intent": "",
        "selected_tool": "",
        "tool_query": "",
        "tool_output": "",
        "final_answer": "",
        "debug_steps": [],
    }


def run_agent_query(query: str, config: Optional[GraphRAGConfig] = None) -> str:
    cfg = config or DEFAULT_CONFIG
    rag_service = RAGService(cfg)
    registry = ToolRegistry(rag_service=rag_service)
    graph = build_multi_agent_graph(registry)
    result = graph.invoke(_build_initial_state(query))
    registry.close()
    return result.get("final_answer", "")


def run_agent_interactive(config: Optional[GraphRAGConfig] = None):
    cfg = config or DEFAULT_CONFIG
    rag_service = RAGService(cfg)
    registry = ToolRegistry(rag_service=rag_service)
    graph = build_multi_agent_graph(registry)

    print("\nMulti-Agent 模式已启动（当前工具: rag_knowledge_search）")
    print("输入 'quit' 退出。")
    while True:
        query = input("\nAgent问题: ").strip()
        if not query:
            continue
        if query.lower() == "quit":
            break

        result = graph.invoke(_build_initial_state(query))
        print(f"\n回答:\n{result.get('final_answer', '')}")

    registry.close()
