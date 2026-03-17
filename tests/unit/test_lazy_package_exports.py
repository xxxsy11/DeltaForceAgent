from importlib import import_module


def test_agents_package_lazy_exports():
    agents = import_module("agents")
    assert hasattr(agents, "run_agent_query")
    assert hasattr(agents, "run_agent_interactive")


def test_tools_package_lazy_exports():
    tools = import_module("tools")
    assert hasattr(tools, "ToolRegistry")
    assert hasattr(tools, "build_df_latest_price_tool")
    assert hasattr(tools, "build_rag_knowledge_tool")


def test_services_and_memory_lazy_exports():
    services = import_module("services")
    memory = import_module("memory")
    assert hasattr(services, "RAGService")
    assert hasattr(services, "DFPriceService")
    assert hasattr(memory, "SessionMemoryManager")
    assert hasattr(memory, "PersistentMemoryStore")
