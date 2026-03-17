"""Multi-Agent 统一入口。"""

import asyncio
import logging
import os
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

load_dotenv(PROJECT_ROOT / ".env")
os.environ.setdefault("RAG_RUN_MODE", "agent")

from config import DEFAULT_CONFIG, validate_runtime_config
from agents.runner import run_agent_interactive


def _configure_console_logging() -> None:
    # 减少三方与底层库的噪声日志，仅保留告警与错误。
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("rag_modules").setLevel(logging.WARNING)
    logging.getLogger("services.rag_service").setLevel(logging.WARNING)
    logging.getLogger("agents.local_qwen_runtime").setLevel(logging.WARNING)


def _parse_host_port_from_uri(uri: str, default_port: int) -> tuple[str, int]:
    raw = str(uri or "").strip()
    if not raw:
        return "", int(default_port)
    parsed = urlparse(raw)
    if parsed.hostname:
        return str(parsed.hostname), int(parsed.port or default_port)
    if ":" in raw:
        host, port = raw.rsplit(":", 1)
        if port.isdigit():
            return host.strip(), int(port)
    return raw, int(default_port)


def _tcp_reachable(host: str, port: int, timeout_s: float = 1.2) -> bool:
    if not host or int(port) <= 0:
        return False
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_s):
            return True
    except Exception:
        return False


def _service_preflight() -> None:
    neo4j_host, neo4j_port = _parse_host_port_from_uri(DEFAULT_CONFIG.neo4j_uri, 7687)
    milvus_host = str(DEFAULT_CONFIG.milvus_host or "").strip()
    milvus_port = int(DEFAULT_CONFIG.milvus_port or 19530)

    print(f"RAG后端配置: neo4j={neo4j_host}:{neo4j_port}, milvus={milvus_host}:{milvus_port}")
    if not _tcp_reachable(neo4j_host, neo4j_port):
        print(f"⚠ Neo4j 连接预检失败: {neo4j_host}:{neo4j_port} 不可达")
    if not _tcp_reachable(milvus_host, milvus_port):
        print(f"⚠ Milvus 连接预检失败: {milvus_host}:{milvus_port} 不可达")

    if DEFAULT_CONFIG.memory_persistent_enabled:
        dsn = str(DEFAULT_CONFIG.memory_persistent_dsn or "").strip()
        if "@" in dsn:
            addr = dsn.split("@", 1)[1]
            host_port = addr.split("/", 1)[0]
            if ":" in host_port:
                host, port = host_port.rsplit(":", 1)
                if port.isdigit() and not _tcp_reachable(host.strip(), int(port)):
                    print(f"⚠ 长期记忆数据库预检失败: {host.strip()}:{int(port)} 不可达")


def main():
    _configure_console_logging()
    validate_runtime_config(DEFAULT_CONFIG)
    print("启动 Multi-Agent 系统（intent -> route -> tool）")
    if DEFAULT_CONFIG.langsmith_enabled and DEFAULT_CONFIG.langsmith_api_key:
        print(f"LangSmith 已启用（project: {DEFAULT_CONFIG.langsmith_project}）")
    _service_preflight()
    asyncio.run(run_agent_interactive(config=DEFAULT_CONFIG))


if __name__ == "__main__":
    main()
