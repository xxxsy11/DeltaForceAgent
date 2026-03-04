"""Multi-Agent 统一入口。"""

import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

load_dotenv(PROJECT_ROOT / ".env")
os.environ.setdefault("RAG_RUN_MODE", "agent")

from config import DEFAULT_CONFIG
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


def main():
    _configure_console_logging()
    print("启动 Multi-Agent 系统（intent -> route -> tool）")
    asyncio.run(run_agent_interactive(config=DEFAULT_CONFIG))


if __name__ == "__main__":
    main()
