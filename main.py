"""Multi-Agent 统一入口。"""

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


def main():
    print("启动 Multi-Agent 系统（intent -> route -> tool）")
    run_agent_interactive(config=DEFAULT_CONFIG)


if __name__ == "__main__":
    main()
