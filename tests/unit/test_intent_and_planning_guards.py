import asyncio
from dataclasses import replace

from agents.intent_analyzer import IntentAnalyzer
from agents.task_planning import TaskPlanningAgent
from agents.tool_planner import LLMToolPlanner
from config import DEFAULT_CONFIG
from tools.registry import ToolRegistry


def test_place_profit_keywords_route_to_profit_rank():
    analyzer = IntentAnalyzer()
    queries = [
        "制造台现在制造什么利润最高",
        "特勤处制造什么利润最高",
        "工作台制造什么利润最高",
    ]
    for query in queries:
        decision = analyzer.analyze(query)
        assert decision.intent == "place_profit_query"
        assert decision.tool_name == "df_place_profit_rank"


def test_task_planning_keeps_place_profit_tool_locked():
    cfg = replace(DEFAULT_CONFIG)
    cfg.agent_local_enabled = False

    planner = TaskPlanningAgent(
        planner=LLMToolPlanner(config=cfg),
        registry=ToolRegistry(),
    )
    state = {
        "user_query": "制造台现在制造什么利润最高",
        "tool_query": "制造台现在制造什么利润最高",
        "intent": "place_profit_query",
        "selected_tool": "df_place_profit_rank",
        "flow_type": "complex",
        "requires_task_planning": True,
        "agent_messages": [],
        "debug_steps": [],
    }

    result = asyncio.run(planner.run(state))
    assert result["plan_source"] == "place_profit_locked_tool"
    assert result["task_plan"] == [
        {"tool_name": "df_place_profit_rank", "tool_query": "制造台现在制造什么利润最高"}
    ]
