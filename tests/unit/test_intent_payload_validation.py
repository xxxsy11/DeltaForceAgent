from agents.intent_recognition import IntentRecognitionAgent


def _agent_stub() -> IntentRecognitionAgent:
    agent = IntentRecognitionAgent.__new__(IntentRecognitionAgent)
    agent._prompt_template = None
    return agent


def test_validate_llm_payload_success():
    agent = _agent_stub()
    payload = {
        "intent": "market_latest_price_query",
        "tool_name": "df_market_latest_price",
        "reason": "price request",
        "entities": ["非洲之心"],
        "confidence": 0.91,
    }
    out = agent._validate_llm_payload(payload)
    assert out["tool_name"] == "df_market_latest_price"
    assert out["entities"] == ["非洲之心"]


def test_validate_llm_payload_invalid_shape_returns_empty():
    agent = _agent_stub()
    payload = {
        "intent": "market_latest_price_query",
        "tool_name": "df_market_latest_price",
        "entities": "非洲之心",  # must be list
    }
    out = agent._validate_llm_payload(payload)
    assert out == {}
