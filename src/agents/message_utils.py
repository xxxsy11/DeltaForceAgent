"""Agent 间消息共享工具。"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional


def build_agent_message(
    *,
    from_agent: str,
    to_agent: str,
    message_type: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "from_agent": str(from_agent or "").strip(),
        "to_agent": str(to_agent or "").strip(),
        "message_type": str(message_type or "").strip(),
        "payload": dict(payload or {}),
    }


def append_agent_message(
    messages: Iterable[Mapping[str, Any]] | None,
    *,
    from_agent: str,
    to_agent: str,
    message_type: str,
    payload: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    next_messages = [dict(item) for item in (messages or []) if isinstance(item, Mapping)]
    next_messages.append(
        build_agent_message(
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=message_type,
            payload=payload,
        )
    )
    return next_messages


def find_latest_message_payload(
    messages: Iterable[Mapping[str, Any]] | None,
    *,
    message_type: str,
    to_agents: Iterable[str] | None = None,
) -> Dict[str, Any]:
    target_agents = {str(item).strip() for item in (to_agents or []) if str(item).strip()}
    for item in reversed(list(messages or [])):
        if not isinstance(item, Mapping):
            continue
        if str(item.get("message_type", "") or "").strip() != message_type:
            continue
        if target_agents:
            to_agent = str(item.get("to_agent", "") or "").strip()
            if to_agent not in target_agents:
                continue
        payload = item.get("payload")
        if isinstance(payload, Mapping):
            return dict(payload)
    return {}
