"""输出质量共享规则。"""

from __future__ import annotations

from typing import Any, Iterable, Mapping


FAILURE_MARKERS = (
    "工具调用失败",
    "查询失败",
    "未找到工具",
    "系统错误",
    "未获得可用结果",
    "不可用",
)
EMPTY_RESULT_TEXT = "未获得可用结果。"
MISSING_COMPARE_ENTITIES_TEXT = "请至少提供两个物品名称"
ENTITY_NOT_MATCHED_TEXT = "未能根据 objectName 匹配到交易物品ID"


def is_failure_text(text: str, *, extra_markers: Iterable[str] = ()) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return True
    markers = tuple(FAILURE_MARKERS) + tuple(str(item) for item in extra_markers)
    return any(token in raw for token in markers)


def has_success_tool_result(results: list[Mapping[str, Any]]) -> bool:
    for item in results:
        ok = item.get("ok")
        if isinstance(ok, bool):
            if ok:
                return True
            continue
        output = str(item.get("output", "") or "")
        if output and not is_failure_text(output):
            return True
    return False


def needs_compare_entity_resolution(text: str) -> bool:
    raw = str(text or "")
    return MISSING_COMPARE_ENTITIES_TEXT in raw or ENTITY_NOT_MATCHED_TEXT in raw
