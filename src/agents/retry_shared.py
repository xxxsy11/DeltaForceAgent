"""重试链路共享工具。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_retry_trace_entry(
    *,
    stage: str,
    reason: str,
    retry_requested: bool,
    target_stage: str = "",
    accepted: bool | None = None,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "stage": stage,
        "reason": str(reason or "").strip(),
        "retry_requested": bool(retry_requested),
        "target_stage": str(target_stage or "").strip(),
        "at_utc": now_utc_iso(),
    }
    if accepted is not None:
        entry["accepted"] = bool(accepted)
    if extra:
        entry.update(extra)
    return entry


def append_retry_trace(
    state_trace: List[Dict[str, Any]] | None,
    *,
    stage: str,
    reason: str,
    retry_requested: bool,
    target_stage: str = "",
    accepted: bool | None = None,
    extra: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    trace = list(state_trace or [])
    trace.append(
        build_retry_trace_entry(
            stage=stage,
            reason=reason,
            retry_requested=retry_requested,
            target_stage=target_stage,
            accepted=accepted,
            extra=extra,
        )
    )
    return trace
