#!/usr/bin/env python3
"""Export PostgreSQL memory data to readable local text/json files."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import psycopg
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"psycopg 未安装，无法导出记忆数据: {exc}")


DEFAULT_DSN = "postgresql://delta_agent:delta_agent@127.0.0.1:5432/delta_agent"
DEFAULT_OUT_BASE = Path("data/memory/exports")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出长期记忆为可读文本")
    parser.add_argument("--dsn", default=DEFAULT_DSN, help="PostgreSQL DSN")
    parser.add_argument("--session-id", default="", help="只导出指定 session_id（可选）")
    parser.add_argument("--limit", type=int, default=0, help="每张表导出条数上限，0 表示不限制")
    parser.add_argument("--output-dir", default="", help="输出目录（默认自动时间目录）")
    return parser.parse_args()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _json_safe(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(_json_safe(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(_json_safe(row) + "\n")


def _query_rows(cur, sql: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
    cur.execute(sql, params or ())
    cols = [desc[0] for desc in cur.description]
    data: List[Dict[str, Any]] = []
    for row in cur.fetchall() or []:
        data.append({cols[i]: row[i] for i in range(len(cols))})
    return data


def _where_clause(session_id: str) -> tuple[str, tuple]:
    if session_id:
        return " WHERE session_id = %s ", (session_id,)
    return "", ()


def _limit_clause(limit: int) -> str:
    return f" LIMIT {int(limit)} " if limit and limit > 0 else ""


def _build_relation_map(cur) -> Dict[str, Any]:
    relation_rows = _query_rows(
        cur,
        """
        SELECT
            c.relname,
            c.relkind,
            c.relfilenode,
            pg_relation_filepath(c.oid) AS relation_filepath,
            ts.spcname AS tablespace_name
        FROM pg_class c
        LEFT JOIN pg_tablespace ts ON ts.oid = c.reltablespace
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
          AND c.relname IN (
            'chat_sessions', 'chat_turns', 'memory_summaries', 'memory_facts',
            'chat_sessions_pkey', 'chat_turns_pkey', 'idx_chat_turns_session_turn',
            'memory_facts_pkey', 'memory_facts_session_id_fact_key_key',
            'idx_memory_facts_session_active', 'idx_memory_facts_ttl',
            'memory_summaries_pkey', 'memory_summaries_session_id_merge_count_key'
          )
        ORDER BY c.relkind, c.relname;
        """,
    )
    data_directory = ""
    try:
        cur.execute("SHOW data_directory;")
        data_directory = str((cur.fetchone() or [""])[0] or "")
    except Exception:
        data_directory = ""

    for row in relation_rows:
        rel = str(row.get("relation_filepath") or "")
        row["absolute_hint"] = f"{data_directory}/{rel}" if data_directory and rel else ""
    return {"data_directory": data_directory, "relations": relation_rows}


def _render_chat_turns_md(rows: List[Dict[str, Any]]) -> str:
    lines: List[str] = ["# Chat Turns（可读导出）", ""]
    if not rows:
        lines.append("无数据。")
        return "\n".join(lines)

    current_sid = None
    for row in rows:
        sid = str(row.get("session_id") or "")
        if sid != current_sid:
            current_sid = sid
            lines.append(f"## Session: {sid}")
            lines.append("")
        role = str(row.get("role") or "")
        turn_index = row.get("turn_index")
        content = str(row.get("content") or "")
        tool_name = str(row.get("tool_name") or "")

        # 兼容历史数据：如果一条 assistant 里打包了 json payload，做可读展开
        parsed: Dict[str, Any] = {}
        if role == "assistant":
            try:
                maybe = json.loads(content)
                if isinstance(maybe, dict) and ("user_query" in maybe or "assistant_answer" in maybe):
                    parsed = maybe
            except Exception:
                parsed = {}

        lines.append(f"### turn={turn_index} role={role}")
        if parsed:
            uq = str(parsed.get("user_query") or "").strip()
            ans = str(parsed.get("assistant_answer") or "").strip()
            tool_results = parsed.get("tool_results") or []
            if uq:
                lines.append(f"- user_query: {uq}")
            if ans:
                lines.append(f"- assistant_answer: {ans}")
            if tool_results:
                lines.append(f"- tool_results_count: {len(tool_results)}")
        else:
            if tool_name:
                lines.append(f"- tool_name: {tool_name}")
            lines.append(f"- content: {content}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir).resolve() if args.output_dir else (DEFAULT_OUT_BASE / ts)
    _ensure_dir(out_dir)

    with psycopg.connect(args.dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            where_sql, where_params = _where_clause(args.session_id)
            limit_sql = _limit_clause(args.limit)

            relation_map = _build_relation_map(cur)

            sessions = _query_rows(
                cur,
                f"SELECT * FROM chat_sessions {where_sql} ORDER BY updated_at DESC {limit_sql};",
                where_params,
            )
            chat_turns = _query_rows(
                cur,
                f"SELECT * FROM chat_turns {where_sql} ORDER BY session_id, turn_index ASC {limit_sql};",
                where_params,
            )
            summaries = _query_rows(
                cur,
                f"SELECT * FROM memory_summaries {where_sql} ORDER BY session_id, merge_count ASC {limit_sql};",
                where_params,
            )
            facts = _query_rows(
                cur,
                f"""
                SELECT
                    id, session_id, fact_key, fact_value, fact_type, keywords, confidence,
                    is_active, ttl_until, source_turn_id, created_at, updated_at
                FROM memory_facts
                {where_sql}
                ORDER BY session_id, updated_at DESC
                {limit_sql};
                """,
                where_params,
            )

    _write_json(out_dir / "relation_map.json", relation_map)
    _write_json(out_dir / "chat_sessions.json", sessions)
    _write_jsonl(out_dir / "chat_turns.jsonl", chat_turns)
    _write_jsonl(out_dir / "memory_summaries.jsonl", summaries)
    _write_jsonl(out_dir / "memory_facts.jsonl", facts)
    (out_dir / "chat_turns_readable.md").write_text(_render_chat_turns_md(chat_turns), encoding="utf-8")

    summary = {
        "output_dir": str(out_dir),
        "session_filter": args.session_id or "",
        "counts": {
            "chat_sessions": len(sessions),
            "chat_turns": len(chat_turns),
            "memory_summaries": len(summaries),
            "memory_facts": len(facts),
        },
        "files": [
            "relation_map.json",
            "chat_sessions.json",
            "chat_turns.jsonl",
            "chat_turns_readable.md",
            "memory_summaries.jsonl",
            "memory_facts.jsonl",
        ],
    }
    print(_json_safe(summary))


if __name__ == "__main__":
    main()

