"""
Query Logging

What it is: appends one JSON-line log record per RAG query (testbench
generation or debug analysis) to a local log file.

Why required: §25 requires every RAG query to be logged — timestamp,
mode, the query, retrieved sources, retrieval distances, the LLM model
used, and the generation result status — so that later you can show
*what the system actually did* for a given interaction, not just its
final answer. This is also what makes the "retrieval inspection" UI
requirement auditable after the fact, not just visible live.

Input: mode ("testbench" | "debug"), the query text, a RetrievedContext,
the LLM model name, and a status string ("success" | "error: ...").
Output: nothing returned — the durable output is one JSON line appended
to settings.log_path / "query_log.jsonl".

How it connects: TestbenchGenerator.generate() and DebugAnalyzer.analyze()
call log_query() once per call, wrapped so a logging failure (e.g.
unwritable disk) never breaks the actual generation/analysis result.

What would happen if it were removed: there would be no record of what
was retrieved or sent for any past query — every "why did it answer that
way?" question would be unanswerable after the fact.

IMPORTANT: never logs settings.llm_api_key or any other secret. Only the
model *name* (e.g. "claude-sonnet-4-6") is recorded, never the key used
to authenticate with it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.schemas.models import RetrievedContext

LOG_FILENAME = "query_log.jsonl"


def log_query(
    log_path: Path,
    mode: str,
    query: str,
    context: Optional[RetrievedContext],
    llm_model: str,
    status: str,
) -> None:
    """
    Append one query record to <log_path>/query_log.jsonl. Never raises —
    a logging failure must not break the actual testbench/debug result,
    so any exception here is swallowed after being written to stderr via
    a plain print (avoiding a dependency on Python's logging module
    being configured, which is out of scope for this project's size).
    """
    try:
        log_path = Path(log_path)
        log_path.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "query": query,
            "retrieved_sources": sorted({r.chunk.source for r in context.results}) if context else [],
            "retrieval_distances": [round(r.score, 4) for r in context.results] if context else [],
            "llm_model": llm_model,
            "status": status,
        }
        with open(log_path / LOG_FILENAME, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:  # noqa: BLE001 - logging must never break the caller
        print(f"[warning] failed to write query log: {exc}")


def read_recent_queries(log_path: Path, limit: int = 20) -> list[dict]:
    """Read the most recent `limit` query log records, newest last. Used
    by the Streamlit Knowledge Base tab to show recent activity. Returns
    [] if no log file exists yet."""
    log_file = Path(log_path) / LOG_FILENAME
    if not log_file.exists():
        return []
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    records = []
    for line in lines[-limit:]:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records
