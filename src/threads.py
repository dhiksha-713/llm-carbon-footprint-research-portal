"""Research thread management: save, load, list, delete, export."""

from __future__ import annotations

import datetime
import json
import uuid
from pathlib import Path

from src.config import THREADS_DIR


def _thread_path(thread_id: str) -> Path:
    return THREADS_DIR / f"{thread_id}.json"


def save_thread(
    query: str,
    answer: str,
    retrieved_chunks: list[dict],
    citation_validation: dict,
    mode: str,
    provider: str,
    *,
    metadata: dict | None = None,
) -> dict:
    """Persist a research thread to disk and return the full thread dict."""
    thread_id = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    thread = {
        "thread_id": thread_id,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "query": query,
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "citation_validation": citation_validation,
        "mode": mode,
        "provider": provider,
        "metadata": metadata or {},
    }
    _thread_path(thread_id).write_text(
        json.dumps(thread, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return thread


def load_thread(thread_id: str) -> dict | None:
    p = _thread_path(thread_id)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def list_threads() -> list[dict]:
    """Return thread summaries sorted newest-first."""
    threads: list[dict] = []
    for p in sorted(THREADS_DIR.glob("*.json"), reverse=True):
        try:
            t = json.loads(p.read_text(encoding="utf-8"))
            threads.append({
                "thread_id": t["thread_id"],
                "timestamp": t["timestamp"],
                "query": t["query"],
                "mode": t.get("mode", ""),
                "provider": t.get("provider", ""),
                "citations": t.get("citation_validation", {}).get("valid_citations", 0),
                "total_citations": t.get("citation_validation", {}).get("total_citations", 0),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return threads


def delete_thread(thread_id: str) -> bool:
    p = _thread_path(thread_id)
    if p.exists():
        p.unlink()
        return True
    return False


def export_thread_markdown(thread: dict) -> str:
    """Export a single thread as Markdown."""
    cv = thread.get("citation_validation", {})
    lines = [
        f"# Research Thread",
        f"**Date**: {thread['timestamp']}",
        f"**Mode**: {thread['mode']} | **Provider**: {thread['provider']}",
        f"**Citations**: {cv.get('valid_citations', 0)}/{cv.get('total_citations', 0)} valid",
        "",
        "---",
        "",
        f"## Query",
        f"{thread['query']}",
        "",
        "## Answer",
        f"{thread['answer']}",
        "",
        "---",
        "",
        "## Retrieved Evidence",
        "",
    ]
    for i, c in enumerate(thread.get("retrieved_chunks", []), 1):
        sid = c.get("source_id", "?")
        cid = c.get("chunk_id", "?")
        title = c.get("title", "")
        year = c.get("year", "")
        score = c.get("retrieval_score", 0)
        preview = c.get("chunk_text_preview", c.get("chunk_text", ""))[:300]
        lines.append(f"### Chunk {i}: ({sid}, {cid})")
        lines.append(f"**Source**: {title} ({year}) | **Score**: {score:.4f}")
        lines.append(f"> {preview}")
        lines.append("")
    return "\n".join(lines)
