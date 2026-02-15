"""Shared utilities: sanitization, citation parsing, chunk formatting, averages."""

from __future__ import annotations

import re
import glob
import json
from pathlib import Path

from src.config import CHUNK_PREVIEW_LEN, OUTPUTS_DIR

# ── Citation regex (shared by baseline and enhanced RAG) ─────────────────
CITE_RE = re.compile(r"\((\w+\d{4}(?:_\w+)?),\s*(chunk_\d+)\)")

# ── Prompt injection detection ───────────────────────────────────────────
MAX_QUERY_LEN = 1000
_INJECTION_RE = re.compile(
    r"|".join([
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"forget\s+(all\s+)?previous",
        r"disregard\s+(all\s+)?(above|instructions|rules)",
        r"you\s+are\s+now",
        r"new\s+instructions\s*:",
        r"system\s*:\s*",
        r"<\s*/?\s*system\s*>",
        r"<\s*/?\s*prompt\s*>",
        r"\]\s*\[",
        r"ASSISTANT\s*:",
        r"USER\s*:",
    ]),
    re.IGNORECASE,
)


def sanitize_query(query: str) -> str:
    """Strip control chars, enforce length, reject prompt-injection patterns."""
    query = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", query).strip()
    if not query:
        raise ValueError("Query is empty after sanitization.")
    if len(query) > MAX_QUERY_LEN:
        query = query[:MAX_QUERY_LEN]
    if _INJECTION_RE.search(query):
        raise ValueError("Query rejected: potential prompt-injection pattern detected.")
    return query


# ── Chunk formatting (used by both RAG pipelines and evaluation) ─────────

def build_chunk_context(chunks: list[dict]) -> str:
    """Format a list of retrieved chunks into a prompt-ready context block."""
    blocks = []
    for i, c in enumerate(chunks):
        blocks.append(
            f"--- CHUNK {i + 1} ---\n"
            f"source_id: {c['source_id']}\nchunk_id: {c['chunk_id']}\n"
            f"title: {c['title']}\nauthors: {c['authors']} ({c['year']})\n"
            f"section: {c['section_header']}\nscore: {c['retrieval_score']:.4f}\n\n"
            f"{c['chunk_text']}"
        )
    return "\n\n".join(blocks)


def summarize_chunk(c: dict) -> dict:
    """Produce a log-safe summary of a retrieved chunk."""
    return {
        "source_id": c["source_id"],
        "chunk_id": c["chunk_id"],
        "title": c["title"],
        "year": c["year"],
        "section_header": c["section_header"],
        "retrieval_score": c["retrieval_score"],
        "chunk_text_preview": c["chunk_text"][:CHUNK_PREVIEW_LEN],
    }


def chunk_preview_ctx(chunks: list[dict], limit: int | None = None) -> str:
    """One-line-per-chunk context for LLM-judge prompts."""
    subset = chunks[:limit] if limit else chunks
    return "\n".join(
        f"[{c['source_id']}, {c['chunk_id']}]: "
        f"{c.get('chunk_text_preview', c.get('chunk_text', ''))[:CHUNK_PREVIEW_LEN]}"
        for c in subset
    )


# ── Averages (used by eval, report, UI) ──────────────────────────────────

def safe_avg(vals, decimals: int = 2) -> float | None:
    """Average of non-None values, or None if empty."""
    clean = [v for v in vals if v is not None]
    return round(sum(clean) / len(clean), decimals) if clean else None


# ── Eval results loader (used by UI and report) ─────────────────────────

def load_eval_results(mode: str) -> list[dict]:
    """Load the latest eval results JSON for a given mode."""
    files = sorted(glob.glob(str(OUTPUTS_DIR / f"eval_results_{mode}_*.json")))
    if not files:
        return []
    return json.loads(Path(files[-1]).read_text(encoding="utf-8"))
