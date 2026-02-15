"""Utility functions: input sanitization, prompt-injection protection."""

from __future__ import annotations

import re

MAX_QUERY_LEN = 1000

# Patterns that indicate prompt-injection attempts
_INJECTION_PATTERNS = re.compile(
    r"|".join([
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"forget\s+(all\s+)?previous",
        r"disregard\s+(all\s+)?(above|instructions|rules)",
        r"you\s+are\s+now",
        r"new\s+instructions\s*:",
        r"system\s*:\s*",
        r"<\s*/?\s*system\s*>",
        r"<\s*/?\s*prompt\s*>",
        r"\]\s*\[",           # bracket injection
        r"ASSISTANT\s*:",     # role injection
        r"USER\s*:",          # role injection
    ]),
    re.IGNORECASE,
)


def sanitize_query(query: str) -> str:
    """Sanitize user input before it enters the RAG pipeline.

    1. Strip control characters (keep newlines, tabs, spaces).
    2. Enforce maximum length.
    3. Reject prompt-injection patterns.

    Returns the cleaned query, or raises ValueError on injection.
    """
    # Strip control characters
    query = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", query)

    # Trim whitespace and enforce length
    query = query.strip()
    if not query:
        raise ValueError("Query is empty after sanitization.")
    if len(query) > MAX_QUERY_LEN:
        query = query[:MAX_QUERY_LEN]

    # Block prompt injection
    if _INJECTION_PATTERNS.search(query):
        raise ValueError(
            "Query rejected: potential prompt-injection pattern detected."
        )

    return query
