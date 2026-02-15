"""Enhanced RAG: query rewriting, decomposition, multi-sub-query retrieval, synthesis."""

from __future__ import annotations

import datetime
import json
import re

from sentence_transformers import SentenceTransformer

from src.config import (
    GENERATION_TEMPERATURE,
    MAX_OUTPUT_TOKENS, TOP_K, ENHANCED_TOP_N, ENHANCED_PROMPT_VERSION,
    EMBED_MODEL_NAME, DECOMPOSE_TEMPERATURE, DECOMPOSE_MAX_TOKENS,
    REWRITE_TEMPERATURE, REWRITE_MAX_TOKENS, MAX_SUB_QUERIES,
    CHUNK_PREVIEW_LEN,
)
from src.llm_client import LLMClient, get_llm_client
from src.rag.rag import retrieve, validate_citations, save_log, load_index, print_result
from src.utils import sanitize_query

# ── Regex / prompts ───────────────────────────────────────────────────────
_CITE_RE = re.compile(r"\((\w+\d{4}(?:_\w+)?),\s*(chunk_\d+)\)")

_DECOMPOSE_INSTRUCTION = (
    "You decompose complex research questions into 2-4 focused sub-queries. "
    "Each sub-query should be retrievable from a single passage. "
    "Output ONLY a JSON array of strings."
)

_REWRITE_INSTRUCTION = (
    "You are a search query optimizer for academic literature retrieval. "
    "Rewrite the query with domain-specific terminology, under 20 words. "
    "Output ONLY the rewritten query."
)

_SYNTHESIS_INSTRUCTION = (
    "You synthesize evidence from multiple retrieved passages.\n\n"
    "RULES:\n"
    "1. Use ONLY the provided context chunks.\n"
    "2. Every claim MUST have an inline citation: (source_id, chunk_id).\n"
    "3. If context lacks evidence, state: 'The corpus does not contain evidence for this claim.'\n"
    "4. When sources agree, note agreement and cite both.\n"
    "5. When sources disagree, flag: '[CONFLICT: source A says X; source B says Y]'\n"
    "6. Preserve hedging language.\n"
    "7. End with a REFERENCE LIST."
)

# ── Query classification keywords ────────────────────────────────────────
_SYNTHESIS_KW = {"compare", "contrast", "difference", "agree", "disagree",
                 "both", "across", "versus", "vs", "relationship"}
_MULTIHOP_KW  = {"why", "what causes", "what leads", "how does", "explain",
                 "what factors", "what determines"}
_EDGE_KW      = {"does the corpus", "is there evidence", "are there", "does any"}


# ── Helpers ───────────────────────────────────────────────────────────────
def classify_query(query: str) -> str:
    q = query.lower()
    if any(k in q for k in _EDGE_KW):
        return "edge_case"
    if any(k in q for k in _SYNTHESIS_KW):
        return "synthesis"
    if any(k in q for k in _MULTIHOP_KW):
        return "multihop"
    return "direct"


def decompose_query(query: str, client: LLMClient) -> list[str]:
    resp = client.generate(
        f"Decompose this research query:\n\n{query}",
        system=_DECOMPOSE_INSTRUCTION,
        temperature=DECOMPOSE_TEMPERATURE,
        max_tokens=DECOMPOSE_MAX_TOKENS,
    )
    text = resp.text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(q) for q in parsed][:MAX_SUB_QUERIES]
    except json.JSONDecodeError:
        lines = [ln.strip().strip('"\'- ') for ln in text.split("\n") if ln.strip()]
        return [ln for ln in lines if len(ln) > 10][:MAX_SUB_QUERIES]
    return [query]


def rewrite_query(query: str, client: LLMClient) -> str:
    resp = client.generate(
        query,
        system=_REWRITE_INSTRUCTION,
        temperature=REWRITE_TEMPERATURE,
        max_tokens=REWRITE_MAX_TOKENS,
    )
    return resp.text.strip() or query


def _merge_chunks(chunk_lists: list[list[dict]], top_n: int = ENHANCED_TOP_N) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    merged: list[dict] = []
    pool = sorted(
        (c for cl in chunk_lists for c in cl),
        key=lambda c: c["retrieval_score"],
        reverse=True,
    )
    for c in pool:
        key = (c["source_id"], c["chunk_id"])
        if key not in seen:
            seen.add(key)
            merged.append(c)
        if len(merged) >= top_n:
            break
    return merged


def _build_synthesis_prompt(
    query: str, chunks: list[dict], sub_queries: list[str], rewritten: str,
) -> str:
    blocks: list[str] = []
    for i, c in enumerate(chunks):
        blocks.append(
            f"--- CHUNK {i + 1} ---\n"
            f"source_id: {c['source_id']}\nchunk_id: {c['chunk_id']}\n"
            f"title: {c['title']}\nauthors: {c['authors']} ({c['year']})\n"
            f"section: {c['section_header']}\nscore: {c['retrieval_score']:.4f}\n\n"
            f"{c['chunk_text']}"
        )
    subs = "\n".join(f"  {i + 1}. {q}" for i, q in enumerate(sub_queries))
    return (
        f"ORIGINAL QUERY: {query}\nREWRITTEN: {rewritten}\n"
        f"SUB-QUERIES:\n{subs}\n\nCONTEXT ({len(chunks)} chunks):\n"
        + "\n\n".join(blocks)
        + "\n\n---\nSynthesize the chunks to answer the original query. "
        "Cite every claim. Flag conflicts. End with REFERENCE LIST."
    )


# ── Pipeline ──────────────────────────────────────────────────────────────
def run_enhanced_rag(
    query: str,
    index,
    store: list[dict],
    embed_model: SentenceTransformer,
    client: LLMClient,
) -> dict:
    """Enhanced RAG: sanitize → classify → rewrite → decompose → multi-retrieve → synthesize."""
    query = sanitize_query(query)
    ts = datetime.datetime.utcnow().isoformat() + "Z"
    qtype = classify_query(query)

    rewritten = rewrite_query(query, client)
    sub_queries = (
        decompose_query(query, client) if qtype in ("synthesis", "multihop") else [rewritten]
    )

    chunk_lists = [retrieve(sq, index, store, embed_model, TOP_K) for sq in sub_queries]
    merged = _merge_chunks(chunk_lists)

    resp = client.generate(
        _build_synthesis_prompt(query, merged, sub_queries, rewritten),
        system=_SYNTHESIS_INSTRUCTION,
        temperature=GENERATION_TEMPERATURE,
        max_tokens=MAX_OUTPUT_TOKENS,
    )

    cites = [{"source_id": s, "chunk_id": c} for s, c in _CITE_RE.findall(resp.text)]
    cv = validate_citations(cites, merged)

    entry = {
        "run_id": f"{ts}_{hash(query) % 100000:05d}",
        "timestamp": ts,
        "prompt_version": ENHANCED_PROMPT_VERSION,
        "mode": "enhanced_rewrite",
        "query_type": qtype,
        "query": query,
        "rewritten_query": rewritten,
        "sub_queries": sub_queries,
        "top_k": TOP_K,
        "retrieved_chunks": [
            {
                "source_id": c["source_id"],
                "chunk_id": c["chunk_id"],
                "title": c["title"],
                "year": c["year"],
                "section_header": c["section_header"],
                "retrieval_score": c["retrieval_score"],
                "chunk_text_preview": c["chunk_text"][:CHUNK_PREVIEW_LEN],
            }
            for c in merged
        ],
        "answer": resp.text,
        "citations_extracted": cites,
        "citation_validation": cv,
        "tokens": {"input": resp.input_tokens, "output": resp.output_tokens},
    }
    save_log(entry)
    return entry


# ── CLI ───────────────────────────────────────────────────────────────────
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced RAG CLI")
    parser.add_argument("--query", type=str)
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    index, store = load_index()
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    client = get_llm_client()

    if args.query:
        print_result(run_enhanced_rag(args.query, index, store, embed_model, client))
    elif args.eval:
        from src.eval.evaluation import EVAL_QUERIES
        for q in (q for q in EVAL_QUERIES if q["type"] in ("synthesis", "multihop")):
            print(f"\n[{q['id']}] {q['query'][:70]}...")
            print_result(run_enhanced_rag(q["query"], index, store, embed_model, client))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
