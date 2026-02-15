"""Enhanced RAG: query rewriting, decomposition, multi-sub-query retrieval, synthesis."""

from __future__ import annotations

import datetime
import json
import logging
import re

from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

from src.config import (
    TOP_K, ENHANCED_TOP_N, ENHANCED_PROMPT_VERSION, EMBED_MODEL_NAME,
    DECOMPOSE_TEMPERATURE, DECOMPOSE_MAX_TOKENS,
    REWRITE_TEMPERATURE, REWRITE_MAX_TOKENS, MAX_SUB_QUERIES,
)
from src.llm_client import LLMClient, get_llm_client
from src.rag.rag import retrieve, validate_citations, save_log, load_index, print_result
from src.utils import sanitize_query, CITE_RE, build_chunk_context, summarize_chunk

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

_SYNTHESIS_KW = {"compare", "contrast", "difference", "agree", "disagree",
                 "both", "across", "versus", "vs", "relationship"}
_MULTIHOP_KW  = {"why", "what causes", "what leads", "how does", "explain",
                 "what factors", "what determines"}
_EDGE_KW      = {"does the corpus", "is there evidence", "are there", "does any"}


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
    log.info("decompose_query: '%s'", query[:80])
    resp = client.generate(
        f"Decompose this research query:\n\n{query}",
        system=_DECOMPOSE_INSTRUCTION,
        temperature=DECOMPOSE_TEMPERATURE, max_tokens=DECOMPOSE_MAX_TOKENS,
    )
    log.debug("decompose raw response: %s", resp.text[:300])
    text = re.sub(r"```(?:json)?\s*", "", resp.text).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            subs = [str(q).strip() for q in parsed if str(q).strip()][:MAX_SUB_QUERIES]
            if subs:
                log.info("decompose -> %d sub-queries (JSON parse)", len(subs))
                return subs
    except (json.JSONDecodeError, TypeError):
        pass
    lines = [ln.strip().strip('"\'- ').rstrip(",") for ln in text.split("\n") if ln.strip()]
    subs = [ln for ln in lines if len(ln) > 10][:MAX_SUB_QUERIES]
    result = subs if subs else [query]
    log.info("decompose -> %d sub-queries (line fallback)", len(result))
    return result


def rewrite_query(query: str, client: LLMClient) -> str:
    resp = client.generate(
        query, system=_REWRITE_INSTRUCTION,
        temperature=REWRITE_TEMPERATURE, max_tokens=REWRITE_MAX_TOKENS,
    )
    rewritten = resp.text.strip() or query
    log.info("rewrite_query: '%s' -> '%s'", query[:60], rewritten[:60])
    return rewritten


def _merge_chunks(chunk_lists: list[list[dict]], top_n: int = ENHANCED_TOP_N) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    merged: list[dict] = []
    for c in sorted((c for cl in chunk_lists for c in cl),
                    key=lambda c: c["retrieval_score"], reverse=True):
        key = (c["source_id"], c["chunk_id"])
        if key not in seen:
            seen.add(key)
            merged.append(c)
        if len(merged) >= top_n:
            break
    return merged


def run_enhanced_rag(
    query: str, index, store: list[dict],
    embed_model: SentenceTransformer, client: LLMClient,
) -> dict:
    query = sanitize_query(query)
    log.info("run_enhanced_rag query='%s'", query[:80])
    ts = datetime.datetime.utcnow().isoformat() + "Z"
    qtype = classify_query(query)
    log.info("classify_query -> %s", qtype)

    rewritten = rewrite_query(query, client)
    sub_queries = (
        decompose_query(query, client) if qtype in ("synthesis", "multihop") else [rewritten]
    )
    if not sub_queries:
        log.warning("decompose returned empty list, falling back to original query")
        sub_queries = [query]
    log.info("sub_queries=%d: %s", len(sub_queries), [q[:50] for q in sub_queries])

    merged = _merge_chunks([retrieve(sq, index, store, embed_model, TOP_K) for sq in sub_queries])
    log.info("merged chunks: %d from %d sub-queries", len(merged), len(sub_queries))

    subs = "\n".join(f"  {i + 1}. {q}" for i, q in enumerate(sub_queries))
    prompt = (
        f"ORIGINAL QUERY: {query}\nREWRITTEN: {rewritten}\n"
        f"SUB-QUERIES:\n{subs}\n\nCONTEXT ({len(merged)} chunks):\n"
        f"{build_chunk_context(merged)}\n\n---\n"
        "Synthesize the chunks to answer the original query. "
        "Cite every claim. Flag conflicts. End with REFERENCE LIST."
    )
    resp = client.generate(prompt, system=_SYNTHESIS_INSTRUCTION)

    cites = [{"source_id": s, "chunk_id": c} for s, c in CITE_RE.findall(resp.text)]
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
        "retrieved_chunks": [summarize_chunk(c) for c in merged],
        "answer": resp.text,
        "citations_extracted": cites,
        "citation_validation": cv,
        "tokens": {"input": resp.input_tokens, "output": resp.output_tokens},
    }
    entry["_log_path"] = save_log(entry)
    return entry


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
        result = run_enhanced_rag(args.query, index, store, embed_model, client)
        print_result(result, log_path=result.get("_log_path"))
    elif args.eval:
        from src.eval.evaluation import EVAL_QUERIES
        for q in (q for q in EVAL_QUERIES if q["type"] in ("synthesis", "multihop")):
            print(f"\n[{q['id']}] {q['query'][:70]}...")
            result = run_enhanced_rag(q["query"], index, store, embed_model, client)
            print_result(result, log_path=result.get("_log_path"))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
