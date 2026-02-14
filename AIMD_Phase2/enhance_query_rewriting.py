"""
enhance_query_rewriting.py
--------------------------
Phase 2 Enhancement: Query Rewriting + Decomposition

For complex/synthesis queries, this module:
1. Rewrites the query for better retrieval
2. Decomposes multi-hop queries into sub-queries
3. Retrieves for each sub-query, merges + deduplicates results
4. Generates a synthesis answer citing all supporting chunks

This addresses the Phase 1 finding that synthesis queries require
multi-source grounding — directly mapping to our Cross-Source Synthesis task.

Run: python enhance_query_rewriting.py --query "Compare BLOOM and GPT energy estimates"
     python enhance_query_rewriting.py --eval
"""

import os
import json
import datetime
import re
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import anthropic

from rag import load_index, retrieve, validate_citations, save_log, EMBED_MODEL, TOP_K, CLAUDE_MODEL

LOGS_DIR      = Path("logs")
PROMPT_VERSION_ENHANCED = "RAG-ENHANCED-REWRITE-V1"

# ── Sub-query decomposition prompt ─────────────────────────────────────
DECOMPOSE_SYSTEM = """You are a research assistant helping decompose complex research questions.
Given a complex query, output 2-4 focused sub-queries that together cover the full question.
Each sub-query should be retrievable from a single passage.
Output ONLY a JSON array of strings. No explanation."""

REWRITE_SYSTEM = """You are a search query optimizer for academic literature retrieval.
Rewrite the given query to be more specific and retrieval-friendly.
- Replace vague terms with domain-specific terminology
- Add relevant synonyms in parentheses if helpful
- Keep it under 20 words
Output ONLY the rewritten query string. No explanation."""

# ── Generation prompt for enhanced (multi-chunk) answers ───────────────
ENHANCED_SYSTEM = """You are a research assistant synthesizing evidence from multiple retrieved passages.

CRITICAL RULES:
1. ONLY use information from the provided context chunks.
2. Every factual claim MUST be followed by an inline citation: (source_id, chunk_id).
3. If the context does NOT support a claim, write "The corpus does not contain evidence for this claim."
4. Do NOT invent, guess, or extrapolate beyond the provided text.
5. When sources AGREE on a point, note the agreement and cite both.
6. When sources DISAGREE, explicitly flag the conflict: "[CONFLICT: source A says X; source B says Y]"
7. Preserve hedging language (e.g., 'approximately', 'estimated', 'may').
8. End with a REFERENCE LIST of all cited sources."""

def decompose_query(query: str, client: anthropic.Anthropic) -> list[str]:
    """Use Claude to decompose a complex query into sub-queries."""
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=300,
        system=DECOMPOSE_SYSTEM,
        messages=[{"role": "user", "content": f"Decompose this research query:\n\n{query}"}],
    )
    text = response.content[0].text.strip()
    # Parse JSON array
    try:
        sub_queries = json.loads(text)
        if isinstance(sub_queries, list):
            return [str(q) for q in sub_queries]
    except json.JSONDecodeError:
        # Fallback: extract lines
        lines = [l.strip().strip('"').strip("'").strip("-").strip()
                 for l in text.split("\n") if l.strip()]
        return [l for l in lines if l and len(l) > 10][:4]
    return [query]

def rewrite_query(query: str, client: anthropic.Anthropic) -> str:
    """Use Claude to rewrite query for better retrieval."""
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=100,
        system=REWRITE_SYSTEM,
        messages=[{"role": "user", "content": query}],
    )
    return response.content[0].text.strip()

def classify_query(query: str) -> str:
    """Classify query type to decide retrieval strategy."""
    q_lower = query.lower()
    synthesis_keywords = ["compare", "contrast", "difference", "agree", "disagree",
                          "both", "across", "versus", "vs", "relationship", "how do"]
    multihop_keywords  = ["why", "what causes", "what leads", "how does", "explain",
                          "what factors", "what determines"]
    edge_keywords      = ["does the corpus", "is there evidence", "are there", "does any"]

    if any(k in q_lower for k in edge_keywords):
        return "edge_case"
    elif any(k in q_lower for k in synthesis_keywords):
        return "synthesis"
    elif any(k in q_lower for k in multihop_keywords):
        return "multihop"
    else:
        return "direct"

def merge_chunks(chunk_lists: list[list[dict]], top_n: int = 8) -> list[dict]:
    """Merge and deduplicate chunks from multiple sub-queries, ranked by score."""
    seen = set()
    merged = []
    # Collect all chunks
    all_chunks = []
    for chunks in chunk_lists:
        all_chunks.extend(chunks)
    # Sort by score descending
    all_chunks.sort(key=lambda c: c["retrieval_score"], reverse=True)
    for c in all_chunks:
        key = (c["source_id"], c["chunk_id"])
        if key not in seen:
            seen.add(key)
            merged.append(c)
        if len(merged) >= top_n:
            break
    return merged

def build_enhanced_prompt(query: str, chunks: list[dict],
                          sub_queries: list[str], rewritten: str) -> str:
    context_blocks = []
    for i, c in enumerate(chunks):
        block = (
            f"--- CHUNK {i+1} ---\n"
            f"source_id: {c['source_id']}\n"
            f"chunk_id: {c['chunk_id']}\n"
            f"title: {c['title']}\n"
            f"authors: {c['authors']} ({c['year']})\n"
            f"section: {c['section_header']}\n"
            f"retrieval_score: {c['retrieval_score']:.4f}\n\n"
            f"{c['chunk_text']}\n"
        )
        context_blocks.append(block)
    context_str = "\n".join(context_blocks)

    sub_q_str = "\n".join(f"  {i+1}. {q}" for i, q in enumerate(sub_queries))

    return f"""ORIGINAL QUERY: {query}
REWRITTEN QUERY: {rewritten}
SUB-QUERIES USED FOR RETRIEVAL:
{sub_q_str}

CONTEXT CHUNKS ({len(chunks)} total):
{context_str}

---
TASK: Answer the original query by synthesizing the provided chunks.
- Use inline citations (source_id, chunk_id) for every claim.
- Note agreements and conflicts across sources explicitly.
- If the corpus lacks evidence for part of the query, say so clearly.
- End with a REFERENCE LIST.
"""

def generate_enhanced_answer(query: str, chunks: list[dict],
                              sub_queries: list[str], rewritten: str,
                              client: anthropic.Anthropic) -> dict:
    user_prompt = build_enhanced_prompt(query, chunks, sub_queries, rewritten)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2000,
        system=ENHANCED_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )
    answer_text = response.content[0].text
    citations_found = re.findall(r'\((\w+\d{4}(?:_\w+)?),\s*(chunk_\d+)\)', answer_text)
    return {
        "answer": answer_text,
        "citations_extracted": [{"source_id": s, "chunk_id": c} for s, c in citations_found],
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }

def run_enhanced_rag(query: str, index, chunk_store: list,
                     embed_model: SentenceTransformer,
                     client: anthropic.Anthropic) -> dict:
    """Enhanced RAG pipeline with query rewriting and decomposition."""
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    query_type = classify_query(query)

    # Step 1: Rewrite query
    rewritten = rewrite_query(query, client)
    print(f"  Rewritten: {rewritten}")

    # Step 2: Decompose if synthesis/multihop
    if query_type in ("synthesis", "multihop"):
        sub_queries = decompose_query(query, client)
        print(f"  Sub-queries: {sub_queries}")
    else:
        sub_queries = [rewritten]

    # Step 3: Retrieve for each sub-query
    chunk_lists = []
    for sq in sub_queries:
        chunks = retrieve(sq, index, chunk_store, embed_model, TOP_K)
        chunk_lists.append(chunks)

    # Step 4: Merge + deduplicate
    merged_chunks = merge_chunks(chunk_lists, top_n=8)
    print(f"  Retrieved {len(merged_chunks)} unique chunks across {len(sub_queries)} sub-queries")

    # Step 5: Generate
    gen = generate_enhanced_answer(query, merged_chunks, sub_queries, rewritten, client)

    # Step 6: Validate citations
    citation_validation = validate_citations(gen["citations_extracted"], merged_chunks)

    # Step 7: Log
    log_entry = {
        "run_id":        f"{timestamp}_{hash(query) % 100000:05d}",
        "timestamp":     timestamp,
        "prompt_version": PROMPT_VERSION_ENHANCED,
        "mode":          "enhanced_rewrite",
        "query_type":    query_type,
        "query":         query,
        "rewritten_query": rewritten,
        "sub_queries":   sub_queries,
        "top_k":         TOP_K,
        "retrieved_chunks": [
            {
                "source_id":       c["source_id"],
                "chunk_id":        c["chunk_id"],
                "title":           c["title"],
                "year":            c["year"],
                "section_header":  c["section_header"],
                "retrieval_score": c["retrieval_score"],
                "chunk_text_preview": c["chunk_text"][:200] + "...",
            }
            for c in merged_chunks
        ],
        "answer": gen["answer"],
        "citations_extracted": gen["citations_extracted"],
        "citation_validation": citation_validation,
        "tokens": {
            "input":  gen["input_tokens"],
            "output": gen["output_tokens"],
        },
    }

    save_log(log_entry)
    return log_entry

# ── CLI ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from rag import load_index, print_result

    parser = argparse.ArgumentParser(description="Enhanced RAG with Query Rewriting")
    parser.add_argument("--query", type=str)
    parser.add_argument("--eval",  action="store_true")
    args = parser.parse_args()

    print("Loading index and models...")
    index, chunk_store = load_index()
    embed_model = SentenceTransformer(EMBED_MODEL)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    if args.query:
        result = run_enhanced_rag(args.query, index, chunk_store, embed_model, client)
        print_result(result)
    elif args.eval:
        from evaluation import EVAL_QUERIES
        synthesis_queries = [q for q in EVAL_QUERIES if q["type"] in ("synthesis", "multihop")]
        print(f"Running enhanced RAG on {len(synthesis_queries)} synthesis/multihop queries...")
        for i, q in enumerate(synthesis_queries, 1):
            print(f"\n[{i}] {q['query'][:70]}...")
            result = run_enhanced_rag(q["query"], index, chunk_store, embed_model, client)
            print_result(result)
    else:
        parser.print_help()
