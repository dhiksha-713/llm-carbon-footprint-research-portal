"""
rag.py
------
Phase 2 — Baseline RAG Engine
- Semantic retrieval from FAISS index
- Citation-backed answer generation via Claude API
- Full logging of query, retrieved chunks, prompt, response
- Trust behavior: refuses to invent citations, flags missing evidence

Run: python rag.py --query "What are the main sources of LLM carbon emissions?"
     python rag.py --eval      (runs full 20-query evaluation set)
"""

import argparse
import json
import os
import re
import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import anthropic

# ── Config ─────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
LOGS_DIR      = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

TOP_K         = 5          # chunks to retrieve
EMBED_MODEL   = "all-MiniLM-L6-v2"
CLAUDE_MODEL  = "claude-sonnet-4-5-20250929"
PROMPT_VERSION = "RAG-BASELINE-V1"

# ── Load index and chunk store ─────────────────────────────────────────
def load_index():
    index = faiss.read_index(str(PROCESSED_DIR / "faiss_index.bin"))
    with open(PROCESSED_DIR / "chunk_store.json", encoding="utf-8") as f:
        chunk_store = json.load(f)
    return index, chunk_store

# ── Retrieval ──────────────────────────────────────────────────────────
def retrieve(query: str, index, chunk_store: list[dict],
             model: SentenceTransformer, top_k: int = TOP_K) -> list[dict]:
    """Embed query and return top-k chunks with scores."""
    q_emb = model.encode([query], show_progress_bar=False).astype("float32")
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-10)
    scores, idxs = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        chunk = chunk_store[idx].copy()
        chunk["retrieval_score"] = float(score)
        results.append(chunk)
    return results

# ── Prompt Construction ────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a research assistant for a systematic review on LLM carbon footprints.

CRITICAL RULES:
1. ONLY use information from the provided context chunks.
2. Every factual claim MUST be followed by an inline citation in format (source_id, chunk_id).
3. If the context does NOT support a claim, write "The corpus does not contain evidence for this claim."
4. Do NOT invent, guess, or extrapolate beyond the provided text.
5. Preserve hedging language from sources (e.g., "approximately", "estimated", "may").
6. If evidence is conflicting between sources, explicitly note the conflict.
7. End your answer with a REFERENCE LIST of all sources you cited."""

def build_user_prompt(query: str, chunks: list[dict]) -> str:
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
    return f"""CONTEXT CHUNKS:
{context_str}

---
RESEARCH QUESTION: {query}

Instructions:
- Answer using ONLY the chunks above.
- Cite every claim as (source_id, chunk_id), e.g., (strubell2019, chunk_003).
- If the chunks don't contain enough information, say so explicitly.
- End with a REFERENCE LIST.
"""

# ── Generation ─────────────────────────────────────────────────────────
def generate_answer(query: str, chunks: list[dict], client: anthropic.Anthropic) -> dict:
    """Call Claude and return structured response dict."""
    user_prompt = build_user_prompt(query, chunks)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1500,
        system=SYSTEM_PROMPT,
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

# ── Citation Validation ────────────────────────────────────────────────
def validate_citations(citations: list[dict], retrieved_chunks: list[dict]) -> dict:
    """Check that all cited chunks actually appear in retrieved context."""
    retrieved_ids = {(c["source_id"], c["chunk_id"]) for c in retrieved_chunks}
    valid, invalid = [], []
    for cit in citations:
        pair = (cit["source_id"], cit["chunk_id"])
        if pair in retrieved_ids:
            valid.append(cit)
        else:
            invalid.append(cit)
    return {
        "total_citations": len(citations),
        "valid_citations": len(valid),
        "invalid_citations": len(invalid),
        "citation_precision": len(valid) / len(citations) if citations else None,
        "invalid_list": invalid,
    }

# ── Logging ────────────────────────────────────────────────────────────
def save_log(log_entry: dict):
    """Append a log entry to JSONL log file."""
    log_path = LOGS_DIR / "rag_runs.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

# ── Main RAG pipeline ──────────────────────────────────────────────────
def run_rag(query: str, index, chunk_store: list,
            embed_model: SentenceTransformer, client: anthropic.Anthropic,
            top_k: int = TOP_K, mode: str = "baseline") -> dict:
    """End-to-end RAG: retrieve → generate → validate → log."""
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"

    # 1. Retrieve
    chunks = retrieve(query, index, chunk_store, embed_model, top_k)

    # 2. Generate
    gen = generate_answer(query, chunks, client)

    # 3. Validate citations
    citation_validation = validate_citations(gen["citations_extracted"], chunks)

    # 4. Build log entry
    log_entry = {
        "run_id":        f"{timestamp}_{hash(query) % 100000:05d}",
        "timestamp":     timestamp,
        "prompt_version": PROMPT_VERSION,
        "mode":          mode,
        "query":         query,
        "top_k":         top_k,
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
            for c in chunks
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
def print_result(result: dict):
    print("\n" + "=" * 70)
    print(f"QUERY: {result['query']}")
    print("=" * 70)
    print(f"\nRetrieved {len(result['retrieved_chunks'])} chunks:")
    for c in result["retrieved_chunks"]:
        print(f"  [{c['source_id']}, {c['chunk_id']}] score={c['retrieval_score']:.4f} | {c['title'][:50]}")
    print("\nANSWER:")
    print("-" * 70)
    print(result["answer"])
    print("-" * 70)
    cv = result["citation_validation"]
    print(f"\nCitation check: {cv['valid_citations']}/{cv['total_citations']} valid "
          f"| precision={cv['citation_precision']}")
    if cv["invalid_list"]:
        print(f"  ⚠ Invalid citations: {cv['invalid_list']}")
    print(f"Tokens used: {result['tokens']['input']} in / {result['tokens']['output']} out")

def main():
    parser = argparse.ArgumentParser(description="Phase 2 Baseline RAG")
    parser.add_argument("--query", type=str, help="Single query to run")
    parser.add_argument("--eval",  action="store_true", help="Run full evaluation set")
    parser.add_argument("--top_k", type=int, default=TOP_K)
    args = parser.parse_args()

    # Load everything
    print("Loading index and models...")
    index, chunk_store = load_index()
    embed_model = SentenceTransformer(EMBED_MODEL)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    if args.query:
        result = run_rag(args.query, index, chunk_store, embed_model, client, args.top_k)
        print_result(result)

    elif args.eval:
        from evaluation import EVAL_QUERIES
        print(f"\nRunning evaluation set: {len(EVAL_QUERIES)} queries...")
        for i, q in enumerate(EVAL_QUERIES, 1):
            print(f"\n[{i}/{len(EVAL_QUERIES)}] {q['type'].upper()}: {q['query'][:70]}...")
            result = run_rag(q["query"], index, chunk_store, embed_model, client, args.top_k)
            print_result(result)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
