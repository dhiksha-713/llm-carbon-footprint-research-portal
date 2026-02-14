"""Baseline RAG engine: semantic retrieval + Gemini generation + citation validation + logging."""

import argparse
import json
import re
import datetime
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

from src.config import (
    PROCESSED_DIR, LOGS_DIR, TOP_K, EMBED_MODEL_NAME,
    GENERATION_MODEL, GENERATION_TEMPERATURE,
    MAX_OUTPUT_TOKENS, BASELINE_PROMPT_VERSION,
)

SYSTEM_PROMPT = (
    "You are a research assistant for a systematic review on LLM carbon footprints.\n\n"
    "RULES:\n"
    "1. Use ONLY the provided context chunks.\n"
    "2. Every factual claim MUST have an inline citation: (source_id, chunk_id).\n"
    "3. If context lacks evidence, state: 'The corpus does not contain evidence for this claim.'\n"
    "4. Do NOT invent or extrapolate beyond the provided text.\n"
    "5. Preserve hedging language (approximately, estimated, may).\n"
    "6. Flag conflicting evidence explicitly.\n"
    "7. End with a REFERENCE LIST of cited sources."
)

_CITE_RE = re.compile(r"\((\w+\d{4}(?:_\w+)?),\s*(chunk_\d+)\)")


# ── Index & Store ────────────────────────────────────────────────────────
def load_index():
    index = faiss.read_index(str(PROCESSED_DIR / "faiss_index.bin"))
    store = json.loads((PROCESSED_DIR / "chunk_store.json").read_text(encoding="utf-8"))
    return index, store


# ── Retrieval ────────────────────────────────────────────────────────────
def retrieve(
    query: str,
    index: faiss.Index,
    store: list[dict],
    embed_model: SentenceTransformer,
    top_k: int = TOP_K,
) -> list[dict]:
    q_emb = embed_model.encode([query], show_progress_bar=False).astype("float32")
    q_emb /= np.linalg.norm(q_emb) + 1e-10
    scores, idxs = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        chunk = store[idx].copy()
        chunk["retrieval_score"] = float(score)
        results.append(chunk)
    return results


# ── Prompt ───────────────────────────────────────────────────────────────
def _build_context(chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks):
        blocks.append(
            f"--- CHUNK {i+1} ---\n"
            f"source_id: {c['source_id']}\nchunk_id: {c['chunk_id']}\n"
            f"title: {c['title']}\nauthors: {c['authors']} ({c['year']})\n"
            f"section: {c['section_header']}\nscore: {c['retrieval_score']:.4f}\n\n"
            f"{c['chunk_text']}"
        )
    return "\n\n".join(blocks)


def _build_user_prompt(query: str, chunks: list[dict]) -> str:
    return (
        f"CONTEXT CHUNKS:\n{_build_context(chunks)}\n\n---\n"
        f"RESEARCH QUESTION: {query}\n\n"
        "Cite every claim as (source_id, chunk_id). "
        "If evidence is insufficient, say so. End with a REFERENCE LIST."
    )


# ── Generation ───────────────────────────────────────────────────────────
def _get_client() -> genai.Client:
    return genai.Client()


def generate_answer(query: str, chunks: list[dict], client: genai.Client) -> dict:
    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=_build_user_prompt(query, chunks),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=GENERATION_TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        ),
    )
    text = response.text or ""
    cites = [{"source_id": s, "chunk_id": c} for s, c in _CITE_RE.findall(text)]
    usage = response.usage_metadata
    return {
        "answer": text,
        "citations_extracted": cites,
        "input_tokens": getattr(usage, "prompt_token_count", 0),
        "output_tokens": getattr(usage, "candidates_token_count", 0),
    }


# ── Citation Validation ──────────────────────────────────────────────────
def validate_citations(citations: list[dict], chunks: list[dict]) -> dict:
    retrieved = {(c["source_id"], c["chunk_id"]) for c in chunks}
    valid = [c for c in citations if (c["source_id"], c["chunk_id"]) in retrieved]
    invalid = [c for c in citations if (c["source_id"], c["chunk_id"]) not in retrieved]
    total = len(citations)
    return {
        "total_citations": total,
        "valid_citations": len(valid),
        "invalid_citations": len(invalid),
        "citation_precision": len(valid) / total if total else None,
        "invalid_list": invalid,
    }


# ── Logging ──────────────────────────────────────────────────────────────
def save_log(entry: dict) -> None:
    path = LOGS_DIR / "rag_runs.jsonl"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Pipeline ─────────────────────────────────────────────────────────────
def run_rag(
    query: str,
    index: faiss.Index,
    store: list[dict],
    embed_model: SentenceTransformer,
    client: genai.Client,
    top_k: int = TOP_K,
    mode: str = "baseline",
) -> dict:
    ts = datetime.datetime.utcnow().isoformat() + "Z"
    chunks = retrieve(query, index, store, embed_model, top_k)
    gen = generate_answer(query, chunks, client)
    cv = validate_citations(gen["citations_extracted"], chunks)

    entry = {
        "run_id": f"{ts}_{hash(query) % 100000:05d}",
        "timestamp": ts,
        "prompt_version": BASELINE_PROMPT_VERSION,
        "mode": mode,
        "query": query,
        "top_k": top_k,
        "retrieved_chunks": [
            {
                "source_id": c["source_id"],
                "chunk_id": c["chunk_id"],
                "title": c["title"],
                "year": c["year"],
                "section_header": c["section_header"],
                "retrieval_score": c["retrieval_score"],
                "chunk_text_preview": c["chunk_text"][:200],
            }
            for c in chunks
        ],
        "answer": gen["answer"],
        "citations_extracted": gen["citations_extracted"],
        "citation_validation": cv,
        "tokens": {"input": gen["input_tokens"], "output": gen["output_tokens"]},
    }
    save_log(entry)
    return entry


# ── CLI helpers ──────────────────────────────────────────────────────────
def print_result(result: dict) -> None:
    print(f"\n{'='*70}\nQUERY: {result['query']}\n{'='*70}")
    for c in result["retrieved_chunks"]:
        print(f"  [{c['source_id']}, {c['chunk_id']}] score={c['retrieval_score']:.4f}")
    print(f"\n{result['answer']}\n{'-'*70}")
    cv = result["citation_validation"]
    print(f"Citations: {cv['valid_citations']}/{cv['total_citations']} valid | precision={cv['citation_precision']}")
    if cv["invalid_list"]:
        print(f"  Invalid: {cv['invalid_list']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--top_k", type=int, default=TOP_K)
    args = parser.parse_args()

    index, store = load_index()
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    client = _get_client()

    if args.query:
        print_result(run_rag(args.query, index, store, embed_model, client, args.top_k))
    elif args.eval:
        from src.eval.evaluation import EVAL_QUERIES
        for i, q in enumerate(EVAL_QUERIES, 1):
            print(f"\n[{i}/{len(EVAL_QUERIES)}] {q['type']}: {q['query'][:70]}...")
            print_result(run_rag(q["query"], index, store, embed_model, client, args.top_k))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
