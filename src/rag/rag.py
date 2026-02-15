"""Baseline RAG: semantic retrieval, LLM generation, citation validation, logging."""

from __future__ import annotations

import argparse
import datetime
import json
import logging

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

from src.config import (
    PROCESSED_DIR, LOGS_DIR, TOP_K, EMBED_MODEL_NAME, BASELINE_PROMPT_VERSION,
)
from src.llm_client import LLMClient, get_llm_client
from src.utils import sanitize_query, CITE_RE, build_chunk_context, summarize_chunk

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


def load_index() -> tuple[faiss.Index, list[dict]]:
    log.info("Loading FAISS index from %s", PROCESSED_DIR)
    index = faiss.read_index(str(PROCESSED_DIR / "faiss_index.bin"))
    store = json.loads((PROCESSED_DIR / "chunk_store.json").read_text(encoding="utf-8"))
    log.info("Index loaded: %d vectors, %d chunks", index.ntotal, len(store))
    return index, store


def retrieve(
    query: str, index: faiss.Index, store: list[dict],
    embed_model: SentenceTransformer, top_k: int = TOP_K,
) -> list[dict]:
    q_emb = embed_model.encode([query], show_progress_bar=False).astype("float32")
    q_emb /= np.linalg.norm(q_emb) + 1e-10
    scores, idxs = index.search(q_emb, top_k)
    results: list[dict] = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        chunk = store[idx].copy()
        chunk["retrieval_score"] = float(score)
        results.append(chunk)
    log.info("retrieve top_k=%d -> %d chunks (scores %.4f..%.4f)",
             top_k, len(results),
             results[0]["retrieval_score"] if results else 0,
             results[-1]["retrieval_score"] if results else 0)
    return results


def validate_citations(citations: list[dict], chunks: list[dict]) -> dict:
    retrieved = {(c["source_id"], c["chunk_id"]) for c in chunks}
    valid = [c for c in citations if (c["source_id"], c["chunk_id"]) in retrieved]
    total = len(citations)
    return {
        "total_citations": total,
        "valid_citations": len(valid),
        "invalid_citations": total - len(valid),
        "citation_precision": len(valid) / total if total else None,
        "invalid_list": [c for c in citations if (c["source_id"], c["chunk_id"]) not in retrieved],
    }


def generate_answer(query: str, chunks: list[dict], client: LLMClient) -> dict:
    prompt = (
        f"CONTEXT CHUNKS:\n{build_chunk_context(chunks)}\n\n---\n"
        f"RESEARCH QUESTION: {query}\n\n"
        "Cite every claim as (source_id, chunk_id). "
        "If evidence is insufficient, say so. End with a REFERENCE LIST."
    )
    resp = client.generate(prompt, system=SYSTEM_PROMPT)
    cites = [{"source_id": s, "chunk_id": c} for s, c in CITE_RE.findall(resp.text)]
    log.info("generate_answer -> %d citations extracted, answer_len=%d", len(cites), len(resp.text))
    return {"answer": resp.text, "citations_extracted": cites,
            "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens}


def save_log(entry: dict) -> str:
    log_path = LOGS_DIR / "rag_runs.jsonl"
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log.info("Log entry saved -> %s (run_id=%s)", log_path, entry.get("run_id", "?"))
    return str(log_path)


def run_rag(
    query: str, index: faiss.Index, store: list[dict],
    embed_model: SentenceTransformer, client: LLMClient,
    top_k: int = TOP_K, mode: str = "baseline",
) -> dict:
    query = sanitize_query(query)
    log.info("run_rag mode=%s top_k=%d query='%s'", mode, top_k, query[:80])
    ts = datetime.datetime.utcnow().isoformat() + "Z"
    chunks = retrieve(query, index, store, embed_model, top_k)
    gen = generate_answer(query, chunks, client)
    cv = validate_citations(gen["citations_extracted"], chunks)
    log.info("run_rag -> %d/%d valid citations, precision=%s",
             cv["valid_citations"], cv["total_citations"], cv["citation_precision"])

    entry = {
        "run_id": f"{ts}_{hash(query) % 100000:05d}",
        "timestamp": ts,
        "prompt_version": BASELINE_PROMPT_VERSION,
        "mode": mode,
        "query": query,
        "top_k": top_k,
        "retrieved_chunks": [summarize_chunk(c) for c in chunks],
        "answer": gen["answer"],
        "citations_extracted": gen["citations_extracted"],
        "citation_validation": cv,
        "tokens": {"input": gen["input_tokens"], "output": gen["output_tokens"]},
    }
    entry["_log_path"] = save_log(entry)
    return entry


def print_result(result: dict, log_path: str | None = None) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"QUERY: {result['query']}")
    print(sep)
    print("\n--- RETRIEVAL RESULTS ---")
    for c in result["retrieved_chunks"]:
        print(f"  [{c['source_id']}, {c['chunk_id']}] score={c['retrieval_score']:.4f}")
    print(f"\n--- ANSWER WITH CITATIONS ---\n{result['answer']}")
    print("-" * 70)
    cv = result["citation_validation"]
    print(f"Citations: {cv['valid_citations']}/{cv['total_citations']} "
          f"valid | precision={cv['citation_precision']}")
    if log_path:
        print(f"\n--- LOG ENTRY SAVED ---\n  {log_path}  (run_id: {result['run_id']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline RAG CLI")
    parser.add_argument("--query", type=str)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--top_k", type=int, default=TOP_K)
    args = parser.parse_args()

    index, store = load_index()
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    client = get_llm_client()

    if args.query:
        result = run_rag(args.query, index, store, embed_model, client, args.top_k)
        print_result(result, log_path=result.get("_log_path"))
    elif args.eval:
        from src.eval.evaluation import EVAL_QUERIES
        for i, q in enumerate(EVAL_QUERIES, 1):
            print(f"\n[{i}/{len(EVAL_QUERIES)}] {q['type']}: {q['query'][:70]}...")
            result = run_rag(q["query"], index, store, embed_model, client, args.top_k)
            print_result(result, log_path=result.get("_log_path"))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
