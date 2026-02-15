"""Evaluation module: 20-query set, LLM-as-judge scoring, metric computation."""

import json
import re
import datetime
import argparse
from pathlib import Path

from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

from src.config import (
    OUTPUTS_DIR, JUDGE_MODEL, JUDGE_TEMPERATURE,
    EMBED_MODEL_NAME, TOP_K, JUDGE_MAX_TOKENS, CHUNK_PREVIEW_LEN,
    MAX_OUTPUT_TOKENS,
)

# ── 20-Query Evaluation Set ─────────────────────────────────────────────
EVAL_QUERIES = [
    {"id": "D01", "type": "direct",
     "query": "What does GPU-hour energy measurement measure, and what are its known failure modes?",
     "expected_sources": ["strubell2019", "patterson2021"]},
    {"id": "D02", "type": "direct",
     "query": "How much CO2 was emitted during the training of BERT according to Strubell et al.?",
     "expected_sources": ["strubell2019"]},
    {"id": "D03", "type": "direct",
     "query": "What is the total lifecycle carbon footprint of BLOOM in tonnes CO2 equivalent?",
     "expected_sources": ["luccioni2022"]},
    {"id": "D04", "type": "direct",
     "query": "What factors does Patterson et al. identify as most impactful for reducing LLM carbon emissions?",
     "expected_sources": ["patterson2021"]},
    {"id": "D05", "type": "direct",
     "query": "How does carbon intensity of the electricity grid affect LLM training emissions?",
     "expected_sources": ["patterson2021", "luccioni2022", "dodge2022"]},
    {"id": "D06", "type": "direct",
     "query": "What is the difference between operational and embodied carbon emissions in AI systems?",
     "expected_sources": ["luccioni2022", "ligozat2022"]},
    {"id": "D07", "type": "direct",
     "query": "What tools exist for tracking carbon emissions during machine learning training?",
     "expected_sources": ["anthony2020", "bannour2021", "lacoste2019"]},
    {"id": "D08", "type": "direct",
     "query": "How does Schwartz et al. define the concept of Red AI versus Green AI?",
     "expected_sources": ["schwartz2020"]},
    {"id": "D09", "type": "direct",
     "query": "What metrics does Henderson et al. recommend for reporting ML energy consumption?",
     "expected_sources": ["henderson2020"]},
    {"id": "D10", "type": "direct",
     "query": "What is the carbon footprint of neural architecture search compared to standard training?",
     "expected_sources": ["strubell2019"]},
    {"id": "S01", "type": "synthesis",
     "query": "Compare Strubell et al. and Patterson et al. on their measurement methodology: where do they agree and disagree on how to estimate training carbon emissions?",
     "expected_sources": ["strubell2019", "patterson2021"]},
    {"id": "S02", "type": "synthesis",
     "query": "How do Luccioni et al. and Patterson et al. differ in their assumptions about hardware efficiency and carbon intensity?",
     "expected_sources": ["luccioni2022", "patterson2021"]},
    {"id": "S03", "type": "synthesis",
     "query": "Across all sources in the corpus, what are the three most commonly cited factors that explain variation in LLM carbon footprint estimates?",
     "expected_sources": ["strubell2019", "patterson2021", "luccioni2022", "henderson2020"]},
    {"id": "S04", "type": "multihop",
     "query": "Why do different studies report dramatically different carbon estimates for similar model sizes, and what does this imply for standardization?",
     "expected_sources": ["henderson2020", "luccioni2022", "dodge2022"]},
    {"id": "S05", "type": "synthesis",
     "query": "How have carbon measurement methods for LLMs evolved from 2019 to 2023, and what gaps remain?",
     "expected_sources": ["strubell2019", "luccioni2022", "luccioni2023", "ligozat2022"]},
    {"id": "E01", "type": "edge_case",
     "query": "Does the corpus contain evidence that LLM inference emissions exceed training emissions over a model's deployment lifetime?",
     "expected_sources": ["luccioni2023", "wu2022"]},
    {"id": "E02", "type": "edge_case",
     "query": "Does the corpus contain evidence about the carbon footprint of GPT-4 specifically?",
     "expected_sources": []},
    {"id": "E03", "type": "edge_case",
     "query": "Is there evidence in the corpus that carbon offset programs effectively neutralize LLM training emissions?",
     "expected_sources": []},
    {"id": "E04", "type": "edge_case",
     "query": "What does the corpus say about the carbon footprint of quantum computing for AI?",
     "expected_sources": []},
    {"id": "E05", "type": "edge_case",
     "query": "Do all sources agree on the carbon intensity of the French electricity grid?",
     "expected_sources": ["luccioni2022", "patterson2021"]},
]


# ── LLM-as-Judge Scoring ────────────────────────────────────────────────
def _judge(client: genai.Client, prompt: str, max_tokens: int = JUDGE_MAX_TOKENS) -> dict:
    """Call the judge model and parse JSON response."""
    resp = client.models.generate_content(
        model=JUDGE_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=JUDGE_TEMPERATURE,
            max_output_tokens=max_tokens,
        ),
    )
    text = re.sub(r"```json|```", "", (resp.text or "")).strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {"score": None, "reasoning": "JSON parse error"}


def score_groundedness(answer: str, chunks: list[dict], client: genai.Client) -> dict:
    ctx = "\n".join(
        f"[{c['source_id']}, {c['chunk_id']}]: "
        f"{c.get('chunk_text_preview', c.get('chunk_text', ''))[:CHUNK_PREVIEW_LEN]}"
        for c in chunks[:TOP_K + 1]
    )
    prompt = (
        "Evaluate groundedness of this research answer.\n\n"
        "Score 1-4:\n"
        "4 = All claims supported by chunks, no fabrication\n"
        "3 = Most supported, minor gap\n"
        "2 = Several claims lack grounding\n"
        "1 = Major fabrication\n\n"
        f"CONTEXT:\n{ctx}\n\nANSWER:\n{answer[:MAX_OUTPUT_TOKENS]}\n\n"
        'Output ONLY JSON: {"score": <1-4>, "reasoning": "<1-2 sentences>", "unsupported_claims": [...]}'
    )
    result = _judge(client, prompt)
    result.setdefault("unsupported_claims", [])
    return result


def score_answer_relevance(query: str, answer: str, client: genai.Client) -> dict:
    prompt = (
        "Does this answer address the research question?\n\n"
        f"QUESTION: {query}\nANSWER: {answer[:MAX_OUTPUT_TOKENS]}\n\n"
        "Score 1-4:\n4 = Directly and completely addresses it\n"
        "3 = Mostly addresses, minor gap\n2 = Partially addresses\n"
        "1 = Does not address\n\n"
        'Output ONLY JSON: {"score": <1-4>, "reasoning": "<1 sentence>"}'
    )
    return _judge(client, prompt)


def score_context_precision(answer: str, chunks: list[dict], client: genai.Client) -> dict:
    """Judge whether the retrieved chunks are relevant to the answer produced."""
    ctx = "\n".join(
        f"[{c['source_id']}, {c['chunk_id']}]: "
        f"{c.get('chunk_text_preview', c.get('chunk_text', ''))[:CHUNK_PREVIEW_LEN]}"
        for c in chunks[:TOP_K + 1]
    )
    prompt = (
        "How many of the retrieved chunks were actually useful for the answer?\n\n"
        f"RETRIEVED CHUNKS:\n{ctx}\n\nANSWER:\n{answer[:MAX_OUTPUT_TOKENS]}\n\n"
        "Score 1-4:\n4 = All chunks relevant\n3 = Most relevant\n"
        "2 = Half irrelevant\n1 = Mostly irrelevant\n\n"
        'Output ONLY JSON: {"score": <1-4>, "reasoning": "<1 sentence>"}'
    )
    return _judge(client, prompt)


def compute_source_recall(retrieved_sources: list[str], expected_sources: list[str]) -> float | None:
    """Fraction of expected sources that appear in retrieval results."""
    if not expected_sources:
        return None
    found = set(retrieved_sources) & set(expected_sources)
    return len(found) / len(expected_sources)


def score_uncertainty_handling(answer: str) -> dict:
    _missing = ["corpus does not contain", "not found in", "no evidence",
                "not addressed", "cannot find", "not available in", "no specific"]
    _hedge = ["approximately", "estimated", "may", "suggests", "likely"]
    lower = answer.lower()
    return {
        "flags_missing_evidence": any(p in lower for p in _missing),
        "preserves_hedging": any(w in lower for w in _hedge),
    }


# ── Runner ───────────────────────────────────────────────────────────────
def run_evaluation(mode: str = "baseline") -> list[dict]:
    from src.rag.rag import load_index, run_rag
    from src.rag.enhance_query_rewriting import run_enhanced_rag

    index, store = load_index()
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    client = genai.Client()

    print(f"\nEvaluation — mode={mode.upper()}, queries={len(EVAL_QUERIES)}")
    results = []

    for i, q in enumerate(EVAL_QUERIES, 1):
        print(f"  [{i:02d}/{len(EVAL_QUERIES)}] {q['id']} ({q['type']}): {q['query'][:55]}...")

        if mode == "enhanced" and q["type"] in ("synthesis", "multihop"):
            run_result = run_enhanced_rag(q["query"], index, store, embed_model, client)
        else:
            run_result = run_rag(q["query"], index, store, embed_model, client, mode=mode)

        ground = score_groundedness(run_result["answer"], run_result["retrieved_chunks"], client)
        relevance = score_answer_relevance(q["query"], run_result["answer"], client)
        ctx_prec = score_context_precision(run_result["answer"], run_result["retrieved_chunks"], client)
        uncertainty = score_uncertainty_handling(run_result["answer"])
        cv = run_result["citation_validation"]
        ret_sources = [c["source_id"] for c in run_result["retrieved_chunks"]]
        src_recall = compute_source_recall(ret_sources, q["expected_sources"])

        row = {
            "query_id": q["id"],
            "query_type": q["type"],
            "query": q["query"],
            "mode": mode,
            "groundedness": ground,
            "answer_relevance": relevance,
            "context_precision": ctx_prec,
            "uncertainty": uncertainty,
            "citation_precision": cv.get("citation_precision"),
            "citations_total": cv.get("total_citations"),
            "citations_valid": cv.get("valid_citations"),
            "source_recall": src_recall,
            "answer_preview": run_result["answer"][:CHUNK_PREVIEW_LEN * 2],
            "retrieved_sources": ret_sources,
            "expected_sources": q["expected_sources"],
        }
        results.append(row)

        g = ground.get("score", "?")
        r = relevance.get("score", "?")
        cp = f"{cv.get('citation_precision', 0):.2f}" if cv.get("citation_precision") is not None else "N/A"
        sr = f"{src_recall:.2f}" if src_recall is not None else "N/A"
        print(f"    ground={g}/4 rel={r}/4 cite_p={cp} src_recall={sr}")

    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = OUTPUTS_DIR / f"eval_results_{mode}_{ts}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved -> {out}")
    return results


def compute_summary(results: list[dict]) -> dict:
    def _avg(vals):
        clean = [v for v in vals if v is not None]
        return round(sum(clean) / len(clean), 3) if clean else None

    by_type = {}
    for qt in ("direct", "synthesis", "multihop", "edge_case"):
        sub = [r for r in results if r["query_type"] == qt]
        if sub:
            by_type[qt] = {
                "count": len(sub),
                "avg_groundedness": _avg([r["groundedness"].get("score") for r in sub]),
                "avg_relevance": _avg([r["answer_relevance"].get("score") for r in sub]),
                "avg_ctx_precision": _avg([r["context_precision"].get("score") for r in sub]),
                "avg_cite_precision": _avg([r["citation_precision"] for r in sub]),
                "avg_source_recall": _avg([r["source_recall"] for r in sub]),
            }
    return {
        "overall": {
            "avg_groundedness": _avg([r["groundedness"].get("score") for r in results]),
            "avg_relevance": _avg([r["answer_relevance"].get("score") for r in results]),
            "avg_ctx_precision": _avg([r["context_precision"].get("score") for r in results]),
            "avg_cite_precision": _avg([r["citation_precision"] for r in results]),
            "avg_source_recall": _avg([r["source_recall"] for r in results]),
            "flags_missing": sum(1 for r in results if r["uncertainty"]["flags_missing_evidence"]),
        },
        "by_type": by_type,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "enhanced", "both"], default="baseline")
    args = parser.parse_args()

    for m in (["baseline", "enhanced"] if args.mode == "both" else [args.mode]):
        res = run_evaluation(m)
        print(f"\n{m.upper()} Summary:")
        print(json.dumps(compute_summary(res), indent=2))
