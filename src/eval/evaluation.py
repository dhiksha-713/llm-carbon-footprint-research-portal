"""Evaluation: 20-query test set, LLM-as-judge scoring, metric computation."""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import re

from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

from src.config import (
    OUTPUTS_DIR, JUDGE_TEMPERATURE, EMBED_MODEL_NAME, TOP_K,
    JUDGE_MAX_TOKENS, CHUNK_PREVIEW_LEN, MAX_OUTPUT_TOKENS,
)
from src.llm_client import LLMClient, get_llm_client
from src.utils import safe_avg, chunk_preview_ctx

# ── 20-Query Evaluation Set ──────────────────────────────────────────────
EVAL_QUERIES: list[dict] = [
    {"id": "D01", "type": "direct",    "query": "What does GPU-hour energy measurement measure, and what are its known failure modes?", "expected_sources": ["strubell2019", "patterson2021"]},
    {"id": "D02", "type": "direct",    "query": "How much CO2 was emitted during the training of BERT according to Strubell et al.?", "expected_sources": ["strubell2019"]},
    {"id": "D03", "type": "direct",    "query": "What is the total lifecycle carbon footprint of BLOOM in tonnes CO2 equivalent?", "expected_sources": ["luccioni2022"]},
    {"id": "D04", "type": "direct",    "query": "What factors does Patterson et al. identify as most impactful for reducing LLM carbon emissions?", "expected_sources": ["patterson2021"]},
    {"id": "D05", "type": "direct",    "query": "How does carbon intensity of the electricity grid affect LLM training emissions?", "expected_sources": ["patterson2021", "luccioni2022", "dodge2022"]},
    {"id": "D06", "type": "direct",    "query": "What is the difference between operational and embodied carbon emissions in AI systems?", "expected_sources": ["luccioni2022", "ligozat2022"]},
    {"id": "D07", "type": "direct",    "query": "What tools exist for tracking carbon emissions during machine learning training?", "expected_sources": ["anthony2020", "bannour2021", "lacoste2019"]},
    {"id": "D08", "type": "direct",    "query": "How does Schwartz et al. define the concept of Red AI versus Green AI?", "expected_sources": ["schwartz2020"]},
    {"id": "D09", "type": "direct",    "query": "What metrics does Henderson et al. recommend for reporting ML energy consumption?", "expected_sources": ["henderson2020"]},
    {"id": "D10", "type": "direct",    "query": "What is the carbon footprint of neural architecture search compared to standard training?", "expected_sources": ["strubell2019"]},
    {"id": "S01", "type": "synthesis", "query": "Compare Strubell et al. and Patterson et al. on their measurement methodology: where do they agree and disagree on how to estimate training carbon emissions?", "expected_sources": ["strubell2019", "patterson2021"]},
    {"id": "S02", "type": "synthesis", "query": "How do Luccioni et al. and Patterson et al. differ in their assumptions about hardware efficiency and carbon intensity?", "expected_sources": ["luccioni2022", "patterson2021"]},
    {"id": "S03", "type": "synthesis", "query": "Across all sources in the corpus, what are the three most commonly cited factors that explain variation in LLM carbon footprint estimates?", "expected_sources": ["strubell2019", "patterson2021", "luccioni2022", "henderson2020"]},
    {"id": "S04", "type": "multihop",  "query": "Why do different studies report dramatically different carbon estimates for similar model sizes, and what does this imply for standardization?", "expected_sources": ["henderson2020", "luccioni2022", "dodge2022"]},
    {"id": "S05", "type": "synthesis", "query": "How have carbon measurement methods for LLMs evolved from 2019 to 2023, and what gaps remain?", "expected_sources": ["strubell2019", "luccioni2022", "luccioni2023", "ligozat2022"]},
    {"id": "E01", "type": "edge_case", "query": "Does the corpus contain evidence that LLM inference emissions exceed training emissions over a model's deployment lifetime?", "expected_sources": ["luccioni2023", "wu2022"]},
    {"id": "E02", "type": "edge_case", "query": "Does the corpus contain evidence about the carbon footprint of GPT-4 specifically?", "expected_sources": []},
    {"id": "E03", "type": "edge_case", "query": "Is there evidence in the corpus that carbon offset programs effectively neutralize LLM training emissions?", "expected_sources": []},
    {"id": "E04", "type": "edge_case", "query": "What does the corpus say about the carbon footprint of quantum computing for AI?", "expected_sources": []},
    {"id": "E05", "type": "edge_case", "query": "Do all sources agree on the carbon intensity of the French electricity grid?", "expected_sources": ["luccioni2022", "patterson2021"]},
]


# ── LLM-as-Judge ─────────────────────────────────────────────────────────

def _judge(client: LLMClient, prompt: str, max_tokens: int = JUDGE_MAX_TOKENS) -> dict:
    resp = client.generate(prompt, temperature=JUDGE_TEMPERATURE, max_tokens=max_tokens)
    raw = resp.text or ""
    log.debug("_judge raw response (%d chars): %s", len(raw), raw[:300])
    text = re.sub(r"```(?:json)?\s*", "", raw).strip()

    # 1. Direct JSON parse
    try:
        result = json.loads(text)
        log.info("_judge -> score=%s (direct JSON)", result.get("score"))
        return result
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. Extract JSON object containing "score" from surrounding prose
    m = re.search(r"\{[^{}]*['\"]score['\"].*?\}", text, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group())
            log.info("_judge -> score=%s (extracted JSON object)", result.get("score"))
            return result
        except (json.JSONDecodeError, TypeError):
            pass

    # 3. Regex: "score": 3 or 'score': 3 (inside JSON-like text)
    m = re.search(r"""['\"]score['\"]\s*:\s*(\d+)""", text)
    if m:
        log.info("_judge -> score=%s (regex from partial JSON)", m.group(1))
        return {"score": int(m.group(1)), "reasoning": "Extracted from partial JSON"}

    # 4. Plain-text: Score: 3 or score: 3/4 (no JSON at all)
    m = re.search(r"(?i)\bscore\b\s*[:=]\s*(\d)", text)
    if m:
        log.info("_judge -> score=%s (plain-text regex)", m.group(1))
        return {"score": int(m.group(1)), "reasoning": "Extracted from plain-text response"}

    log.warning("_judge -> FAILED to parse. Raw: %s", raw[:300])
    return {"score": None, "reasoning": f"Unparseable response: {raw[:200]}"}


def score_groundedness(answer: str, chunks: list[dict], client: LLMClient) -> dict:
    prompt = (
        "Evaluate groundedness of this research answer.\n\n"
        "Score 1-4:\n4 = All claims supported by chunks, no fabrication\n"
        "3 = Most supported, minor gap\n2 = Several claims lack grounding\n"
        "1 = Major fabrication\n\n"
        f"CONTEXT:\n{chunk_preview_ctx(chunks, TOP_K + 1)}\n\n"
        f"ANSWER:\n{answer[:MAX_OUTPUT_TOKENS]}\n\n"
        'Output ONLY JSON: {"score": <1-4>, "reasoning": "<1-2 sentences>", "unsupported_claims": [...]}'
    )
    result = _judge(client, prompt)
    result.setdefault("unsupported_claims", [])
    return result


def score_answer_relevance(query: str, answer: str, client: LLMClient) -> dict:
    prompt = (
        "Does this answer address the research question?\n\n"
        f"QUESTION: {query}\nANSWER: {answer[:MAX_OUTPUT_TOKENS]}\n\n"
        "Score 1-4:\n4 = Directly and completely addresses it\n"
        "3 = Mostly addresses, minor gap\n2 = Partially addresses\n"
        "1 = Does not address\n\n"
        'Output ONLY JSON: {"score": <1-4>, "reasoning": "<1 sentence>"}'
    )
    return _judge(client, prompt)


def score_context_precision(answer: str, chunks: list[dict], client: LLMClient) -> dict:
    prompt = (
        "How many of the retrieved chunks were actually useful for the answer?\n\n"
        f"RETRIEVED CHUNKS:\n{chunk_preview_ctx(chunks, TOP_K + 1)}\n\n"
        f"ANSWER:\n{answer[:MAX_OUTPUT_TOKENS]}\n\n"
        "Score 1-4:\n4 = All chunks relevant\n3 = Most relevant\n"
        "2 = Half irrelevant\n1 = Mostly irrelevant\n\n"
        'Output ONLY JSON: {"score": <1-4>, "reasoning": "<1 sentence>"}'
    )
    return _judge(client, prompt)


def compute_source_recall(retrieved_sources: list[str], expected_sources: list[str]) -> float | None:
    if not expected_sources:
        return None
    return len(set(retrieved_sources) & set(expected_sources)) / len(expected_sources)


def score_uncertainty_handling(answer: str) -> dict:
    lower = answer.lower()
    return {
        "flags_missing_evidence": any(p in lower for p in [
            "corpus does not contain", "not found in", "no evidence",
            "not addressed", "cannot find", "not available in", "no specific",
        ]),
        "preserves_hedging": any(w in lower for w in [
            "approximately", "estimated", "may", "suggests", "likely",
        ]),
    }


# ── Runner ────────────────────────────────────────────────────────────────

def run_evaluation(mode: str = "baseline") -> list[dict]:
    from src.rag.rag import load_index, run_rag
    from src.rag.enhance_query_rewriting import run_enhanced_rag

    index, store = load_index()
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    client = get_llm_client()

    print(f"\nEvaluation - mode={mode.upper()}, queries={len(EVAL_QUERIES)}")
    results: list[dict] = []

    for i, q in enumerate(EVAL_QUERIES, 1):
        print(f"  [{i:02d}/{len(EVAL_QUERIES)}] {q['id']} ({q['type']}): {q['query'][:55]}...")

        if mode == "enhanced" and q["type"] in ("synthesis", "multihop"):
            rr = run_enhanced_rag(q["query"], index, store, embed_model, client)
        else:
            rr = run_rag(q["query"], index, store, embed_model, client, mode=mode)

        cv = rr["citation_validation"]
        ret_srcs = [c["source_id"] for c in rr["retrieved_chunks"]]
        src_rec = compute_source_recall(ret_srcs, q["expected_sources"])

        results.append({
            "query_id": q["id"], "query_type": q["type"], "query": q["query"], "mode": mode,
            "groundedness": score_groundedness(rr["answer"], rr["retrieved_chunks"], client),
            "answer_relevance": score_answer_relevance(q["query"], rr["answer"], client),
            "context_precision": score_context_precision(rr["answer"], rr["retrieved_chunks"], client),
            "uncertainty": score_uncertainty_handling(rr["answer"]),
            "citation_precision": cv.get("citation_precision"),
            "citations_total": cv.get("total_citations"),
            "citations_valid": cv.get("valid_citations"),
            "source_recall": src_rec,
            "answer_preview": rr["answer"][:CHUNK_PREVIEW_LEN * 2],
            "retrieved_sources": ret_srcs,
            "expected_sources": q["expected_sources"],
        })

        r = results[-1]
        g = r["groundedness"].get("score", "?")
        rv = r["answer_relevance"].get("score", "?")
        cp = f"{cv.get('citation_precision', 0):.2f}" if cv.get("citation_precision") is not None else "N/A"
        sr = f"{src_rec:.2f}" if src_rec is not None else "N/A"
        print(f"    ground={g}/4 rel={rv}/4 cite_p={cp} src_recall={sr}")

    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = OUTPUTS_DIR / f"eval_results_{mode}_{ts}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved -> {out}")
    return results


def compute_summary(results: list[dict]) -> dict:
    def _metrics(sub):
        return {
            "avg_groundedness":  safe_avg([r["groundedness"].get("score") for r in sub], 3),
            "avg_relevance":     safe_avg([r["answer_relevance"].get("score") for r in sub], 3),
            "avg_ctx_precision": safe_avg([r["context_precision"].get("score") for r in sub], 3),
            "avg_cite_precision": safe_avg([r["citation_precision"] for r in sub], 3),
            "avg_source_recall": safe_avg([r["source_recall"] for r in sub], 3),
        }

    by_type = {}
    for qt in ("direct", "synthesis", "multihop", "edge_case"):
        sub = [r for r in results if r["query_type"] == qt]
        if sub:
            by_type[qt] = {"count": len(sub), **_metrics(sub)}

    overall = _metrics(results)
    overall["flags_missing"] = sum(1 for r in results if r["uncertainty"]["flags_missing_evidence"])
    return {"overall": overall, "by_type": by_type}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation runner")
    parser.add_argument("--mode", choices=["baseline", "enhanced", "both"], default="baseline")
    args = parser.parse_args()
    for m in (["baseline", "enhanced"] if args.mode == "both" else [args.mode]):
        res = run_evaluation(m)
        print(f"\n{m.upper()} Summary:")
        print(json.dumps(compute_summary(res), indent=2))
