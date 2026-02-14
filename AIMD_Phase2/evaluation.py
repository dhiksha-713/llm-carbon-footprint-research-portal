"""
evaluation.py
-------------
Phase 2 Evaluation Module

Contains:
- EVAL_QUERIES: 20-query evaluation set (10 direct, 5 synthesis, 5 edge-case)
- Scoring functions: groundedness, citation precision, answer relevance
- run_evaluation(): runs full eval set on baseline and enhanced RAG
- generate_report(): produces evaluation report

Run: python evaluation.py --mode baseline
     python evaluation.py --mode enhanced
     python evaluation.py --mode both
"""

import os
import json
import datetime
import re
import argparse
from pathlib import Path

import anthropic
from sentence_transformers import SentenceTransformer
import numpy as np

LOGS_DIR    = Path("logs")
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 20-Query Evaluation Set ─────────────────────────────────────────────
# Aligned with Phase 1 sub-questions and Phase 2 requirements
EVAL_QUERIES = [
    # ── 10 DIRECT queries ──────────────────────────────────────────────
    {
        "id": "D01", "type": "direct",
        "query": "What does GPU-hour energy measurement measure, and what are its known failure modes?",
        "expected_sources": ["strubell2019", "patterson2021"],
        "notes": "Tests core methodology understanding",
    },
    {
        "id": "D02", "type": "direct",
        "query": "How much CO2 was emitted during the training of BERT according to Strubell et al.?",
        "expected_sources": ["strubell2019"],
        "notes": "Specific numerical fact check",
    },
    {
        "id": "D03", "type": "direct",
        "query": "What is the total lifecycle carbon footprint of BLOOM in tonnes CO2 equivalent?",
        "expected_sources": ["luccioni2022"],
        "notes": "Lifecycle figure from Luccioni",
    },
    {
        "id": "D04", "type": "direct",
        "query": "What factors does Patterson et al. identify as most impactful for reducing LLM carbon emissions?",
        "expected_sources": ["patterson2021"],
        "notes": "Key factors from Patterson",
    },
    {
        "id": "D05", "type": "direct",
        "query": "How does carbon intensity of the electricity grid affect LLM training emissions?",
        "expected_sources": ["patterson2021", "luccioni2022", "dodge2022"],
        "notes": "Grid carbon intensity factor",
    },
    {
        "id": "D06", "type": "direct",
        "query": "What is the difference between operational and embodied carbon emissions in AI systems?",
        "expected_sources": ["luccioni2022", "ligozat2022"],
        "notes": "Embodied vs operational distinction",
    },
    {
        "id": "D07", "type": "direct",
        "query": "What tools exist for tracking carbon emissions during machine learning training?",
        "expected_sources": ["anthony2020", "bannour2021", "lacoste2019"],
        "notes": "Carbon tracking tools",
    },
    {
        "id": "D08", "type": "direct",
        "query": "How does Schwartz et al. define the concept of Red AI versus Green AI?",
        "expected_sources": ["schwartz2020"],
        "notes": "Green AI definition",
    },
    {
        "id": "D09", "type": "direct",
        "query": "What metrics does Henderson et al. recommend for reporting ML energy consumption?",
        "expected_sources": ["henderson2020"],
        "notes": "Reporting standards",
    },
    {
        "id": "D10", "type": "direct",
        "query": "What is the carbon footprint of neural architecture search compared to standard training?",
        "expected_sources": ["strubell2019"],
        "notes": "NAS carbon cost",
    },

    # ── 5 SYNTHESIS / MULTI-HOP queries ────────────────────────────────
    {
        "id": "S01", "type": "synthesis",
        "query": "Compare Strubell et al. and Patterson et al. on their measurement methodology: where do they agree and disagree on how to estimate training carbon emissions?",
        "expected_sources": ["strubell2019", "patterson2021"],
        "notes": "Direct reuse of Phase 1 TC2A synthesis task",
    },
    {
        "id": "S02", "type": "synthesis",
        "query": "How do Luccioni et al. and Patterson et al. differ in their assumptions about hardware efficiency and carbon intensity?",
        "expected_sources": ["luccioni2022", "patterson2021"],
        "notes": "Direct reuse of Phase 1 TC2B",
    },
    {
        "id": "S03", "type": "synthesis",
        "query": "Across all sources in the corpus, what are the three most commonly cited factors that explain variation in LLM carbon footprint estimates?",
        "expected_sources": ["strubell2019", "patterson2021", "luccioni2022", "henderson2020"],
        "notes": "Multi-source aggregation",
    },
    {
        "id": "S04", "type": "multihop",
        "query": "Why do different studies report dramatically different carbon estimates for similar model sizes, and what does this imply for standardization?",
        "expected_sources": ["henderson2020", "luccioni2022", "dodge2022"],
        "notes": "Multi-hop: causes → implications",
    },
    {
        "id": "S05", "type": "synthesis",
        "query": "How have carbon measurement methods for LLMs evolved from 2019 to 2023, and what gaps remain?",
        "expected_sources": ["strubell2019", "luccioni2022", "luccioni2023", "ligozat2022"],
        "notes": "Temporal synthesis across years",
    },

    # ── 5 EDGE CASE / AMBIGUITY queries ────────────────────────────────
    {
        "id": "E01", "type": "edge_case",
        "query": "Does the corpus contain evidence that LLM inference emissions exceed training emissions over a model's deployment lifetime?",
        "expected_sources": ["luccioni2023", "wu2022"],
        "notes": "Corpus boundary test — should find partial evidence",
    },
    {
        "id": "E02", "type": "edge_case",
        "query": "Does the corpus contain evidence about the carbon footprint of GPT-4 specifically?",
        "expected_sources": [],
        "notes": "Expected: corpus does NOT contain this — should say so",
    },
    {
        "id": "E03", "type": "edge_case",
        "query": "Is there evidence in the corpus that carbon offset programs effectively neutralize LLM training emissions?",
        "expected_sources": [],
        "notes": "Out of scope per Phase 1 framing — should flag absence",
    },
    {
        "id": "E04", "type": "edge_case",
        "query": "What does the corpus say about the carbon footprint of quantum computing for AI?",
        "expected_sources": [],
        "notes": "Completely out of domain — should refuse to guess",
    },
    {
        "id": "E05", "type": "edge_case",
        "query": "Do all sources agree on the carbon intensity of the French electricity grid?",
        "expected_sources": ["luccioni2022", "patterson2021"],
        "notes": "Conflict detection test — values may differ slightly",
    },
]

# ── Groundedness Scoring ────────────────────────────────────────────────
GROUNDEDNESS_PROMPT = """You are evaluating the groundedness of an AI-generated research answer.

Score the answer on a 1-4 scale:
4 = All major claims are supported by the provided chunks; no fabrication detected
3 = Most claims supported; minor unsupported claim or missing nuance  
2 = Several claims lack grounding or are imprecise
1 = Major claims fabricated or not supported by chunks

CONTEXT CHUNKS PROVIDED:
{context}

ANSWER TO EVALUATE:
{answer}

Output ONLY a JSON object: {{"score": <1-4>, "reasoning": "<1-2 sentences>", "unsupported_claims": ["..."]}}"""

def score_groundedness(answer: str, chunks: list[dict], client: anthropic.Anthropic) -> dict:
    context_summary = "\n".join(
        f"[{c['source_id']}, {c['chunk_id']}]: {c.get('chunk_text_preview', c.get('chunk_text',''))[:150]}"
        for c in chunks[:5]
    )
    prompt = GROUNDEDNESS_PROMPT.format(context=context_summary, answer=answer[:1500])
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    try:
        text = re.sub(r"```json|```", "", text).strip()
        return json.loads(text)
    except Exception:
        return {"score": None, "reasoning": "Parse error", "unsupported_claims": []}

def score_answer_relevance(query: str, answer: str, client: anthropic.Anthropic) -> dict:
    """Score how well the answer addresses the query."""
    prompt = f"""Does this answer address the research question?

QUESTION: {query}

ANSWER: {answer[:1000]}

Score 1-4:
4 = Directly and completely addresses the question
3 = Mostly addresses it; minor tangent or gap
2 = Partially addresses; misses key aspect
1 = Doesn't address the question

Output ONLY JSON: {{"score": <1-4>, "reasoning": "<1 sentence>"}}"""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )
    text = re.sub(r"```json|```", "", response.content[0].text).strip()
    try:
        return json.loads(text)
    except Exception:
        return {"score": None, "reasoning": "Parse error"}

def score_uncertainty_handling(answer: str) -> dict:
    """Check if answer correctly flags missing evidence."""
    flags_missing = any(phrase in answer.lower() for phrase in [
        "corpus does not contain",
        "not found in",
        "no evidence",
        "not addressed",
        "cannot find",
        "not available in",
    ])
    has_hedging = any(w in answer.lower() for w in
                      ["approximately", "estimated", "may", "suggests", "likely"])
    return {
        "flags_missing_evidence": flags_missing,
        "preserves_hedging": has_hedging,
    }

# ── Main evaluation runner ──────────────────────────────────────────────
def run_evaluation(mode: str = "baseline"):
    from rag import load_index, run_rag, EMBED_MODEL, TOP_K
    from enhance_query_rewriting import run_enhanced_rag

    print(f"\n{'='*60}")
    print(f"Phase 2 Evaluation — Mode: {mode.upper()}")
    print(f"Queries: {len(EVAL_QUERIES)}")
    print("=" * 60)

    index, chunk_store = load_index()
    embed_model = SentenceTransformer(EMBED_MODEL)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    results = []
    for i, q in enumerate(EVAL_QUERIES, 1):
        print(f"\n[{i:02d}/{len(EVAL_QUERIES)}] {q['id']} ({q['type']}): {q['query'][:60]}...")

        # Run RAG
        if mode == "enhanced" and q["type"] in ("synthesis", "multihop"):
            run_result = run_enhanced_rag(q["query"], index, chunk_store, embed_model, client)
        else:
            run_result = run_rag(q["query"], index, chunk_store, embed_model, client,
                                 mode=mode)

        # Score
        ground_score = score_groundedness(
            run_result["answer"], run_result["retrieved_chunks"], client
        )
        relevance_score = score_answer_relevance(q["query"], run_result["answer"], client)
        uncertainty = score_uncertainty_handling(run_result["answer"])
        cv = run_result["citation_validation"]

        eval_result = {
            "query_id":          q["id"],
            "query_type":        q["type"],
            "query":             q["query"],
            "mode":              mode,
            "groundedness":      ground_score,
            "answer_relevance":  relevance_score,
            "uncertainty":       uncertainty,
            "citation_precision": cv.get("citation_precision"),
            "citations_total":   cv.get("total_citations"),
            "citations_valid":   cv.get("valid_citations"),
            "answer_preview":    run_result["answer"][:300] + "...",
            "retrieved_sources": [c["source_id"] for c in run_result["retrieved_chunks"]],
        }
        results.append(eval_result)

        g = ground_score.get("score", "?")
        r = relevance_score.get("score", "?")
        cp = f"{cv.get('citation_precision', 0):.2f}" if cv.get("citation_precision") is not None else "N/A"
        print(f"  Groundedness={g}/4 | Relevance={r}/4 | CitePrecision={cp}")

    # Save results
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    results_path = OUTPUTS_DIR / f"eval_results_{mode}_{ts}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_path}")

    return results

def compute_summary(results: list[dict]) -> dict:
    """Compute aggregate metrics from evaluation results."""
    def avg(vals):
        clean = [v for v in vals if v is not None]
        return round(sum(clean) / len(clean), 3) if clean else None

    by_type = {}
    for qtype in ["direct", "synthesis", "multihop", "edge_case"]:
        subset = [r for r in results if r["query_type"] == qtype]
        if subset:
            by_type[qtype] = {
                "count": len(subset),
                "avg_groundedness": avg([r["groundedness"].get("score") for r in subset]),
                "avg_relevance":    avg([r["answer_relevance"].get("score") for r in subset]),
                "avg_cite_precision": avg([r["citation_precision"] for r in subset]),
            }

    return {
        "overall": {
            "avg_groundedness":   avg([r["groundedness"].get("score") for r in results]),
            "avg_relevance":      avg([r["answer_relevance"].get("score") for r in results]),
            "avg_cite_precision": avg([r["citation_precision"] for r in results]),
            "flags_missing_evidence": sum(1 for r in results
                                         if r["uncertainty"]["flags_missing_evidence"]),
        },
        "by_type": by_type,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "enhanced", "both"], default="baseline")
    args = parser.parse_args()

    if args.mode in ("baseline", "both"):
        baseline_results = run_evaluation("baseline")
        print("\nBaseline Summary:")
        print(json.dumps(compute_summary(baseline_results), indent=2))

    if args.mode in ("enhanced", "both"):
        enhanced_results = run_evaluation("enhanced")
        print("\nEnhanced Summary:")
        print(json.dumps(compute_summary(enhanced_results), indent=2))
