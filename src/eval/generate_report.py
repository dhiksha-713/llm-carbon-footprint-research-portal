"""Generates the Phase 2 evaluation report from saved eval results."""

import datetime
from pathlib import Path

from src.config import (
    OUTPUTS_DIR, REPORT_DIR, SCORE_PASS_THRESHOLD, SCORE_WARN_THRESHOLD,
    GROK_MODEL, AZURE_MODEL, EMBED_MODEL_NAME, CHUNK_SIZE_TOKENS,
    CHUNK_OVERLAP_TOKENS, TOP_K, ENHANCED_TOP_N, LLM_PROVIDER,
)
from src.eval.evaluation import EVAL_QUERIES
from src.utils import safe_avg, load_eval_results


def _label(s):
    if s is None:
        return "---"
    tag = " PASS" if s >= SCORE_PASS_THRESHOLD else " WARN" if s >= SCORE_WARN_THRESHOLD else " FAIL"
    return f"{s:.2f}{tag}"


def _scores_table(results: list[dict], title: str) -> list[str]:
    rows = [
        f"### {title}", "",
        "| ID | Type | Ground. | Relev. | Ctx Prec. | Cite Prec. | Src Recall | Flags Missing |",
        "|----|------|:-------:|:------:|:---------:|:----------:|:----------:|:-------------:|",
    ]
    for r in results:
        g = r["groundedness"].get("score")
        rv = r["answer_relevance"].get("score")
        cxp = r.get("context_precision", {}).get("score")
        cp = r.get("citation_precision")
        sr = r.get("source_recall")
        fm = "Y" if r["uncertainty"]["flags_missing_evidence"] else "---"
        rows.append(
            f"| {r['query_id']} | {r['query_type']} "
            f"| {_label(g)} | {_label(rv)} | {_label(cxp)} "
            f"| {f'{cp:.2f}' if cp is not None else '---'} "
            f"| {f'{sr:.2f}' if sr is not None else '---'} | {fm} |"
        )
    rows.append(
        f"| **AVG** | --- "
        f"| **{safe_avg([r['groundedness'].get('score') for r in results])}** "
        f"| **{safe_avg([r['answer_relevance'].get('score') for r in results])}** "
        f"| **{safe_avg([r.get('context_precision', {}).get('score') for r in results])}** "
        f"| **{safe_avg([r.get('citation_precision') for r in results])}** "
        f"| **{safe_avg([r.get('source_recall') for r in results])}** | --- |"
    )
    rows.append("")
    return rows


def _by_type_table(results: list[dict]) -> list[str]:
    rows = [
        "### Scores by Query Type", "",
        "| Type | Count | Avg Ground. | Avg Relev. | Avg Cite Prec. | Avg Src Recall |",
        "|------|:-----:|:-----------:|:----------:|:--------------:|:--------------:|",
    ]
    for qt in ("direct", "synthesis", "multihop", "edge_case"):
        sub = [r for r in results if r["query_type"] == qt]
        if not sub:
            continue
        rows.append(
            f"| {qt} | {len(sub)} "
            f"| {safe_avg([r['groundedness'].get('score') for r in sub])} "
            f"| {safe_avg([r['answer_relevance'].get('score') for r in sub])} "
            f"| {safe_avg([r.get('citation_precision') for r in sub])} "
            f"| {safe_avg([r.get('source_recall') for r in sub])} |"
        )
    rows.append("")
    return rows


def _per_query_logs(results: list[dict], title: str) -> list[str]:
    rows = [f"### {title}", ""]
    for r in results:
        answer = r.get("answer_text") or r.get("answer_preview", "")
        excerpt = answer[:600].replace("\n", " ").strip()
        if len(answer) > 600:
            excerpt += "..."
        g_reason = r["groundedness"].get("reasoning", "")
        rel_reason = r["answer_relevance"].get("reasoning", "")
        unsup = r["groundedness"].get("unsupported_claims", [])
        rows += [
            f"#### {r['query_id']} ({r['query_type']})",
            f"**Query**: {r['query']}",
            f"**Retrieved sources**: {', '.join(r['retrieved_sources']) if r['retrieved_sources'] else '(none)'}",
            f"**Expected sources**: {', '.join(r['expected_sources']) if r['expected_sources'] else '(none)'}",
            f"**Citations**: {r.get('citations_valid', 0)}/{r.get('citations_total', 0)} valid "
            f"(precision: {r['citation_precision']:.2f})" if r.get("citation_precision") is not None
            else f"**Citations**: {r.get('citations_valid', 0)}/{r.get('citations_total', 0)} valid",
            f"**Source recall**: {r['source_recall']:.2f}" if r.get("source_recall") is not None
            else "**Source recall**: N/A (no expected sources)",
            f"**Groundedness**: {r['groundedness'].get('score', '?')}/4 - {g_reason}",
            f"**Relevance**: {r['answer_relevance'].get('score', '?')}/4 - {rel_reason}",
        ]
        if unsup:
            rows.append(f"**Unsupported claims flagged**: {', '.join(str(c) for c in unsup[:3])}")
        if r["uncertainty"]["flags_missing_evidence"]:
            rows.append("**Uncertainty**: Answer correctly flags missing/insufficient evidence")
        if r["uncertainty"]["preserves_hedging"]:
            rows.append("**Hedging**: Answer preserves hedging language (approximately, estimated, may)")
        rows += [
            "",
            f"> **Answer excerpt**: {excerpt}",
            "",
            "---", "",
        ]
    return rows


def _detect_failures(results: list[dict]) -> list[dict]:
    """Auto-detect failure cases from eval results. Returns worst-first."""
    failures = []
    for r in results:
        issues = []
        g = r["groundedness"].get("score")
        rv = r["answer_relevance"].get("score")
        cp = r.get("citation_precision")
        sr = r.get("source_recall")
        unsup = r["groundedness"].get("unsupported_claims", [])

        if g is not None and g <= 2:
            issues.append(f"Low groundedness ({g}/4)")
        if rv is not None and rv <= 2:
            issues.append(f"Low relevance ({rv}/4)")
        if cp is not None and cp < 0.5:
            issues.append(f"Low citation precision ({cp:.2f})")
        if sr is not None and sr == 0.0 and r["expected_sources"]:
            issues.append(f"Zero source recall (expected: {', '.join(r['expected_sources'])})")
        if r["query_type"] == "edge_case" and r["expected_sources"] == [] and not r["uncertainty"]["flags_missing_evidence"]:
            issues.append("Edge-case query with no expected sources but answer did not flag missing evidence")
        if unsup:
            issues.append(f"Unsupported claims: {', '.join(str(c) for c in unsup[:3])}")
        if cp is not None and cp == 0.0 and r.get("citations_total", 0) > 0:
            issues.append("All citations are hallucinated (none match retrieved chunks)")

        if issues:
            severity = 0
            if g is not None:
                severity += max(0, 3 - g)
            if rv is not None:
                severity += max(0, 3 - rv)
            if cp is not None:
                severity += max(0, 0.5 - cp) * 4
            failures.append({"result": r, "issues": issues, "severity": severity})

    failures.sort(key=lambda f: f["severity"], reverse=True)
    return failures


def _weakest_results(results: list[dict], exclude_ids: set) -> list[dict]:
    """Pick the lowest-scoring results as borderline/near-miss cases."""
    scored = []
    for r in results:
        rid = f"{r['query_id']}_{r['mode']}"
        if rid in exclude_ids:
            continue
        g = r["groundedness"].get("score") or 4
        rv = r["answer_relevance"].get("score") or 4
        cp = r.get("citation_precision") if r.get("citation_precision") is not None else 1.0
        composite = g + rv + cp * 4
        issues = []
        if g <= 3:
            issues.append(f"Below-perfect groundedness ({g}/4)")
        if rv <= 3:
            issues.append(f"Below-perfect relevance ({rv}/4)")
        if cp < 1.0:
            issues.append(f"Imperfect citation precision ({cp:.2f})")
        sr = r.get("source_recall")
        if sr is not None and sr < 1.0 and r["expected_sources"]:
            issues.append(f"Incomplete source recall ({sr:.2f})")
        if not issues:
            issues.append("Lowest composite score in evaluation set (near-miss)")
        scored.append({"result": r, "issues": issues, "severity": 12 - composite})
    scored.sort(key=lambda x: x["severity"], reverse=True)
    return scored


def _failure_section(baseline: list[dict], enhanced: list[dict]) -> list[str]:
    all_results = baseline + enhanced
    failures = _detect_failures(all_results)

    seen_ids = set()
    unique = []
    for f in failures:
        fid = f"{f['result']['query_id']}_{f['result']['mode']}"
        if fid not in seen_ids:
            seen_ids.add(fid)
            unique.append(f)

    if len(unique) < 3:
        weak = _weakest_results(all_results, seen_ids)
        for w in weak:
            wid = f"{w['result']['query_id']}_{w['result']['mode']}"
            if wid not in seen_ids:
                seen_ids.add(wid)
                unique.append(w)
            if len(unique) >= 3:
                break

    failures = unique[:5]

    rows = [
        "## 6. Representative Failure Cases", "",
        "The following failures were auto-detected from evaluation results "
        "(sorted by severity). Each includes the evidence that triggered detection.", "",
    ]

    if not failures:
        rows.append("No significant failures detected across all evaluated queries.")
        rows.append("")
        return rows

    for i, f in enumerate(failures, 1):
        r = f["result"]
        answer = r.get("answer_text") or r.get("answer_preview", "")
        excerpt = answer[:500].replace("\n", " ").strip()
        if len(answer) > 500:
            excerpt += "..."
        g_reason = r["groundedness"].get("reasoning", "N/A")
        rel_reason = r["answer_relevance"].get("reasoning", "N/A")

        rows += [
            f"### Failure {i}: {r['query_id']} ({r['query_type']}, {r['mode']})", "",
            f"**Query**: {r['query']}", "",
            f"**Issues detected**:",
        ]
        for issue in f["issues"]:
            rows.append(f"- {issue}")
        rows += [
            "",
            f"**Scores**: Groundedness={r['groundedness'].get('score', '?')}/4, "
            f"Relevance={r['answer_relevance'].get('score', '?')}/4, "
            f"Citation Precision={r.get('citation_precision', 'N/A')}, "
            f"Source Recall={r.get('source_recall', 'N/A')}",
            "",
            f"**Judge reasoning (groundedness)**: {g_reason}",
            f"**Judge reasoning (relevance)**: {rel_reason}",
            "",
            f"**Retrieved sources**: {', '.join(r['retrieved_sources']) if r['retrieved_sources'] else '(none)'}",
            f"**Expected sources**: {', '.join(r['expected_sources']) if r['expected_sources'] else '(none)'}",
            "",
            f"> **Answer excerpt**: {excerpt}",
            "",
        ]

        if r["query_type"] == "edge_case" and not r["uncertainty"]["flags_missing_evidence"]:
            rows.append("**Root cause**: System did not flag missing evidence for an out-of-corpus query. "
                        "The answer may overstate confidence in claims not supported by the corpus.")
        elif r.get("citation_precision") is not None and r["citation_precision"] < 0.5:
            rows.append("**Root cause**: Most citations reference chunks that were not in the retrieved set, "
                        "indicating the LLM hallucinated citation IDs.")
        elif r.get("source_recall") is not None and r["source_recall"] == 0.0:
            rows.append("**Root cause**: None of the expected sources appeared in the top-K retrieval. "
                        "The embedding model may not capture the semantic match for this query.")
        elif r["groundedness"].get("score") is not None and r["groundedness"]["score"] <= 3:
            rows.append("**Root cause**: Answer contains claims that go beyond what the retrieved chunks "
                        "explicitly state, likely due to the LLM relying on parametric knowledge rather "
                        "than restricting itself to provided context.")
        else:
            rows.append("**Root cause**: Near-miss case. While not a critical failure, this query "
                        "produced the lowest composite score in the evaluation set, indicating room for "
                        "improvement in retrieval precision or answer grounding.")

        rows += ["", "---", ""]

    return rows


def generate_report() -> Path:
    baseline = load_eval_results("baseline")
    enhanced = load_eval_results("enhanced")
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    L = [
        "# Phase 2 Evaluation Report",
        f"**Project**: Personal Research Portal - Carbon Footprint of LLMs",
        f"**Group**: Group 4 - Dhiksha Rathis, Shreya Verma",
        f"**Generated**: {ts}", "", "---", "",

        "## 1. System Overview", "",
        "- **Corpus**: 20 sources (14 peer-reviewed, 3 technical reports, 3 tools/workshops)",
        f"- **Chunking**: Section-aware sliding window, {CHUNK_SIZE_TOKENS} tokens, {CHUNK_OVERLAP_TOKENS}-token overlap",
        f"- **Embeddings**: `{EMBED_MODEL_NAME}` (384-dim, sentence-transformers)",
        "- **Index**: FAISS IndexFlatIP (inner product on L2-normalised vectors = cosine similarity)",
        f"- **Generation**: Grok-3=`{GROK_MODEL}`, Azure=`{AZURE_MODEL}` (active: {LLM_PROVIDER})",
        f"- **Baseline retrieval**: top-{TOP_K} semantic search",
        f"- **Enhanced retrieval**: query rewriting + decomposition into sub-queries, top-{ENHANCED_TOP_N} merged chunks",
        f"- **LLM-as-Judge**: Same LLM used for scoring (temperature=0.0, structured JSON output)",
        "", "---", "",

        "## 2. Query Set Design", "",
        "The 20-query evaluation set tests three dimensions of research retrieval quality:", "",
        "| Category | Count | Purpose |",
        "|----------|:-----:|---------|",
        "| Direct (D01-D10) | 10 | Single-source factual retrieval - tests basic grounding and citation |",
        "| Synthesis (S01-S05) | 5 | Cross-source comparison - tests multi-source integration |",
        "| Edge Case (E01-E05) | 5 | Corpus boundary detection - tests trust and uncertainty handling |",
        "",
        "**Full query listing:**", "",
        "| ID | Type | Query | Expected Sources |",
        "|----|------|-------|------------------|",
    ]

    for q in EVAL_QUERIES:
        exp = ", ".join(q["expected_sources"]) if q["expected_sources"] else "(none - out of corpus)"
        L.append(f"| {q['id']} | {q['type']} | {q['query'][:90]}{'...' if len(q['query']) > 90 else ''} | {exp} |")

    L += [
        "", "---", "",
        "## 3. Metrics", "",
        "Six metrics evaluate RAG quality across complementary dimensions:", "",
        "| # | Metric | Method | Scale | What it measures |",
        "|---|--------|--------|-------|------------------|",
        "| 1 | Groundedness | LLM-judge | 1-4 | Are factual claims in the answer supported by retrieved chunks? |",
        "| 2 | Answer Relevance | LLM-judge | 1-4 | Does the answer address the research question? |",
        "| 3 | Context Precision | LLM-judge | 1-4 | Were the retrieved chunks actually useful? |",
        "| 4 | Citation Precision | Deterministic | 0-1 | valid_citations / total_citations |",
        "| 5 | Source Recall | Deterministic | 0-1 | expected_sources_found / total_expected |",
        "| 6 | Uncertainty Handling | Rule-based | Y/N | Does answer flag missing evidence when appropriate? |",
        "",
        "**Thresholds**: PASS >= {p}, WARN >= {w}, FAIL < {w}".format(p=SCORE_PASS_THRESHOLD, w=SCORE_WARN_THRESHOLD),
        "", "---", "",
        "## 4. Results", "",
    ]

    if not baseline and not enhanced:
        L += ["*No evaluation results found. Run `make eval-both` first.*", ""]

    if baseline:
        L += _scores_table(baseline, "4.1 Baseline RAG Results")
        L += _by_type_table(baseline)

    if enhanced:
        L += _scores_table(enhanced, "4.2 Enhanced RAG Results")
        L += _by_type_table(enhanced)

    if baseline and enhanced:
        L += [
            "### 4.3 Enhancement Delta (Baseline -> Enhanced)", "",
            "| Metric | Baseline | Enhanced | Delta | Improved? |",
            "|--------|:--------:|:--------:|:-----:|:---------:|",
        ]
        for label, key, sub in [
            ("Groundedness", "groundedness", "score"),
            ("Relevance", "answer_relevance", "score"),
            ("Ctx Precision", "context_precision", "score"),
        ]:
            ba = safe_avg([r[key].get(sub) for r in baseline])
            ea = safe_avg([r[key].get(sub) for r in enhanced])
            d = round(ea - ba, 2) if ba is not None and ea is not None else None
            arrow = "Y" if d and d > 0 else ("---" if d == 0 or d is None else "N")
            L.append(f"| {label} | {ba} | {ea} | {'+' if d and d > 0 else ''}{d} | {arrow} |")

        for label, key in [("Citation Prec.", "citation_precision"), ("Source Recall", "source_recall")]:
            ba = safe_avg([r.get(key) for r in baseline])
            ea = safe_avg([r.get(key) for r in enhanced])
            d = round(ea - ba, 2) if ba is not None and ea is not None else None
            arrow = "Y" if d and d > 0 else ("---" if d == 0 or d is None else "N")
            L.append(f"| {label} | {ba} | {ea} | {'+' if d and d > 0 else ''}{d} | {arrow} |")

        L.append("")

        bm = sum(1 for r in baseline if r["uncertainty"]["flags_missing_evidence"])
        em = sum(1 for r in enhanced if r["uncertainty"]["flags_missing_evidence"])
        L += [
            f"**Uncertainty flags**: Baseline={bm}/{len(baseline)}, Enhanced={em}/{len(enhanced)}",
            "",
        ]

    L += ["---", ""]

    # Section 5: Per-query detail logs
    L += ["## 5. Per-Query Detail Logs", "",
          "Full outputs for every query including answer excerpts, retrieved sources, "
          "citation counts, judge reasoning, and uncertainty flags.", ""]

    if baseline:
        L += _per_query_logs(baseline, "5.1 Baseline - Per-Query Logs")
    if enhanced:
        L += _per_query_logs(enhanced, "5.2 Enhanced - Per-Query Logs")

    # Section 6: Failure cases
    L += _failure_section(baseline or [], enhanced or [])

    # Section 7: Reproducibility
    L += [
        "## 7. Reproducibility", "",
        "```bash",
        "pip install -r requirements.txt",
        "python -m src.ingest.download_sources",
        "python -m src.ingest.ingest",
        "python -m src.eval.evaluation --mode both",
        "python -m src.eval.generate_report",
        "```", "",
        "All dependencies pinned in `requirements.txt`. "
        "FAISS index and chunk store are fully reproducible from scratch using the manifest.",
        "",
    ]

    path = REPORT_DIR / "evaluation_report.md"
    content = "\n".join(L)
    path.write_text(content, encoding="utf-8")
    (OUTPUTS_DIR / "evaluation_report.md").write_text(content, encoding="utf-8")
    print(f"Report saved to {path} ({len(content)} chars)")
    return path


if __name__ == "__main__":
    generate_report()
