"""Generates the Phase 2 evaluation report from saved eval results."""

import datetime
from pathlib import Path

from src.config import (
    OUTPUTS_DIR, REPORT_DIR, SCORE_PASS_THRESHOLD, SCORE_WARN_THRESHOLD,
    GROK_MODEL, AZURE_MODEL, EMBED_MODEL_NAME, CHUNK_SIZE_TOKENS,
    CHUNK_OVERLAP_TOKENS, TOP_K, ENHANCED_TOP_N, LLM_PROVIDER,
)
from src.utils import safe_avg, load_eval_results


def _label(s):
    if s is None:
        return "---"
    tag = " pass" if s >= SCORE_PASS_THRESHOLD else " warn" if s >= SCORE_WARN_THRESHOLD else " fail"
    return f"{s:.2f}{tag}"


def _results_table(results: list[dict], title: str) -> list[str]:
    rows = [
        f"### {title}", "",
        "| ID | Type | Ground. | Relev. | Ctx Prec. | Cite Prec. | Src Recall | Missing |",
        "|----|------|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in results:
        g, rv = r["groundedness"].get("score"), r["answer_relevance"].get("score")
        cxp = r.get("context_precision", {}).get("score")
        cp, sr = r.get("citation_precision"), r.get("source_recall")
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
        f"- **Chunking**: Section-aware, {CHUNK_SIZE_TOKENS} tokens, {CHUNK_OVERLAP_TOKENS}-token overlap",
        f"- **Embeddings**: {EMBED_MODEL_NAME}",
        "- **Index**: FAISS IndexFlatIP (cosine similarity)",
        f"- **Generation**: Grok-3={GROK_MODEL}, Azure={AZURE_MODEL} (active: {LLM_PROVIDER})",
        f"- **Baseline**: top-{TOP_K} semantic retrieval",
        f"- **Enhanced**: query rewriting + decomposition -> top-{ENHANCED_TOP_N} merged chunks",
        "", "---", "",
        "## 2. Query Set Design", "",
        "| Category | Count | Rationale |",
        "|----------|-------|-----------|",
        "| Direct (D01-D10) | 10 | Single-source factual retrieval |",
        "| Synthesis (S01-S05) | 5 | Cross-source comparison |",
        "| Edge Case (E01-E05) | 5 | Corpus boundary detection |",
        "", "---", "",
        "## 3. Metrics", "",
        "| Metric | Definition |",
        "|--------|-----------|",
        "| Groundedness | LLM-judge (1-4): are claims supported by retrieved chunks? |",
        "| Answer Relevance | LLM-judge (1-4): does answer address the query? |",
        "| Context Precision | LLM-judge (1-4): are retrieved chunks relevant? |",
        "| Citation Precision | valid citations / total citations (deterministic) |",
        "| Source Recall | expected sources found / total expected sources |",
        "| Uncertainty Handling | Does answer flag missing evidence? (boolean) |",
        "", "---", "",
        "## 4. Results", "",
    ]

    if baseline:
        L += _results_table(baseline, "4.1 Baseline RAG")
    if enhanced:
        L += _results_table(enhanced, "4.2 Enhanced RAG (Query Rewriting)")

    if baseline and enhanced:
        L += ["### 4.3 Enhancement Delta", "",
              "| Metric | Baseline | Enhanced | Delta |",
              "|--------|----------|----------|-------|"]
        for label, key, sub in [
            ("Groundedness", "groundedness", "score"),
            ("Relevance", "answer_relevance", "score"),
            ("Ctx Precision", "context_precision", "score"),
        ]:
            ba = safe_avg([r[key].get(sub) for r in baseline])
            ea = safe_avg([r[key].get(sub) for r in enhanced])
            d = round(ea - ba, 2) if ba and ea else None
            L.append(f"| {label} | {ba} | {ea} | {'+' if d and d > 0 else ''}{d} |")
        ba = safe_avg([r.get("citation_precision") for r in baseline])
        ea = safe_avg([r.get("citation_precision") for r in enhanced])
        d = round(ea - ba, 2) if ba and ea else None
        L.append(f"| Citation Prec. | {ba} | {ea} | {'+' if d and d > 0 else ''}{d} |")
        L.append("")

    L += ["---", "", "## 5. Reproducibility", "",
          "```bash", "pip install -r requirements.txt",
          "python -m src.ingest.download_sources", "python -m src.ingest.ingest",
          "python -m src.eval.evaluation --mode both",
          "python -m src.eval.generate_report", "```", ""]

    path = REPORT_DIR / "evaluation_report.md"
    content = "\n".join(L)
    path.write_text(content, encoding="utf-8")
    (OUTPUTS_DIR / "evaluation_report.md").write_text(content, encoding="utf-8")
    print(f"Report saved to {path}")
    return path


if __name__ == "__main__":
    generate_report()
