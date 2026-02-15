"""Generates the Phase 2 evaluation report from saved eval results."""

import json
import glob
import datetime
from pathlib import Path

from src.config import (
    OUTPUTS_DIR, REPORT_DIR, SCORE_PASS_THRESHOLD, SCORE_WARN_THRESHOLD,
    GENERATION_MODEL, EMBED_MODEL_NAME, CHUNK_SIZE_TOKENS,
    CHUNK_OVERLAP_TOKENS, TOP_K, ENHANCED_TOP_N, LLM_PROVIDER,
)


def _load_latest(mode: str) -> list[dict]:
    files = sorted(glob.glob(str(OUTPUTS_DIR / f"eval_results_{mode}_*.json")))
    if not files:
        return []
    return json.loads(Path(files[-1]).read_text(encoding="utf-8"))


def _avg(vals):
    clean = [v for v in vals if v is not None]
    return round(sum(clean) / len(clean), 2) if clean else None


def _label(s):
    if s is None:
        return "---"
    return f"{s:.2f}" + (" pass" if s >= SCORE_PASS_THRESHOLD else " warn" if s >= SCORE_WARN_THRESHOLD else " fail")


def _results_table(results: list[dict], title: str) -> list[str]:
    rows = [
        f"### {title}", "",
        "| ID | Type | Ground. | Relev. | Ctx Prec. | Cite Prec. | Src Recall | Missing |",
        "|----|------|:---:|:---:|:---:|:---:|:---:|:---:|",
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
        f"| **{_avg([r['groundedness'].get('score') for r in results])}** "
        f"| **{_avg([r['answer_relevance'].get('score') for r in results])}** "
        f"| **{_avg([r.get('context_precision', {}).get('score') for r in results])}** "
        f"| **{_avg([r.get('citation_precision') for r in results])}** "
        f"| **{_avg([r.get('source_recall') for r in results])}** | --- |"
    )
    rows.append("")
    return rows


def generate_report() -> Path:
    baseline = _load_latest("baseline")
    enhanced = _load_latest("enhanced")
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    L = [
        "# Phase 2 Evaluation Report",
        f"**Project**: Personal Research Portal -- Carbon Footprint of LLMs",
        f"**Group**: Group 4 -- Dhiksha Rathis, Shreya Verma",
        f"**Generated**: {ts}", "", "---", "",
        "## 1. System Overview", "",
        "- **Corpus**: 15 sources (8 peer-reviewed, 4 technical reports, 2 tools, 1 standards)",
        f"- **Chunking**: Section-aware, {CHUNK_SIZE_TOKENS} tokens, {CHUNK_OVERLAP_TOKENS}-token overlap",
        f"- **Embeddings**: {EMBED_MODEL_NAME}",
        "- **Index**: FAISS IndexFlatIP (cosine similarity)",
        f"- **Generation**: {GENERATION_MODEL} via {LLM_PROVIDER}",
        f"- **Baseline**: top-{TOP_K} semantic retrieval",
        f"- **Enhanced**: query rewriting + decomposition -> top-{ENHANCED_TOP_N} merged chunks", "", "---", "",
        "## 2. Query Set Design", "",
        "| Category | Count | Rationale |",
        "|----------|-------|-----------|",
        "| Direct (D01-D10) | 10 | Single-source factual retrieval |",
        "| Synthesis (S01-S05) | 5 | Cross-source comparison, mirrors Phase 1 |",
        "| Edge Case (E01-E05) | 5 | Trust behavior, corpus boundary detection |",
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
            ba = _avg([r[key].get(sub) for r in baseline])
            ea = _avg([r[key].get(sub) for r in enhanced])
            d = round(ea - ba, 2) if ba and ea else None
            L.append(f"| {label} | {ba} | {ea} | {'+' if d and d > 0 else ''}{d} |")
        ba = _avg([r.get("citation_precision") for r in baseline])
        ea = _avg([r.get("citation_precision") for r in enhanced])
        d = round(ea - ba, 2) if ba and ea else None
        L.append(f"| Citation Prec. | {ba} | {ea} | {'+' if d and d > 0 else ''}{d} |")
        L.append("")

    L += ["---", "",
          "## 5. Failure Cases", "",
          "### 5.1 Citation Hallucination (E02)",
          "**Query**: GPT-4 carbon footprint?",
          "**Expected**: Flag corpus absence. **Observed**: Baseline retrieves tangential GPT chunks.",
          "**Fix**: Explicit uncertainty prompt guardrail.", "",
          "### 5.2 Source Coverage Gap (S03)",
          "**Query**: Common factors across all sources.",
          "**Expected**: 4+ sources. **Observed**: Baseline concentrates on 2 sources.",
          "**Fix**: Query decomposition broadens retrieval.", "",
          "### 5.3 Conflict Blindness (E05)",
          "**Query**: French grid carbon intensity agreement.",
          "**Expected**: Flag differing values. **Observed**: Single value without conflict note.",
          "**Fix**: CONFLICT detection instruction in enhanced prompt.", "",
          "---", "",
          "## 6. Reproducibility", "",
          "```bash",
          "pip install -r requirements.txt",
          "python -m src.ingest.download_sources",
          "python -m src.ingest.ingest",
          "python -m src.eval.evaluation --mode both",
          "python -m src.eval.generate_report",
          "```", ""]

    path = REPORT_DIR / "evaluation_report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    (OUTPUTS_DIR / "evaluation_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"Report saved to {path}")
    return path


if __name__ == "__main__":
    generate_report()
