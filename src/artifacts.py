"""Research artifact generation: evidence tables, annotated bibliographies, synthesis memos."""

from __future__ import annotations

import csv
import datetime
import io
import logging
from pathlib import Path

log = logging.getLogger(__name__)

from src.config import ARTIFACTS_DIR, MANIFEST_PATH
from src.llm_client import LLMClient
from src.utils import CITE_RE, build_chunk_context, extract_json_array


def _load_manifest_map() -> dict[str, dict]:
    """Return {source_id: row_dict} from the CSV manifest."""
    import pandas as pd
    if not MANIFEST_PATH.exists():
        return {}
    df = pd.read_csv(MANIFEST_PATH)
    return {row["source_id"]: row.to_dict() for _, row in df.iterrows()}


def _save_artifact(name: str, content: str, ext: str = "md") -> Path:
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    p = ARTIFACTS_DIR / f"{name}_{ts}.{ext}"
    p.write_text(content, encoding="utf-8")
    return p


# ── Evidence Table ────────────────────────────────────────────────────────

_EVIDENCE_TABLE_SYSTEM = (
    "You are a research assistant building an evidence table from academic text. "
    "For each distinct factual claim in the answer, extract one row with these fields:\n"
    "1. claim — one sentence\n"
    "2. evidence — direct quote or close paraphrase from the context, ≤50 words\n"
    "3. citation — as (source_id, chunk_id)\n"
    "4. confidence — HIGH if directly stated, MEDIUM if inferred, LOW if weakly supported\n"
    "5. notes — caveats, hedging, or conflicts\n\n"
    "IMPORTANT: Output ONLY a valid JSON array. No prose, no explanation, no markdown fences.\n"
    "Example format: [{\"claim\":\"...\",\"evidence\":\"...\",\"citation\":\"...\",\"confidence\":\"HIGH\",\"notes\":\"...\"}]\n"
    "Extract 3-10 rows. If a claim lacks support, set confidence=LOW and notes='Not directly supported'."
)


def generate_evidence_table(
    query: str, answer: str, chunks: list[dict], client: LLMClient,
) -> dict:
    """Generate a structured evidence table from a RAG result."""
    prompt = (
        f"RESEARCH QUESTION: {query}\n\n"
        f"ANSWER TO ANALYZE:\n{answer}\n\n"
        f"CONTEXT CHUNKS:\n{build_chunk_context(chunks)}\n\n"
        "Extract an evidence table (JSON array)."
    )
    resp = client.generate(prompt, system=_EVIDENCE_TABLE_SYSTEM, max_tokens=2000)
    rows = extract_json_array(resp.text)
    if rows is None:
        rows = [{"claim": "Parse error", "evidence": resp.text[:500],
                 "citation": "", "confidence": "LOW", "notes": "Failed to parse LLM output"}]

    md_lines = [
        "# Evidence Table",
        f"**Query**: {query}",
        f"**Generated**: {datetime.datetime.utcnow().isoformat()}Z",
        "",
        "| # | Claim | Evidence Snippet | Citation | Confidence | Notes |",
        "|---|-------|-----------------|----------|:----------:|-------|",
    ]
    for i, r in enumerate(rows, 1):
        md_lines.append(
            f"| {i} | {r.get('claim', '')} | {r.get('evidence', '')} "
            f"| {r.get('citation', '')} | {r.get('confidence', '')} "
            f"| {r.get('notes', '')} |"
        )

    md_content = "\n".join(md_lines)
    path = _save_artifact("evidence_table", md_content)

    csv_buf = io.StringIO()
    writer = csv.DictWriter(csv_buf, fieldnames=["claim", "evidence", "citation", "confidence", "notes"])
    writer.writeheader()
    for r in rows:
        writer.writerow({k: r.get(k, "") for k in writer.fieldnames})
    csv_path = _save_artifact("evidence_table", csv_buf.getvalue(), ext="csv")

    return {"rows": rows, "markdown": md_content, "md_path": str(path),
            "csv_path": str(csv_path), "count": len(rows)}


# ── Annotated Bibliography ────────────────────────────────────────────────

_ANNOT_BIB_SYSTEM = (
    "You are a research librarian creating an annotated bibliography. "
    "For each source referenced in the answer or context, produce an entry with:\n"
    "1. source_id\n2. Full citation (authors, year, title, venue)\n"
    "3. Key claim or contribution (1-2 sentences)\n4. Method summary (1 sentence)\n"
    "5. Limitations (1 sentence)\n6. Why it matters for the research question (1 sentence)\n\n"
    "Output ONLY a JSON array of objects with keys: source_id, citation, claim, method, "
    "limitations, relevance.\n"
    "Include 8-12 sources. If a source was retrieved but not useful, still include it with "
    "relevance='Tangential to this specific query'."
)


def generate_annotated_bibliography(
    query: str, answer: str, chunks: list[dict], client: LLMClient,
) -> dict:
    manifest = _load_manifest_map()
    prompt = (
        f"RESEARCH QUESTION: {query}\n\n"
        f"ANSWER:\n{answer}\n\n"
        f"CONTEXT CHUNKS:\n{build_chunk_context(chunks)}\n\n"
        "Create an annotated bibliography (JSON array)."
    )
    resp = client.generate(prompt, system=_ANNOT_BIB_SYSTEM, max_tokens=3000)
    entries = extract_json_array(resp.text)
    if entries is None:
        entries = [{"source_id": "parse_error", "citation": resp.text[:300],
                    "claim": "", "method": "", "limitations": "", "relevance": ""}]

    for e in entries:
        sid = e.get("source_id", "")
        if sid in manifest:
            m = manifest[sid]
            e["url_or_doi"] = m.get("url_or_doi", "")
            e["venue"] = m.get("venue", "")
            e["year"] = m.get("year", "")

    md_lines = [
        "# Annotated Bibliography",
        f"**Research Question**: {query}",
        f"**Generated**: {datetime.datetime.utcnow().isoformat()}Z",
        f"**Sources**: {len(entries)}",
        "",
    ]
    for i, e in enumerate(entries, 1):
        md_lines += [
            f"## {i}. {e.get('source_id', 'Unknown')}",
            f"**Citation**: {e.get('citation', 'N/A')}",
            f"**URL/DOI**: {e.get('url_or_doi', 'N/A')}",
            f"**Key Claim**: {e.get('claim', 'N/A')}",
            f"**Method**: {e.get('method', 'N/A')}",
            f"**Limitations**: {e.get('limitations', 'N/A')}",
            f"**Relevance**: {e.get('relevance', 'N/A')}",
            "",
        ]

    md_content = "\n".join(md_lines)
    path = _save_artifact("annotated_bibliography", md_content)
    return {"entries": entries, "markdown": md_content, "md_path": str(path),
            "count": len(entries)}


# ── Synthesis Memo ────────────────────────────────────────────────────────

_SYNTHESIS_MEMO_SYSTEM = (
    "You are a senior research analyst writing a synthesis memo. Write 800-1200 words "
    "that integrates findings from the provided context chunks to answer the research question.\n\n"
    "STRUCTURE:\n"
    "1. **Executive Summary** (2-3 sentences)\n"
    "2. **Key Findings** (3-5 paragraphs with inline citations as (source_id, chunk_id))\n"
    "3. **Areas of Agreement** (what sources converge on)\n"
    "4. **Areas of Disagreement or Gaps** (conflicts, missing evidence)\n"
    "5. **Implications and Next Steps** (what this means for the research question)\n\n"
    "RULES:\n"
    "- Every factual claim MUST cite (source_id, chunk_id)\n"
    "- Flag conflicts explicitly\n"
    "- If evidence is insufficient, say so and suggest what evidence would help\n"
    "- End with a REFERENCE LIST of all cited sources\n"
    "- Target 800-1200 words"
)


def generate_synthesis_memo(
    query: str, answer: str, chunks: list[dict], client: LLMClient,
) -> dict:
    manifest = _load_manifest_map()
    prompt = (
        f"RESEARCH QUESTION: {query}\n\n"
        f"PRIOR ANSWER (use as starting point, but expand with full context):\n{answer}\n\n"
        f"CONTEXT CHUNKS:\n{build_chunk_context(chunks)}\n\n"
        "Write a synthesis memo (800-1200 words) with inline citations."
    )
    resp = client.generate(prompt, system=_SYNTHESIS_MEMO_SYSTEM, max_tokens=4000)
    memo_text = resp.text.strip()

    cites = CITE_RE.findall(memo_text)
    cited_sources = sorted(set(s for s, _ in cites))
    ref_block = []
    for sid in cited_sources:
        if sid in manifest:
            m = manifest[sid]
            ref_block.append(
                f"- **{sid}**: {m.get('authors', '')} ({m.get('year', '')}). "
                f"*{m.get('title', '')}*. {m.get('venue', '')}. {m.get('url_or_doi', '')}"
            )

    md_lines = [
        "# Synthesis Memo",
        f"**Research Question**: {query}",
        f"**Generated**: {datetime.datetime.utcnow().isoformat()}Z",
        f"**Sources cited**: {len(cited_sources)}",
        "",
        "---",
        "",
        memo_text,
    ]
    if ref_block:
        md_lines += ["", "---", "", "## Full Reference List", ""] + ref_block

    md_content = "\n".join(md_lines)
    word_count = len(memo_text.split())
    path = _save_artifact("synthesis_memo", md_content)
    return {"memo": memo_text, "markdown": md_content, "md_path": str(path),
            "word_count": word_count, "citations_count": len(cites),
            "sources_cited": cited_sources}


# ── PDF Export ────────────────────────────────────────────────────────────

def artifact_to_pdf(md_text: str) -> bytes:
    """Convert artifact Markdown to a simple PDF."""
    from fpdf import FPDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    for line in md_text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, stripped[2:], new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=10)
        elif stripped.startswith("## "):
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 9, stripped[3:], new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=10)
        elif stripped.startswith("### "):
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 8, stripped[4:], new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=10)
        elif stripped.startswith("---"):
            pdf.cell(0, 4, "", new_x="LMARGIN", new_y="NEXT")
        elif stripped.startswith("|"):
            pdf.set_font("Courier", size=7)
            pdf.cell(0, 5, stripped[:120], new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=10)
        elif stripped.startswith("```"):
            continue
        elif stripped.startswith(">"):
            pdf.set_font("Helvetica", "I", 9)
            clean = stripped.lstrip("> ").replace("**", "").replace("*", "")
            pdf.multi_cell(0, 5, clean)
            pdf.set_font("Helvetica", size=10)
        elif stripped:
            clean = stripped.replace("**", "").replace("*", "")
            pdf.multi_cell(0, 5, clean)
        else:
            pdf.cell(0, 3, "", new_x="LMARGIN", new_y="NEXT")
    return pdf.output()
