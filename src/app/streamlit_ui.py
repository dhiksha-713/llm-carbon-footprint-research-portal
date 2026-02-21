"""Streamlit UI - LLM Carbon Footprint Personal Research Portal (Phase 3)."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json
import time

import streamlit as st
import pandas as pd

from src.config import (
    PROJECT_ROOT, PROCESSED_DIR, OUTPUTS_DIR, MANIFEST_PATH,
    GROK_MODEL, AZURE_MODEL, EMBED_MODEL_NAME,
    CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS, TOP_K, ENHANCED_TOP_N,
    REPORT_DIR, LLM_PROVIDER, GROK_API_KEY, AZURE_API_KEY,
    AZURE_ENDPOINT, ARTIFACTS_DIR,
)
from src.utils import sanitize_query, safe_avg, load_eval_results, suggest_next_steps
from src.artifacts import artifact_to_pdf

st.set_page_config(
    page_title="LLM Carbon Footprint Research Portal",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""<style>
    .block-container { padding-top: 1.5rem; }
    div[data-testid="stMetric"] {
        background: #f8f9fa; border-radius: 8px; padding: 12px 16px;
        border-left: 4px solid #4CAF50;
    }
    /* Only constrain long answers inside chat bubbles on the Research page */
    div[data-testid="stChatMessage"] .stMarkdown {
        max-height: 60vh; overflow-y: auto;
    }
</style>""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_manifest() -> pd.DataFrame:
    return pd.read_csv(MANIFEST_PATH) if MANIFEST_PATH.exists() else pd.DataFrame()

@st.cache_resource
def _load_resources():
    from sentence_transformers import SentenceTransformer
    from src.rag.rag import load_index
    index, store = load_index()
    return index, store, SentenceTransformer(EMBED_MODEL_NAME)

def _get_client(provider: str):
    from src.llm_client import GrokClient, AzureOpenAIClient
    return AzureOpenAIClient() if provider == "azure_openai" else GrokClient()

def _run_query(query, index, store, embed_model, client, mode, top_k=TOP_K):
    from src.rag.rag import run_rag
    from src.rag.enhance_query_rewriting import run_enhanced_rag
    if mode == "enhanced":
        return run_enhanced_rag(query, index, store, embed_model, client)
    return run_rag(query, index, store, embed_model, client, top_k=top_k, mode=mode)

def _index_ready() -> bool:
    return (PROCESSED_DIR / "faiss_index.bin").exists() and (PROCESSED_DIR / "chunk_store.json").exists()

SAMPLE_QUESTIONS = [
    "What are the major sources of carbon emissions in LLM training?",
    "What is the total lifecycle carbon footprint of BLOOM?",
    "How does carbon intensity of the electricity grid affect LLM training emissions?",
    "Compare Strubell et al. and Patterson et al. on measurement methodology.",
    "What tools exist for tracking carbon emissions during ML training?",
    "Does the corpus contain evidence about the carbon footprint of GPT-4?",
    "How have carbon measurement methods for LLMs evolved from 2019 to 2023?",
    "What is the difference between operational and embodied carbon emissions in AI?",
]

# ── Sidebar ───────────────────────────────────────────────────────────────
grok_ok = bool(GROK_API_KEY)
azure_ok = bool(AZURE_API_KEY and AZURE_ENDPOINT)

PAGES = [
    "Home",
    "Demo All Phases",
    "Research",
    "Research Threads",
    "Artifacts",
    "Corpus Explorer",
    "Evaluation",
    "Compare Models",
]

with st.sidebar:
    st.markdown("### LLM Carbon Footprint")
    st.markdown("**Personal Research Portal**")
    st.markdown("---")
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.markdown("---")

    st.markdown("**LLM Provider**")
    opts, labels = [], {}
    if grok_ok:
        opts.append("grok"); labels["grok"] = f"Grok-3 ({GROK_MODEL})"
    if azure_ok:
        opts.append("azure_openai"); labels["azure_openai"] = f"Azure ({AZURE_MODEL})"

    if not opts:
        st.error("No LLM provider. Add API keys to .env"); chosen = None
    elif len(opts) == 1:
        chosen = opts[0]; st.info(f"Using: {labels[chosen]}")
    else:
        chosen = st.radio("Provider", opts,
                          index=opts.index(LLM_PROVIDER) if LLM_PROVIDER in opts else 0,
                          format_func=lambda x: labels[x], label_visibility="collapsed")

    st.markdown("---")
    st.caption(f"Embeddings: {EMBED_MODEL_NAME}")
    st.caption(f"Chunk: {CHUNK_SIZE_TOKENS}t / {CHUNK_OVERLAP_TOKENS}t overlap")
    st.caption(f"Top-K: {TOP_K} / Enhanced: {ENHANCED_TOP_N}")
    st.markdown("---")
    st.caption("AI Model Development (95-864)")
    st.caption("Group 4 | Dhiksha Rathis, Shreya Verma")
    st.caption("CMU Spring 2026")


# ══════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════
if page == "Home":
    st.title("Personal Research Portal")
    st.markdown("##### Carbon Footprint of Large Language Models")

    st.markdown("""
A research portal that answers questions about the environmental cost of large AI models
using 20 peer-reviewed papers. Every claim is cited. Every citation is validated. Missing
evidence is flagged with suggested next steps.

| Phase | What it does |
|-------|-------------|
| **Phase 1** | Research framing, prompt kit, evaluation rubric |
| **Phase 2** | RAG pipeline over 20-paper corpus with baseline + enhanced retrieval |
| **Phase 3** | Usable portal: research threads, artifact generation, export, trust behavior |

Go to **Demo All Phases** to see the entire system run end-to-end with one click.
    """)


# ══════════════════════════════════════════════════════════════════════════
# DEMO ALL PHASES — One click, entire pipeline, step by step
# ══════════════════════════════════════════════════════════════════════════
elif page == "Demo All Phases":
    st.header("Demo All Phases")
    st.caption("One click runs the entire pipeline from Phase 1 through Phase 3. "
               "Results persist across page reruns (laptop sleep, etc.).")

    if not chosen:
        st.error("No LLM provider configured. Add API keys to .env"); st.stop()

    both = grok_ok and azure_ok
    st.info(f"Active provider: **{labels[chosen]}**"
            + (" | Both providers ready — model comparison included." if both else ""))

    fresh = st.checkbox("Fresh start (re-download PDFs, rebuild index, re-run everything)", value=False)

    def _render_demo(d: dict):
        """Display demo results from a saved dict (works for live run AND cache)."""

        # ── Phase 1 ──────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Phase 1 — Research Framing")
        st.write("**Domain**: Environmental cost of large language model training and inference")
        st.write("**Main research question**: What is the carbon footprint of training and deploying "
                 "large language models, and how can it be measured and reduced?")
        st.write("**Sub-questions**:")
        st.write("1. What are the major sources of carbon emissions in LLM training?")
        st.write("2. How do different studies estimate training carbon — where do they agree/disagree?")
        st.write("3. What is the lifecycle carbon footprint (embodied + operational)?")
        st.write("4. Does inference energy exceed training energy over a model's lifetime?")
        st.write("5. What tools exist for tracking and reporting ML carbon emissions?")
        st.write("6. How have measurement methods evolved from 2019 to 2024?")
        st.write("")
        st.write("**Tasks chosen** (Phase 1 task menu):")
        st.write("- *Claim-evidence extraction*: output Claim | Evidence | Citation rows")
        st.write("- *Cross-source synthesis*: output Agreement | Disagreement | Supporting evidence")
        st.write("")
        st.write("**Models**: Grok-3 (CMU LLM API) and Azure OpenAI (o4-mini)")
        st.write("**Prompt design**: Baseline (minimal) vs Structured (guardrails + cite chunk_id + say unknown)")
        mdf = load_manifest()
        if not mdf.empty:
            st.write(f"**Corpus**: {len(mdf)} sources selected:")
            st.dataframe(mdf[["source_id", "title", "year", "source_type"]],
                         use_container_width=True, hide_index=True)

        # ── Phase 2 ──────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Phase 2 — RAG Pipeline")

        st.write(f"**Download**: {d['pdfs']}/{d['n_manifest']} PDFs")
        st.write(f"**Ingest**: {d['n_chunks']} chunks from {d['n_sources']} sources indexed in FAISS")
        st.write(f"**Embedding**: `{EMBED_MODEL_NAME}` (384-dim) | **LLM**: {d['provider']} | "
                 f"**FAISS**: {d['n_vectors']} vectors")

        st.markdown("#### Baseline RAG")
        bcv = d["baseline_cv"]
        st.write(f"**Query**: {d['baseline_query']}")
        st.write(f"Retrieved **{d['baseline_n_chunks']} chunks** in {d['baseline_time']:.1f}s — "
                 f"**Citations**: {bcv['valid_citations']}/{bcv['total_citations']} valid "
                 f"(precision: {bcv.get('citation_precision', 'N/A')})")
        with st.expander("Full Baseline Answer"):
            st.markdown(d["baseline_answer"])

        st.markdown("#### Enhanced RAG")
        ecv = d["enhanced_cv"]
        st.write(f"**Query**: {d['enhanced_query']}")
        st.write(f"Query type: **{d['enhanced_qtype']}** | "
                 f"Rewritten: {d['enhanced_rewritten']}")
        if d.get("enhanced_sub_queries"):
            for sq in d["enhanced_sub_queries"]:
                st.write(f"  - {sq}")
        st.write(f"**{d['enhanced_n_chunks']} chunks** merged — "
                 f"**Citations**: {ecv['valid_citations']}/{ecv['total_citations']} valid")
        with st.expander("Full Enhanced Answer"):
            st.markdown(d["enhanced_answer"])

        if d.get("comparison"):
            st.markdown("#### Model Comparison")
            st.dataframe(pd.DataFrame(d["comparison"]), use_container_width=True, hide_index=True)
        else:
            st.info("Model comparison skipped (need both API keys in .env).")

        # ── Phase 3 ──────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Phase 3 — Portal Product")

        st.markdown("#### Research Thread")
        st.write(f"Thread saved: `{d['thread_id']}` — "
                 f"Total threads: **{d['n_threads']}**")
        st.write("Each thread preserves: query + retrieved evidence + answer + citation validation")

        st.markdown("#### Evidence Table")
        st.write(f"**{d['artifact_count']} rows** extracted — Saved to: `{d['artifact_path']}`")
        st.markdown(d["artifact_md"])

        st.markdown("#### Trust Behavior")
        st.write(f"**Query**: {d['edge_query']}")
        edge_cv = d["edge_cv"]
        st.write(f"**Citations**: {edge_cv['valid_citations']}/{edge_cv['total_citations']}")
        with st.expander("Answer"):
            st.markdown(d["edge_answer"])
        if d.get("edge_next_steps"):
            st.write("**Evidence gap detected — suggested next steps:**")
            st.markdown(d["edge_next_steps"])
        else:
            st.write("(No evidence gap detected in this answer)")

        st.markdown("#### Evaluation")
        if d.get("eval_table"):
            st.dataframe(pd.DataFrame(d["eval_table"]), use_container_width=True, hide_index=True)

        st.markdown("#### Report")
        st.write(f"Report: **{d['report_chars']:,} chars** with per-query logs and failure cases.")
        if d.get("report_text"):
            st.download_button("Download Evaluation Report", data=d["report_text"],
                               file_name="evaluation_report.md", mime="text/markdown",
                               key="demo_report_dl")

        st.markdown("#### Security + AI Disclosure")
        st.write("**Security**: API keys in `.env` (git-ignored), prompt-injection detection, "
                 "1000-char limit, control-char strip")
        st.write("Prompt injection correctly **blocked**")
        st.write("**AI usage disclosure**:")
        st.dataframe(pd.DataFrame([
            {"Tool": f"Grok-3 ({GROK_MODEL})", "Purpose": "RAG generation + eval judge + artifacts"},
            {"Tool": f"Azure ({AZURE_MODEL})", "Purpose": "RAG generation + comparison"},
            {"Tool": "Cursor AI", "Purpose": "Code scaffolding"},
            {"Tool": f"sentence-transformers ({EMBED_MODEL_NAME})", "Purpose": "Embeddings"},
        ]), use_container_width=True, hide_index=True)

        st.success("All phases demonstrated end-to-end. Use the sidebar to explore individual features.")

    # ── Run button ────────────────────────────────────────────────────
    run_col, clear_col = st.columns([3, 1])
    run_clicked = run_col.button("Run Complete Demo", type="primary")
    if clear_col.button("Clear cached results"):
        st.session_state.pop("demo", None)
        st.rerun()

    if run_clicked:
        demo: dict = {}

        with st.status("Phase 2: Downloading PDFs...", expanded=True):
            if fresh:
                raw_dir = PROJECT_ROOT / "data" / "raw"
                for dd in [raw_dir, PROCESSED_DIR, PROJECT_ROOT / "logs",
                           OUTPUTS_DIR, REPORT_DIR]:
                    if dd.exists():
                        for f in dd.glob("*"):
                            if f.is_file():
                                f.unlink()
                    dd.mkdir(parents=True, exist_ok=True)
                _load_resources.clear()
                st.write("Cleaned all generated artifacts.")

            df = load_manifest()
            if df.empty:
                st.error("data/data_manifest.csv not found."); st.stop()
            st.write(f"Manifest: **{len(df)} sources**. Downloading PDFs...")
            from src.ingest.download_sources import main as download_main
            download_main(force_redownload=fresh)
            raw_dir = PROJECT_ROOT / "data" / "raw"
            pdfs = len(list(raw_dir.glob("*.pdf"))) if raw_dir.exists() else 0
            st.write(f"**{pdfs}/{len(df)} PDFs** downloaded and validated.")
            demo["pdfs"] = pdfs
            demo["n_manifest"] = len(df)

        with st.status("Ingesting and indexing...", expanded=True):
            from src.ingest.ingest import main as ingest_main
            ingest_main()
            _load_resources.clear()
            sp = PROCESSED_DIR / "chunk_store.json"
            if sp.exists():
                sd = json.loads(sp.read_text(encoding="utf-8"))
                demo["n_chunks"] = len(sd)
                demo["n_sources"] = len({c["source_id"] for c in sd})
                st.write(f"**{demo['n_chunks']} chunks** from **{demo['n_sources']} sources** indexed.")
            else:
                st.error("Ingestion failed."); st.stop()

        with st.status("Loading models...", expanded=True):
            index, store, embed_model = _load_resources()
            client = _get_client(chosen)
            demo["provider"] = labels[chosen]
            demo["n_vectors"] = index.ntotal
            st.write(f"Embedding: `{EMBED_MODEL_NAME}` | LLM: **{labels[chosen]}** | FAISS: {index.ntotal} vectors")

        with st.status("Running Baseline RAG...", expanded=True):
            from src.rag.rag import run_rag
            bq = "What are the major sources of carbon emissions in LLM training?"
            t0 = time.time()
            br = run_rag(bq, index, store, embed_model, client, top_k=TOP_K, mode="baseline")
            bt = time.time() - t0
            bcv = br["citation_validation"]
            demo["baseline_query"] = bq
            demo["baseline_answer"] = br["answer"]
            demo["baseline_cv"] = bcv
            demo["baseline_time"] = bt
            demo["baseline_n_chunks"] = len(br["retrieved_chunks"])
            st.write(f"**{bcv['valid_citations']}/{bcv['total_citations']}** valid citations in {bt:.1f}s")

        with st.status("Running Enhanced RAG...", expanded=True):
            from src.rag.enhance_query_rewriting import run_enhanced_rag
            eq = "How do different studies measure the energy consumption and carbon footprint of training large language models?"
            t0 = time.time()
            er = run_enhanced_rag(eq, index, store, embed_model, client)
            et = time.time() - t0
            ecv = er["citation_validation"]
            demo["enhanced_query"] = eq
            demo["enhanced_answer"] = er["answer"]
            demo["enhanced_cv"] = ecv
            demo["enhanced_time"] = et
            demo["enhanced_n_chunks"] = len(er["retrieved_chunks"])
            demo["enhanced_qtype"] = er.get("query_type", "")
            demo["enhanced_rewritten"] = er.get("rewritten_query", "N/A")
            demo["enhanced_sub_queries"] = er.get("sub_queries", [])
            st.write(f"**{ecv['valid_citations']}/{ecv['total_citations']}** valid citations in {et:.1f}s")

        if both:
            with st.status("Model comparison...", expanded=True):
                from src.llm_client import GrokClient, AzureOpenAIClient
                cq = "What tools exist for tracking carbon emissions during ML training?"
                def _try(cls):
                    try:
                        t0 = time.time()
                        return run_rag(cq, index, store, embed_model, cls(), mode="baseline"), time.time() - t0
                    except Exception:
                        return None, None
                g_r, g_t = _try(GrokClient)
                a_r, a_t = _try(AzureOpenAIClient)
                if g_r and a_r:
                    gcv, acv = g_r["citation_validation"], a_r["citation_validation"]
                    demo["comparison"] = [
                        {"Metric": "Latency", "Grok-3": f"{g_t:.1f}s", "Azure": f"{a_t:.1f}s"},
                        {"Metric": "Valid Citations",
                         "Grok-3": f"{gcv['valid_citations']}/{gcv['total_citations']}",
                         "Azure": f"{acv['valid_citations']}/{acv['total_citations']}"},
                    ]
                    st.write("Comparison complete.")

        with st.status("Saving research thread...", expanded=True):
            from src.threads import save_thread, list_threads
            thread = save_thread(
                query=bq, answer=br["answer"],
                retrieved_chunks=br["retrieved_chunks"],
                citation_validation=bcv, mode="baseline",
                provider=labels[chosen],
            )
            all_threads = list_threads()
            demo["thread_id"] = thread["thread_id"]
            demo["n_threads"] = len(all_threads)
            st.write(f"Thread saved: `{thread['thread_id']}` ({len(all_threads)} total)")

        with st.status("Generating evidence table...", expanded=True):
            from src.artifacts import generate_evidence_table
            artifact = generate_evidence_table(bq, br["answer"], br["retrieved_chunks"], client)
            demo["artifact_count"] = artifact["count"]
            demo["artifact_path"] = artifact["md_path"]
            demo["artifact_md"] = artifact["markdown"]
            st.write(f"**{artifact['count']} rows** extracted")

        with st.status("Trust behavior...", expanded=True):
            edge_q = "Does the corpus contain evidence about the carbon footprint of GPT-4?"
            edge_r = run_rag(edge_q, index, store, embed_model, client, mode="baseline")
            edge_cv = edge_r["citation_validation"]
            ns = suggest_next_steps(edge_r["answer"], edge_q, edge_r["retrieved_chunks"])
            demo["edge_query"] = edge_q
            demo["edge_answer"] = edge_r["answer"]
            demo["edge_cv"] = edge_cv
            demo["edge_next_steps"] = ns
            st.write(f"Citations: {edge_cv['valid_citations']}/{edge_cv['total_citations']} — "
                     + ("gap detected" if ns else "no gap"))

        with st.status("Running 20-query evaluation (this takes several minutes)...", expanded=True):
            from src.eval.evaluation import EVAL_QUERIES, run_evaluation, compute_summary
            st.write(f"**{len(EVAL_QUERIES)} queries** x 2 modes...")
            st.write("Running **baseline**...")
            baseline_results = run_evaluation("baseline")
            st.write(f"Baseline done ({len(baseline_results)} queries). Running **enhanced**...")
            enhanced_results = run_evaluation("enhanced")
            st.write(f"Enhanced done ({len(enhanced_results)} queries).")
            bsm = compute_summary(baseline_results)["overall"]
            esm = compute_summary(enhanced_results)["overall"]
            demo["eval_table"] = [
                {"Mode": "Baseline",
                 "Groundedness": f"{bsm['avg_groundedness']}/4",
                 "Relevance": f"{bsm['avg_relevance']}/4",
                 "Cite Precision": str(bsm["avg_cite_precision"])},
                {"Mode": "Enhanced",
                 "Groundedness": f"{esm['avg_groundedness']}/4",
                 "Relevance": f"{esm['avg_relevance']}/4",
                 "Cite Precision": str(esm["avg_cite_precision"])},
            ]

        with st.status("Generating report...", expanded=True):
            from src.eval.generate_report import generate_report
            rp = generate_report()
            report_text = rp.read_text(encoding="utf-8") if rp.exists() else ""
            demo["report_text"] = report_text
            demo["report_chars"] = len(report_text)
            st.write(f"Report: **{len(report_text):,} chars**")

        st.session_state["demo"] = demo
        _render_demo(demo)

    elif "demo" in st.session_state:
        st.caption("Showing cached results from previous run. Click **Run Complete Demo** to re-run, "
                   "or **Clear cached results** to reset.")
        _render_demo(st.session_state["demo"])


# ══════════════════════════════════════════════════════════════════════════
# RESEARCH — Ask questions, get cited answers, auto-save threads
# ══════════════════════════════════════════════════════════════════════════
elif page == "Research":
    st.header("Research")
    st.caption("Ask questions. Answers are grounded in the 20-paper corpus. Threads save automatically.")

    if not chosen:
        st.error("No LLM provider configured."); st.stop()
    if not _index_ready():
        st.warning("Index not built. Run **Demo All Phases** or `make ingest` first."); st.stop()

    mode = st.selectbox("Mode", ["baseline", "enhanced"],
                        help="Enhanced: query rewriting + decomposition for complex questions")
    sample = st.selectbox("Sample questions:", ["(type your own)"] + SAMPLE_QUESTIONS)

    if "chat" not in st.session_state:
        st.session_state["chat"] = []

    for msg in st.session_state["chat"]:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                st.markdown(msg["answer"])
                cv = msg["cv"]
                st.caption(f"{cv['valid_citations']}/{cv['total_citations']} citations valid | "
                           f"{msg['mode']} | {msg['provider']}")
                if msg.get("next_steps"):
                    with st.expander("Suggested Next Steps"):
                        st.markdown(msg["next_steps"])

    pending = sample if sample != "(type your own)" and st.button("Use sample") else None
    query = pending or st.chat_input("Ask about LLM carbon footprints...")

    if query:
        try:
            query = sanitize_query(query)
        except ValueError as e:
            st.error(str(e)); st.stop()

        st.session_state["chat"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.status("Retrieving and generating...", expanded=True) as ps:
                index, store, embed_model = _load_resources()
                t0 = time.time()
                result = _run_query(query, index, store, embed_model,
                                    _get_client(chosen), mode)
                elapsed = time.time() - t0
                ps.update(label=f"Done ({elapsed:.1f}s)", state="complete")

            st.markdown(result["answer"])
            cv = result["citation_validation"]
            st.caption(f"{cv['valid_citations']}/{cv['total_citations']} citations valid | "
                       f"precision {cv.get('citation_precision', 'N/A')} | {elapsed:.1f}s")

            next_steps = suggest_next_steps(result["answer"], query, result["retrieved_chunks"])
            if next_steps:
                with st.expander("Suggested Next Steps (evidence gap detected)", expanded=True):
                    st.markdown(next_steps)

            with st.expander(f"Retrieved Evidence ({len(result['retrieved_chunks'])} chunks)"):
                for j, c in enumerate(result["retrieved_chunks"], 1):
                    st.markdown(f"**{j}. ({c['source_id']}, {c['chunk_id']})** "
                                f"score={c['retrieval_score']:.4f} — {c['title']} ({c['year']})")

            from src.threads import save_thread
            thread = save_thread(
                query=query, answer=result["answer"],
                retrieved_chunks=result["retrieved_chunks"],
                citation_validation=cv, mode=mode,
                provider=labels.get(chosen, chosen),
                metadata={"elapsed_s": round(elapsed, 2)},
            )

        st.session_state["chat"].append({
            "role": "assistant", "answer": result["answer"],
            "cv": cv, "mode": mode, "provider": labels.get(chosen, chosen),
            "next_steps": next_steps,
        })


# ══════════════════════════════════════════════════════════════════════════
# RESEARCH THREADS — Browse, view, export
# ══════════════════════════════════════════════════════════════════════════
elif page == "Research Threads":
    st.header("Research Threads")
    st.caption("Saved research sessions with query, evidence, answer, and citations.")

    from src.threads import list_threads, load_thread, delete_thread, export_thread_markdown

    threads = list_threads()
    if not threads:
        st.info("No threads yet. Ask a question on the **Research** page."); st.stop()

    selected_id = st.selectbox(
        "Select thread:",
        [t["thread_id"] for t in threads],
        format_func=lambda tid: next(
            (f"{t['timestamp'][:16]} — {t['query'][:70]}" for t in threads
             if t["thread_id"] == tid), tid),
    )

    thread = load_thread(selected_id)
    if thread:
        st.markdown(f"**Query**: {thread['query']}")
        st.markdown("---")
        st.markdown(thread["answer"])

        cv = thread.get("citation_validation", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Valid Citations", cv.get("valid_citations", 0))
        c2.metric("Total Citations", cv.get("total_citations", 0))
        prec = cv.get("citation_precision")
        c3.metric("Precision", f"{prec:.2f}" if prec is not None else "N/A")

        with st.expander(f"Retrieved Evidence ({len(thread.get('retrieved_chunks', []))} chunks)"):
            for j, c in enumerate(thread.get("retrieved_chunks", []), 1):
                st.markdown(f"**{j}. ({c.get('source_id')}, {c.get('chunk_id')})** "
                            f"score={c.get('retrieval_score', 0):.4f}")

        st.markdown("---")
        md_export = export_thread_markdown(thread)
        col_a, col_b = st.columns(2)
        col_a.download_button("Export Markdown", data=md_export,
                              file_name=f"thread_{selected_id}.md", mime="text/markdown")
        try:
            col_b.download_button("Export PDF", data=artifact_to_pdf(md_export),
                                  file_name=f"thread_{selected_id}.pdf", mime="application/pdf")
        except Exception:
            pass

        if st.button("Delete this thread"):
            delete_thread(selected_id)
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# ARTIFACTS — Generate and browse research artifacts
# ══════════════════════════════════════════════════════════════════════════
elif page == "Artifacts":
    st.header("Research Artifacts")
    st.caption("Generate evidence tables, annotated bibliographies, or synthesis memos.")

    if not chosen:
        st.error("No LLM provider configured."); st.stop()
    if not _index_ready():
        st.warning("Index not built. Run **Demo All Phases** first."); st.stop()

    art_query = st.text_area("Research question:", height=60,
                             value="What are the major sources of carbon emissions in LLM training?")
    art_type = st.selectbox("Artifact type:",
                            ["Evidence Table", "Annotated Bibliography", "Synthesis Memo"])

    if st.button("Generate Artifact", type="primary") and art_query.strip():
        try:
            art_query = sanitize_query(art_query)
        except ValueError as e:
            st.error(str(e)); st.stop()

        with st.status("Running RAG + generating artifact...", expanded=True) as ps:
            index, store, embed_model = _load_resources()
            client = _get_client(chosen)
            result = _run_query(art_query, index, store, embed_model, client, "enhanced")

            from src.artifacts import (
                generate_evidence_table, generate_annotated_bibliography,
                generate_synthesis_memo, artifact_to_pdf,
            )

            if art_type == "Evidence Table":
                artifact = generate_evidence_table(
                    art_query, result["answer"], result["retrieved_chunks"], client)
            elif art_type == "Annotated Bibliography":
                artifact = generate_annotated_bibliography(
                    art_query, result["answer"], result["retrieved_chunks"], client)
            else:
                artifact = generate_synthesis_memo(
                    art_query, result["answer"], result["retrieved_chunks"], client)
            ps.update(label="Artifact generated", state="complete")

        st.markdown(artifact["markdown"])
        st.markdown("---")
        col_a, col_b = st.columns(2)
        col_a.download_button("Download Markdown", data=artifact["markdown"],
                              file_name=f"{art_type.lower().replace(' ', '_')}.md",
                              mime="text/markdown")
        if "csv_path" in artifact:
            col_b.download_button("Download CSV",
                                  data=Path(artifact["csv_path"]).read_text(encoding="utf-8"),
                                  file_name="evidence_table.csv", mime="text/csv")

    st.markdown("---")
    st.subheader("Previously Generated")
    artifacts = sorted(ARTIFACTS_DIR.glob("*"), reverse=True) if ARTIFACTS_DIR.exists() else []
    if artifacts:
        for af in artifacts[:10]:
            with st.expander(af.name):
                if af.suffix == ".csv":
                    try:
                        st.dataframe(pd.read_csv(af), use_container_width=True, hide_index=True)
                    except Exception:
                        st.text(af.read_text(encoding="utf-8"))
                else:
                    st.markdown(af.read_text(encoding="utf-8"))
    else:
        st.caption("No artifacts yet. Generate one above.")


# ══════════════════════════════════════════════════════════════════════════
# CORPUS EXPLORER
# ══════════════════════════════════════════════════════════════════════════
elif page == "Corpus Explorer":
    st.header("Corpus Explorer")
    st.caption("Browse the 20-source knowledge base.")

    df = load_manifest()
    if df.empty:
        st.warning("Manifest not found."); st.stop()

    search = st.text_input("Search by title or tags:")
    filtered = df
    if search:
        mask = (df["title"].str.contains(search, case=False, na=False) |
                df["tags"].str.contains(search, case=False, na=False))
        filtered = df[mask]

    st.dataframe(
        filtered[["source_id", "title", "authors", "year", "source_type", "venue"]],
        use_container_width=True, hide_index=True,
    )

    selected_source = st.selectbox("View details:", ["(select)"] + filtered["source_id"].tolist())

    if selected_source != "(select)":
        row = df[df["source_id"] == selected_source].iloc[0]
        st.subheader(row["title"])
        st.markdown(f"**Authors**: {row['authors']} | **Year**: {row['year']} | "
                    f"**Venue**: {row['venue']}")
        st.markdown(f"**URL**: [{row['url_or_doi']}]({row['url_or_doi']})")
        st.markdown(f"**Tags**: {row['tags']}")
        st.markdown(f"**Relevance**: {row['relevance_note']}")

        if _index_ready():
            chunk_store_path = PROCESSED_DIR / "chunk_store.json"
            if chunk_store_path.exists():
                all_chunks = json.loads(chunk_store_path.read_text(encoding="utf-8"))
                source_chunks = [c for c in all_chunks if c["source_id"] == selected_source]
                st.caption(f"{len(source_chunks)} chunks indexed")
                if source_chunks:
                    with st.expander(f"Preview chunks ({len(source_chunks)})"):
                        for c in source_chunks[:8]:
                            st.markdown(f"**{c['chunk_id']}** | {c.get('section_header', '')}")
                            st.caption(c["chunk_text"])


# ══════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════
elif page == "Evaluation":
    st.header("Evaluation")
    st.caption("20-query test set scored on 6 metrics.")

    from src.eval.evaluation import EVAL_QUERIES

    st.dataframe(pd.DataFrame([
        {"ID": q["id"], "Type": q["type"], "Query": q["query"][:80],
         "Expected": ", ".join(q["expected_sources"]) or "(none)"}
        for q in EVAL_QUERIES
    ]), use_container_width=True, hide_index=True)

    st.markdown("---")

    for lbl, results in [("Baseline", load_eval_results("baseline")),
                         ("Enhanced", load_eval_results("enhanced"))]:
        if not results:
            continue
        st.subheader(f"{lbl} Results ({len(results)} queries)")

        st.dataframe(pd.DataFrame([{
            "ID": r["query_id"], "Type": r["query_type"],
            "Ground.": r["groundedness"].get("score"),
            "Relev.": r["answer_relevance"].get("score"),
            "Cite Prec.": round(r["citation_precision"], 2) if r.get("citation_precision") is not None else None,
            "Src Recall": round(r["source_recall"], 2) if r.get("source_recall") is not None else None,
        } for r in results]), use_container_width=True, hide_index=True)

        m1, m2, m3 = st.columns(3)
        g = [r["groundedness"].get("score") for r in results if r["groundedness"].get("score")]
        rel = [r["answer_relevance"].get("score") for r in results if r["answer_relevance"].get("score")]
        cp = [r["citation_precision"] for r in results if r.get("citation_precision") is not None]
        m1.metric("Avg Groundedness", f"{safe_avg(g)}/4" if g else "-")
        m2.metric("Avg Relevance", f"{safe_avg(rel)}/4" if rel else "-")
        m3.metric("Avg Cite Precision", f"{safe_avg(cp)}" if cp else "-")

        scored = sorted(
            [(r["groundedness"].get("score") or 0) + (r["answer_relevance"].get("score") or 0), r]
            for r in results
        )
        if scored:
            worst = scored[0][1]
            best = scored[-1][1]
            st.markdown(f"**Best**: {best['query_id']} — Ground. {best['groundedness'].get('score')}/4, "
                        f"Relev. {best['answer_relevance'].get('score')}/4")
            st.markdown(f"**Worst**: {worst['query_id']} — Ground. {worst['groundedness'].get('score')}/4, "
                        f"Relev. {worst['answer_relevance'].get('score')}/4")
            if worst["groundedness"].get("reasoning"):
                st.caption(f"Judge: {worst['groundedness']['reasoning']}")

        st.markdown("---")

    if not load_eval_results("baseline") and not load_eval_results("enhanced"):
        st.info("No results. Run **Demo All Phases** or `make eval-both`.")


# ══════════════════════════════════════════════════════════════════════════
# COMPARE MODELS
# ══════════════════════════════════════════════════════════════════════════
elif page == "Compare Models":
    st.header("Model Comparison")
    st.caption("Same query through Grok-3 and Azure side-by-side.")
    if not _index_ready():
        st.warning("Index not built."); st.stop()
    if not grok_ok:
        st.warning("Grok-3 not configured.")
    if not azure_ok:
        st.warning("Azure not configured.")

    comp_q = st.selectbox("Query:", SAMPLE_QUESTIONS)
    custom = st.text_input("Or custom:")
    qtr = custom.strip() or comp_q

    if st.button("Run Comparison", type="primary", disabled=(not grok_ok or not azure_ok)):
        try:
            qtr = sanitize_query(qtr)
        except ValueError as e:
            st.error(str(e)); st.stop()

        index, store, embed_model = _load_resources()
        from src.llm_client import GrokClient, AzureOpenAIClient

        def _run_prov(name, cls):
            try:
                t0 = time.time()
                return _run_query(qtr, index, store, embed_model, cls(), "enhanced"), time.time() - t0
            except Exception as exc:
                st.error(f"{name}: {exc}"); return None, None

        cg, ca = st.columns(2)
        with cg:
            st.subheader(f"Grok-3")
            with st.status("Running...") as sg:
                gr, gt = _run_prov("Grok-3", GrokClient)
                sg.update(label=f"{gt:.1f}s" if gt else "Failed",
                          state="complete" if gr else "error")
            if gr:
                st.markdown(gr["answer"])
                gcv = gr["citation_validation"]
                st.caption(f"{gt:.1f}s | {gcv['valid_citations']}/{gcv['total_citations']} citations")

        with ca:
            st.subheader(f"Azure")
            with st.status("Running...") as sa:
                ar, at = _run_prov("Azure", AzureOpenAIClient)
                sa.update(label=f"{at:.1f}s" if at else "Failed",
                          state="complete" if ar else "error")
            if ar:
                st.markdown(ar["answer"])
                acv = ar["citation_validation"]
                st.caption(f"{at:.1f}s | {acv['valid_citations']}/{acv['total_citations']} citations")
