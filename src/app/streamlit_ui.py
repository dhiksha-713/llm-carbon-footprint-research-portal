"""Streamlit UI -- LLM Carbon Footprint Research Portal (Phase 2)."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json
import glob
import time

import streamlit as st
import pandas as pd

from src.config import (
    PROJECT_ROOT, PROCESSED_DIR, LOGS_DIR, OUTPUTS_DIR,
    MANIFEST_PATH, GENERATION_MODEL, JUDGE_MODEL, EMBED_MODEL_NAME,
    CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS, TOP_K, ENHANCED_TOP_N,
    REPORT_DIR, CHUNK_PREVIEW_LEN, LLM_PROVIDER,
    GEMINI_API_KEY, AZURE_API_KEY, AZURE_ENDPOINT,
)
from src.utils import sanitize_query

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Carbon Footprint Research Portal",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Helpers ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_manifest() -> pd.DataFrame:
    if MANIFEST_PATH.exists():
        return pd.read_csv(MANIFEST_PATH)
    return pd.DataFrame()


@st.cache_resource
def load_rag_resources():
    from sentence_transformers import SentenceTransformer
    from src.rag.rag import load_index
    from src.llm_client import get_llm_client
    index, store = load_index()
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    client = get_llm_client()
    return index, store, embed_model, client


def load_eval_results(mode: str) -> list[dict]:
    files = sorted(glob.glob(str(OUTPUTS_DIR / f"eval_results_{mode}_*.json")))
    if not files:
        return []
    return json.loads(Path(files[-1]).read_text(encoding="utf-8"))


def safe_avg(vals):
    clean = [v for v in vals if v is not None]
    return round(sum(clean) / len(clean), 2) if clean else None


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
PAGES = [
    "Overview",
    "Run Pipeline",
    "Ask a Question",
    "Compare Models",
    "Evaluation",
    "Deliverables",
]

with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio("Page", PAGES, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Active Configuration**")
    st.text(f"Provider:    {LLM_PROVIDER}")
    st.text(f"Generation:  {GENERATION_MODEL}")
    st.text(f"Judge:       {JUDGE_MODEL}")
    st.text(f"Embeddings:  {EMBED_MODEL_NAME}")
    st.text(f"Chunk:       {CHUNK_SIZE_TOKENS}t / {CHUNK_OVERLAP_TOKENS}t")
    st.text(f"Top-K:       {TOP_K} / {ENHANCED_TOP_N}")

    gemini_ok = bool(GEMINI_API_KEY)
    azure_ok = bool(AZURE_API_KEY and AZURE_ENDPOINT)
    st.markdown("---")
    st.markdown("**Provider Status**")
    st.text(f"Gemini:      {'ready' if gemini_ok else 'not configured'}")
    st.text(f"Azure OpenAI:{'ready' if azure_ok else 'not configured'}")

    st.markdown("---")
    st.caption("AI Model Development (95-864)")
    st.caption("Group 4 -- Dhiksha Rathis, Shreya Verma")
    st.caption("CMU, Spring 2026")


# ══════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.header("LLM Carbon Footprint Research Portal")
    st.markdown(
        "A research-grade RAG system for systematic review of carbon emissions "
        "in Large Language Models. Supports **multi-provider comparison** "
        "(Google Gemini and Azure OpenAI o4-mini)."
    )

    st.subheader("Research Question")
    st.markdown(
        "> **How do we accurately measure and compare the carbon footprint of "
        "different LLMs across their lifecycle?**"
    )
    st.markdown("""
**Sub-questions:**
1. What are the major sources of emissions in LLM training vs. inference?
2. How do different studies measure and report carbon metrics?
3. What factors (model size, hardware, location) most impact carbon footprint?
4. How do carbon estimates vary across different LLM families?
5. What data is missing or inconsistent in current carbon reporting?
    """)

    # ── Lifecycle diagram ─────────────────────────────────────────────────
    st.subheader("Pipeline Lifecycle")
    st.code("""
    DATA LAYER                      QUERY LAYER                     OUTPUT LAYER
    ----------                      -----------                     ------------
    20 PDF sources                  User query                      Cited answer
         |                               |                              ^
         v                               v                              |
    [Download]                     [Sanitize input]               [Validate citations]
         |                               |                              ^
         v                               v                              |
    [Parse PDF]                    [Classify query type]          [Generate answer]
    (PyMuPDF)                      direct / synthesis / edge           ^
         |                               |                              |
         v                          +---------+---------+               |
    [Section-aware chunking]        |         |         |               |
    (500t / 100t overlap)     [Baseline]  [Enhanced]  [Edge]           |
         |                      top-K     rewrite +    flag            |
         v                     retrieve   decompose   missing          |
    [Embed chunks]                  |     multi-query    |             |
    (all-MiniLM-L6-v2)             |     merge chunks   |             |
         |                          +--------+---------+              |
         v                                   |                         |
    [FAISS IndexFlatIP]  <--- semantic search |                        |
                                             v                         |
                                    [LLM Generation] -----------------+
                                    (Gemini / Azure OpenAI o4-mini)

    EVALUATION: 20-query set x 6 metrics x LLM-as-judge scoring
    COMPARISON: Same queries, same corpus, different LLM providers
    """, language=None)

    st.subheader("Corpus")
    df = load_manifest()
    if not df.empty:
        st.write(f"**{len(df)} sources**, spanning {int(df['year'].min())}--{int(df['year'].max())}.")
        st.dataframe(
            df[["source_id", "title", "authors", "year", "source_type", "venue", "tags"]],
            width="stretch", hide_index=True,
        )
    else:
        st.warning("Manifest not found. Run `make download` first.")

    st.subheader("Phase 2 Deliverables")
    st.markdown("""
- **D1 -- Code repository**: Modular `src/` packages, `Makefile`, env-driven config, multi-provider LLM abstraction.
- **D2 -- Data manifest**: 20-source corpus with full A3 schema in `data/data_manifest.csv`.
- **D3 -- RAG pipeline**: Baseline and enhanced pipelines with citation validation and trust behaviors.
- **D4 -- Evaluation framework**: 20-query set scored on 6 metrics using LLM-as-judge.
- **D5 -- Evaluation report**: Auto-generated at `report/phase2/evaluation_report.md`.
- **D6 -- API backend**: FastAPI REST API with `/query`, `/corpus`, `/evaluation`, `/logs` endpoints.
- **D7 -- Interactive UI**: This Streamlit app with pipeline orchestrator, chat, and model comparison.
- **D8 -- Model comparison**: Side-by-side Gemini vs Azure OpenAI on identical queries.
- **D9 -- Security**: Input sanitization, prompt-injection protection, API key isolation.
    """)


# ══════════════════════════════════════════════════════════════════════════
# PAGE: RUN PIPELINE
# ══════════════════════════════════════════════════════════════════════════
elif page == "Run Pipeline":
    st.header("Run Full Pipeline")
    st.markdown(
        "Execute the entire Phase 2 pipeline end-to-end. "
        "Each step shows exactly what happens behind the scenes."
    )

    if st.button("Run Full Pipeline", type="primary"):

        # Step 1
        with st.status("Step 1/6: Loading corpus manifest...", expanded=True) as s1:
            df = load_manifest()
            if df.empty:
                st.error("data_manifest.csv not found. Run `make download` first.")
                st.stop()
            st.write(f"Loaded **{len(df)} sources**.")
            st.write(f"Types: {df['source_type'].value_counts().to_dict()}")
            st.dataframe(
                df[["source_id", "title", "year", "source_type"]],
                width="stretch", hide_index=True,
            )
            s1.update(label=f"Step 1/6: Corpus loaded ({len(df)} sources)", state="complete")

        # Step 2
        with st.status("Step 2/6: Verifying ingestion...", expanded=True) as s2:
            idx_path = PROCESSED_DIR / "faiss_index.bin"
            store_path = PROCESSED_DIR / "chunk_store.json"
            if not idx_path.exists() or not store_path.exists():
                st.error("FAISS index or chunk store missing. Run `make ingest`.")
                st.stop()
            store_data = json.loads(store_path.read_text(encoding="utf-8"))
            st.write(f"FAISS index: `{idx_path.name}` exists.")
            st.write(f"Chunk store: **{len(store_data)} chunks** across "
                      f"**{len({c['source_id'] for c in store_data})} sources**.")
            s2.update(label=f"Step 2/6: Ingestion verified ({len(store_data)} chunks)", state="complete")

        # Step 3
        with st.status(f"Step 3/6: Loading models ({LLM_PROVIDER})...", expanded=True) as s3:
            t0 = time.time()
            index, store, embed_model, client = load_rag_resources()
            lt = time.time() - t0
            st.write(f"Embedding model: `{EMBED_MODEL_NAME}` | LLM provider: `{LLM_PROVIDER}`")
            st.write(f"Generation model: `{GENERATION_MODEL}` | FAISS vectors: {index.ntotal}")
            st.write(f"Load time: {lt:.1f}s")
            s3.update(label=f"Step 3/6: Models loaded ({lt:.1f}s)", state="complete")

        # Step 4
        with st.status("Step 4/6: Baseline RAG...", expanded=True) as s4:
            from src.rag.rag import run_rag
            bq = "What are the major sources of carbon emissions in LLM training?"
            st.write(f"**Query:** {bq}")
            t0 = time.time()
            br = run_rag(bq, index, store, embed_model, client, top_k=TOP_K, mode="baseline")
            et = time.time() - t0
            st.write(f"Retrieved {len(br['retrieved_chunks'])} chunks in {et:.1f}s")
            st.write(f"**Answer** (preview): {br['answer'][:400]}...")
            cv = br["citation_validation"]
            st.write(f"Citations: {cv['valid_citations']}/{cv['total_citations']} valid "
                      f"(precision={cv.get('citation_precision', 'N/A')})")
            s4.update(label=f"Step 4/6: Baseline RAG ({et:.1f}s, {cv['valid_citations']}/{cv['total_citations']} cites)", state="complete")

        # Step 5
        with st.status("Step 5/6: Enhanced RAG...", expanded=True) as s5:
            from src.rag.enhance_query_rewriting import run_enhanced_rag
            eq = "Compare Strubell et al. and Patterson et al. on measurement methodology."
            st.write(f"**Query:** {eq}")
            t0 = time.time()
            er = run_enhanced_rag(eq, index, store, embed_model, client)
            et = time.time() - t0
            st.write(f"Query type: {er.get('query_type')} | Rewritten: {er.get('rewritten_query', 'N/A')}")
            sqs = er.get("sub_queries", [])
            if sqs:
                st.write(f"Sub-queries: {len(sqs)}")
                for i, sq in enumerate(sqs, 1):
                    st.text(f"  {i}. {sq}")
            st.write(f"Merged chunks: {len(er['retrieved_chunks'])}")
            st.write(f"**Synthesis** (preview): {er['answer'][:400]}...")
            ecv = er["citation_validation"]
            st.write(f"Citations: {ecv['valid_citations']}/{ecv['total_citations']} valid")
            s5.update(label=f"Step 5/6: Enhanced RAG ({et:.1f}s, {len(sqs)} sub-queries)", state="complete")

        # Step 6
        with st.status("Step 6/6: Summary...", expanded=True) as s6:
            st.write("**Pipeline executed successfully.**")
            st.write(f"- Corpus: {len(df)} sources loaded and verified")
            st.write(f"- Ingestion: {len(store_data)} chunks indexed in FAISS")
            st.write(f"- Baseline RAG: top-{TOP_K} retrieval + {GENERATION_MODEL} generation")
            st.write(f"- Enhanced RAG: rewriting + decomposition + merged retrieval")
            st.write(f"- Trust: citation validation, uncertainty flagging active")
            st.write(f"- Provider: {LLM_PROVIDER} ({GENERATION_MODEL})")
            st.write("")
            st.write("Full evaluation: `make eval-both` | Report: `make report`")
            s6.update(label="Step 6/6: Done", state="complete")

        st.success("Pipeline complete.")


# ══════════════════════════════════════════════════════════════════════════
# PAGE: ASK A QUESTION
# ══════════════════════════════════════════════════════════════════════════
elif page == "Ask a Question":
    st.header("Ask a Research Question")
    st.markdown(
        f"Queries the **{LLM_PROVIDER}** ({GENERATION_MODEL}) RAG pipeline. "
        "Input is sanitized for prompt-injection protection."
    )

    col1, col2 = st.columns(2)
    with col1:
        mode = st.selectbox("Pipeline mode", ["baseline", "enhanced"])
    with col2:
        top_k = st.slider("Top-K", 1, 20, TOP_K)

    sample = st.selectbox("Sample questions:", ["(type your own below)"] + SAMPLE_QUESTIONS)

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                st.markdown(msg["answer"])
                cv = msg["citation_validation"]
                st.caption(
                    f"Citations: {cv['valid_citations']}/{cv['total_citations']} | "
                    f"Precision: {cv.get('citation_precision', 'N/A')} | "
                    f"Mode: {msg.get('mode', 'N/A')}"
                )

    pending = None
    if sample != "(type your own below)":
        if st.button("Use this sample question"):
            pending = sample

    user_input = st.chat_input("Ask about LLM carbon footprints...")
    query = pending or user_input

    if query:
        # Sanitize
        try:
            query = sanitize_query(query)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        st.session_state["chat_history"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.status("Running RAG pipeline...", expanded=True) as ps:
                st.write(f"Provider: {LLM_PROVIDER} | Mode: {mode} | Top-K: {top_k}")
                index, store, embed_model, client = load_rag_resources()

                from src.rag.rag import run_rag
                from src.rag.enhance_query_rewriting import run_enhanced_rag

                t0 = time.time()
                if mode == "enhanced":
                    result = run_enhanced_rag(query, index, store, embed_model, client)
                else:
                    result = run_rag(query, index, store, embed_model, client,
                                     top_k=top_k, mode=mode)
                elapsed = time.time() - t0
                ps.update(label=f"Pipeline complete ({elapsed:.1f}s)", state="complete")

            st.markdown(result["answer"])

            cv = result["citation_validation"]
            st.caption(
                f"Citations: {cv['valid_citations']}/{cv['total_citations']} valid | "
                f"Precision: {cv.get('citation_precision', 'N/A')} | "
                f"Time: {elapsed:.1f}s"
            )

            with st.expander(f"{len(result['retrieved_chunks'])} retrieved chunks"):
                for j, c in enumerate(result["retrieved_chunks"], 1):
                    st.text(
                        f"  {j}. [{c['source_id']}, {c['chunk_id']}] "
                        f"score={c['retrieval_score']:.4f} -- {c['title']} ({c['year']})"
                    )

        st.session_state["chat_history"].append({
            "role": "assistant",
            "answer": result["answer"],
            "retrieved_chunks": result["retrieved_chunks"],
            "citation_validation": cv,
            "tokens": result["tokens"],
            "mode": mode,
        })


# ══════════════════════════════════════════════════════════════════════════
# PAGE: COMPARE MODELS
# ══════════════════════════════════════════════════════════════════════════
elif page == "Compare Models":
    st.header("Model Comparison: Gemini vs Azure OpenAI")
    st.markdown(
        "Run the **same query** through both LLM providers on the same corpus "
        "and compare answer quality, latency, citations, and token usage side-by-side."
    )

    if not gemini_ok:
        st.warning("Gemini API key not set. Add `GEMINI_API_KEY` to `.env`.")
    if not azure_ok:
        st.warning("Azure OpenAI not configured. Add `AZURE_ENDPOINT` and `AZURE_API_KEY` to `.env`.")

    comp_mode = st.selectbox("Pipeline mode", ["baseline", "enhanced"], key="comp_mode")
    comp_query = st.selectbox("Pick a query:", SAMPLE_QUESTIONS, key="comp_query")
    custom = st.text_input("Or type a custom query:", key="comp_custom")
    query_to_run = custom.strip() if custom.strip() else comp_query

    if st.button("Run Comparison", type="primary", disabled=(not gemini_ok or not azure_ok)):
        try:
            query_to_run = sanitize_query(query_to_run)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        # Load shared resources (index, store, embeddings -- these are provider-independent)
        from sentence_transformers import SentenceTransformer
        from src.rag.rag import load_index, run_rag
        from src.rag.enhance_query_rewriting import run_enhanced_rag
        from src.llm_client import GeminiClient, AzureOpenAIClient

        index, store = load_index()
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)

        st.markdown(f"**Query:** {query_to_run}")
        st.markdown(f"**Mode:** {comp_mode}")
        st.markdown("---")

        col_g, col_a = st.columns(2)

        # ── Gemini ────────────────────────────────────────────────────────
        with col_g:
            st.subheader("Gemini")
            with st.status("Running...", expanded=True) as sg:
                try:
                    gc = GeminiClient()
                    t0 = time.time()
                    if comp_mode == "enhanced":
                        gr = run_enhanced_rag(query_to_run, index, store, embed_model, gc)
                    else:
                        gr = run_rag(query_to_run, index, store, embed_model, gc, mode="baseline")
                    g_time = time.time() - t0
                    sg.update(label=f"Gemini ({g_time:.1f}s)", state="complete")
                except Exception as exc:
                    st.error(f"Gemini failed: {exc}")
                    gr = None
                    g_time = None
                    sg.update(label="Gemini: failed", state="error")

            if gr:
                st.markdown(gr["answer"][:600] + ("..." if len(gr["answer"]) > 600 else ""))
                gcv = gr["citation_validation"]
                st.metric("Latency", f"{g_time:.1f}s")
                st.metric("Citations (valid/total)", f"{gcv['valid_citations']}/{gcv['total_citations']}")
                st.metric("Citation Precision",
                          f"{gcv['citation_precision']:.2f}" if gcv.get("citation_precision") is not None else "N/A")
                st.metric("Tokens (in/out)", f"{gr['tokens'].get('input', 0)} / {gr['tokens'].get('output', 0)}")
                st.metric("Chunks Retrieved", str(len(gr["retrieved_chunks"])))

        # ── Azure OpenAI ──────────────────────────────────────────────────
        with col_a:
            st.subheader("Azure OpenAI (o4-mini)")
            with st.status("Running...", expanded=True) as sa:
                try:
                    ac = AzureOpenAIClient()
                    t0 = time.time()
                    if comp_mode == "enhanced":
                        ar = run_enhanced_rag(query_to_run, index, store, embed_model, ac)
                    else:
                        ar = run_rag(query_to_run, index, store, embed_model, ac, mode="baseline")
                    a_time = time.time() - t0
                    sa.update(label=f"Azure OpenAI ({a_time:.1f}s)", state="complete")
                except Exception as exc:
                    st.error(f"Azure OpenAI failed: {exc}")
                    ar = None
                    a_time = None
                    sa.update(label="Azure: failed", state="error")

            if ar:
                st.markdown(ar["answer"][:600] + ("..." if len(ar["answer"]) > 600 else ""))
                acv = ar["citation_validation"]
                st.metric("Latency", f"{a_time:.1f}s")
                st.metric("Citations (valid/total)", f"{acv['valid_citations']}/{acv['total_citations']}")
                st.metric("Citation Precision",
                          f"{acv['citation_precision']:.2f}" if acv.get("citation_precision") is not None else "N/A")
                st.metric("Tokens (in/out)", f"{ar['tokens'].get('input', 0)} / {ar['tokens'].get('output', 0)}")
                st.metric("Chunks Retrieved", str(len(ar["retrieved_chunks"])))

        # ── Summary table ─────────────────────────────────────────────────
        if gr and ar:
            st.markdown("---")
            st.subheader("Comparison Summary")
            gcv = gr["citation_validation"]
            acv = ar["citation_validation"]
            comp_data = pd.DataFrame([
                {
                    "Metric": "Latency (s)",
                    "Gemini": f"{g_time:.1f}",
                    "Azure OpenAI": f"{a_time:.1f}",
                    "Winner": "Gemini" if g_time < a_time else "Azure",
                },
                {
                    "Metric": "Answer Length (chars)",
                    "Gemini": str(len(gr["answer"])),
                    "Azure OpenAI": str(len(ar["answer"])),
                    "Winner": "--",
                },
                {
                    "Metric": "Valid Citations",
                    "Gemini": str(gcv["valid_citations"]),
                    "Azure OpenAI": str(acv["valid_citations"]),
                    "Winner": "Gemini" if gcv["valid_citations"] > acv["valid_citations"] else
                              "Azure" if acv["valid_citations"] > gcv["valid_citations"] else "Tie",
                },
                {
                    "Metric": "Citation Precision",
                    "Gemini": f"{gcv['citation_precision']:.2f}" if gcv.get("citation_precision") is not None else "N/A",
                    "Azure OpenAI": f"{acv['citation_precision']:.2f}" if acv.get("citation_precision") is not None else "N/A",
                    "Winner": "--",
                },
                {
                    "Metric": "Input Tokens",
                    "Gemini": str(gr["tokens"].get("input", 0)),
                    "Azure OpenAI": str(ar["tokens"].get("input", 0)),
                    "Winner": "--",
                },
                {
                    "Metric": "Output Tokens",
                    "Gemini": str(gr["tokens"].get("output", 0)),
                    "Azure OpenAI": str(ar["tokens"].get("output", 0)),
                    "Winner": "--",
                },
            ])
            st.dataframe(comp_data, width="stretch", hide_index=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE: EVALUATION
# ══════════════════════════════════════════════════════════════════════════
elif page == "Evaluation":
    st.header("Evaluation")

    st.subheader("Metrics (6)")
    st.dataframe(pd.DataFrame([
        {"Metric": "Groundedness", "Type": "LLM-judge (1-4)", "Description": "Are claims supported by retrieved chunks?"},
        {"Metric": "Answer Relevance", "Type": "LLM-judge (1-4)", "Description": "Does the answer address the query?"},
        {"Metric": "Context Precision", "Type": "LLM-judge (1-4)", "Description": "Are retrieved chunks relevant?"},
        {"Metric": "Citation Precision", "Type": "Deterministic", "Description": "valid citations / total citations"},
        {"Metric": "Source Recall", "Type": "Deterministic", "Description": "expected sources found / total expected"},
        {"Metric": "Uncertainty Handling", "Type": "Rule-based", "Description": "Does answer flag missing evidence?"},
    ]), width="stretch", hide_index=True)

    st.subheader("20-Query Test Set")
    from src.eval.evaluation import EVAL_QUERIES
    q_rows = [{"ID": q["id"], "Type": q["type"], "Query": q["query"],
               "Expected Sources": ", ".join(q["expected_sources"]) or "(none)"}
              for q in EVAL_QUERIES]
    st.dataframe(pd.DataFrame(q_rows), width="stretch", hide_index=True)

    st.subheader("Results")
    baseline_results = load_eval_results("baseline")
    enhanced_results = load_eval_results("enhanced")

    if not baseline_results and not enhanced_results:
        st.info("No evaluation results found. Run: `make eval-both`")
    else:
        for label, results in [("Baseline", baseline_results), ("Enhanced", enhanced_results)]:
            if not results:
                continue
            st.markdown(f"**{label} Pipeline** ({len(results)} queries)")
            rows = []
            for r in results:
                rows.append({
                    "ID": r["query_id"],
                    "Type": r["query_type"],
                    "Ground.": r["groundedness"].get("score"),
                    "Relev.": r["answer_relevance"].get("score"),
                    "Ctx P.": r.get("context_precision", {}).get("score"),
                    "Cite P.": round(r["citation_precision"], 2) if r.get("citation_precision") is not None else None,
                    "Src Rec.": round(r["source_recall"], 2) if r.get("source_recall") is not None else None,
                    "Missing?": "yes" if r["uncertainty"]["flags_missing_evidence"] else "--",
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

            ground = [r["groundedness"].get("score") for r in results if r["groundedness"].get("score")]
            rel = [r["answer_relevance"].get("score") for r in results if r["answer_relevance"].get("score")]
            cite = [r["citation_precision"] for r in results if r.get("citation_precision") is not None]

            m1, m2, m3 = st.columns(3)
            m1.metric("Avg Groundedness", f"{safe_avg(ground)}/4" if ground else "--")
            m2.metric("Avg Relevance", f"{safe_avg(rel)}/4" if rel else "--")
            m3.metric("Avg Cite Precision", f"{safe_avg(cite):.2f}" if cite else "--")
            st.markdown("---")

        if baseline_results and enhanced_results:
            st.subheader("Baseline vs Enhanced")
            def _avgs(res):
                return {
                    "Groundedness": safe_avg([r["groundedness"].get("score") for r in res]),
                    "Relevance": safe_avg([r["answer_relevance"].get("score") for r in res]),
                    "Ctx Precision": safe_avg([r.get("context_precision", {}).get("score") for r in res]),
                    "Cite Precision": safe_avg([r.get("citation_precision") for r in res]),
                    "Source Recall": safe_avg([r.get("source_recall") for r in res]),
                }
            ba, ea = _avgs(baseline_results), _avgs(enhanced_results)
            comp = []
            for m in ba:
                bv, ev = ba[m], ea[m]
                delta = round(ev - bv, 3) if bv is not None and ev is not None else None
                comp.append({"Metric": m, "Baseline": bv, "Enhanced": ev, "Delta": delta})
            st.dataframe(pd.DataFrame(comp), width="stretch", hide_index=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE: DELIVERABLES
# ══════════════════════════════════════════════════════════════════════════
elif page == "Deliverables":
    st.header("Phase 2 Deliverables")

    def _check(path: Path) -> str:
        return "OK" if path.exists() else "MISSING"

    st.subheader("D1: Code Repository")
    for item, path in [
        ("src/config.py",                   PROJECT_ROOT / "src" / "config.py"),
        ("src/llm_client.py",               PROJECT_ROOT / "src" / "llm_client.py"),
        ("src/utils.py",                    PROJECT_ROOT / "src" / "utils.py"),
        ("src/ingest/",                     PROJECT_ROOT / "src" / "ingest" / "ingest.py"),
        ("src/rag/",                        PROJECT_ROOT / "src" / "rag" / "rag.py"),
        ("src/eval/",                       PROJECT_ROOT / "src" / "eval" / "evaluation.py"),
        ("src/app/",                        PROJECT_ROOT / "src" / "app" / "app.py"),
        ("Makefile",                        PROJECT_ROOT / "Makefile"),
        ("requirements.txt",               PROJECT_ROOT / "requirements.txt"),
        (".env.example",                    PROJECT_ROOT / ".env.example"),
    ]:
        st.text(f"  [{_check(path)}] {item}")

    st.subheader("D2: Data Manifest")
    df = load_manifest()
    if not df.empty:
        st.write(f"{len(df)} sources in data/data_manifest.csv")
        pdf_count = len(list((PROJECT_ROOT / "data" / "raw").glob("*.pdf"))) if (PROJECT_ROOT / "data" / "raw").exists() else 0
        st.write(f"Raw PDFs on disk: {pdf_count}")

    st.subheader("D3: RAG Pipeline")
    st.text(f"  [{_check(PROCESSED_DIR / 'faiss_index.bin')}] FAISS index")
    st.text(f"  [{_check(PROCESSED_DIR / 'chunk_store.json')}] Chunk store")
    st.text(f"  [{_check(PROJECT_ROOT / 'src' / 'rag' / 'rag.py')}] Baseline RAG")
    st.text(f"  [{_check(PROJECT_ROOT / 'src' / 'rag' / 'enhance_query_rewriting.py')}] Enhanced RAG")

    st.subheader("D4: Evaluation Framework")
    st.write("20-query test set: 10 direct, 5 synthesis, 5 edge-case.")
    st.write("6 metrics: groundedness, relevance, context precision (LLM-judge), "
             "citation precision, source recall (deterministic), uncertainty (rule-based).")

    st.subheader("D5: Evaluation Report")
    report_path = REPORT_DIR / "evaluation_report.md"
    if report_path.exists():
        with st.expander("View report"):
            st.markdown(report_path.read_text(encoding="utf-8"))
    else:
        st.warning("Report not generated. Run: `make report`")

    st.subheader("D6: API Backend")
    st.dataframe(pd.DataFrame([
        {"Method": "GET", "Endpoint": "/health", "Description": "Server health + provider info"},
        {"Method": "POST", "Endpoint": "/query", "Description": "Run RAG query (sanitized)"},
        {"Method": "GET", "Endpoint": "/corpus", "Description": "Corpus manifest"},
        {"Method": "GET", "Endpoint": "/evaluation", "Description": "Eval results + metrics"},
        {"Method": "GET", "Endpoint": "/logs", "Description": "Run logs"},
    ]), width="stretch", hide_index=True)

    st.subheader("D7-D8: UI + Model Comparison")
    st.write("This Streamlit app. Pages: Overview, Run Pipeline, Ask a Question, "
             "Compare Models, Evaluation, Deliverables.")

    st.subheader("D9: Security")
    st.markdown("""
- **API key isolation**: Keys stored in `.env` (git-ignored), never in source code.
- **Prompt-injection protection**: `src/utils.sanitize_query()` strips control characters,
  enforces length limits, and rejects injection patterns before any query enters the LLM.
- **Input validation**: FastAPI Pydantic schemas validate all API request fields.
- **PDF exclusion**: Downloaded PDFs are git-ignored; re-downloaded at runtime.
    """)

    st.subheader("AI Usage Disclosure")
    st.dataframe(pd.DataFrame([
        {"Tool": "Gemini (gemini-3-flash-preview)", "Purpose": "RAG generation + eval judging", "Review": "Prompt engineering, guardrails"},
        {"Tool": "Azure OpenAI (o4-mini)", "Purpose": "RAG generation + comparison", "Review": "Same prompts, side-by-side evaluation"},
        {"Tool": "Cursor AI", "Purpose": "Code scaffolding", "Review": "Full code review and testing"},
        {"Tool": "sentence-transformers", "Purpose": "Embeddings (all-MiniLM-L6-v2)", "Review": "Configuration only"},
    ]), width="stretch", hide_index=True)
