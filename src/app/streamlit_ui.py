"""Streamlit UI for the LLM Carbon Footprint Research Portal."""

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
import plotly.express as px
import plotly.graph_objects as go

from src.config import (
    PROJECT_ROOT, DATA_DIR, PROCESSED_DIR, LOGS_DIR, OUTPUTS_DIR,
    MANIFEST_PATH, GENERATION_MODEL, JUDGE_MODEL, EMBED_MODEL_NAME,
    CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS, TOP_K, ENHANCED_TOP_N,
    REPORT_DIR,
)

# ── Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Carbon Footprint Research Portal",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.4rem; font-weight: 700; color: #1B5E20; margin-bottom: 0; }
    .sub-header  { font-size: 1.05rem; color: #555; margin-bottom: 1.2rem; }
    .metric-card {
        background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
        border-radius: 12px; padding: 1.2rem; text-align: center;
        border: 1px solid #A5D6A7;
    }
    .metric-card h3 { margin: 0; font-size: 1.8rem; color: #2E7D32; }
    .metric-card p  { margin: 0.3rem 0 0; font-size: 0.85rem; color: #555; }
    .phase-badge {
        display: inline-block; background: #2E7D32; color: white;
        padding: 0.25rem 0.75rem; border-radius: 20px;
        font-size: 0.8rem; font-weight: 600; margin-right: 0.5rem;
    }
    .del-card {
        background: #FAFAFA; border-left: 4px solid #2E7D32;
        padding: 1rem 1.2rem; margin-bottom: 0.8rem; border-radius: 0 8px 8px 0;
    }
    .del-card h4 { margin: 0 0 0.3rem; color: #1B5E20; }
    .del-card p  { margin: 0; color: #666; font-size: 0.9rem; }
    .del-status-ok  { color: #2E7D32; font-weight: 600; }
    .del-status-miss { color: #E65100; font-weight: 600; }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #E8F5E9 0%, #FFFFFF 100%);
    }
    .sample-q {
        background: #F1F8E9; border: 1px solid #C5E1A5; border-radius: 8px;
        padding: 0.6rem 1rem; margin-bottom: 0.4rem; cursor: pointer;
        font-size: 0.9rem; color: #33691E;
    }
    .chat-sources {
        background: #F5F5F5; border-radius: 8px; padding: 0.8rem;
        margin-top: 0.5rem; font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def load_manifest() -> pd.DataFrame:
    if MANIFEST_PATH.exists():
        return pd.read_csv(MANIFEST_PATH)
    return pd.DataFrame()


@st.cache_resource
def load_rag_resources():
    from sentence_transformers import SentenceTransformer
    from src.rag.rag import load_index, _get_client
    index, store = load_index()
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    client = _get_client()
    return index, store, embed_model, client


def load_eval_results(mode: str) -> list[dict]:
    files = sorted(glob.glob(str(OUTPUTS_DIR / f"eval_results_{mode}_*.json")))
    if not files:
        return []
    return json.loads(Path(files[-1]).read_text(encoding="utf-8"))


def load_logs() -> list[dict]:
    path = LOGS_DIR / "rag_runs.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    return [json.loads(line) for line in lines if line.strip()]


def safe_avg(vals):
    clean = [v for v in vals if v is not None]
    return round(sum(clean) / len(clean), 2) if clean else None


def _file_exists(p: Path) -> bool:
    return p.exists()


def _dir_count(p: Path, pattern: str = "*") -> int:
    if not p.exists():
        return 0
    return len(list(p.glob(pattern)))


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


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🌍 Navigation")
    page = st.radio(
        "Go to",
        [
            "🏠 Home",
            "💬 Ask a Question",
            "📚 Corpus Explorer",
            "📊 Evaluation Dashboard",
            "📋 Phase 2 Deliverables",
            "📝 Run Logs",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### ⚙️ System Config")
    st.caption(f"**Generation**: `{GENERATION_MODEL}`")
    st.caption(f"**Judge**: `{JUDGE_MODEL}`")
    st.caption(f"**Embeddings**: `{EMBED_MODEL_NAME}`")
    st.caption(f"**Chunk**: {CHUNK_SIZE_TOKENS}t / {CHUNK_OVERLAP_TOKENS}t overlap")
    st.caption(f"**Top-K**: {TOP_K} baseline / {ENHANCED_TOP_N} enhanced")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#888; font-size:0.75rem;'>"
        "AI Model Development (95-864)<br>"
        "Group 4 — Dhiksha Rathis, Shreya Verma<br>"
        "CMU • Spring 2026"
        "</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown('<div class="main-header">🌍 LLM Carbon Footprint Research Portal</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'A research-grade RAG system for systematic review of carbon emissions in Large Language Models'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Research Question ────────────────────────────────────────────────
    st.markdown("### 🔬 Research Question")
    st.info(
        "**How do we accurately measure and compare the carbon footprint of different LLMs "
        "across their lifecycle?**"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Sub-questions")
        st.markdown("""
1. What are the major sources of emissions in LLM training vs. inference?
2. How do different studies measure and report carbon metrics?
3. What factors (model size, hardware, location) most impact carbon footprint?
4. How do carbon estimates vary across different LLM families?
5. What data is missing or inconsistent in current carbon reporting?
        """)
    with col2:
        st.markdown("#### Corpus at a Glance")
        df = load_manifest()
        if not df.empty:
            c1, c2 = st.columns(2)
            c1.metric("Total Sources", len(df))
            c2.metric("Year Range", f"{int(df['year'].min())}–{int(df['year'].max())}")
            c3, c4 = st.columns(2)
            c3.metric("Peer-reviewed", len(df[df["source_type"] == "peer-reviewed paper"]))
            c4.metric("Technical Reports", len(df[df["source_type"] == "technical report"]))

    st.markdown("---")

    # ── Pipeline Architecture ────────────────────────────────────────────
    st.markdown("### 🏗️ Pipeline Architecture")
    cols = st.columns(4)
    steps = [
        ("📄 Ingest", "PDF → chunk → embed → FAISS index"),
        ("🔍 Retrieve", "Semantic search (cosine similarity)"),
        ("🤖 Generate", "Gemini + citation-constrained prompt"),
        ("📏 Evaluate", "6 metrics • LLM-judge + deterministic"),
    ]
    for col, (title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f'<div class="metric-card"><h3>{title}</h3><p>{desc}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Quick start ──────────────────────────────────────────────────────
    st.markdown("### 🚀 Quick Start")
    st.markdown(
        "Use the **sidebar** to navigate. Head to **💬 Ask a Question** to query the corpus interactively, "
        "or browse **📊 Evaluation Dashboard** to review system performance. "
        "All **📋 Phase 2 Deliverables** are accessible from their dedicated page."
    )

    st.markdown("---")

    # ── Metrics overview ─────────────────────────────────────────────────
    st.markdown("### 📏 Evaluation Metrics (6)")
    metrics_data = pd.DataFrame([
        {"Metric": "Groundedness", "Type": "LLM-judge (1–4)", "Description": "Are claims supported by retrieved chunks?"},
        {"Metric": "Answer Relevance", "Type": "LLM-judge (1–4)", "Description": "Does the answer address the query?"},
        {"Metric": "Context Precision", "Type": "LLM-judge (1–4)", "Description": "Are retrieved chunks relevant to the answer?"},
        {"Metric": "Citation Precision", "Type": "Deterministic", "Description": "valid citations / total citations"},
        {"Metric": "Source Recall", "Type": "Deterministic", "Description": "expected sources found / total expected"},
        {"Metric": "Uncertainty Handling", "Type": "Rule-based", "Description": "Does answer flag missing evidence?"},
    ])
    st.dataframe(metrics_data, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE: ASK A QUESTION (CHAT)
# ══════════════════════════════════════════════════════════════════════════
elif page == "💬 Ask a Question":
    st.markdown("## 💬 Ask a Research Question")
    st.markdown(
        "Type a question about LLM carbon footprints and the RAG pipeline will retrieve "
        "relevant evidence from the 15-source corpus, generate an answer with inline citations, "
        "and validate every citation against the retrieved chunks."
    )

    # ── Mode & config ────────────────────────────────────────────────────
    with st.expander("⚙️ Pipeline Settings", expanded=False):
        cfg_col1, cfg_col2 = st.columns(2)
        with cfg_col1:
            mode = st.selectbox(
                "Pipeline Mode",
                ["baseline", "enhanced"],
                help=(
                    "**Baseline**: Top-K semantic retrieval → Gemini generation.\n\n"
                    "**Enhanced**: Query rewriting + decomposition for synthesis/multi-hop "
                    "questions, merged deduplication, broader retrieval."
                ),
            )
        with cfg_col2:
            top_k = st.slider("Top-K chunks to retrieve", 1, 20, TOP_K)

    # ── Sample questions ─────────────────────────────────────────────────
    st.markdown("#### 💡 Sample Questions (click to use)")
    sample_cols = st.columns(2)
    for i, q in enumerate(SAMPLE_QUESTIONS):
        col = sample_cols[i % 2]
        if col.button(q, key=f"sample_{i}", use_container_width=True):
            st.session_state["pending_query"] = q

    st.markdown("---")

    # ── Chat history ─────────────────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Display past conversations
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                st.markdown(msg["answer"])
                # Show metrics bar
                cv = msg["citation_validation"]
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Citations", f"{cv['valid_citations']}/{cv['total_citations']}")
                prec = cv.get("citation_precision")
                mc2.metric("Cite Precision", f"{prec:.0%}" if prec is not None else "N/A")
                mc3.metric("Input Tokens", msg["tokens"].get("input", 0))
                mc4.metric("Output Tokens", msg["tokens"].get("output", 0))
                # Sources
                with st.expander(f"📎 {len(msg['retrieved_chunks'])} Retrieved Chunks"):
                    for j, c in enumerate(msg["retrieved_chunks"], 1):
                        st.markdown(
                            f"**{j}. [{c['source_id']}, {c['chunk_id']}]** "
                            f"score={c['retrieval_score']:.4f} — {c['title']} ({c['year']})"
                        )
                        st.caption(c.get("chunk_text_preview", c.get("chunk_text", ""))[:300])
                if cv.get("invalid_list"):
                    st.warning(f"⚠️ {cv['invalid_citations']} invalid citation(s): {cv['invalid_list']}")

    # ── Input ────────────────────────────────────────────────────────────
    pending = st.session_state.pop("pending_query", None)
    user_input = st.chat_input("Ask about LLM carbon footprints…")
    query = pending or user_input

    if query:
        # Show user message
        st.session_state["chat_history"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Run RAG
        with st.chat_message("assistant"):
            with st.spinner("Loading models…"):
                index, store, embed_model, client = load_rag_resources()

            from src.rag.rag import run_rag
            from src.rag.enhance_query_rewriting import run_enhanced_rag

            with st.spinner(f"Running **{mode}** RAG pipeline…"):
                t0 = time.time()
                if mode == "enhanced":
                    result = run_enhanced_rag(query, index, store, embed_model, client)
                else:
                    result = run_rag(query, index, store, embed_model, client, top_k=top_k, mode=mode)
                elapsed = time.time() - t0

            # Answer
            st.markdown(result["answer"])

            # Metrics
            cv = result["citation_validation"]
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Citations", f"{cv['valid_citations']}/{cv['total_citations']}")
            prec = cv.get("citation_precision")
            mc2.metric("Cite Precision", f"{prec:.0%}" if prec is not None else "N/A")
            mc3.metric("Input Tokens", result["tokens"].get("input", 0))
            mc4.metric("Output Tokens", result["tokens"].get("output", 0))

            # Retrieved chunks
            with st.expander(f"📎 {len(result['retrieved_chunks'])} Retrieved Chunks"):
                for j, c in enumerate(result["retrieved_chunks"], 1):
                    st.markdown(
                        f"**{j}. [{c['source_id']}, {c['chunk_id']}]** "
                        f"score={c['retrieval_score']:.4f} — {c['title']} ({c['year']})"
                    )
                    st.caption(c.get("chunk_text_preview", c.get("chunk_text", ""))[:300])

            if cv.get("invalid_list"):
                st.warning(f"⚠️ {cv['invalid_citations']} invalid citation(s): {cv['invalid_list']}")

            st.caption(f"⏱ {elapsed:.1f}s | mode={mode} | top_k={top_k}")

        # Save to history
        st.session_state["chat_history"].append({
            "role": "assistant",
            "answer": result["answer"],
            "retrieved_chunks": result["retrieved_chunks"],
            "citation_validation": cv,
            "tokens": result["tokens"],
        })


# ══════════════════════════════════════════════════════════════════════════
# PAGE: CORPUS EXPLORER
# ══════════════════════════════════════════════════════════════════════════
elif page == "📚 Corpus Explorer":
    st.markdown("## 📚 Corpus Explorer")
    st.markdown("Browse the 15-source research corpus used for retrieval-augmented generation.")

    df = load_manifest()

    if df.empty:
        st.warning("No data manifest found. Run `make download` first.")
    else:
        # Filters
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            types = ["All"] + sorted(df["source_type"].dropna().unique().tolist())
            sel_type = st.selectbox("Source Type", types)
        with col_f2:
            years = ["All"] + sorted(df["year"].dropna().astype(str).unique().tolist())
            sel_year = st.selectbox("Year", years)
        with col_f3:
            search = st.text_input("Search (title, tags)", "")

        filtered = df.copy()
        if sel_type != "All":
            filtered = filtered[filtered["source_type"] == sel_type]
        if sel_year != "All":
            filtered = filtered[filtered["year"].astype(str) == sel_year]
        if search:
            mask = (
                filtered["title"].str.contains(search, case=False, na=False)
                | filtered["tags"].str.contains(search, case=False, na=False)
            )
            filtered = filtered[mask]

        # Summary
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Showing", len(filtered))
        c2.metric("Peer-reviewed", len(filtered[filtered["source_type"] == "peer-reviewed paper"]))
        c3.metric("Reports", len(filtered[filtered["source_type"] == "technical report"]))
        c4.metric("Tool Papers", len(filtered[filtered["source_type"].str.contains("tool", case=False, na=False)]))

        st.markdown("---")

        # Source cards
        for _, row in filtered.iterrows():
            with st.expander(f"**{row['source_id']}** — {row['title']} ({row['year']})"):
                ca, cb = st.columns([2, 1])
                with ca:
                    st.markdown(f"**Authors**: {row['authors']}")
                    st.markdown(f"**Venue**: {row['venue']}")
                    st.markdown(f"**Type**: {row['source_type']}")
                    st.markdown(f"**Relevance**: {row.get('relevance_note', '')}")
                with cb:
                    tags = row.get("tags", "")
                    if tags:
                        for tag in str(tags).split(";"):
                            st.code(tag.strip(), language=None)
                    url = row.get("url_or_doi", "")
                    if url:
                        st.link_button("Open Source ↗", url)

        # Year distribution
        st.markdown("---")
        st.markdown("### Distribution by Year")
        year_counts = df.groupby("year").size().reset_index(name="count")
        fig = px.bar(
            year_counts, x="year", y="count",
            color_discrete_sequence=["#2E7D32"],
            labels={"year": "Publication Year", "count": "Sources"},
        )
        fig.update_layout(xaxis_dtick=1, bargap=0.3, height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Type distribution
        st.markdown("### Distribution by Type")
        type_counts = df.groupby("source_type").size().reset_index(name="count")
        fig2 = px.pie(
            type_counts, values="count", names="source_type",
            color_discrete_sequence=["#2E7D32", "#66BB6A", "#A5D6A7", "#C8E6C9"],
            hole=0.4,
        )
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE: EVALUATION DASHBOARD
# ══════════════════════════════════════════════════════════════════════════
elif page == "📊 Evaluation Dashboard":
    st.markdown("## 📊 Evaluation Dashboard")
    st.markdown("Review evaluation results across all 6 metrics for both pipeline modes.")

    baseline_results = load_eval_results("baseline")
    enhanced_results = load_eval_results("enhanced")

    if not baseline_results and not enhanced_results:
        st.warning(
            "No evaluation results found yet. Run the evaluation first:"
        )
        st.code("python3 -m src.eval.evaluation --mode both", language="bash")
        st.markdown("Or from the Makefile:")
        st.code("make eval-both", language="bash")

        # Still show the 20-query set
        st.markdown("---")
        st.markdown("### 📝 20-Query Evaluation Set")
        from src.eval.evaluation import EVAL_QUERIES
        q_rows = []
        for q in EVAL_QUERIES:
            q_rows.append({
                "ID": q["id"],
                "Type": q["type"],
                "Query": q["query"],
                "Expected Sources": ", ".join(q["expected_sources"]) if q["expected_sources"] else "—",
            })
        st.dataframe(pd.DataFrame(q_rows), use_container_width=True, hide_index=True)
    else:
        tabs = st.tabs(["Overview", "Per-Query Detail", "Comparison", "Query Set"])

        # TAB: Overview
        with tabs[0]:
            for label, results in [("Baseline", baseline_results), ("Enhanced", enhanced_results)]:
                if not results:
                    continue
                st.markdown(f"### {label} Pipeline")

                ground = [r["groundedness"].get("score") for r in results if r["groundedness"].get("score")]
                rel = [r["answer_relevance"].get("score") for r in results if r["answer_relevance"].get("score")]
                ctx = [r.get("context_precision", {}).get("score") for r in results if r.get("context_precision", {}).get("score")]
                cite = [r["citation_precision"] for r in results if r.get("citation_precision") is not None]
                recall = [r["source_recall"] for r in results if r.get("source_recall") is not None]
                missing = sum(1 for r in results if r["uncertainty"]["flags_missing_evidence"])

                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Groundedness", f"{safe_avg(ground)}/4" if ground else "—")
                m2.metric("Relevance", f"{safe_avg(rel)}/4" if rel else "—")
                m3.metric("Ctx Precision", f"{safe_avg(ctx)}/4" if ctx else "—")
                m4.metric("Cite Precision", f"{safe_avg(cite):.0%}" if cite else "—")
                m5.metric("Source Recall", f"{safe_avg(recall):.0%}" if recall else "—")
                m6.metric("Flags Missing", f"{missing}/{len(results)}")

                # Radar
                cats = ["Groundedness", "Relevance", "Ctx Prec.", "Cite Prec.", "Src Recall"]
                vals = [
                    safe_avg(ground) or 0,
                    safe_avg(rel) or 0,
                    safe_avg(ctx) or 0,
                    (safe_avg(cite) or 0) * 4,
                    (safe_avg(recall) or 0) * 4,
                ]
                fig = go.Figure(data=go.Scatterpolar(
                    r=vals + [vals[0]], theta=cats + [cats[0]],
                    fill="toself", fillcolor="rgba(46,125,50,0.15)", line_color="#2E7D32",
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 4])),
                    showlegend=False, height=350, margin=dict(t=30, b=30),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("---")

        # TAB: Per-Query Detail
        with tabs[1]:
            available = [m for m, r in [("baseline", baseline_results), ("enhanced", enhanced_results)] if r]
            sel_mode = st.selectbox("Mode", available) if available else None
            results = baseline_results if sel_mode == "baseline" else enhanced_results

            if results:
                rows = []
                for r in results:
                    rows.append({
                        "ID": r["query_id"],
                        "Type": r["query_type"],
                        "Query": r["query"][:55] + "…" if len(r["query"]) > 55 else r["query"],
                        "Ground.": r["groundedness"].get("score"),
                        "Relev.": r["answer_relevance"].get("score"),
                        "Ctx Prec.": r.get("context_precision", {}).get("score"),
                        "Cite Prec.": round(r["citation_precision"], 2) if r.get("citation_precision") is not None else None,
                        "Src Recall": round(r["source_recall"], 2) if r.get("source_recall") is not None else None,
                        "Missing?": "✓" if r["uncertainty"]["flags_missing_evidence"] else "—",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                st.markdown("### Query Details")
                for r in results:
                    with st.expander(f"{r['query_id']} ({r['query_type']}): {r['query'][:65]}…"):
                        st.markdown(f"**Answer preview**: {r['answer_preview']}")
                        st.markdown(f"**Groundedness**: {r['groundedness'].get('score')}/4 — _{r['groundedness'].get('reasoning', '')}_")
                        st.markdown(f"**Relevance**: {r['answer_relevance'].get('score')}/4 — _{r['answer_relevance'].get('reasoning', '')}_")
                        st.markdown(f"**Context Precision**: {r.get('context_precision', {}).get('score')}/4")
                        st.markdown(f"**Citation Precision**: {r.get('citation_precision')}")
                        st.markdown(f"**Source Recall**: {r.get('source_recall')}")
                        st.markdown(f"**Retrieved**: `{', '.join(r.get('retrieved_sources', []))}`")
                        st.markdown(f"**Expected**: `{', '.join(r.get('expected_sources', []))}`")

        # TAB: Comparison
        with tabs[2]:
            if baseline_results and enhanced_results:
                st.markdown("### Baseline vs. Enhanced")

                def _avgs(res):
                    return {
                        "Groundedness": safe_avg([r["groundedness"].get("score") for r in res]),
                        "Relevance": safe_avg([r["answer_relevance"].get("score") for r in res]),
                        "Ctx Prec.": safe_avg([r.get("context_precision", {}).get("score") for r in res]),
                        "Cite Prec.": safe_avg([r.get("citation_precision") for r in res]),
                        "Src Recall": safe_avg([r.get("source_recall") for r in res]),
                    }

                ba, ea = _avgs(baseline_results), _avgs(enhanced_results)
                comp = []
                for m in ba:
                    bv, ev = ba[m], ea[m]
                    d = round(ev - bv, 3) if bv is not None and ev is not None else None
                    comp.append({"Metric": m, "Baseline": bv, "Enhanced": ev, "Delta": d,
                                 "↑": "✓" if d and d > 0 else ("✗" if d and d < 0 else "—")})
                st.dataframe(pd.DataFrame(comp), use_container_width=True, hide_index=True)

                fig = go.Figure()
                fig.add_trace(go.Bar(name="Baseline", x=list(ba.keys()), y=[ba[m] or 0 for m in ba], marker_color="#81C784"))
                fig.add_trace(go.Bar(name="Enhanced", x=list(ea.keys()), y=[ea[m] or 0 for m in ea], marker_color="#2E7D32"))
                fig.update_layout(barmode="group", height=400, yaxis_title="Score")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### By Query Type")
                for qt in ("direct", "synthesis", "multihop", "edge_case"):
                    bsub = [r for r in baseline_results if r["query_type"] == qt]
                    esub = [r for r in enhanced_results if r["query_type"] == qt]
                    if bsub or esub:
                        st.markdown(f"**{qt.replace('_', ' ').title()}** ({len(bsub)} queries)")
                        qc1, qc2, qc3 = st.columns(3)
                        bg = safe_avg([r["groundedness"].get("score") for r in bsub]) if bsub else None
                        eg = safe_avg([r["groundedness"].get("score") for r in esub]) if esub else None
                        br = safe_avg([r["answer_relevance"].get("score") for r in bsub]) if bsub else None
                        er = safe_avg([r["answer_relevance"].get("score") for r in esub]) if esub else None
                        bc = safe_avg([r.get("citation_precision") for r in bsub]) if bsub else None
                        ec = safe_avg([r.get("citation_precision") for r in esub]) if esub else None
                        qc1.metric("Ground.", f"{bg} → {eg}" if bg and eg else "—")
                        qc2.metric("Relev.", f"{br} → {er}" if br and er else "—")
                        qc3.metric("Cite P.", f"{bc} → {ec}" if bc and ec else "—")
            else:
                st.info("Run both baseline and enhanced evaluations to see the comparison.")

        # TAB: Query Set
        with tabs[3]:
            st.markdown("### 📝 20-Query Evaluation Set")
            from src.eval.evaluation import EVAL_QUERIES
            q_rows = []
            for q in EVAL_QUERIES:
                q_rows.append({
                    "ID": q["id"], "Type": q["type"], "Query": q["query"],
                    "Expected Sources": ", ".join(q["expected_sources"]) if q["expected_sources"] else "— (none expected)",
                })
            st.dataframe(pd.DataFrame(q_rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE: PHASE 2 DELIVERABLES
# ══════════════════════════════════════════════════════════════════════════
elif page == "📋 Phase 2 Deliverables":
    st.markdown("## 📋 Phase 2 Deliverables")
    st.markdown(
        '<span class="phase-badge">PHASE 2</span> '
        '<span class="phase-badge" style="background:#1565C0;">95-864 AI Model Development</span> '
        "Research-Grade RAG System",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── D1: Code Repository ──────────────────────────────────────────────
    st.markdown("### D1: Code Repository")
    st.markdown(
        "Complete, modular Python codebase with hierarchical structure, "
        "`Makefile` automation, env-driven configuration (`.env`), "
        "FastAPI backend, and Streamlit UI."
    )
    d1_checks = {
        "src/config.py (centralized config)": _file_exists(PROJECT_ROOT / "src" / "config.py"),
        "src/ingest/ (download + ingest)": _file_exists(PROJECT_ROOT / "src" / "ingest" / "ingest.py"),
        "src/rag/ (baseline + enhanced RAG)": _file_exists(PROJECT_ROOT / "src" / "rag" / "rag.py"),
        "src/eval/ (evaluation + report)": _file_exists(PROJECT_ROOT / "src" / "eval" / "evaluation.py"),
        "src/app/ (FastAPI + Streamlit)": _file_exists(PROJECT_ROOT / "src" / "app" / "app.py"),
        "Makefile": _file_exists(PROJECT_ROOT / "Makefile"),
        "requirements.txt": _file_exists(PROJECT_ROOT / "requirements.txt"),
        ".env.example": _file_exists(PROJECT_ROOT / ".env.example"),
        ".env (configured)": _file_exists(PROJECT_ROOT / ".env"),
        "README.md": _file_exists(PROJECT_ROOT / "README.md"),
    }
    for item, ok in d1_checks.items():
        icon = "✅" if ok else "❌"
        st.markdown(f"- {icon} `{item}`")

    st.markdown("---")

    # ── D2: Data Manifest ────────────────────────────────────────────────
    st.markdown("### D2: Data Manifest")
    st.markdown(
        "15-source corpus with full metadata following the A3 schema: "
        "`source_id`, `title`, `authors`, `year`, `source_type`, `venue`, "
        "`url_or_doi`, `raw_path`, `processed_path`, `tags`, `relevance_note`."
    )
    df = load_manifest()
    if not df.empty:
        st.success(f"✅ {len(df)} sources loaded from `data/data_manifest.csv`")
        st.dataframe(df[["source_id", "title", "year", "source_type", "venue", "tags"]],
                      use_container_width=True, hide_index=True)
    else:
        st.error("❌ data_manifest.csv not found")

    raw_ct = _dir_count(PROJECT_ROOT / "data" / "raw", "*.pdf")
    proc_ct = _dir_count(PROJECT_ROOT / "data" / "processed", "*.json")
    rc1, rc2 = st.columns(2)
    rc1.metric("Raw PDFs (data/raw/)", raw_ct)
    rc2.metric("Processed JSONs (data/processed/)", proc_ct)

    st.markdown("---")

    # ── D3: RAG Pipeline ─────────────────────────────────────────────────
    st.markdown("### D3: RAG Pipeline")
    st.markdown("""
**Baseline RAG**: Top-K semantic retrieval + Gemini generation with citation constraints.

**Enhanced RAG**: Query rewriting + sub-query decomposition → independent retrieval → merged deduplication → synthesis generation.

**Trust Behaviors**:
- Refuses to fabricate citations
- Flags missing evidence ("corpus does not contain…")
- Preserves hedging language (approximately, estimated, may)
- Detects conflicting evidence between sources
    """)
    d3_checks = {
        "FAISS index (data/processed/faiss_index.bin)": _file_exists(PROCESSED_DIR / "faiss_index.bin"),
        "Chunk store (data/processed/chunk_store.json)": _file_exists(PROCESSED_DIR / "chunk_store.json"),
        "Baseline RAG (src/rag/rag.py)": _file_exists(PROJECT_ROOT / "src" / "rag" / "rag.py"),
        "Enhanced RAG (src/rag/enhance_query_rewriting.py)": _file_exists(PROJECT_ROOT / "src" / "rag" / "enhance_query_rewriting.py"),
    }
    for item, ok in d3_checks.items():
        st.markdown(f"- {'✅' if ok else '❌'} `{item}`")

    if _file_exists(PROCESSED_DIR / "chunk_store.json"):
        chunks = json.loads((PROCESSED_DIR / "chunk_store.json").read_text(encoding="utf-8"))
        st.metric("Total Chunks in Index", len(chunks))

    st.markdown("---")

    # ── D4: Evaluation Framework ─────────────────────────────────────────
    st.markdown("### D4: Evaluation Framework")
    st.markdown(
        "20-query test set (10 direct, 5 synthesis, 5 edge-case) scored on 6 metrics "
        "using LLM-as-judge (Gemini) and deterministic measures."
    )

    eval_tab1, eval_tab2 = st.tabs(["Metrics", "20-Query Set"])
    with eval_tab1:
        st.dataframe(pd.DataFrame([
            {"Metric": "Groundedness", "Scoring": "LLM-judge 1–4", "Measures": "Claim support from chunks"},
            {"Metric": "Answer Relevance", "Scoring": "LLM-judge 1–4", "Measures": "Query alignment"},
            {"Metric": "Context Precision", "Scoring": "LLM-judge 1–4", "Measures": "Retrieved chunk relevance"},
            {"Metric": "Citation Precision", "Scoring": "Deterministic", "Measures": "valid / total citations"},
            {"Metric": "Source Recall", "Scoring": "Deterministic", "Measures": "expected sources found"},
            {"Metric": "Uncertainty Handling", "Scoring": "Rule-based", "Measures": "Flags missing evidence"},
        ]), use_container_width=True, hide_index=True)

    with eval_tab2:
        from src.eval.evaluation import EVAL_QUERIES
        qr = [{"ID": q["id"], "Type": q["type"], "Query": q["query"],
               "Expected": ", ".join(q["expected_sources"]) or "—"} for q in EVAL_QUERIES]
        st.dataframe(pd.DataFrame(qr), use_container_width=True, hide_index=True)

    baseline_files = sorted(glob.glob(str(OUTPUTS_DIR / "eval_results_baseline_*.json")))
    enhanced_files = sorted(glob.glob(str(OUTPUTS_DIR / "eval_results_enhanced_*.json")))
    ec1, ec2 = st.columns(2)
    ec1.metric("Baseline Eval Runs", len(baseline_files))
    ec2.metric("Enhanced Eval Runs", len(enhanced_files))

    st.markdown("---")

    # ── D5: Evaluation Report ────────────────────────────────────────────
    st.markdown("### D5: Evaluation Report")
    report_path = REPORT_DIR / "evaluation_report.md"
    if report_path.exists():
        st.success(f"✅ Report generated at `report/phase2/evaluation_report.md`")
        with st.expander("📄 View Report Contents"):
            st.markdown(report_path.read_text(encoding="utf-8"))
    else:
        st.warning("❌ Report not yet generated. Run: `make report`")

    st.markdown("---")

    # ── D6: API Backend ──────────────────────────────────────────────────
    st.markdown("### D6: API Backend")
    st.markdown("FastAPI REST API with structured endpoints and Pydantic validation.")

    endpoints = pd.DataFrame([
        {"Method": "GET", "Endpoint": "/health", "Description": "Server health, model info, index status"},
        {"Method": "POST", "Endpoint": "/query", "Description": "Run a RAG query (body: query, mode, top_k)"},
        {"Method": "GET", "Endpoint": "/corpus", "Description": "Corpus manifest + chunking strategy"},
        {"Method": "GET", "Endpoint": "/evaluation", "Description": "Latest eval results + summary metrics"},
        {"Method": "GET", "Endpoint": "/evaluation/queries", "Description": "The 20-query evaluation set"},
        {"Method": "GET", "Endpoint": "/logs", "Description": "Recent run logs"},
        {"Method": "GET", "Endpoint": "/logs/{run_id}", "Description": "Full detail for a specific run"},
    ])
    st.dataframe(endpoints, use_container_width=True, hide_index=True)
    st.caption("Launch with: `make serve` → interactive docs at `http://localhost:8000/docs`")

    st.markdown("---")

    # ── D7: Phase 1 Artifacts ────────────────────────────────────────────
    st.markdown("### D7: Phase 1 Artifacts")
    phase1_dir = PROJECT_ROOT / "report" / "phase1"
    expected_phase1 = [
        "Group 4- Framing brief.pdf",
        "Group 4- Prompt kit.pdf",
        "Group 4- Analysis memo.pdf",
        "Group 4- Evaluation sheet.pdf",
    ]
    if phase1_dir.exists():
        found = list(phase1_dir.glob("*.pdf"))
        st.success(f"✅ {len(found)} Phase 1 documents in `report/phase1/`")
        for f in found:
            st.markdown(f"- 📄 `{f.name}`")
    else:
        st.warning("❌ report/phase1/ directory not found")

    st.markdown("---")

    # ── D8: Streamlit UI ─────────────────────────────────────────────────
    st.markdown("### D8: Interactive Streamlit UI")
    st.success("✅ You are viewing it right now!")
    st.markdown("""
| Page | Purpose |
|------|---------|
| 🏠 Home | Project overview, research question, pipeline architecture |
| 💬 Ask a Question | Interactive chat interface to query the RAG pipeline |
| 📚 Corpus Explorer | Browse and filter the 15-source corpus |
| 📊 Evaluation Dashboard | View scores, radar charts, baseline vs. enhanced comparison |
| 📋 Phase 2 Deliverables | This page — status of every deliverable |
| 📝 Run Logs | Full audit trail of all RAG executions |
    """)

    st.markdown("---")

    # ── AI Usage Disclosure ──────────────────────────────────────────────
    st.markdown("### AI Usage Disclosure")
    st.dataframe(pd.DataFrame([
        {"Tool": f"Gemini ({GENERATION_MODEL})", "Purpose": "RAG generation + evaluation judging", "Review": "Prompt engineering, guardrail design"},
        {"Tool": "Cursor AI", "Purpose": "Code scaffolding", "Review": "Full code review and testing"},
        {"Tool": "sentence-transformers", "Purpose": "Embedding generation", "Review": "Configuration only"},
    ]), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE: RUN LOGS
# ══════════════════════════════════════════════════════════════════════════
elif page == "📝 Run Logs":
    st.markdown("## 📝 Run Logs")
    st.markdown("Audit trail of every RAG query execution with full detail.")

    logs = load_logs()

    if not logs:
        st.info("No logs yet. Ask a question on the 💬 page to generate your first log entry.")
    else:
        st.metric("Total Runs", len(logs))
        st.markdown("---")

        # Summary table
        log_rows = []
        for e in reversed(logs):
            cv = e.get("citation_validation", {})
            log_rows.append({
                "Timestamp": e.get("timestamp", "")[:19],
                "Mode": e.get("mode", ""),
                "Query": e.get("query", "")[:55],
                "Top-K": e.get("top_k", ""),
                "Citations": f"{cv.get('valid_citations', 0)}/{cv.get('total_citations', 0)}",
                "Precision": f"{cv.get('citation_precision', 0):.0%}" if cv.get("citation_precision") is not None else "N/A",
                "Tokens": f"{e.get('tokens', {}).get('input', 0)} / {e.get('tokens', {}).get('output', 0)}",
            })
        st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)

        # Detail
        st.markdown("### Detailed View")
        for e in reversed(logs):
            label = f"{e.get('timestamp', '')[:19]} | {e.get('mode', '')} | {e.get('query', '')[:45]}"
            with st.expander(label):
                st.markdown(f"**Run ID**: `{e.get('run_id', '')}`")
                st.markdown(f"**Prompt Version**: `{e.get('prompt_version', '')}`")
                st.markdown(f"**Query**: {e.get('query', '')}")

                st.markdown("**Retrieved Chunks:**")
                for c in e.get("retrieved_chunks", []):
                    st.markdown(
                        f"- `[{c['source_id']}, {c['chunk_id']}]` "
                        f"score={c['retrieval_score']:.4f} — {c['title']} ({c['year']})"
                    )

                st.markdown("**Answer:**")
                st.markdown(e.get("answer", ""))

                cv = e.get("citation_validation", {})
                st.markdown(
                    f"**Citations**: {cv.get('valid_citations', 0)}/{cv.get('total_citations', 0)} valid "
                    f"(precision: {cv.get('citation_precision', 'N/A')})"
                )
                if cv.get("invalid_list"):
                    st.warning(f"Invalid: {cv['invalid_list']}")

                tokens = e.get("tokens", {})
                st.caption(f"Tokens — input: {tokens.get('input', 0)}, output: {tokens.get('output', 0)}")
