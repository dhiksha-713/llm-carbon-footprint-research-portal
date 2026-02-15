"""Streamlit UI — LLM Carbon Footprint Research Portal (Phase 2)."""

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
    REPORT_DIR, CHUNK_PREVIEW_LEN,
)

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Carbon Footprint Research Portal",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Helpers ──────────────────────────────────────────────────────────────
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


# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio(
        "Page",
        ["Overview", "Run Pipeline", "Ask a Question", "Evaluation", "Deliverables"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**System Config**")
    st.text(f"Generation:  {GENERATION_MODEL}")
    st.text(f"Judge:       {JUDGE_MODEL}")
    st.text(f"Embeddings:  {EMBED_MODEL_NAME}")
    st.text(f"Chunk:       {CHUNK_SIZE_TOKENS}t / {CHUNK_OVERLAP_TOKENS}t overlap")
    st.text(f"Top-K:       {TOP_K} baseline / {ENHANCED_TOP_N} enhanced")

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
        "in Large Language Models."
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

    st.subheader("Corpus")
    df = load_manifest()
    if not df.empty:
        st.write(f"{len(df)} sources, spanning {int(df['year'].min())}--{int(df['year'].max())}.")
        st.dataframe(
            df[["source_id", "title", "authors", "year", "source_type", "venue", "tags"]],
            use_container_width=True, hide_index=True,
        )
    else:
        st.warning("Manifest not found. Run `make download` first.")

    st.subheader("Pipeline")
    st.markdown(f"""
1. **Ingest** -- PyMuPDF PDF parsing, section-aware chunking ({CHUNK_SIZE_TOKENS}t / {CHUNK_OVERLAP_TOKENS}t overlap), {EMBED_MODEL_NAME} embeddings, FAISS IndexFlatIP.
2. **Baseline RAG** -- Top-{TOP_K} semantic retrieval + {GENERATION_MODEL} generation with citation constraints.
3. **Enhanced RAG** -- Query rewriting + decomposition for synthesis/multi-hop questions, top-{ENHANCED_TOP_N} merged chunks.
4. **Trust behaviors** -- Refuses fabricated citations, flags missing evidence, preserves hedging, detects conflicts.
5. **Evaluation** -- 20-query test set, 6 metrics (3 LLM-judge + 3 deterministic), baseline vs enhanced comparison.
6. **Report** -- Auto-generated Markdown evaluation report.
    """)

    st.subheader("Phase 2 Deliverables")
    st.markdown("""
- **D1 -- Code repository**: Modular `src/` packages, `Makefile`, env-driven config, FastAPI + Streamlit.
- **D2 -- Data manifest**: 15-source corpus with full A3 schema in `data/data_manifest.csv`.
- **D3 -- RAG pipeline**: Baseline and enhanced pipelines with citation validation and trust behaviors.
- **D4 -- Evaluation framework**: 20-query set scored on 6 metrics using LLM-as-judge.
- **D5 -- Evaluation report**: Auto-generated at `report/phase2/evaluation_report.md`.
- **D6 -- API backend**: FastAPI REST API with `/query`, `/corpus`, `/evaluation`, `/logs` endpoints.
- **D7 -- Interactive UI**: This Streamlit application with pipeline orchestrator and chat interface.
- **D8 -- Phase 1 artifacts**: Framing brief, prompt kit, analysis memo, evaluation sheet in `report/phase1/`.
    """)


# ══════════════════════════════════════════════════════════════════════════
# PAGE: RUN PIPELINE
# ══════════════════════════════════════════════════════════════════════════
elif page == "Run Pipeline":
    st.header("Run Full Pipeline")
    st.markdown(
        "Click the button below to execute the entire Phase 2 pipeline end-to-end. "
        "Each step shows exactly what is happening behind the scenes."
    )

    if st.button("Run Full Pipeline", type="primary", use_container_width=True):

        # ── Step 1: Corpus manifest ──────────────────────────────────────
        with st.status("Step 1/7: Loading corpus manifest...", expanded=True) as s1:
            df = load_manifest()
            if df.empty:
                st.error("data_manifest.csv not found. Run `make download` first.")
                st.stop()
            st.write(f"Loaded **{len(df)} sources** from `data/data_manifest.csv`.")
            st.write(f"Types: {df['source_type'].value_counts().to_dict()}")
            st.write(f"Year range: {int(df['year'].min())}--{int(df['year'].max())}")
            st.dataframe(
                df[["source_id", "title", "year", "source_type"]],
                use_container_width=True, hide_index=True,
            )
            s1.update(label="Step 1/7: Corpus manifest loaded (15 sources)", state="complete")

        # ── Step 2: Verify ingestion ─────────────────────────────────────
        with st.status("Step 2/7: Verifying ingestion (FAISS index + chunk store)...", expanded=True) as s2:
            index_path = PROCESSED_DIR / "faiss_index.bin"
            store_path = PROCESSED_DIR / "chunk_store.json"
            strat_path = PROCESSED_DIR / "chunking_strategy.json"

            if not index_path.exists() or not store_path.exists():
                st.error("FAISS index or chunk store missing. Run `make ingest` first.")
                st.stop()

            store_data = json.loads(store_path.read_text(encoding="utf-8"))
            st.write(f"FAISS index: `{index_path.name}` exists.")
            st.write(f"Chunk store: **{len(store_data)} chunks** across "
                      f"**{len({c['source_id'] for c in store_data})} sources**.")

            if strat_path.exists():
                strategy = json.loads(strat_path.read_text(encoding="utf-8"))
                st.write(f"Chunking strategy: {strategy.get('chunk_size_tokens')}t size, "
                          f"{strategy.get('chunk_overlap_tokens')}t overlap, "
                          f"{strategy.get('embed_model')} embeddings ({strategy.get('embed_dim')}d).")

            # Show a sample chunk
            sample = store_data[0]
            st.write("**Sample chunk:**")
            st.text(f"  source_id:  {sample['source_id']}")
            st.text(f"  chunk_id:   {sample['chunk_id']}")
            st.text(f"  section:    {sample['section_header']}")
            st.text(f"  text:       {sample['chunk_text'][:150]}...")

            s2.update(label=f"Step 2/7: Ingestion verified ({len(store_data)} chunks indexed)", state="complete")

        # ── Step 3: Load models ──────────────────────────────────────────
        with st.status("Step 3/7: Loading embedding model, FAISS index, Gemini client...", expanded=True) as s3:
            t0 = time.time()
            index, store, embed_model, client = load_rag_resources()
            load_time = time.time() - t0
            st.write(f"Embedding model: `{EMBED_MODEL_NAME}`")
            st.write(f"Generation model: `{GENERATION_MODEL}`")
            st.write(f"FAISS index loaded: {index.ntotal} vectors")
            st.write(f"Load time: {load_time:.1f}s")
            s3.update(label=f"Step 3/7: Models loaded ({load_time:.1f}s)", state="complete")

        # ── Step 4: Baseline RAG ─────────────────────────────────────────
        with st.status("Step 4/7: Running baseline RAG on a sample query...", expanded=True) as s4:
            from src.rag.rag import run_rag

            baseline_query = "What are the major sources of carbon emissions in LLM training?"
            st.write(f"**Query:** {baseline_query}")
            st.write(f"**Mode:** baseline | **Top-K:** {TOP_K}")

            t0 = time.time()
            baseline_result = run_rag(baseline_query, index, store, embed_model, client,
                                       top_k=TOP_K, mode="baseline")
            elapsed = time.time() - t0

            st.write(f"**Retrieval:** {len(baseline_result['retrieved_chunks'])} chunks retrieved in {elapsed:.1f}s")
            st.write("Retrieved sources:")
            for c in baseline_result["retrieved_chunks"]:
                st.text(f"  [{c['source_id']}, {c['chunk_id']}] score={c['retrieval_score']:.4f} -- {c['title']}")

            st.write("**Generated answer** (first 500 chars):")
            st.markdown(f"> {baseline_result['answer'][:500]}...")

            cv = baseline_result["citation_validation"]
            st.write(f"**Citation validation:** {cv['valid_citations']}/{cv['total_citations']} valid "
                      f"(precision: {cv['citation_precision']:.2f})" if cv['citation_precision'] is not None
                      else f"**Citation validation:** {cv['valid_citations']}/{cv['total_citations']} valid")
            if cv["invalid_list"]:
                st.write(f"Invalid citations: {cv['invalid_list']}")

            st.write(f"Tokens: {baseline_result['tokens'].get('input', 0)} in / "
                      f"{baseline_result['tokens'].get('output', 0)} out")

            s4.update(label=f"Step 4/7: Baseline RAG complete ({elapsed:.1f}s, "
                            f"cite precision={cv.get('citation_precision', 'N/A')})", state="complete")

        # ── Step 5: Enhanced RAG ─────────────────────────────────────────
        with st.status("Step 5/7: Running enhanced RAG on a synthesis query...", expanded=True) as s5:
            from src.rag.enhance_query_rewriting import run_enhanced_rag

            enhanced_query = ("Compare Strubell et al. and Patterson et al. on their measurement "
                              "methodology: where do they agree and disagree?")
            st.write(f"**Query:** {enhanced_query}")
            st.write(f"**Mode:** enhanced (query rewriting + decomposition)")

            t0 = time.time()
            enhanced_result = run_enhanced_rag(enhanced_query, index, store, embed_model, client)
            elapsed = time.time() - t0

            st.write(f"**Query type classified as:** {enhanced_result.get('query_type', 'N/A')}")
            st.write(f"**Rewritten query:** {enhanced_result.get('rewritten_query', 'N/A')}")
            sub_qs = enhanced_result.get("sub_queries", [])
            if sub_qs:
                st.write(f"**Decomposed into {len(sub_qs)} sub-queries:**")
                for i, sq in enumerate(sub_qs, 1):
                    st.text(f"  {i}. {sq}")

            st.write(f"**Retrieval:** {len(enhanced_result['retrieved_chunks'])} merged chunks")
            for c in enhanced_result["retrieved_chunks"]:
                st.text(f"  [{c['source_id']}, {c['chunk_id']}] score={c['retrieval_score']:.4f}")

            st.write("**Synthesized answer** (first 500 chars):")
            st.markdown(f"> {enhanced_result['answer'][:500]}...")

            ecv = enhanced_result["citation_validation"]
            st.write(f"**Citation validation:** {ecv['valid_citations']}/{ecv['total_citations']} valid")

            s5.update(label=f"Step 5/7: Enhanced RAG complete ({elapsed:.1f}s, "
                            f"{len(sub_qs)} sub-queries)", state="complete")

        # ── Step 6: Evaluation (subset) ──────────────────────────────────
        with st.status("Step 6/7: Running evaluation on 5 sample queries...", expanded=True) as s6:
            from src.eval.evaluation import (
                EVAL_QUERIES, score_groundedness, score_answer_relevance,
                score_context_precision, score_uncertainty_handling,
                compute_source_recall,
            )

            eval_subset = [EVAL_QUERIES[0], EVAL_QUERIES[2], EVAL_QUERIES[9],
                           EVAL_QUERIES[10], EVAL_QUERIES[16]]
            st.write(f"Running {len(eval_subset)} queries (D01, D03, D10, S01, E02) "
                      f"through baseline pipeline with LLM-as-judge scoring...")

            eval_rows = []
            for i, q in enumerate(eval_subset, 1):
                st.write(f"  [{i}/{len(eval_subset)}] {q['id']} ({q['type']}): {q['query'][:60]}...")
                result = run_rag(q["query"], index, store, embed_model, client, mode="baseline")
                ground = score_groundedness(result["answer"], result["retrieved_chunks"], client)
                rel = score_answer_relevance(q["query"], result["answer"], client)
                ctx = score_context_precision(result["answer"], result["retrieved_chunks"], client)
                unc = score_uncertainty_handling(result["answer"])
                ret_sources = [c["source_id"] for c in result["retrieved_chunks"]]
                sr = compute_source_recall(ret_sources, q["expected_sources"])
                cp = result["citation_validation"].get("citation_precision")

                eval_rows.append({
                    "ID": q["id"],
                    "Type": q["type"],
                    "Groundedness": ground.get("score"),
                    "Relevance": rel.get("score"),
                    "Ctx Precision": ctx.get("score"),
                    "Cite Precision": round(cp, 2) if cp is not None else None,
                    "Source Recall": round(sr, 2) if sr is not None else None,
                    "Flags Missing": "yes" if unc["flags_missing_evidence"] else "no",
                })

            st.dataframe(pd.DataFrame(eval_rows), use_container_width=True, hide_index=True)

            g_avg = safe_avg([r["Groundedness"] for r in eval_rows])
            r_avg = safe_avg([r["Relevance"] for r in eval_rows])
            cp_avg = safe_avg([r["Cite Precision"] for r in eval_rows])
            st.write(f"**Averages:** groundedness={g_avg}/4, relevance={r_avg}/4, "
                      f"cite precision={cp_avg}")

            s6.update(label=f"Step 6/7: Evaluation complete (avg ground={g_avg}, rel={r_avg})",
                      state="complete")

        # ── Step 7: Summary ──────────────────────────────────────────────
        with st.status("Step 7/7: Pipeline summary...", expanded=True) as s7:
            st.write("**Full pipeline executed successfully.**")
            st.write("")
            st.write("What was demonstrated:")
            st.write("- Corpus: 15 sources loaded and verified")
            st.write(f"- Ingestion: {len(store_data)} chunks indexed in FAISS")
            st.write(f"- Baseline RAG: query -> top-{TOP_K} retrieval -> {GENERATION_MODEL} generation -> citation validation")
            st.write(f"- Enhanced RAG: query rewriting + decomposition -> merged retrieval -> synthesis")
            st.write(f"- Trust behavior: citation precision = {cv.get('citation_precision', 'N/A')}, "
                      f"uncertainty flagging active")
            st.write(f"- Evaluation: {len(eval_subset)} queries scored on 6 metrics via LLM-as-judge")
            st.write("")
            st.write("To run the full 20-query evaluation: `make eval-both`")
            st.write("To generate the report: `make report`")
            s7.update(label="Step 7/7: Pipeline demo complete", state="complete")

        st.success("All pipeline steps completed. See each step above for details.")


# ══════════════════════════════════════════════════════════════════════════
# PAGE: ASK A QUESTION
# ══════════════════════════════════════════════════════════════════════════
elif page == "Ask a Question":
    st.header("Ask a Research Question")
    st.markdown(
        "Query the RAG pipeline interactively. The system retrieves relevant evidence "
        "from the 15-source corpus, generates an answer with inline citations, "
        "and validates every citation."
    )

    # Config
    col1, col2 = st.columns(2)
    with col1:
        mode = st.selectbox("Pipeline mode", ["baseline", "enhanced"])
    with col2:
        top_k = st.slider("Top-K", 1, 20, TOP_K)

    # Sample query selector
    sample = st.selectbox(
        "Or pick a sample question:",
        ["(type your own below)"] + SAMPLE_QUESTIONS,
    )

    # Chat history
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
                    f"Citations: {cv['valid_citations']}/{cv['total_citations']} valid | "
                    f"Precision: {cv.get('citation_precision', 'N/A')} | "
                    f"Tokens: {msg['tokens'].get('input', 0)} in / {msg['tokens'].get('output', 0)} out | "
                    f"Mode: {msg.get('mode', 'N/A')}"
                )
                with st.expander(f"{len(msg['retrieved_chunks'])} retrieved chunks"):
                    for j, c in enumerate(msg["retrieved_chunks"], 1):
                        st.text(
                            f"  {j}. [{c['source_id']}, {c['chunk_id']}] "
                            f"score={c['retrieval_score']:.4f} -- {c['title']} ({c['year']})"
                        )

    # Input
    pending = None
    if sample != "(type your own below)":
        if st.button("Use this sample question"):
            pending = sample

    user_input = st.chat_input("Ask about LLM carbon footprints...")
    query = pending or user_input

    if query:
        st.session_state["chat_history"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.status("Running RAG pipeline...", expanded=True) as ps:
                st.write(f"Mode: {mode} | Top-K: {top_k}")

                st.write("Loading models...")
                index, store, embed_model, client = load_rag_resources()

                from src.rag.rag import run_rag
                from src.rag.enhance_query_rewriting import run_enhanced_rag

                if mode == "enhanced":
                    st.write("Rewriting query and decomposing into sub-queries...")
                else:
                    st.write(f"Retrieving top-{top_k} chunks via semantic search...")

                t0 = time.time()
                if mode == "enhanced":
                    result = run_enhanced_rag(query, index, store, embed_model, client)
                else:
                    result = run_rag(query, index, store, embed_model, client,
                                     top_k=top_k, mode=mode)
                elapsed = time.time() - t0

                st.write(f"Retrieved {len(result['retrieved_chunks'])} chunks.")
                st.write(f"Generating answer with {GENERATION_MODEL}...")
                st.write(f"Validating {len(result.get('citations_extracted', []))} citations...")
                ps.update(label=f"Pipeline complete ({elapsed:.1f}s)", state="complete")

            # Answer
            st.markdown(result["answer"])

            # Summary line
            cv = result["citation_validation"]
            st.caption(
                f"Citations: {cv['valid_citations']}/{cv['total_citations']} valid | "
                f"Precision: {cv.get('citation_precision', 'N/A')} | "
                f"Tokens: {result['tokens'].get('input', 0)} in / "
                f"{result['tokens'].get('output', 0)} out | "
                f"Time: {elapsed:.1f}s"
            )

            # Chunks
            with st.expander(f"{len(result['retrieved_chunks'])} retrieved chunks"):
                for j, c in enumerate(result["retrieved_chunks"], 1):
                    st.text(
                        f"  {j}. [{c['source_id']}, {c['chunk_id']}] "
                        f"score={c['retrieval_score']:.4f} -- {c['title']} ({c['year']})"
                    )

            if cv.get("invalid_list"):
                st.warning(f"Invalid citations: {cv['invalid_list']}")

        st.session_state["chat_history"].append({
            "role": "assistant",
            "answer": result["answer"],
            "retrieved_chunks": result["retrieved_chunks"],
            "citation_validation": cv,
            "tokens": result["tokens"],
            "mode": mode,
        })


# ══════════════════════════════════════════════════════════════════════════
# PAGE: EVALUATION
# ══════════════════════════════════════════════════════════════════════════
elif page == "Evaluation":
    st.header("Evaluation")

    # Metrics definitions
    st.subheader("Metrics (6)")
    st.dataframe(pd.DataFrame([
        {"Metric": "Groundedness", "Type": "LLM-judge (1-4)", "Description": "Are claims supported by retrieved chunks?"},
        {"Metric": "Answer Relevance", "Type": "LLM-judge (1-4)", "Description": "Does the answer address the query?"},
        {"Metric": "Context Precision", "Type": "LLM-judge (1-4)", "Description": "Are retrieved chunks relevant?"},
        {"Metric": "Citation Precision", "Type": "Deterministic", "Description": "valid citations / total citations"},
        {"Metric": "Source Recall", "Type": "Deterministic", "Description": "expected sources found / total expected"},
        {"Metric": "Uncertainty Handling", "Type": "Rule-based", "Description": "Does answer flag missing evidence?"},
    ]), use_container_width=True, hide_index=True)

    # 20-query set
    st.subheader("20-Query Evaluation Set")
    from src.eval.evaluation import EVAL_QUERIES
    q_rows = [{"ID": q["id"], "Type": q["type"], "Query": q["query"],
               "Expected Sources": ", ".join(q["expected_sources"]) or "(none)"}
              for q in EVAL_QUERIES]
    st.dataframe(pd.DataFrame(q_rows), use_container_width=True, hide_index=True)

    # Results
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
                    "Ctx Prec.": r.get("context_precision", {}).get("score"),
                    "Cite Prec.": round(r["citation_precision"], 2) if r.get("citation_precision") is not None else None,
                    "Src Recall": round(r["source_recall"], 2) if r.get("source_recall") is not None else None,
                    "Missing?": "yes" if r["uncertainty"]["flags_missing_evidence"] else "--",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            ground = [r["groundedness"].get("score") for r in results if r["groundedness"].get("score")]
            rel = [r["answer_relevance"].get("score") for r in results if r["answer_relevance"].get("score")]
            cite = [r["citation_precision"] for r in results if r.get("citation_precision") is not None]
            recall = [r["source_recall"] for r in results if r.get("source_recall") is not None]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Avg Groundedness", f"{safe_avg(ground)}/4" if ground else "--")
            m2.metric("Avg Relevance", f"{safe_avg(rel)}/4" if rel else "--")
            m3.metric("Avg Cite Precision", f"{safe_avg(cite):.2f}" if cite else "--")
            m4.metric("Avg Source Recall", f"{safe_avg(recall):.2f}" if recall else "--")
            st.markdown("---")

        # Comparison
        if baseline_results and enhanced_results:
            st.subheader("Baseline vs Enhanced Comparison")

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
            st.dataframe(pd.DataFrame(comp), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE: DELIVERABLES
# ══════════════════════════════════════════════════════════════════════════
elif page == "Deliverables":
    st.header("Phase 2 Deliverables")
    st.markdown("AI Model Development (95-864) -- Group 4 -- Dhiksha Rathis, Shreya Verma")

    def _check(path: Path) -> str:
        return "OK" if path.exists() else "MISSING"

    def _count(path: Path, pattern: str) -> int:
        return len(list(path.glob(pattern))) if path.exists() else 0

    st.subheader("D1: Code Repository")
    checks = {
        "src/config.py": _check(PROJECT_ROOT / "src" / "config.py"),
        "src/ingest/": _check(PROJECT_ROOT / "src" / "ingest" / "ingest.py"),
        "src/rag/": _check(PROJECT_ROOT / "src" / "rag" / "rag.py"),
        "src/eval/": _check(PROJECT_ROOT / "src" / "eval" / "evaluation.py"),
        "src/app/": _check(PROJECT_ROOT / "src" / "app" / "app.py"),
        "Makefile": _check(PROJECT_ROOT / "Makefile"),
        "requirements.txt": _check(PROJECT_ROOT / "requirements.txt"),
        ".env.example": _check(PROJECT_ROOT / ".env.example"),
        "README.md": _check(PROJECT_ROOT / "README.md"),
    }
    for item, status in checks.items():
        st.text(f"  [{status}] {item}")

    st.subheader("D2: Data Manifest")
    df = load_manifest()
    if not df.empty:
        st.write(f"{len(df)} sources in data/data_manifest.csv")
        st.write(f"Raw PDFs: {_count(PROJECT_ROOT / 'data' / 'raw', '*.pdf')}")
        st.write(f"Processed JSONs: {_count(PROCESSED_DIR, '*.json')}")
    else:
        st.warning("data_manifest.csv not found.")

    st.subheader("D3: RAG Pipeline")
    st.text(f"  [{_check(PROCESSED_DIR / 'faiss_index.bin')}] FAISS index")
    st.text(f"  [{_check(PROCESSED_DIR / 'chunk_store.json')}] Chunk store")
    st.text(f"  [{_check(PROJECT_ROOT / 'src' / 'rag' / 'rag.py')}] Baseline RAG")
    st.text(f"  [{_check(PROJECT_ROOT / 'src' / 'rag' / 'enhance_query_rewriting.py')}] Enhanced RAG")
    if (PROCESSED_DIR / "chunk_store.json").exists():
        chunks = json.loads((PROCESSED_DIR / "chunk_store.json").read_text(encoding="utf-8"))
        st.write(f"Total chunks in index: {len(chunks)}")

    st.subheader("D4: Evaluation Framework")
    st.write("20-query test set: 10 direct, 5 synthesis, 5 edge-case.")
    st.write("6 metrics: groundedness, relevance, context precision (LLM-judge), "
             "citation precision, source recall (deterministic), uncertainty handling (rule-based).")
    baseline_ct = len(glob.glob(str(OUTPUTS_DIR / "eval_results_baseline_*.json")))
    enhanced_ct = len(glob.glob(str(OUTPUTS_DIR / "eval_results_enhanced_*.json")))
    st.write(f"Baseline eval runs: {baseline_ct} | Enhanced eval runs: {enhanced_ct}")

    st.subheader("D5: Evaluation Report")
    report_path = REPORT_DIR / "evaluation_report.md"
    if report_path.exists():
        st.write(f"Report at: report/phase2/evaluation_report.md")
        with st.expander("View report"):
            st.markdown(report_path.read_text(encoding="utf-8"))
    else:
        st.warning("Report not generated. Run: `make report`")

    st.subheader("D6: API Backend")
    st.text(f"  [{_check(PROJECT_ROOT / 'src' / 'app' / 'app.py')}] FastAPI app")
    st.dataframe(pd.DataFrame([
        {"Method": "GET", "Endpoint": "/health", "Description": "Server health + model info"},
        {"Method": "POST", "Endpoint": "/query", "Description": "Run RAG query"},
        {"Method": "GET", "Endpoint": "/corpus", "Description": "Corpus manifest"},
        {"Method": "GET", "Endpoint": "/evaluation", "Description": "Eval results + metrics"},
        {"Method": "GET", "Endpoint": "/logs", "Description": "Run logs"},
    ]), use_container_width=True, hide_index=True)
    st.caption("Start with: `make serve` -- docs at http://localhost:8000/docs")

    st.subheader("D7: Interactive UI")
    st.write("This Streamlit application. Pages: Overview, Run Pipeline, Ask a Question, "
             "Evaluation, Deliverables.")

    st.subheader("D8: Phase 1 Artifacts")
    phase1_dir = PROJECT_ROOT / "report" / "phase1"
    if phase1_dir.exists():
        found = list(phase1_dir.glob("*.pdf"))
        st.write(f"{len(found)} documents in report/phase1/")
        for f in found:
            st.text(f"  {f.name}")
    else:
        st.warning("report/phase1/ not found.")

    st.subheader("AI Usage Disclosure")
    st.dataframe(pd.DataFrame([
        {"Tool": GENERATION_MODEL, "Purpose": "RAG generation + eval judging", "Review": "Prompt engineering, guardrails"},
        {"Tool": "Cursor AI", "Purpose": "Code scaffolding", "Review": "Full code review and testing"},
        {"Tool": "sentence-transformers", "Purpose": "Embeddings", "Review": "Configuration only"},
    ]), use_container_width=True, hide_index=True)
