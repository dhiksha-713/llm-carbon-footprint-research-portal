# LLM Carbon Footprint Research Portal

**Personal Research Portal (PRP)** that helps you move from a research question to a grounded synthesis on the
environmental cost of large AI models. It ingests 20 research sources, retrieves evidence via semantic search,
produces citation-backed answers, generates exportable research artifacts, and logs evaluation results.

**Course**: AI Model Development (95-864) | **Group 4**: Dhiksha Rathis, Shreya Verma | CMU Spring 2026

[![Demo Video](https://img.shields.io/badge/Demo-YouTube-red?logo=youtube)](https://youtu.be/Sir30x56_Dk) **[Watch the demo video](https://youtu.be/Sir30x56_Dk)**

---

## Quick Start (5 minutes)

```bash
git clone <repo-url> && cd llm-carbon-footprint-research-portal
python3 -m venv venv
source venv/bin/activate
make install                      # pip install -r requirements.txt
cp .env.example .env              # then edit .env with your API keys
make download && make ingest      # download 20 PDFs, build FAISS index
make ui                           # http://localhost:8501
```

### Windows (PowerShell)

```powershell
git clone <repo-url>; cd llm-carbon-footprint-research-portal
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env            # then edit .env with your API keys
python -m src.ingest.download_sources
python -m src.ingest.ingest
python -m streamlit run src/app/streamlit_ui.py --server.port 8501
```

### Verify It Works (single command)

```bash
make query
# Runs a sample query, prints retrieved chunks + cited answer + log path
```

---

## Deliverables Checklist

| Deliverable | Location | Status |
|-------------|----------|--------|
| Working PRP app | `src/app/streamlit_ui.py` — run with `make ui` | Done |
| Demo recording (3–6 min) | [YouTube](https://youtu.be/Sir30x56_Dk) | Done |
| Final report (6–10 pages) | `report/phase3/` | Done |
| Phase 2 evaluation report | `report/phase2/evaluation_report.md` | Done |
| Data manifest | `data/data_manifest.csv` (20 sources, full metadata) | Done |
| Raw corpus | `data/raw/` (20 PDFs, downloaded via `make download`) | Done |
| Generated artifacts | `outputs/artifacts/` (evidence tables as MD + CSV) | Done |
| Run logs | `logs/rag_runs.jsonl` (machine-readable, one entry per RAG run) | Done |
| Evaluation results | `outputs/eval_results_baseline_*.json`, `outputs/eval_results_enhanced_*.json` | Done |
| Research threads | `data/threads/` (JSON, auto-saved per query) | Done |
| Pinned dependencies | `requirements.txt` | Done |
| AI usage disclosure | See section below + included in report | Done |

---

## What This Portal Does (MVP Capabilities)

1. **Ingest a domain corpus** — 20 PDFs with metadata (author, year, source type, link/DOI)
2. **Retrieve evidence** — semantic retrieval via FAISS + sentence-transformers
3. **Generate answers** — every major claim backed by inline citations `(source_id, chunk_id)`
4. **Produce research artifacts** — evidence tables, annotated bibliographies, synthesis memos
5. **Save research threads** — query + retrieved chunks + answer; export as Markdown/CSV/PDF
6. **Run evaluation** — 20-query test set, LLM-as-judge scoring, 6 metrics
7. **Trust behavior** — refuse to fabricate citations; flag missing evidence with suggested next steps

---

## Latest Evaluation Results

20-query evaluation set (10 direct, 5 synthesis/multihop, 5 edge case):

| Mode | Groundedness | Relevance | Citation Precision | Source Recall | Flags Missing |
|------|:-----------:|:---------:|:-----------------:|:------------:|:-------------:|
| **Baseline** | 3.1 / 4 | 3.1 / 4 | 0.94 | 0.37 | 9 / 20 |
| **Enhanced** | 2.95 / 4 | 3.4 / 4 | 0.94 | 0.42 | 6 / 20 |

**Key findings**: Enhanced mode trades marginal groundedness for better relevance (+0.3) and source coverage (+0.05 recall). Query decomposition pulls in evidence from more diverse sources. Edge-case queries correctly flag missing evidence instead of hallucinating.

---

## Phase 3 — Personal Research Portal Product

Phase 3 wraps the Phase 2 RAG pipeline into a usable research product:

| Feature | Description | Location |
|---------|-------------|----------|
| **Research Interface** | Ask questions, get cited answers, threads saved automatically | Research page |
| **Research Threads** | Browse, view, export, delete saved research sessions | Research Threads page |
| **Artifact Generator** | Evidence tables, annotated bibliographies, synthesis memos | Artifacts page |
| **Export** | Download any artifact or thread as Markdown, CSV, or PDF | All pages |
| **Corpus Explorer** | Browse and filter the 20-source corpus with chunk previews | Corpus Explorer page |
| **Evaluation Dashboard** | 20-query test set, metrics, representative examples | Evaluation page |
| **Trust Behavior** | Missing evidence flagged explicitly with suggested next retrieval steps | Research page |
| **Model Comparison** | Side-by-side Grok-3 vs Azure OpenAI | Compare Models page |
| **Demo All Phases** | One-click walkthrough of Phases 1 → 2 → 3 with cached results | Home page |

### Streamlit UI Pages

| Page | Purpose |
|------|---------|
| **Home / Demo All Phases** | One-click demo: downloads, ingests, runs baseline + enhanced RAG, generates artifacts, evaluates 20 queries, caches results |
| **Research** | Main ask interface — cited answers, auto-saved threads, trust warnings |
| **Research Threads** | Browse, view, export (MD/JSON/PDF), delete saved sessions |
| **Artifacts** | Generate from query or thread: evidence table, bibliography, synthesis memo |
| **Corpus Explorer** | Filter by type/year/keyword, view source details and chunk previews |
| **Evaluation** | Query set, results tables, representative examples, run evaluations |
| **Compare Models** | Same query through Grok-3 and Azure side-by-side |

---

## Pipeline

```
User Query
  |-> Sanitize (prompt-injection detection, 1000-char limit, control-char strip)
  |-> [Enhanced] Classify (direct / synthesis / multihop / edge_case)
  |-> [Enhanced] Rewrite query + decompose into sub-queries
  |-> Retrieve
  |     +-- Baseline: embed query -> FAISS top-5 cosine search
  |     +-- Enhanced: per sub-query retrieval -> merge + deduplicate -> top-8
  |-> Generate (LLM with citation rules: cite (source_id, chunk_id))
  |-> Validate Citations (deterministic check: each cited ID exists in retrieved set)
  |-> Trust Check (flag missing evidence, suggest next retrieval steps)
  |-> Save Thread (query + evidence + answer -> data/threads/)
  |-> [Optional] Generate Artifact (evidence table / bibliography / synthesis memo)
  +-> Cited answer with reference list + export options
```

---

## Corpus

**20 sources** — 16 peer-reviewed papers, 2 technical reports, 2 tool/workshop papers.

Covering: LLM training energy, inference costs, lifecycle carbon analysis, measurement tools, reporting standards, Green AI, and carbon estimation methodologies. Sources span 2016–2024 from venues including ACL, NeurIPS, JMLR, FAccT, ICLR, MLSys, and IEEE.

Full metadata in `data/data_manifest.csv` with fields: `source_id`, `title`, `authors`, `year`, `source_type`, `venue`, `url_or_doi`, `raw_path`, `processed_path`, `tags`, `relevance_note`.

### Corpus Selection Methodology (Explicit)

To address reproducibility and grading transparency, source selection was done with a fixed, documented process:

1. **Research scope fixed first**: LLM carbon footprint across training, inference, lifecycle, and measurement methods.
2. **Candidate pool creation**: papers/reports gathered from well-known venues plus high-signal references from survey papers.
3. **Inclusion criteria**:
   - Directly measures, models, or critiques energy/carbon impacts of ML/LLMs.
   - Provides methodological detail sufficient for evidence extraction.
   - Has stable source access (PDF and URL/DOI).
   - Fits the 2016-2024 period to capture method evolution.
4. **Exclusion criteria**:
   - Purely opinion/editorial pieces without empirical or methodological evidence.
   - Duplicates/near-duplicates of the same findings.
   - Sources with missing metadata or unavailable full text.
5. **Balance constraints**:
   - Include both foundational and recent work.
   - Cover both training and inference emissions.
   - Include methods/tools papers to support retrieval and evaluation queries.
6. **Final manifest freeze**:
   - Final 20 sources stored in `data/data_manifest.csv`.
   - Raw files reproducibly fetched to `data/raw/` via `make download`.
   - Each source has a relevance note explaining why it was selected.

Selection mode is **manual curation with scripted download/validation** (not random crawl), which keeps the corpus aligned with the research question while remaining fully reproducible.

---

## Research Artifacts

The portal generates three types of research artifacts:

| Artifact | Schema | Output |
|----------|--------|--------|
| **Evidence Table** | Claim \| Evidence Snippet \| Citation \| Confidence \| Notes | Markdown + CSV |
| **Annotated Bibliography** | Citation \| Key Claim \| Method \| Limitations \| Relevance | Markdown |
| **Synthesis Memo** | 800–1200 words with inline citations + reference list | Markdown |

All artifacts are saved to `outputs/artifacts/` and can be exported as Markdown, CSV, or PDF.

---

## Repository Structure

```
llm-carbon-footprint-research-portal/
  README.md
  requirements.txt
  Makefile
  .env.example
  .gitignore
  data/
    data_manifest.csv              # 20 sources with full metadata
    raw/                           # Downloaded PDFs (git-ignored, reproduced via make download)
    processed/                     # chunk_store.json, faiss_index.bin (git-ignored)
    threads/                       # Saved research threads as JSON (git-ignored)
  src/
    config.py                      # Centralized config (all env-driven, no hardcoded secrets)
    llm_client.py                  # Provider-agnostic LLM abstraction (Grok-3 + Azure)
    utils.py                       # Sanitization, citation regex, trust behavior, JSON extraction
    threads.py                     # Research thread save/load/list/delete/export
    artifacts.py                   # Artifact generation + PDF export (evidence table, bib, memo)
    ingest/
      download_sources.py          # Download PDFs from manifest URLs + validate title match
      ingest.py                    # Parse PDFs -> chunk (500t/100t) -> embed -> FAISS index
    rag/
      rag.py                       # Baseline RAG: retrieve, generate, validate citations, log
      enhance_query_rewriting.py   # Enhanced: classify, rewrite, decompose, multi-retrieve, synthesize
    eval/
      evaluation.py                # 20-query eval set, LLM-as-judge, 6 metrics
      generate_report.py           # Markdown evaluation report with failure cases
    app/
      app.py                       # FastAPI backend (7 endpoints)
      streamlit_ui.py              # Streamlit UI (7 pages + demo)
  outputs/
    artifacts/                     # Generated evidence tables, bibliographies, memos
    eval_results_baseline_*.json   # Baseline evaluation results
    eval_results_enhanced_*.json   # Enhanced evaluation results
  logs/
    rag_runs.jsonl                 # Machine-readable run logs (query, chunks, output, prompt version)
  report/
    phase2/                        # Phase 2 evaluation report
      evaluation_report.md
    phase3/                        # Phase 3 final report
  .streamlit/
    config.toml                    # Disables auto-rerun, preserves demo state across idle
```

---

## Commands

| Target | What it does |
|--------|-------------|
| `make install` | `pip install -r requirements.txt` |
| `make download` | Download 20 PDFs from manifest URLs, validate each against expected title |
| `make ingest` | Parse PDFs, chunk (500 tokens, 100 overlap), embed with all-MiniLM-L6-v2, build FAISS index |
| `make query` | Single query: prints retrieved chunks + cited answer + saved log path |
| `make eval-baseline` | Run 20-query evaluation in baseline mode |
| `make eval-enhanced` | Run 20-query evaluation in enhanced mode |
| `make eval-both` | Run both baseline and enhanced evaluations |
| `make report` | Generate evaluation report from saved results |
| `make serve` | Start FastAPI backend on port 8000 |
| `make ui` | Start Streamlit UI on port 8501 |
| `make all` | Full pipeline: install → download → ingest → eval-both → report |
| `make clean` | Remove index, logs, outputs, reports (keep PDFs) |
| `make clean-threads` | Remove saved research threads |
| `make clean-all` | Full clean including downloaded PDFs |

---

## LLM Providers

| Provider | Model | SDK | Endpoint |
|----------|-------|-----|----------|
| Grok-3 (CMU LLM API) | `grok-3` | `openai.OpenAI` | CMU LLM API gateway |
| Azure OpenAI | `o4-mini` | `openai.AzureOpenAI` | Azure resource endpoint |

Both providers are used for RAG generation. Grok-3 also serves as the LLM-as-judge for evaluation scoring.

### .env configuration

Copy `.env.example` to `.env` and fill in your keys:

```env
LLM_PROVIDER=grok
GROK_API_KEY=<your-key>
GROK_ENDPOINT=https://cmu-llm-api-resource.services.ai.azure.com/openai/v1/
GROK_MODEL=grok-3
AZURE_API_KEY=<your-key>
AZURE_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_API_VERSION=2024-12-01-preview
AZURE_MODEL=o4-mini
LLM_MAX_RETRIES=4
LLM_BACKOFF_BASE_S=1.0
LLM_BACKOFF_MAX_S=15.0
LLM_MIN_CALL_INTERVAL_S=0.5
```

---

## Evaluation Framework

**20 queries** across three categories:

| Category | IDs | Count | Tests |
|----------|-----|:-----:|-------|
| Direct | D01–D10 | 10 | Single-source factual retrieval, citation accuracy |
| Synthesis / Multi-hop | S01–S05 | 5 | Cross-source comparison, multi-source integration |
| Edge Case | E01–E05 | 5 | Out-of-corpus detection, uncertainty handling |

**6 metrics**:

| Metric | Method | Scale | What it measures |
|--------|--------|:-----:|----------|
| Groundedness | LLM-judge | 1–4 | Are claims supported by the retrieved chunks? |
| Answer Relevance | LLM-judge | 1–4 | Does the answer address the question? |
| Context Precision | LLM-judge | 1–4 | Were retrieved chunks actually useful? |
| Citation Precision | Deterministic | 0–1 | valid_citations / total_citations |
| Source Recall | Deterministic | 0–1 | expected_sources_found / total_expected |
| Uncertainty Handling | Rule-based | Y/N | Does the answer flag missing evidence when appropriate? |

---

## Data Handling

- **20 sources**: 16 peer-reviewed papers, 2 technical reports, 2 tool/workshop papers (2016–2024)
- **PDFs are git-ignored**: Reproduced on any machine via `make download`
- **PDF validation**: First 3 pages checked against expected title (≥ 40% word match)
- **Chunking**: Section-aware sliding window, 500 tokens/chunk, 100-token overlap
- **Embeddings**: `all-MiniLM-L6-v2` (384-dim), L2-normalized, FAISS `IndexFlatIP` (cosine similarity)
- **Logging**: Every RAG run appends to `logs/rag_runs.jsonl` with query, chunks, output, prompt version
- **Threads**: Saved as JSON in `data/threads/` with full query + retrieved evidence + answer + citation validation

---

## Security

| Measure | Implementation |
|---------|---------------|
| API key isolation | Keys in `.env` (git-ignored), loaded via `python-dotenv` |
| Prompt-injection detection | `sanitize_query()` rejects 10+ injection patterns (regex-based) |
| Input length limit | Queries capped at 1000 characters |
| Input sanitization | Control characters stripped at every entry point |
| Pydantic validation | All FastAPI request bodies validated with schemas |
| Data exclusion | PDFs, processed data, logs, outputs, threads all in `.gitignore` |

## LLM Reliability and Rate Limiting

LLM calls now include production-style resilience safeguards:

- **Retry on transient failures** in `src/llm_client.py` (e.g., 429, timeouts, 5xx).
- **Exponential backoff with jitter** between retries (`LLM_BACKOFF_BASE_S`, `LLM_BACKOFF_MAX_S`).
- **Configurable retry count** via `LLM_MAX_RETRIES`.
- **Minimum call interval throttling** between consecutive requests via `LLM_MIN_CALL_INTERVAL_S`.
- **Structured logging** for retry attempts and final failures for easier debugging and TA verification.

---

## AI Usage Disclosure

| Tool | Purpose | Manual Changes |
|------|---------|----------------|
| Grok-3 (`grok-3`) | RAG answer generation, LLM-as-judge evaluation, artifact generation | Reviewed all outputs, validated citations |
| Azure OpenAI (`o4-mini`) | RAG answer generation, model comparison | Reviewed all outputs |
| sentence-transformers (`all-MiniLM-L6-v2`) | Document and query embeddings | Configured chunking and indexing parameters |
| Cursor AI | Code scaffolding and development assistance | Reviewed, tested, and modified all generated code |

---

## Dependencies

```
PyMuPDF>=1.25.0              # PDF text extraction
sentence-transformers>=3.0.0 # Embedding model
torch>=2.3.0                 # ML backend
faiss-cpu>=1.8.0             # Vector similarity search
openai>=1.40.0               # Grok-3 + Azure OpenAI clients
fastapi>=0.115.0             # REST API
uvicorn>=0.32.0              # ASGI server
streamlit>=1.38.0            # Interactive UI
fpdf2>=2.7.0                 # PDF report/artifact export
pandas>=2.2.0                # Data tables
numpy>=1.26.0                # Array operations
python-dotenv>=1.0.0         # .env loading
pydantic>=2.0.0              # Request validation
```
