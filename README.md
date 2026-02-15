# LLM Carbon Footprint Research Portal

Research-grade RAG system that answers questions about the environmental cost of large AI models
using 20 peer-reviewed academic papers. Every claim is cited; every citation is validated against
the actually-retrieved source text.

**Course**: AI Model Development (95-864) | **Group 4**: Dhiksha Rathis, Shreya Verma | CMU Spring 2026

---

## Quick Start

### macOS / Linux

```bash
git clone <repo-url> && cd llm-carbon-footprint-research-portal
python3 -m venv venv
source venv/bin/activate
make install                      # pip install -r requirements.txt
cp .env.example .env              # then edit .env with your API keys
make download && make ingest      # download PDFs, build FAISS index
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

### One-Click Alternative

Skip steps 4-5 above. Open the Streamlit UI, go to **Demo All Deliverables**, check "Fresh start",
and click **Run Complete Demo**. It downloads PDFs, validates them, builds the index, runs baseline
and enhanced RAG, model comparison, full 20-query evaluation, and generates the report automatically.

---

## Pipeline

```
User Query
  |-> Sanitize (prompt-injection check, 1000-char limit, control-char strip)
  |-> Classify (direct / synthesis / multihop / edge_case by keyword rules)
  |-> Retrieve
  |     +-- Baseline: embed query -> FAISS top-5 cosine search
  |     +-- Enhanced: rewrite + decompose into sub-queries -> multi-retrieve -> merge top-8
  |-> Generate (LLM with strict citation rules, no output token cap)
  |-> Validate (each (source_id, chunk_id) checked against retrieved set)
  |-> Log (JSONL entry appended to logs/rag_runs.jsonl)
  +-> Cited answer with reference list
```

---

## LLM Providers

| Provider | Model | SDK | Endpoint |
|----------|-------|-----|----------|
| Grok-3 (CMU LLM API) | `grok-3` | `openai.OpenAI` | `https://cmu-llm-api-resource.services.ai.azure.com/openai/v1/` |
| Azure OpenAI | `o4-mini` | `openai.AzureOpenAI` | Azure resource endpoint |

Switch providers at runtime in the Streamlit sidebar. **Compare Models** page runs both on the
same query side-by-side with latency, citation quality, and token usage.

### .env configuration

Copy `.env.example` to `.env` and fill in your keys:

```env
LLM_PROVIDER=grok                  # default provider: grok or azure_openai
GROK_API_KEY=<your-key>
GROK_ENDPOINT=https://cmu-llm-api-resource.services.ai.azure.com/openai/v1/
GROK_MODEL=grok-3
AZURE_API_KEY=<your-key>
AZURE_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_API_VERSION=2024-12-01-preview
AZURE_MODEL=o4-mini
```

All config lives in `src/config.py`, driven entirely by environment variables. Nothing is hardcoded.
See `.env.example` for the full list of tunable parameters (temperatures, chunk sizes, thresholds).

---

## Phase 2 Deliverables

| # | Deliverable | Implementation | Verified by |
|---|-------------|----------------|-------------|
| D1 | Code repository | Modular Python (`src/`), Makefile automation | Step 9 in Demo page |
| D2 | Data manifest | 20 sources with metadata, PDF validation | `data/data_manifest.csv` |
| D3 | RAG pipeline | Baseline top-5 + enhanced (rewrite, decompose, merge top-8) | `make query`, Steps 4-5 |
| D4 | Evaluation framework | 20 queries, 6 metrics, LLM-as-judge | `make eval-both`, Step 7 |
| D5 | Evaluation report | Markdown with per-query logs and failure cases | `make report`, Step 8 |
| D6 | API backend | FastAPI: `/health`, `/query`, `/corpus`, `/evaluation`, `/logs` | `make serve` |
| D7 | Interactive UI | Streamlit, 5 pages | `make ui` |
| D8 | Model comparison | Grok-3 vs Azure side-by-side with metrics table | Compare Models page, Step 6 |
| D9 | Security | Sanitization, injection detection, key isolation | Step 10 live test |

---

## Streamlit UI (5 Pages)

| Page | Purpose |
|------|---------|
| **Home** | Project overview, deliverable table, pipeline diagram, corpus table (20 papers) |
| **Ask a Question** | Chat interface - pick baseline/enhanced, top-K slider, sample questions, cited answers |
| **Compare Models** | Same query through Grok-3 and Azure side-by-side with comparison table |
| **Evaluation** | Browse the 20-query test set and scored results (baseline + enhanced) |
| **Demo All Deliverables** | One-click end-to-end: clean slate, download + validate PDFs, ingest, baseline RAG, enhanced RAG, model comparison, 20-query eval (both modes), report generation, code/security checks |

---

## Repository Structure

```
src/
  config.py                    # Centralized config (all env-driven, auto-creates dirs)
  llm_client.py                # Provider-agnostic LLM abstraction (GrokClient, AzureOpenAIClient)
  utils.py                     # Sanitization, citation regex, chunk formatting, eval loader
  ingest/
    download_sources.py        # Download PDFs + validate title against first-page text
    ingest.py                  # Parse PDFs -> section-aware chunking -> embed -> FAISS index
  rag/
    rag.py                     # Baseline RAG: retrieve, generate, validate citations, log
    enhance_query_rewriting.py # Enhanced RAG: classify, rewrite, decompose, multi-retrieve, synthesize
  eval/
    evaluation.py              # 20-query eval set, LLM-as-judge (4-layer parse), 6 metrics
    generate_report.py         # Markdown report: results, per-query logs, failure cases
  app/
    app.py                     # FastAPI backend (7 endpoints, Pydantic validation, CORS)
    streamlit_ui.py            # Streamlit UI (5 pages, cached resources, PDF export)
data/
  data_manifest.csv            # 20 sources: source_id, title, authors, year, URL, venue, tags
  raw/                         # Downloaded PDFs (git-ignored, validated after download)
  processed/                   # chunk_store.json + faiss_index.bin (git-ignored, rebuilt by ingest)
logs/                          # rag_runs.jsonl (git-ignored)
outputs/                       # eval_results_*.json (git-ignored)
report/phase2/                 # evaluation_report.md (git-ignored)
```

---

## Commands

### Makefile (macOS / Linux)

| Target | What it does |
|--------|-------------|
| `make install` | `pip install -r requirements.txt` |
| `make download` | Download 20 PDFs, validate each against manifest title |
| `make ingest` | Parse PDFs, chunk (500t/100t overlap), embed, build FAISS index |
| `make query` | Single query: prints retrieval results + cited answer + log path |
| `make eval-baseline` | Run 20-query evaluation (baseline mode) |
| `make eval-enhanced` | Run 20-query evaluation (enhanced mode) |
| `make eval-both` | Run both baseline and enhanced evaluations |
| `make report` | Generate evaluation report from saved results |
| `make serve` | Start FastAPI backend on port 8000 |
| `make ui` | Start Streamlit UI on port 8501 |
| `make all` | Full pipeline: install, download, ingest, eval-both, report |
| `make clean` | Remove index, logs, outputs, report (keep PDFs) |
| `make clean-all` | Also remove downloaded PDFs (full fresh start) |

### Windows equivalents (PowerShell)

```powershell
# Install
pip install -r requirements.txt

# Download + ingest
python -m src.ingest.download_sources
python -m src.ingest.ingest

# Single query
python -m src.rag.rag --query "What are the major sources of carbon emissions in LLM training?"

# Evaluation
python -m src.eval.evaluation --mode baseline
python -m src.eval.evaluation --mode enhanced
python -m src.eval.evaluation --mode both

# Report
python -m src.eval.generate_report

# Servers
python -m uvicorn src.app.app:app --host 0.0.0.0 --port 8000 --reload
python -m streamlit run src/app/streamlit_ui.py --server.port 8501

# Clean (PowerShell)
Remove-Item -Recurse -Force data/processed, logs, outputs, report/phase2 -ErrorAction SilentlyContinue
Remove-Item -Force data/raw/*.pdf -ErrorAction SilentlyContinue   # full clean
```

---

## Evaluation Framework

**20 queries** across three categories:

| Category | IDs | Count | Tests |
|----------|-----|:-----:|-------|
| Direct | D01-D10 | 10 | Single-source factual retrieval, citation accuracy |
| Synthesis | S01-S05 | 5 | Cross-source comparison, multi-source integration |
| Edge Case | E01-E05 | 5 | Out-of-corpus detection, uncertainty handling |

**6 metrics**:

| Metric | Method | Scale | Measures |
|--------|--------|:-----:|----------|
| Groundedness | LLM-judge | 1-4 | Are claims supported by retrieved chunks? |
| Answer Relevance | LLM-judge | 1-4 | Does the answer address the question? |
| Context Precision | LLM-judge | 1-4 | Were retrieved chunks actually useful? |
| Citation Precision | Deterministic | 0-1 | valid_citations / total_citations |
| Source Recall | Deterministic | 0-1 | expected_sources_found / total_expected |
| Uncertainty Handling | Rule-based | Y/N | Does answer flag missing evidence? |

**Thresholds**: PASS >= 3.5, WARN >= 2.5, FAIL < 2.5

**Evaluation report** (`report/phase2/evaluation_report.md`) contains:
1. System overview (corpus, chunking, embeddings, models)
2. Query set design (full 20-query listing with expected sources)
3. Metric definitions and thresholds
4. Results (baseline + enhanced score tables, by-type breakdown, enhancement delta)
5. Per-query detail logs (query, chunks, sources, judge reasoning, answer excerpts)
6. Representative failure cases (min 3, auto-detected with evidence and root cause)
7. Reproducibility commands

---

## Prompts

All prompts are module-level constants, independently tunable.

| Prompt | Location | Purpose |
|--------|----------|---------|
| Baseline System | `rag.py:SYSTEM_PROMPT` | Strict citation rules, no fabrication, flag missing evidence |
| Decompose | `enhance_query_rewriting.py:_DECOMPOSE_INSTRUCTION` | Break complex queries into 2-4 sub-queries (JSON output) |
| Rewrite | `enhance_query_rewriting.py:_REWRITE_INSTRUCTION` | Optimize query for academic retrieval (< 20 words) |
| Synthesis | `enhance_query_rewriting.py:_SYNTHESIS_INSTRUCTION` | Merge multi-source evidence with conflict flagging |
| Judge: Groundedness | `evaluation.py:score_groundedness` | Score 1-4 + reasoning + unsupported_claims list |
| Judge: Relevance | `evaluation.py:score_answer_relevance` | Score 1-4 + reasoning |
| Judge: Context Precision | `evaluation.py:score_context_precision` | Score 1-4 + reasoning |

**Design choices**:
- `temperature=0.0` for judges (consistency), `0.2` for generation (slight creativity)
- Strict `(source_id, chunk_id)` citation format enables deterministic validation
- "Output ONLY JSON" enforced by 4-layer fallback parser in `_judge()`
- No output token cap on generation calls (judges/decompose/rewrite have specific limits)

---

## Data Handling

- **20 sources**: 14 peer-reviewed papers, 3 technical reports, 3 tool/workshop papers
- **PDFs are git-ignored**: Downloaded on first run via `make download` or Demo page
- **PDF validation**: After download, first 3 pages are checked against expected title (>= 40% word match). Mismatched PDFs are deleted and flagged
- **Fresh ingestion**: `make ingest` clears `data/processed/` and rebuilds from scratch every time
- **Chunk quality filter**: Chunks < 50 characters are discarded during ingestion
- **Chunking**: Section-aware sliding window, 500 tokens/chunk, 100-token overlap, `WORDS_PER_TOKEN=0.75`
- **Embeddings**: `all-MiniLM-L6-v2` (384-dim), L2-normalized, FAISS `IndexFlatIP` (cosine similarity)
- **Logging**: Every RAG run appends a structured JSON entry to `logs/rag_runs.jsonl`

---

## Security

| Measure | Implementation |
|---------|---------------|
| API key isolation | Keys in `.env` (git-ignored), loaded via `python-dotenv`, never in source |
| Prompt-injection detection | `sanitize_query()` rejects 10+ injection patterns (regex-based) |
| Input length limit | Queries capped at 1000 characters |
| Input sanitization | Control characters stripped at every entry point |
| Pydantic validation | All FastAPI request bodies validated with schemas |
| PDF/data exclusion | `data/raw/*.pdf`, `data/processed/`, `logs/`, `outputs/`, `report/` all in `.gitignore` |

---

## Acceptance Tests

Quick checks a grader can run:

1. **Single command produces retrieval + answer + log**:
   `make query` (or `python -m src.rag.rag --query "..."`) prints three labeled sections:
   `--- RETRIEVAL RESULTS ---`, `--- ANSWER WITH CITATIONS ---`, `--- LOG ENTRY SAVED ---`

2. **Citations resolve to real source text**:
   `validate_citations()` checks every `(source_id, chunk_id)` against retrieved chunks.
   Output shows `Citations: X/Y valid | precision=Z`. Invalid citations are logged.

3. **Evaluation report has 3+ failure cases with evidence**:
   `make report` generates `report/phase2/evaluation_report.md` with auto-detected failures
   (low groundedness, hallucinated citations, missed sources, edge-case mishandling) plus
   weakest-scoring near-misses as fallback. Each includes query, scores, judge reasoning,
   sources, answer excerpt, and root cause.

---

## AI Usage Disclosure

| Tool | Purpose |
|------|---------|
| Grok-3 (`grok-3`) | RAG answer generation, LLM-as-judge evaluation scoring |
| Azure OpenAI (`o4-mini`) | RAG answer generation, model comparison |
| sentence-transformers (`all-MiniLM-L6-v2`) | Document and query embeddings |
| Cursor AI | Code scaffolding and development assistance |

---

## Dependencies

```
PyMuPDF>=1.25.0          # PDF text extraction
sentence-transformers>=3.0.0  # Embedding model
torch>=2.3.0             # ML backend
faiss-cpu>=1.8.0         # Vector similarity search
openai>=1.40.0           # Grok-3 + Azure OpenAI clients
fastapi>=0.115.0         # REST API
uvicorn>=0.32.0          # ASGI server
streamlit>=1.38.0        # Interactive UI
fpdf2>=2.7.0             # PDF report export
pandas>=2.2.0            # Data tables
numpy>=1.26.0            # Array operations
python-dotenv>=1.0.0     # .env loading
```
