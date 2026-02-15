# LLM Carbon Footprint Research Portal

Research-grade RAG system that answers questions about AI carbon emissions using 20 peer-reviewed papers.
Every claim is cited; every citation is validated against retrieved source text.

**Course**: AI Model Development (95-864) | **Group 4**: Dhiksha Rathis, Shreya Verma | CMU Spring 2026

---

## Quick Start

```bash
# 1. Clone and create virtual environment
git clone <repo-url> && cd llm-carbon-footprint-research-portal
python3 -m venv venv && source venv/bin/activate

# 2. Install dependencies
make install

# 3. Configure API keys
cp .env.example .env   # then edit .env with your keys

# 4. Build the knowledge base (download PDFs, chunk, embed, index)
make download && make ingest

# 5. Launch the interactive UI
make ui                # opens http://localhost:8501

# 6. Or run a single query from the command line
make query             # produces retrieval results + cited answer + saved log entry
```

The Streamlit **Demo All Deliverables** page automates steps 4-6 in one click (auto-downloads PDFs and builds the index if missing).

---

## Pipeline

```
User Query
  |-> Sanitize (prompt-injection check, length limit)
  |-> Classify (direct / synthesis / multihop / edge-case)
  |-> Retrieve
  |     +-- Baseline: embed query -> FAISS top-K
  |     +-- Enhanced: rewrite query + decompose into sub-queries -> multi-retrieve -> merge top-N
  |-> Generate (LLM with strict citation rules)
  |-> Validate (each (source_id, chunk_id) checked against retrieved set)
  |-> Log (appended to logs/rag_runs.jsonl)
  +-> Answer with inline citations + reference list
```

---

## LLM Providers

| Provider | Model | SDK | Endpoint |
|----------|-------|-----|----------|
| Grok-3 (CMU LLM API) | `grok-3` | `openai.OpenAI` | `https://cmu-llm-api-resource.services.ai.azure.com/openai/v1/` |
| Azure OpenAI | `o4-mini` | `openai.AzureOpenAI` | Azure resource endpoint |

Providers are switchable at runtime via the Streamlit sidebar. The **Compare Models** page runs both simultaneously on the same query and displays answers side-by-side with latency, citation quality, and token usage.

### .env configuration

Copy `.env.example` to `.env` and fill in your keys:

```env
LLM_PROVIDER=grok              # default provider (grok or azure_openai)
GROK_API_KEY=<your-key>
GROK_ENDPOINT=https://cmu-llm-api-resource.services.ai.azure.com/openai/v1/
GROK_MODEL=grok-3
AZURE_API_KEY=<your-key>
AZURE_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_MODEL=o4-mini
AZURE_API_VERSION=2024-12-01-preview
```

All configuration lives in `src/config.py`, driven entirely by environment variables. Nothing is hardcoded.

---

## Phase 2 Deliverables

| # | Deliverable | Implementation | Verified by |
|---|-------------|----------------|-------------|
| D1 | Code repository | Modular Python (`src/`), Makefile automation | File checks in Demo page |
| D2 | Data manifest | 20 sources with metadata | `data/data_manifest.csv` |
| D3 | RAG pipeline | Baseline top-K + enhanced (rewrite, decompose, merge) | `make query`, Demo page |
| D4 | Evaluation framework | 20 queries, 6 metrics, LLM-as-judge | `make eval-both` |
| D5 | Evaluation report | Auto-generated Markdown with failure cases | `make report` |
| D6 | API backend | FastAPI: `/health`, `/query`, `/corpus`, `/evaluation`, `/logs` | `make serve` |
| D7 | Interactive UI | Streamlit, 5 pages | `make ui` |
| D8 | Model comparison | Grok-3 vs Azure side-by-side with metrics | Compare Models page |
| D9 | Security | Sanitization, injection detection, key isolation | Demo page live test |

---

## Streamlit UI

| Page | Purpose |
|------|---------|
| **Home** | Project overview, deliverable checklist, corpus table, pipeline explanation |
| **Ask a Question** | Chat interface - pick baseline/enhanced mode, top-K slider, cited answers |
| **Compare Models** | Same query through Grok-3 and Azure side-by-side with comparison table |
| **Evaluation** | Browse the 20-query test set and all scored results |
| **Demo All Deliverables** | One-click: downloads PDFs if missing, builds index, runs baseline + enhanced RAG, model comparison, full 20-query evaluation (both modes), generates report with download |

---

## Repository Structure

```
src/
  config.py                    # Centralized configuration (all env-driven)
  llm_client.py                # Provider-agnostic LLM abstraction
  utils.py                     # Sanitization, citation regex, chunk formatting, averages
  ingest/
    download_sources.py        # Download + validate PDFs (checks title against first page)
    ingest.py                  # Parse PDFs -> section-aware chunking -> embed -> FAISS index
  rag/
    rag.py                     # Baseline RAG: retrieve, generate, validate citations, log
    enhance_query_rewriting.py # Enhanced RAG: rewrite + decompose + multi-retrieve + synthesize
  eval/
    evaluation.py              # 20-query eval set, LLM-as-judge scoring, metrics
    generate_report.py         # Comprehensive Markdown report with failure analysis
  app/
    app.py                     # FastAPI backend (5 endpoints)
    streamlit_ui.py            # Streamlit interactive UI (5 pages)
data/
  data_manifest.csv            # 20-source metadata (source_id, title, year, URL, venue)
  raw/                         # Downloaded PDFs (git-ignored)
  processed/                   # FAISS index + chunk store (git-ignored, rebuilt by make ingest)
logs/                          # Run logs in JSONL (git-ignored)
outputs/                       # Evaluation result JSONs (git-ignored)
report/phase2/                 # Generated evaluation report (git-ignored)
```

---

## Makefile Targets

| Target | What it does |
|--------|-------------|
| `make install` | `pip install -r requirements.txt` |
| `make download` | Download 20 PDFs from manifest URLs |
| `make ingest` | Parse, chunk, embed, build FAISS index (fresh each run) |
| `make query` | Single query: prints retrieval results + cited answer + log path |
| `make eval-baseline` | Run 20-query evaluation (baseline mode) |
| `make eval-enhanced` | Run 20-query evaluation (enhanced mode) |
| `make eval-both` | Run both baseline and enhanced evaluations |
| `make report` | Generate evaluation report from saved results |
| `make serve` | Start FastAPI backend on port 8000 |
| `make ui` | Start Streamlit UI on port 8501 |
| `make all` | Full pipeline: install, download, ingest, eval-both, report |
| `make clean` | Remove generated artifacts (index, logs, outputs, report) |
| `make clean-all` | Also remove downloaded PDFs - next run re-downloads everything |

---

## Evaluation Framework

**20 queries** across three categories:

| Category | IDs | Count | Tests |
|----------|-----|:-----:|-------|
| Direct | D01-D10 | 10 | Single-source factual retrieval, basic grounding and citation |
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

**Evaluation report** (`report/phase2/evaluation_report.md`) includes: system overview, query set design, metric definitions, baseline vs enhanced results with delta analysis, per-query detail logs (query, chunks, sources, judge reasoning, answer excerpts), and at least 3 representative failure cases with evidence and root cause analysis.

---

## Prompts

All prompts are module-level constants. Each has a single responsibility and is independently tunable.

| Prompt | Location | Purpose | Key rules |
|--------|----------|---------|-----------|
| **Baseline System** | `rag.py:SYSTEM_PROMPT` | Generation with citation rules | Use ONLY context; cite as `(source_id, chunk_id)`; flag missing evidence; preserve hedging; end with reference list |
| **Decompose** | `enhance_query_rewriting.py:_DECOMPOSE_INSTRUCTION` | Break complex queries into 2-4 sub-queries | Output ONLY JSON array of strings |
| **Rewrite** | `enhance_query_rewriting.py:_REWRITE_INSTRUCTION` | Optimize query for academic retrieval | Domain-specific terms, under 20 words |
| **Synthesis** | `enhance_query_rewriting.py:_SYNTHESIS_INSTRUCTION` | Merge multi-source evidence | Same citation rules as baseline + explicit conflict flagging `[CONFLICT: ...]` |
| **Judge: Groundedness** | `evaluation.py:score_groundedness` | Score answer grounding (1-4) | Returns JSON with score, reasoning, unsupported_claims list |
| **Judge: Relevance** | `evaluation.py:score_answer_relevance` | Score answer relevance (1-4) | Returns JSON with score, reasoning |
| **Judge: Context Precision** | `evaluation.py:score_context_precision` | Score retrieval usefulness (1-4) | Returns JSON with score, reasoning |

**Design choices**:
- `temperature=0.0` for judges (scoring consistency) vs `0.2` for generation (slight creativity)
- Strict `(source_id, chunk_id)` citation format enables deterministic validation
- "Output ONLY JSON" enforced by a 4-layer fallback parser (direct parse, regex JSON extract, regex score extract, plain-text fallback)
- Separate decompose/rewrite/synthesize prompts so each step can be tuned independently

---

## Data Handling

- **20 sources**: 14 peer-reviewed papers, 3 technical reports, 3 tool/workshop papers (listed in `data/data_manifest.csv`)
- **PDFs are git-ignored**: Downloaded on first run via `make download` or automatically by the Demo page
- **PDF validation**: After download, each PDF's first-page text is checked against the expected title from the manifest. Mismatched PDFs (wrong paper) are deleted and flagged
- **Fresh ingestion**: `make ingest` clears `data/processed/` and rebuilds from scratch every time
- **Chunk quality filter**: Chunks shorter than 50 characters are discarded during ingestion
- **Chunking**: Section-aware sliding window, 500 tokens per chunk, 100-token overlap
- **Embeddings**: `all-MiniLM-L6-v2` (384-dim), L2-normalized, stored in FAISS `IndexFlatIP` (cosine similarity)
- **Logging**: Every RAG run appends a structured JSON entry to `logs/rag_runs.jsonl` with run_id, timestamp, query, retrieved chunks, answer, citations, and validation results

---

## Security

| Measure | Implementation |
|---------|---------------|
| API key isolation | Keys in `.env` (git-ignored), loaded via `python-dotenv`, never in source |
| Prompt-injection detection | `sanitize_query()` rejects patterns like "ignore previous instructions" |
| Input length limit | Queries capped at 1000 characters |
| Input sanitization | Control characters stripped at every entry point |
| Pydantic validation | All FastAPI request bodies validated with schemas |
| PDF exclusion | `data/raw/*.pdf` and all generated artifacts in `.gitignore` |

---

## Acceptance Tests

These are the checks a grader can run quickly:

1. **Single command produces retrieval + answer + log**:
   `make query` prints `--- RETRIEVAL RESULTS ---`, `--- ANSWER WITH CITATIONS ---`, and `--- LOG ENTRY SAVED ---` with the file path.

2. **Citations resolve to real source text**:
   Every `(source_id, chunk_id)` in the answer is validated by `validate_citations()`. The output shows `Citations: X/Y valid | precision=Z`. Invalid citations are logged.

3. **Evaluation report has 3+ failure cases with evidence**:
   `make report` generates `report/phase2/evaluation_report.md` with auto-detected failures (low groundedness, hallucinated citations, missed sources, edge-case mishandling) plus weakest-scoring near-misses as fallback. Each includes: query, issues, all scores, judge reasoning, sources, answer excerpt, and root cause.

---

## AI Usage Disclosure

| Tool | Purpose |
|------|---------|
| Grok-3 (`grok-3`) | RAG answer generation, LLM-as-judge evaluation scoring |
| Azure OpenAI (`o4-mini`) | RAG answer generation, model comparison |
| sentence-transformers (`all-MiniLM-L6-v2`) | Document and query embeddings |
| Cursor AI | Code scaffolding and development assistance |
