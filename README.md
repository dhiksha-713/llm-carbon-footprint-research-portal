# LLM Carbon Footprint Research Portal

**Personal Research Portal (PRP)** that helps you move from a research question to a grounded synthesis on the
environmental cost of large AI models. It ingests 20 peer-reviewed papers, retrieves evidence via semantic search,
produces citation-backed answers, generates exportable research artifacts, and logs evaluation results.

**Course**: AI Model Development (95-864) | **Group 4**: Dhiksha Rathis, Shreya Verma | CMU Spring 2026

---

## Quick Start (5 minutes)

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

---

## What This Portal Does (MVP Capabilities)

1. **Ingest a domain corpus** (20 PDFs) with metadata (author, year, source type, link/DOI)
2. **Retrieve evidence** via semantic retrieval (FAISS + sentence-transformers)
3. **Generate answers** where every major claim is backed by an inline citation `(source_id, chunk_id)`
4. **Produce research artifacts**: evidence tables, annotated bibliographies, synthesis memos
5. **Save research threads** (query + retrieved chunks + answer) and export outputs (Markdown/CSV/PDF)
6. **Run evaluation** and report groundedness/faithfulness plus additional metrics
7. **Trust behavior**: refuse to fabricate citations; flag missing evidence with suggested next steps

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

### Streamlit UI (7 Pages)

| Page | Purpose |
|------|---------|
| **Home** | Portal overview, metrics dashboard, quick start guide |
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
  |-> Sanitize (prompt-injection check, 1000-char limit, control-char strip)
  |-> Classify (direct / synthesis / multihop / edge_case by keyword rules)
  |-> Retrieve
  |     +-- Baseline: embed query -> FAISS top-5 cosine search
  |     +-- Enhanced: rewrite + decompose into sub-queries -> multi-retrieve -> merge top-8
  |-> Generate (LLM with strict citation rules, no output token cap)
  |-> Validate (each (source_id, chunk_id) checked against retrieved set)
  |-> Trust Check (flag missing evidence, suggest next steps)
  |-> Save Thread (query + evidence + answer persisted to data/threads/)
  |-> [Optional] Generate Artifact (evidence table / bibliography / synthesis memo)
  +-> Cited answer with reference list + export options
```

---

## Research Artifacts

The portal generates three types of research artifacts:

| Artifact | Schema | Output |
|----------|--------|--------|
| **Evidence Table** | Claim \| Evidence Snippet \| Citation \| Confidence \| Notes | Markdown + CSV |
| **Annotated Bibliography** | Citation \| Key Claim \| Method \| Limitations \| Relevance | Markdown |
| **Synthesis Memo** | 800-1200 words with inline citations + reference list | Markdown |

All artifacts are saved to `outputs/artifacts/` and can be exported as Markdown, CSV, or PDF.

---

## Repository Structure

```
repo/
  README.md
  requirements.txt
  Makefile
  .env.example
  data/
    raw/                         # Downloaded PDFs (git-ignored)
    processed/                   # chunk_store.json, faiss_index.bin (git-ignored)
    threads/                     # Saved research threads (git-ignored)
    data_manifest.csv            # 20 sources with full metadata
  src/
    config.py                    # Centralized config (all env-driven)
    llm_client.py                # Provider-agnostic LLM abstraction
    utils.py                     # Sanitization, citation regex, trust behavior
    threads.py                   # Research thread save/load/list/delete/export
    artifacts.py                 # Artifact generation (evidence table, bibliography, memo)
    ingest/
      download_sources.py        # Download PDFs + validate title
      ingest.py                  # Parse -> chunk -> embed -> FAISS index
    rag/
      rag.py                     # Baseline RAG: retrieve, generate, validate, log
      enhance_query_rewriting.py # Enhanced RAG: classify, rewrite, decompose, synthesize
    eval/
      evaluation.py              # 20-query eval set, LLM-as-judge, 6 metrics
      generate_report.py         # Markdown report with failure cases
    app/
      app.py                     # FastAPI backend (7 endpoints)
      streamlit_ui.py            # Streamlit UI (7 pages)
  outputs/
    artifacts/                   # Generated research artifacts (git-ignored)
    eval_results_*.json          # Evaluation results (git-ignored)
  logs/
    rag_runs.jsonl               # Run logs (git-ignored)
  report/
    phase2/                      # Phase 2 evaluation report
    phase3/                      # Phase 3 report
```

---

## Commands

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
| `make clean` | Remove index, logs, outputs, reports (keep PDFs) |
| `make clean-threads` | Remove saved research threads |
| `make clean-all` | Full clean including PDFs |

---

## LLM Providers

| Provider | Model | SDK | Endpoint |
|----------|-------|-----|----------|
| Grok-3 (CMU LLM API) | `grok-3` | `openai.OpenAI` | `https://cmu-llm-api-resource.services.ai.azure.com/openai/v1/` |
| Azure OpenAI | `o4-mini` | `openai.AzureOpenAI` | Azure resource endpoint |

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

---

## Data Handling

- **20 sources**: 14 peer-reviewed papers, 3 technical reports, 3 tool/workshop papers
- **PDFs are git-ignored**: Downloaded on first run via `make download`
- **PDF validation**: First 3 pages checked against expected title (>= 40% word match)
- **Chunking**: Section-aware sliding window, 500 tokens/chunk, 100-token overlap
- **Embeddings**: `all-MiniLM-L6-v2` (384-dim), L2-normalized, FAISS `IndexFlatIP`
- **Logging**: Every RAG run appends to `logs/rag_runs.jsonl`
- **Threads**: Saved as JSON in `data/threads/` with full query + evidence + answer

---

## Security

| Measure | Implementation |
|---------|---------------|
| API key isolation | Keys in `.env` (git-ignored), loaded via `python-dotenv` |
| Prompt-injection detection | `sanitize_query()` rejects 10+ injection patterns |
| Input length limit | Queries capped at 1000 characters |
| Input sanitization | Control characters stripped at every entry point |
| Pydantic validation | All FastAPI request bodies validated with schemas |
| Data exclusion | PDFs, processed data, logs, outputs, threads all in `.gitignore` |

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
pydantic>=2.0.0          # Request validation
```
