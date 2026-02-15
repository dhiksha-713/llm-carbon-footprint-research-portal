# LLM Carbon Footprint Research Portal

A research-grade RAG system for systematic review of carbon emissions in Large Language Models.
Supports **multi-provider LLM comparison** (Google Gemini and Azure OpenAI).

**Course**: AI Model Development (95-864) | **Group 4**: Dhiksha Rathis, Shreya Verma | CMU Spring 2026

---

## Quick Start

```bash

Mac / Linux Setup
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
make install
# (or: pip install -r requirements.txt)

# Configure (edit .env with your API keys)
cp .env.example .env

# Download + ingest corpus
make download && make ingest
# (or: python -m src.ingest.download_sources && python -m src.ingest.ingest)

# Launch UI
make ui
# http://localhost:8501

Windows Setup
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure (edit .env with your API keys)
copy .env.example .env

# Download + ingest corpus
python -m src.ingest.download_sources
python -m src.ingest.ingest

# Launch UI
streamlit run src/app/streamlit_ui.py --server.port 8501
# http://localhost:8501


```

---

## What is this?

A **research assistant** that reads 20 peer-reviewed papers about AI carbon emissions and answers
your questions using only those sources. Every claim is cited; every citation is validated.

| Phase | What we did |
|---|---|
| **Phase 1** | Research design: question, 20 papers, prompt kit, evaluation plan |
| **Phase 2** (this) | Working system: RAG pipeline, evaluation framework, two-model comparison, API, UI |

### How it works

1. **Collect** - 20 academic PDFs downloaded from arXiv
2. **Chunk** - Section-aware splitting (~500 tokens, 100-token overlap)
3. **Index** - Embedded with `all-MiniLM-L6-v2`, stored in FAISS
4. **Ask** - Your question is embedded; top-K similar chunks retrieved
5. **Answer** - LLM (Gemini or Azure) writes a cited answer from retrieved chunks
6. **Verify** - Every citation validated against what was actually retrieved

---

## LLM Providers

| Provider | Model | Token Limits |
|---|---|---|
| Google Gemini | `gemini-2.0-flash-lite` | 1M input / 8K output |
| Azure OpenAI | `o4-mini` | ~100K context |

Switch providers dynamically in the Streamlit sidebar. The **Compare Models** page runs both simultaneously.

### .env configuration

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-2.0-flash-lite
AZURE_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_API_KEY=your_key
AZURE_MODEL=o4-mini
```

Each LLM client automatically uses its own model name from config. No cross-provider conflicts.

---

## Streamlit UI (5 pages)

| Page | What it does |
|---|---|
| **Home** | Overview, deliverables, corpus, provider status |
| **Ask a Question** | Chat interface with cited answers |
| **Compare Models** | Same query through both providers side-by-side |
| **Evaluation** | 20-query test set and scoring results |
| **Demo All Deliverables** | One-click full demo of all D1-D9 deliverables |

---

## Architecture

```
User Query -> [Sanitize] -> [Classify] -> [Retrieve (FAISS)]
                                              |
                              +-- Baseline: top-K
                              +-- Enhanced:  rewrite + decompose + multi-retrieve + merge
                                              |
                                              v
                                        [LLM Generate] -> [Validate Citations] -> Answer
```

---

## Repository Structure

```
src/
  config.py                 # All constants, env-driven
  llm_client.py             # Provider-agnostic LLM abstraction
  utils.py                  # Sanitization, shared helpers (CITE_RE, build_chunk_context, safe_avg)
  ingest/
    download_sources.py     # PDF downloader
    ingest.py               # Parse -> chunk -> embed -> FAISS
  rag/
    rag.py                  # Baseline RAG pipeline
    enhance_query_rewriting.py  # Enhanced RAG (rewrite + decompose)
  eval/
    evaluation.py           # 20-query eval set + LLM-as-judge
    generate_report.py      # Markdown report generator
  app/
    app.py                  # FastAPI backend (5 endpoints)
    streamlit_ui.py         # Streamlit interactive UI
```

---

## Makefile Targets

| Target | Description |
|---|---|
| `make install` | Install dependencies |
| `make download` | Download corpus PDFs |
| `make ingest` | Parse, chunk, embed, build FAISS index |
| `make eval-both` | Evaluate baseline + enhanced pipelines |
| `make report` | Generate evaluation report |
| `make serve` | Start FastAPI backend (port 8000) |
| `make ui` | Start Streamlit UI (port 8501) |
| `make all` | Full pipeline: install -> download -> ingest -> eval -> report |

---

## Evaluation

20 queries (10 direct, 5 synthesis, 5 edge-case) scored on 6 metrics:

| Metric | Type | Scale |
|---|---|---|
| Groundedness | LLM-judge | 1-4 |
| Answer Relevance | LLM-judge | 1-4 |
| Context Precision | LLM-judge | 1-4 |
| Citation Precision | Deterministic | 0-1 |
| Source Recall | Deterministic | 0-1 |
| Uncertainty Handling | Rule-based | boolean |

---

## Security

- **API key isolation**: Keys in `.env` (git-ignored), never in source code
- **Prompt-injection protection**: `sanitize_query()` at every entry point
- **Input validation**: Pydantic schemas on all API requests
- **PDF exclusion**: Downloaded PDFs are git-ignored
