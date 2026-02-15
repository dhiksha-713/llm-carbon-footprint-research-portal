# LLM Carbon Footprint Research Portal

A research-grade RAG system for systematic review of carbon emissions in Large Language Models.  
Supports **multi-provider LLM comparison** (Google Gemini and Azure OpenAI o4-mini).

**Course**: AI Model Development (95-864) | **Group 4**: Dhiksha Rathis, Shreya Verma | CMU Spring 2026

---

## What is this?

This is a **research assistant** that reads 20 peer-reviewed academic papers about the environmental
impact of AI models and answers your questions using only those sources. Think of it like a smart
librarian that quotes every source and never makes things up.

### What is Phase 2?

| Phase | What we did |
|---|---|
| **Phase 1** (completed) | Designed the research question, selected 20 papers, created the prompt kit and evaluation plan. |
| **Phase 2** (this) | Built the complete working system: RAG pipeline, evaluation framework, two-model comparison, API, and interactive UI. |

### How does it work?

1. **Collect** - 20 academic PDFs about AI carbon footprints are downloaded from arXiv.
2. **Chunk** - Each PDF is split into ~500-word overlapping text pieces.
3. **Index** - Each piece is embedded into a vector and stored in a FAISS search index.
4. **Ask** - You type a question. The system finds the most relevant pieces.
5. **Answer** - An LLM (your choice: Gemini or Azure OpenAI) reads the pieces and writes a cited answer.
6. **Verify** - Every citation is validated against what was actually retrieved.

### Choosing your LLM provider

When you open the Streamlit UI, **pick your provider from the sidebar** before doing anything.
Both providers are pre-configured:

- **Google Gemini** (`gemini-2.5-pro`) - 1M input / 65K output token limits
- **Azure OpenAI** (`o4-mini`) - 100K context window, reasoning-focused model

The **Compare Models** page runs the same query through both simultaneously for side-by-side comparison.

---

## Quick Start

### 1. Clone and create virtual environment

```bash
git clone <repo-url>
cd llm-carbon-footprint-research-portal
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
make install
# or: pip install -r requirements.txt
```

### 3. Configure API keys

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

**Required fields in `.env`:**

| Variable | Description |
|---|---|
| `LLM_PROVIDER` | `gemini` or `azure_openai` (default provider for CLI/API) |
| `GEMINI_API_KEY` | Google Gemini API key |
| `GEMINI_MODEL` | Gemini model name (default: `gemini-2.5-pro`) |
| `AZURE_ENDPOINT` | Azure OpenAI endpoint URL |
| `AZURE_API_KEY` | Azure OpenAI API key |
| `AZURE_API_VERSION` | Azure API version (default: `2024-12-01-preview`) |
| `AZURE_MODEL` | Azure deployment name (default: `o4-mini`) |

> **Note**: The Streamlit UI lets you switch providers dynamically via the sidebar - no restart needed.  
> **Security**: `.env` is in `.gitignore` and is never committed. API keys never appear in source code.

### 4. Download corpus PDFs

```bash
make download
```

Downloads 20 peer-reviewed papers from arXiv and ACL Anthology into `data/raw/`.  
PDFs are git-ignored and stored locally only.

### 5. Ingest (parse, chunk, embed, index)

```bash
make ingest
```

Parses PDFs with PyMuPDF, creates section-aware chunks (500t/100t overlap), embeds with `all-MiniLM-L6-v2`, and builds a FAISS index.

### 6. Launch the application

**Streamlit UI** (recommended for demo):

```bash
make ui
# Opens at http://localhost:8501
```

**FastAPI backend** (for programmatic access):

```bash
make serve
# Opens at http://localhost:8000/docs
```

### 7. Run evaluation

```bash
make eval-both    # Run 20-query eval on both baseline and enhanced pipelines
make report       # Generate evaluation report
```

---

## Navigating the Streamlit UI

The UI has 6 pages accessible from the sidebar:

| Page | What it does |
|---|---|
| **Home** | Overview of the project, Phase 2 deliverables, corpus details, and provider status |
| **Run Pipeline** | Executes the full RAG system step-by-step with real-time progress indicators |
| **Ask a Question** | Chat interface to ask any question, with cited answers from the corpus |
| **Compare Models** | Run the same query through Gemini and Azure OpenAI side-by-side |
| **Evaluation** | View the 20-query test set, metric definitions, and scoring results |
| **Demo All Deliverables** | **One-click button** that runs everything (pipeline, comparison, eval, security check) and shows all D1-D9 deliverables |

The **sidebar** also has a **provider selector** that dynamically switches between Gemini and Azure OpenAI for all pages.

---

## Architecture

```
User Query
    |
    v
[Input Sanitization] - prompt-injection protection
    |
    v
[Query Classification] - direct / synthesis / multihop / edge_case
    |
    +-- Baseline: top-K semantic retrieval from FAISS
    |
    +-- Enhanced: query rewriting + decomposition + multi-sub-query retrieval + merge
    |
    v
[LLM Generation] - Gemini or Azure OpenAI (configurable via sidebar / LLM_PROVIDER)
    |
    v
[Citation Validation] - verify every (source_id, chunk_id) against retrieved chunks
    |
    v
[Answer + Reference List]
```

---

## Repository Structure

```
llm-carbon-footprint-research-portal/
├── .env.example              # Template for API keys and config
├── .gitignore                # Excludes .env, PDFs, processed data, venv
├── Makefile                  # Build automation (install, download, ingest, eval, serve, ui)
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── data/
│   ├── data_manifest.csv     # 20-source corpus manifest (A3 schema)
│   ├── raw/                  # Downloaded PDFs (git-ignored)
│   └── processed/            # FAISS index + chunk store (git-ignored)
├── src/
│   ├── config.py             # Centralized env-driven configuration
│   ├── llm_client.py         # Provider-agnostic LLM abstraction (Gemini + Azure)
│   ├── utils.py              # Input sanitization, prompt-injection protection
│   ├── ingest/
│   │   ├── download_sources.py   # PDF downloader (arXiv, ACL Anthology)
│   │   └── ingest.py             # PDF parse -> chunk -> embed -> FAISS index
│   ├── rag/
│   │   ├── rag.py                # Baseline RAG pipeline
│   │   └── enhance_query_rewriting.py  # Enhanced RAG (rewrite + decompose)
│   ├── eval/
│   │   ├── evaluation.py         # 20-query eval set + LLM-as-judge scoring
│   │   └── generate_report.py    # Markdown report generator
│   └── app/
│       ├── app.py                # FastAPI backend
│       └── streamlit_ui.py       # Streamlit interactive UI
├── report/phase2/            # Generated evaluation report
├── logs/                     # RAG run logs (git-ignored)
└── outputs/                  # Evaluation results (git-ignored)
```

---

## Makefile Targets

| Target | Command | Description |
|---|---|---|
| `make install` | `pip install -r requirements.txt` | Install all dependencies |
| `make download` | `python -m src.ingest.download_sources` | Download corpus PDFs |
| `make ingest` | `python -m src.ingest.ingest` | Parse, chunk, embed, build FAISS index |
| `make query` | `python -m src.rag.rag --query "..."` | Run a single RAG query (CLI) |
| `make eval-baseline` | `python -m src.eval.evaluation --mode baseline` | Evaluate baseline pipeline |
| `make eval-enhanced` | `python -m src.eval.evaluation --mode enhanced` | Evaluate enhanced pipeline |
| `make eval-both` | `python -m src.eval.evaluation --mode both` | Evaluate both pipelines |
| `make report` | `python -m src.eval.generate_report` | Generate evaluation report |
| `make serve` | `uvicorn src.app.app:app` | Start FastAPI backend |
| `make ui` | `streamlit run src/app/streamlit_ui.py` | Start Streamlit UI |
| `make clean` | Remove generated artifacts | Clean processed data, logs, outputs |
| `make all` | Full pipeline | install -> download -> ingest -> eval -> report |

---

## LLM Provider Configuration

The system supports two LLM providers. Each provider has its own model name in `.env`:

### Google Gemini

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-pro
```

Token limits: 1,048,576 input / 65,536 output.

### Azure OpenAI (o4-mini)

```env
AZURE_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_API_KEY=your_key_here
AZURE_API_VERSION=2024-12-01-preview
AZURE_MODEL=o4-mini
```

Token limits: ~100K context window. Uses `max_completion_tokens` (o-series reasoning model).

### How provider selection works

- **CLI / API**: Uses `LLM_PROVIDER` from `.env` to pick the default provider.
- **Streamlit UI**: The sidebar has a radio button to switch between providers at runtime. This overrides the `.env` default for that session.
- **Compare Models page**: Always runs both providers regardless of which is selected.

Each LLM client (`GeminiClient`, `AzureOpenAIClient`) automatically uses its own model name from `.env`. No cross-provider model name conflicts.

---

## Corpus (20 Sources)

| # | Source ID | Year | Venue | Topic |
|---|---|---|---|---|
| 1 | strubell2019 | 2019 | ACL | NLP training energy costs |
| 2 | luccioni2022 | 2022 | arXiv | BLOOM lifecycle carbon |
| 3 | patterson2021 | 2021 | arXiv/Google | Training emissions analysis |
| 4 | schwartz2020 | 2020 | CACM | Green AI vs Red AI |
| 5 | henderson2020 | 2020 | JMLR | ML energy reporting framework |
| 6 | luccioni2023 | 2023 | FAccT | Inference energy costs |
| 7 | bannour2021 | 2021 | EMNLP | Carbon estimation tools |
| 8 | dodge2022 | 2022 | FAccT | Cloud carbon intensity |
| 9 | lacoste2019 | 2019 | NeurIPS | ML CO2 calculator |
| 10 | canziani2016 | 2016 | arXiv | CNN energy benchmarking |
| 11 | wu2022 | 2022 | MLSys | Meta sustainability |
| 12 | anthony2020 | 2020 | ICML | Carbontracker tool |
| 13 | ligozat2022 | 2022 | ACM Surveys | Carbon estimation survey |
| 14 | desislavov2023 | 2023 | Sustainable Computing | Inference energy trends |
| 15 | faiz2024 | 2024 | ICLR | LLMCarbon modeling |
| 16 | lannelongue2021 | 2021 | Advanced Science | Green Algorithms |
| 17 | samsi2023 | 2023 | IEEE HPEC | LLM inference benchmarks |
| 18 | garcia_martin2019 | 2019 | JMLR | ML energy estimation |
| 19 | verdecchia2023 | 2023 | WIREs | Green AI systematic review |
| 20 | thompson2020 | 2020 | IEEE | Computational limits of DL |

All sources are peer-reviewed or published at top-tier venues (ACL, EMNLP, NeurIPS, ICML, FAccT, MLSys, ICLR, IEEE, JMLR, ACM).

---

## Security

- **API key isolation**: All keys stored in `.env` (git-ignored). No keys in source code.
- **Prompt-injection protection**: `src/utils.sanitize_query()` is called at every entry point (RAG pipeline, FastAPI endpoint, Streamlit UI). It strips control characters, enforces a 1000-character limit, and rejects queries matching known injection patterns.
- **Input validation**: FastAPI uses Pydantic schemas for all request validation.
- **PDF exclusion**: Downloaded PDFs are git-ignored; re-downloaded at runtime via `make download`.

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

Run `make eval-both` then `make report` to generate the full evaluation report.
