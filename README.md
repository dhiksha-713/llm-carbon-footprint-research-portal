# Personal Research Portal -- Carbon Footprint of LLMs

**Course**: AI Model Development (95-864)  
**Group 4**: Dhiksha Rathis, Shreya Verma  
**Domain**: Green AI / Sustainable Computing  

## Research Question

How do we accurately measure and compare the carbon footprint of different LLMs across their lifecycle?

### Sub-questions (Phase 1)

1. What are the major sources of emissions in LLM training vs. inference?
2. How do different studies measure and report carbon metrics?
3. What factors (model size, hardware, location) most impact carbon footprint?
4. How do carbon estimates vary across different LLM families?
5. What data is missing or inconsistent in current carbon reporting?

## Repository Structure

```
├── README.md
├── requirements.txt
├── Makefile
├── .env.example
├── data/
│   ├── data_manifest.csv
│   ├── raw/
│   └── processed/
├── src/
│   ├── config.py                      # Centralized configuration (env-driven)
│   ├── ingest/
│   │   ├── download_sources.py        # PDF downloader
│   │   └── ingest.py                  # Parse -> chunk -> embed -> FAISS index
│   ├── rag/
│   │   ├── rag.py                     # Baseline RAG (Gemini + FAISS)
│   │   └── enhance_query_rewriting.py # Query rewriting + decomposition
│   ├── eval/
│   │   ├── evaluation.py              # 20-query eval set + LLM-as-judge
│   │   └── generate_report.py         # Markdown report generator
│   └── app/
│       ├── app.py                     # FastAPI backend
│       └── streamlit_ui.py            # Streamlit interactive UI
├── outputs/
├── logs/
└── report/
    ├── phase1/                        # Framing brief, prompt kit, eval sheet, analysis memo
    └── phase2/                        # Evaluation report
```

## Quick Start

```bash
# 1. Clone and setup
cp .env.example .env
# Edit .env with your GEMINI_API_KEY

# 2. Install
pip install -r requirements.txt

# 3. Download corpus PDFs
python -m src.ingest.download_sources

# 4. Ingest (parse, chunk, embed, index)
python -m src.ingest.ingest

# 5. Query
python -m src.rag.rag --query "What are the major sources of carbon emissions in LLM training?"

# 6. Evaluate
python -m src.eval.evaluation --mode both

# 7. Generate report
python -m src.eval.generate_report

# 8. Launch FastAPI server
make serve
# or: uvicorn src.app.app:app --host 0.0.0.0 --port 8000 --reload

# 9. Launch Streamlit UI
make ui
# or: streamlit run src/app/streamlit_ui.py --server.port 8502
```

Or: `make all` then `make ui`

### Streamlit UI

The Streamlit interface at `http://localhost:8502` provides:

| Page | Description |
|------|-------------|
| **Home** | Project overview, research question, Phase 2 deliverables, pipeline architecture, evaluation metrics |
| **RAG Query** | Interactive query interface with baseline/enhanced mode, retrieved chunks, citation validation |
| **Corpus Explorer** | Browse all 15 sources with filters (type, year, search), metadata, year distribution chart |
| **Evaluation Dashboard** | Per-query scores, radar charts, baseline vs. enhanced comparison, breakdown by query type |
| **Run Logs** | Full audit trail of every RAG query with expandable detail view |

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server health, model info, index status |
| `POST` | `/query` | Run a RAG query (body: `{"query": "...", "mode": "baseline", "top_k": 5}`) |
| `GET` | `/corpus` | Corpus manifest, chunking strategy |
| `GET` | `/evaluation` | Latest eval results + aggregated metrics |
| `GET` | `/evaluation/queries` | The 20-query evaluation set |
| `GET` | `/logs?limit=50` | Recent run logs |
| `GET` | `/logs/{run_id}` | Full detail for a specific run |

Interactive API docs at `http://localhost:8000/docs` after starting the server.

## Configuration

All parameters are env-driven via `.env` (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | (required) | Google Gemini API key |
| `GENERATION_MODEL` | gemini-3-flash-preview | Model for answer generation |
| `JUDGE_MODEL` | gemini-3-flash-preview | Model for LLM-as-judge evaluation |
| `GENERATION_TEMPERATURE` | 0.2 | Generation sampling temperature |
| `JUDGE_TEMPERATURE` | 0.0 | Judge sampling temperature |
| `MAX_OUTPUT_TOKENS` | 2048 | Max tokens for generation |
| `JUDGE_MAX_TOKENS` | 300 | Max tokens for judge responses |
| `DECOMPOSE_TEMPERATURE` | 0.0 | Temperature for query decomposition |
| `REWRITE_TEMPERATURE` | 0.0 | Temperature for query rewriting |
| `DECOMPOSE_MAX_TOKENS` | 300 | Max tokens for decomposition |
| `REWRITE_MAX_TOKENS` | 100 | Max tokens for rewriting |
| `EMBED_MODEL_NAME` | all-MiniLM-L6-v2 | Sentence-transformer model |
| `EMBED_BATCH_SIZE` | 32 | Embedding batch size |
| `CHUNK_SIZE_TOKENS` | 500 | Chunk size in tokens |
| `CHUNK_OVERLAP_TOKENS` | 100 | Overlap between chunks |
| `TOP_K` | 5 | Baseline retrieval count |
| `ENHANCED_TOP_N` | 8 | Enhanced mode merged chunk count |
| `MAX_SUB_QUERIES` | 4 | Max sub-queries for decomposition |
| `REQUEST_TIMEOUT` | 30 | PDF download timeout (seconds) |
| `REQUEST_DELAY_S` | 2 | Delay between downloads (seconds) |
| `API_HOST` | 0.0.0.0 | FastAPI bind host |
| `API_PORT` | 8000 | FastAPI bind port |
| `STREAMLIT_PORT` | 8501 | Streamlit UI port |
| `CHUNK_PREVIEW_LEN` | 200 | Chunk text preview length |
| `SCORE_PASS_THRESHOLD` | 3.5 | Score threshold for "pass" label |
| `SCORE_WARN_THRESHOLD` | 2.5 | Score threshold for "warn" label |

## Phase 1 Summary

See `report/phase1/` for full deliverables.

- **Tasks**: Claim-Evidence Extraction + Cross-Source Synthesis
- **Models evaluated**: Claude Opus 4.5, Claude Sonnet 4.5, GPT-5, Gemini 3
- **Key finding**: Structured prompts improve all models by ~30%; citation accuracy requires explicit guardrails

## Phase 2: Research-Grade RAG

### Corpus

15 sources (8 peer-reviewed, 4 technical reports, 2 tool papers, 1 standards report), spanning 2016-2023. Full metadata in `data/data_manifest.csv`.

### Pipeline

1. **Ingest**: PyMuPDF PDF parsing, section-aware chunking (500t/100t overlap), all-MiniLM-L6-v2 embeddings, FAISS IndexFlatIP
2. **Baseline RAG**: Top-K semantic retrieval + Gemini generation with citation constraints
3. **Enhanced RAG**: Query rewriting + decomposition for synthesis/multi-hop queries, merged deduplication
4. **Trust behavior**: Refuses fabricated citations, flags missing evidence, preserves hedging, detects conflicts

### Evaluation (6 metrics)

| Metric | Type | Description |
|--------|------|-------------|
| Groundedness | LLM-judge 1-4 | Are claims supported by retrieved chunks? |
| Answer Relevance | LLM-judge 1-4 | Does the answer address the query? |
| Context Precision | LLM-judge 1-4 | Are retrieved chunks relevant? |
| Citation Precision | Deterministic | valid citations / total citations |
| Source Recall | Deterministic | expected sources found / total expected |
| Uncertainty Handling | Rule-based | Does answer flag missing evidence? |

20-query set: 10 direct, 5 synthesis/multi-hop, 5 edge-case.

### Enhancement

Query rewriting + decomposition targets synthesis queries. Sub-queries retrieve independently, results are merged and deduplicated before generation.

### Citation Format

`(source_id, chunk_id)` -- each resolves to `data_manifest.csv` and `chunk_store.json`.

## AI Usage Disclosure

| Tool | Purpose | Manual review |
|------|---------|---------------|
| Gemini 2.0 Flash | RAG generation + evaluation judging | Prompt engineering, guardrail design |
| Cursor AI | Code scaffolding | Full code review and testing |
| sentence-transformers | Embedding | Configuration only |
