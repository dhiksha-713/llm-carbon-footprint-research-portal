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
│       └── app.py                     # FastAPI backend
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
```

Or: `make all` then `make serve`

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
| `GENERATION_MODEL` | gemini-2.0-flash | Model for answer generation |
| `JUDGE_MODEL` | gemini-2.0-flash | Model for LLM-as-judge evaluation |
| `EMBED_MODEL_NAME` | all-MiniLM-L6-v2 | Sentence-transformer model |
| `CHUNK_SIZE_TOKENS` | 500 | Chunk size in tokens |
| `CHUNK_OVERLAP_TOKENS` | 100 | Overlap between chunks |
| `TOP_K` | 5 | Baseline retrieval count |
| `ENHANCED_TOP_N` | 8 | Enhanced mode merged chunk count |

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
