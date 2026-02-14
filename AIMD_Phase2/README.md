# Phase 2 — Research-Grade RAG: Carbon Footprint of LLMs
**Group 4**: Dhiksha Rathis, Shreya Verma  
**Course**: AI Model Development  

---

## Overview

This repository implements a Research-Grade RAG (Retrieval-Augmented Generation) pipeline
over a 15-source corpus on LLM carbon footprint measurement. It builds directly on the
Phase 1 prompt kit and evaluation rubric.

**Domain**: Green AI / Sustainable Computing  
**Main question**: How do we accurately measure and compare the carbon footprint of different LLMs across their lifecycle?

---

## Repository Structure

```
rag_phase2/
├── data/
│   ├── data_manifest.csv          # 15 sources with full metadata
│   ├── raw/                       # Downloaded PDFs (git-ignored)
│   └── processed/                 # FAISS index + chunk store (generated)
│       ├── chunk_store.json
│       ├── embeddings.npy
│       ├── faiss_index.bin
│       └── chunking_strategy.json
├── logs/
│   └── rag_runs.jsonl             # All query/retrieval/answer logs
├── outputs/
│   ├── eval_results_baseline_*.json
│   ├── eval_results_enhanced_*.json
│   └── evaluation_report.md       # Phase 2 evaluation report
├── download_sources.py            # Download PDFs from arXiv
├── ingest.py                      # Parse → chunk → embed → index
├── rag.py                         # Baseline RAG pipeline
├── enhance_query_rewriting.py     # Enhancement: query rewriting + decomposition
├── evaluation.py                  # 20-query eval set + scoring
├── generate_report.py             # Evaluation report generator
├── requirements.txt               # Pinned dependencies
├── Makefile                       # One-command run path
└── README.md                      # This file
```

---

## Quick Start

### 1. Prerequisites

```bash
# Python 3.10+ required
python --version

# Set your API key
export ANTHROPIC_API_KEY=your_key_here
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download sources

```bash
python download_sources.py
```

This downloads PDFs from arXiv. A few sources (Google report, IEA report) require
manual download — the script will tell you exactly which ones and where to save them.

### 4. Ingest corpus

```bash
python ingest.py
```

This will:
- Parse all PDFs with PyMuPDF
- Apply section-aware chunking (500 tokens, 100-token overlap)
- Embed all chunks with `all-MiniLM-L6-v2`
- Build and save a FAISS index

### 5. Run a query

```bash
python rag.py --query "What does GPU-hour energy measurement measure, and what are its failure modes?"
```

### 6. Run full evaluation

```bash
python evaluation.py --mode both
```

### 7. Generate evaluation report

```bash
python generate_report.py
```

### One-command run (all steps)

```bash
make all
```

---

## Chunking Strategy

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Chunk size | 500 tokens | Fits within typical context windows; large enough for substantive claims |
| Overlap | 100 tokens | Prevents claims at chunk boundaries from being split |
| Section awareness | Yes (regex) | Keeps methodology sections intact; improves precision |
| Embedding model | all-MiniLM-L6-v2 | Fast, high quality, 384-dim; good for academic text |
| Index type | FAISS IndexFlatIP | Exact cosine similarity; suitable for corpus of this size |

---

## Enhancement: Query Rewriting + Decomposition

For synthesis and multi-hop queries, the enhanced pipeline:

1. **Rewrites** the query for academic retrieval (adds domain terminology)
2. **Decomposes** into 2–4 sub-queries using Claude
3. **Retrieves** top-5 chunks for each sub-query
4. **Merges** and deduplicates (top-8 unique chunks by score)
5. **Generates** a synthesis answer with conflict detection

This directly addresses the Phase 1 finding that cross-source synthesis requires
multi-source grounding and explicit conflict flagging.

---

## Evaluation Set Design

| Type | Count | Example |
|------|-------|---------|
| Direct | 10 | "How much CO₂ was emitted during BERT training?" |
| Synthesis | 5 | "Compare Strubell and Patterson on measurement methodology" |
| Edge-case | 5 | "Does the corpus contain evidence about GPT-4 carbon footprint?" |

The synthesis queries (S01, S02) directly reuse Phase 1 test cases TC2A and TC2B,
creating explicit continuity between phases.

---

## Citation Format

All answers use the format: `(source_id, chunk_id)` e.g. `(strubell2019, chunk_003)`

Every citation resolves to the data manifest where:
- `source_id` maps to a unique entry in `data/data_manifest.csv`
- `chunk_id` maps to a chunk in `data/processed/chunk_store.json`

The system **refuses to invent citations** — if evidence is missing, it explicitly states:
> "The corpus does not contain evidence for this claim."

---

## Trust Behavior

The RAG system implements the following trust behaviors:
- Never fabricates citations (validated post-generation)
- Flags missing evidence explicitly
- Preserves hedging language from sources
- Detects and reports inter-source conflicts
- Validates all citations against retrieved context

---

## Logging

Every query run is logged to `logs/rag_runs.jsonl` with:
- `run_id`, `timestamp`, `prompt_version`
- Full query and retrieved chunks (with scores)
- Complete answer text
- Citation validation results
- Token usage

---

## Sources

See `data/data_manifest.csv` for full metadata on all 15 sources.  
8 peer-reviewed papers + 4 technical reports + 2 tool papers + 1 standards report.  
Spanning 2016–2023, covering training emissions, inference costs, lifecycle analysis,
measurement tools, and reporting standards.
