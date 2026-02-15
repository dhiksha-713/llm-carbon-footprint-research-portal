"""FastAPI backend for the LLM Carbon Footprint Research Portal."""

from __future__ import annotations

import csv
import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from src.config import (
    GENERATION_MODEL, EMBED_MODEL_NAME, LLM_PROVIDER,
    PROCESSED_DIR, LOGS_DIR, OUTPUTS_DIR, MANIFEST_PATH,
    TOP_K, API_HOST, API_PORT,
)
from src.llm_client import get_llm_client
from src.rag.rag import load_index, run_rag
from src.rag.enhance_query_rewriting import run_enhanced_rag
from src.eval.evaluation import EVAL_QUERIES, compute_summary
from src.utils import sanitize_query


# ── Shared state (loaded once at startup) ─────────────────────────────────
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    index, store = load_index()
    _state["index"] = index
    _state["store"] = store
    _state["embed_model"] = SentenceTransformer(EMBED_MODEL_NAME)
    _state["client"] = get_llm_client()
    yield
    _state.clear()


app = FastAPI(
    title="LLM Carbon Footprint Research Portal",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Research question")
    mode: str = Field("baseline", pattern="^(baseline|enhanced)$")
    top_k: int = Field(TOP_K, ge=1, le=20)


class CitationValidation(BaseModel):
    total_citations: int
    valid_citations: int
    invalid_citations: int
    citation_precision: Optional[float]
    invalid_list: list[dict]


class ChunkSummary(BaseModel):
    source_id: str
    chunk_id: str
    title: str
    year: str
    section_header: str
    retrieval_score: float
    chunk_text_preview: str


class QueryResponse(BaseModel):
    run_id: str
    query: str
    mode: str
    answer: str
    retrieved_chunks: list[ChunkSummary]
    citations_extracted: list[dict]
    citation_validation: CitationValidation
    tokens: dict


class CorpusSource(BaseModel):
    source_id: str
    title: str
    authors: str
    year: str
    source_type: str
    venue: str
    url_or_doi: str
    tags: str
    relevance_note: str


class CorpusResponse(BaseModel):
    total_sources: int
    sources: list[CorpusSource]
    chunking_strategy: Optional[dict] = None


class EvalSummaryResponse(BaseModel):
    total_queries: int
    summary: dict
    results: list[dict]


class LogEntry(BaseModel):
    run_id: str
    timestamp: str
    mode: str
    query: str
    citations_total: int
    citations_valid: int


class LogsResponse(BaseModel):
    total_runs: int
    entries: list[LogEntry]


class LogDetailResponse(BaseModel):
    entry: dict


class HealthResponse(BaseModel):
    status: str
    provider: str
    model: str
    index_loaded: bool
    chunks_count: int


# ── Endpoints ─────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        provider=LLM_PROVIDER,
        model=GENERATION_MODEL,
        index_loaded="index" in _state,
        chunks_count=len(_state.get("store", [])),
    )


@app.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    if not _state:
        raise HTTPException(503, "Models not loaded yet")

    # Sanitize input (raises ValueError on injection)
    try:
        clean_query = sanitize_query(req.query)
    except ValueError as e:
        raise HTTPException(400, str(e))

    index = _state["index"]
    store = _state["store"]
    embed_model = _state["embed_model"]
    client = _state["client"]

    if req.mode == "enhanced":
        result = run_enhanced_rag(clean_query, index, store, embed_model, client)
    else:
        result = run_rag(clean_query, index, store, embed_model, client,
                         top_k=req.top_k, mode=req.mode)

    return QueryResponse(
        run_id=result["run_id"],
        query=result["query"],
        mode=result["mode"],
        answer=result["answer"],
        retrieved_chunks=[ChunkSummary(**c) for c in result["retrieved_chunks"]],
        citations_extracted=result["citations_extracted"],
        citation_validation=CitationValidation(**result["citation_validation"]),
        tokens=result["tokens"],
    )


@app.get("/corpus", response_model=CorpusResponse)
def get_corpus():
    sources: list[CorpusSource] = []
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                sources.append(CorpusSource(
                    source_id=row["source_id"],
                    title=row["title"],
                    authors=row["authors"],
                    year=row["year"],
                    source_type=row["source_type"],
                    venue=row["venue"],
                    url_or_doi=row["url_or_doi"],
                    tags=row.get("tags", ""),
                    relevance_note=row.get("relevance_note", ""),
                ))

    strategy = None
    strat_path = PROCESSED_DIR / "chunking_strategy.json"
    if strat_path.exists():
        strategy = json.loads(strat_path.read_text(encoding="utf-8"))

    return CorpusResponse(
        total_sources=len(sources),
        sources=sources,
        chunking_strategy=strategy,
    )


@app.get("/evaluation", response_model=EvalSummaryResponse)
def get_evaluation():
    files = sorted(OUTPUTS_DIR.glob("eval_results_*.json"))
    if not files:
        return EvalSummaryResponse(total_queries=len(EVAL_QUERIES), summary={}, results=[])
    results = json.loads(files[-1].read_text(encoding="utf-8"))
    return EvalSummaryResponse(
        total_queries=len(results),
        summary=compute_summary(results),
        results=results,
    )


@app.get("/evaluation/queries")
def get_eval_queries():
    return {"queries": EVAL_QUERIES}


@app.get("/logs", response_model=LogsResponse)
def get_logs(limit: int = Query(50, ge=1, le=500)):
    path = LOGS_DIR / "rag_runs.jsonl"
    if not path.exists():
        return LogsResponse(total_runs=0, entries=[])

    raw = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").strip().split("\n")
        if line.strip()
    ]
    entries = [
        LogEntry(
            run_id=e.get("run_id", ""),
            timestamp=e.get("timestamp", ""),
            mode=e.get("mode", ""),
            query=e.get("query", "")[:120],
            citations_total=e.get("citation_validation", {}).get("total_citations", 0),
            citations_valid=e.get("citation_validation", {}).get("valid_citations", 0),
        )
        for e in raw[-limit:]
    ]
    return LogsResponse(total_runs=len(raw), entries=entries)


@app.get("/logs/{run_id}", response_model=LogDetailResponse)
def get_log_detail(run_id: str):
    path = LOGS_DIR / "rag_runs.jsonl"
    if not path.exists():
        raise HTTPException(404, "No logs found")

    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            entry = json.loads(line)
            if entry.get("run_id") == run_id:
                return LogDetailResponse(entry=entry)

    raise HTTPException(404, f"Run {run_id} not found")


# ── Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app.app:app", host=API_HOST, port=API_PORT, reload=True)
