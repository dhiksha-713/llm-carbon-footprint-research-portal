"""Centralized configuration. Every tunable lives here, driven by .env."""

import logging
import os
from pathlib import Path
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ── Paths ────────────────────────────────────────────────────────────────
DATA_DIR      = PROJECT_ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_PATH = DATA_DIR / "data_manifest.csv"
LOGS_DIR      = PROJECT_ROOT / "logs"
OUTPUTS_DIR   = PROJECT_ROOT / "outputs"
REPORT_DIR    = PROJECT_ROOT / "report" / "phase2"

for _d in (RAW_DIR, PROCESSED_DIR, LOGS_DIR, OUTPUTS_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── LLM Provider ─────────────────────────────────────────────────────────
LLM_PROVIDER      = os.getenv("LLM_PROVIDER", "grok")
GROK_API_KEY      = os.getenv("GROK_API_KEY", "")
GROK_ENDPOINT     = os.getenv("GROK_ENDPOINT", "https://cmu-llm-api-resource.services.ai.azure.com/openai/v1/")
GROK_MODEL        = os.getenv("GROK_MODEL", "grok-3")
AZURE_ENDPOINT    = os.getenv("AZURE_ENDPOINT", "")
AZURE_API_KEY     = os.getenv("AZURE_API_KEY", "")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
AZURE_MODEL       = os.getenv("AZURE_MODEL", "o4-mini")

# ── Generation ───────────────────────────────────────────────────────────
GENERATION_TEMPERATURE = float(os.getenv("GENERATION_TEMPERATURE", "0.2"))
JUDGE_TEMPERATURE      = float(os.getenv("JUDGE_TEMPERATURE", "0.0"))
JUDGE_MAX_TOKENS       = int(os.getenv("JUDGE_MAX_TOKENS", "300"))
DECOMPOSE_MAX_TOKENS   = int(os.getenv("DECOMPOSE_MAX_TOKENS", "300"))
REWRITE_MAX_TOKENS     = int(os.getenv("REWRITE_MAX_TOKENS", "100"))
DECOMPOSE_TEMPERATURE  = float(os.getenv("DECOMPOSE_TEMPERATURE", "0.0"))
REWRITE_TEMPERATURE    = float(os.getenv("REWRITE_TEMPERATURE", "0.0"))

# ── Embeddings / Chunking / Retrieval ────────────────────────────────────
EMBED_MODEL_NAME     = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
EMBED_BATCH_SIZE     = int(os.getenv("EMBED_BATCH_SIZE", "32"))
CHUNK_SIZE_TOKENS    = int(os.getenv("CHUNK_SIZE_TOKENS", "500"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "100"))
WORDS_PER_TOKEN      = float(os.getenv("WORDS_PER_TOKEN", "0.75"))
TOP_K                = int(os.getenv("TOP_K", "5"))
ENHANCED_TOP_N       = int(os.getenv("ENHANCED_TOP_N", "8"))
MAX_SUB_QUERIES      = int(os.getenv("MAX_SUB_QUERIES", "4"))

# ── Download / Serving ───────────────────────────────────────────────────
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
REQUEST_DELAY_S = int(os.getenv("REQUEST_DELAY_S", "2"))
API_HOST        = os.getenv("API_HOST", "0.0.0.0")
API_PORT        = int(os.getenv("API_PORT", "8000"))
STREAMLIT_PORT  = int(os.getenv("STREAMLIT_PORT", "8501"))

# ── Evaluation ───────────────────────────────────────────────────────────
CHUNK_PREVIEW_LEN    = int(os.getenv("CHUNK_PREVIEW_LEN", "200"))
SCORE_PASS_THRESHOLD = float(os.getenv("SCORE_PASS_THRESHOLD", "3.5"))
SCORE_WARN_THRESHOLD = float(os.getenv("SCORE_WARN_THRESHOLD", "2.5"))

# ── Prompt versions (logged with every run) ──────────────────────────────
BASELINE_PROMPT_VERSION  = os.getenv("BASELINE_PROMPT_VERSION", "RAG-BASELINE-V2")
ENHANCED_PROMPT_VERSION  = os.getenv("ENHANCED_PROMPT_VERSION", "RAG-ENHANCED-REWRITE-V2")
