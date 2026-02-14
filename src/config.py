"""Centralized configuration. All tunable parameters and paths live here."""

import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ── Paths ────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_PATH = DATA_DIR / "data_manifest.csv"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "report" / "phase2"

for d in [RAW_DIR, PROCESSED_DIR, LOGS_DIR, OUTPUTS_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Gemini API ───────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-3-flash-preview")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-3-flash-preview")
GENERATION_TEMPERATURE = float(os.getenv("GENERATION_TEMPERATURE", "0.2"))
JUDGE_TEMPERATURE = float(os.getenv("JUDGE_TEMPERATURE", "0.0"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "2048"))

# ── Embeddings ───────────────────────────────────────────────────────────
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")

# ── Chunking ─────────────────────────────────────────────────────────────
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "500"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "100"))
WORDS_PER_TOKEN = float(os.getenv("WORDS_PER_TOKEN", "0.75"))

# ── Retrieval ────────────────────────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", "5"))
ENHANCED_TOP_N = int(os.getenv("ENHANCED_TOP_N", "8"))

# ── Prompt Versions ──────────────────────────────────────────────────────
BASELINE_PROMPT_VERSION = "RAG-BASELINE-V2"
ENHANCED_PROMPT_VERSION = "RAG-ENHANCED-REWRITE-V2"
