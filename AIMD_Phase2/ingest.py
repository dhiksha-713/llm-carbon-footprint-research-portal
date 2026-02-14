"""
ingest.py
---------
Phase 2 Ingestion Pipeline
- Parses PDFs from data/raw/ using PyMuPDF
- Applies section-aware chunking with documented strategy
- Creates embeddings via sentence-transformers
- Indexes in FAISS
- Saves chunk store to data/processed/

Chunk strategy (documented per Phase 2 requirement):
  - Chunk size:    500 tokens (~400 words)
  - Overlap:       100 tokens (~80 words)
  - Section-aware: Tries to split on section headers before hard-splitting
  - Metadata kept: source_id, chunk_id, page_num, section_header, char_start, char_end

Run: python ingest.py
"""

import csv
import json
import os
import re
import pickle
import hashlib
from pathlib import Path
from typing import Optional

# ── Third-party (install via requirements.txt) ─────────────────────────
import fitz                          # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ── Config ─────────────────────────────────────────────────────────────
MANIFEST_PATH   = "data/data_manifest.csv"
RAW_DIR         = Path("data/raw")
PROCESSED_DIR   = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE      = 500   # approximate tokens (split by words as proxy)
CHUNK_OVERLAP   = 100   # token overlap between consecutive chunks
EMBED_MODEL     = "all-MiniLM-L6-v2"   # fast, 384-dim, good quality
WORDS_PER_TOKEN = 0.75  # rough conversion

# ── Section header detection regex ─────────────────────────────────────
SECTION_RE = re.compile(
    r"^(?:\d+\.?\s+|[A-Z][A-Z\s]{3,}$|Abstract|Introduction|"
    r"Related Work|Background|Methodology|Methods|Results|"
    r"Discussion|Conclusion|References|Acknowledgements)",
    re.MULTILINE,
)

# ───────────────────────────────────────────────────────────────────────
def load_manifest() -> list[dict]:
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """Return list of {page_num, text} dicts."""
    try:
        doc = fitz.open(str(pdf_path))
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text("text")
            text = re.sub(r"\n{3,}", "\n\n", text)   # collapse excess newlines
            text = re.sub(r"[ \t]+", " ", text)       # normalise spaces
            pages.append({"page_num": i + 1, "text": text.strip()})
        doc.close()
        return pages
    except Exception as e:
        print(f"  [ERROR] Could not parse PDF: {e}")
        return []

def words_to_tokens(n_words: int) -> int:
    return int(n_words / WORDS_PER_TOKEN)

def split_into_chunks(full_text: str, chunk_size_tokens: int, overlap_tokens: int) -> list[dict]:
    """
    Section-aware chunking:
      1. Split text on detected section headers.
      2. Within each section, apply sliding-window word chunking.
    Returns list of {chunk_text, section_header, char_start, char_end}
    """
    # Convert token targets to word counts
    chunk_words  = int(chunk_size_tokens * WORDS_PER_TOKEN)
    overlap_words = int(overlap_tokens * WORDS_PER_TOKEN)

    # Find section boundaries
    sections: list[tuple[str, str]] = []   # (header, content)
    last_end = 0
    current_header = "preamble"
    for match in SECTION_RE.finditer(full_text):
        if match.start() > last_end:
            sections.append((current_header, full_text[last_end:match.start()].strip()))
        current_header = match.group(0).strip()
        last_end = match.end()
    sections.append((current_header, full_text[last_end:].strip()))  # final section

    chunks = []
    global_char_offset = 0

    for header, content in sections:
        if not content:
            continue
        words = content.split()
        start = 0
        while start < len(words):
            end = min(start + chunk_words, len(words))
            chunk_words_list = words[start:end]
            chunk_text = " ".join(chunk_words_list)
            # Approximate char positions
            char_start = global_char_offset
            char_end   = char_start + len(chunk_text)
            chunks.append({
                "chunk_text":      chunk_text,
                "section_header":  header,
                "char_start":      char_start,
                "char_end":        char_end,
            })
            global_char_offset = char_end + 1
            if end == len(words):
                break
            start = end - overlap_words   # slide back for overlap
        global_char_offset += 10  # small gap between sections

    return chunks

def make_chunk_id(source_id: str, idx: int) -> str:
    return f"chunk_{idx:03d}"

def ingest_source(row: dict, embed_model: SentenceTransformer) -> list[dict]:
    """Parse, chunk, and embed a single source. Returns list of chunk dicts."""
    source_id = row["source_id"]
    local_file = Path(row["local_file"])
    pdf_path = RAW_DIR / local_file.name

    if not pdf_path.exists():
        print(f"  [SKIP] PDF not found: {pdf_path}  (run download_sources.py first)")
        return []

    print(f"  Parsing {source_id} ...")
    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        return []

    full_text = "\n\n".join(p["text"] for p in pages)
    raw_chunks = split_into_chunks(full_text, CHUNK_SIZE, CHUNK_OVERLAP)

    chunk_records = []
    for idx, raw in enumerate(raw_chunks):
        chunk_id = make_chunk_id(source_id, idx)
        record = {
            # Identity
            "source_id":      source_id,
            "chunk_id":       chunk_id,
            "full_id":        f"{source_id}_{chunk_id}",
            # Content
            "chunk_text":     raw["chunk_text"],
            "section_header": raw["section_header"],
            # Position
            "char_start":     raw["char_start"],
            "char_end":       raw["char_end"],
            # Source metadata (denormalised for convenience)
            "title":          row["title"],
            "authors":        row["authors"],
            "year":           row["year"],
            "source_type":    row["type"],
            "venue":          row["venue"],
            "link_doi":       row["link_doi"],
        }
        chunk_records.append(record)

    if chunk_records:
        texts = [c["chunk_text"] for c in chunk_records]
        embeddings = embed_model.encode(texts, show_progress_bar=False, batch_size=32)
        for c, emb in zip(chunk_records, embeddings):
            c["embedding"] = emb.tolist()
        print(f"    → {len(chunk_records)} chunks embedded")

    return chunk_records

def build_faiss_index(all_chunks: list[dict]) -> faiss.Index:
    """Build FAISS flat L2 index from chunk embeddings."""
    dim = len(all_chunks[0]["embedding"])
    index = faiss.IndexFlatIP(dim)   # Inner product (cosine on normalised vectors)
    embeddings = np.array([c["embedding"] for c in all_chunks], dtype="float32")
    # Normalise for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-10)
    index.add(embeddings)
    return index

def main():
    print("=" * 60)
    print("Phase 2 — Ingestion Pipeline")
    print(f"Chunk size: {CHUNK_SIZE} tokens | Overlap: {CHUNK_OVERLAP} tokens")
    print(f"Embed model: {EMBED_MODEL}")
    print("=" * 60)

    manifest = load_manifest()
    print(f"\nLoaded {len(manifest)} sources from manifest.\n")

    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)

    all_chunks: list[dict] = []
    for row in manifest:
        print(f"\n[{row['source_id']}] {row['title'][:60]}...")
        chunks = ingest_source(row, model)
        all_chunks.extend(chunks)

    if not all_chunks:
        print("\n[ERROR] No chunks created. Check that PDFs exist in data/raw/")
        return

    print(f"\n{'='*60}")
    print(f"Total chunks: {len(all_chunks)}")
    print("Building FAISS index...")

    index = build_faiss_index(all_chunks)

    # Save chunk store (without embeddings for readability, keep separate)
    chunk_store = []
    embeddings_list = []
    for c in all_chunks:
        emb = c.pop("embedding")
        embeddings_list.append(emb)
        chunk_store.append(c)

    chunk_store_path = PROCESSED_DIR / "chunk_store.json"
    with open(chunk_store_path, "w", encoding="utf-8") as f:
        json.dump(chunk_store, f, indent=2, ensure_ascii=False)

    embeddings_path = PROCESSED_DIR / "embeddings.npy"
    np.save(str(embeddings_path), np.array(embeddings_list, dtype="float32"))

    faiss_path = PROCESSED_DIR / "faiss_index.bin"
    faiss.write_index(index, str(faiss_path))

    # Summary
    print(f"\nSaved:")
    print(f"  {chunk_store_path}   ({len(chunk_store)} chunks)")
    print(f"  {embeddings_path}")
    print(f"  {faiss_path}")

    # Chunking strategy doc
    strategy = {
        "chunk_size_tokens": CHUNK_SIZE,
        "chunk_overlap_tokens": CHUNK_OVERLAP,
        "embed_model": EMBED_MODEL,
        "embed_dim": len(embeddings_list[0]),
        "section_aware": True,
        "section_detection": "regex on common academic paper headers",
        "total_chunks": len(chunk_store),
        "total_sources": len(manifest),
        "sources_ingested": len(set(c["source_id"] for c in chunk_store)),
    }
    with open(PROCESSED_DIR / "chunking_strategy.json", "w") as f:
        json.dump(strategy, f, indent=2)

    print(f"\n✓ Ingestion complete. {len(set(c['source_id'] for c in chunk_store))} sources indexed.")
    print("=" * 60)

if __name__ == "__main__":
    main()
