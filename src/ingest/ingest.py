"""Ingestion pipeline: PDF parsing, section-aware chunking, embedding, FAISS indexing."""

import csv
import json
import logging
import re
from pathlib import Path

import fitz
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.config import (
    MANIFEST_PATH, RAW_DIR, PROCESSED_DIR, PROJECT_ROOT,
    CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS, WORDS_PER_TOKEN,
    EMBED_MODEL_NAME, EMBED_BATCH_SIZE,
)

log = logging.getLogger(__name__)

MIN_CHUNK_CHARS = 50

_SECTION_RE = re.compile(
    r"^(?:\d+\.?\s+|[A-Z][A-Z\s]{3,}$|Abstract|Introduction|"
    r"Related Work|Background|Methodology|Methods|Results|"
    r"Discussion|Conclusion|References|Acknowledgements)",
    re.MULTILINE,
)


def load_manifest() -> list[dict]:
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def extract_text(pdf_path: Path) -> list[dict]:
    """Return per-page text from a PDF."""
    try:
        doc = fitz.open(str(pdf_path))
        pages = []
        for i, page in enumerate(doc):
            txt = page.get_text("text")
            txt = re.sub(r"\n{3,}", "\n\n", txt)
            txt = re.sub(r"[ \t]+", " ", txt)
            pages.append({"page_num": i + 1, "text": txt.strip()})
        doc.close()
        log.info("extract_text(%s) -> %d pages, %d total chars",
                 pdf_path.name, len(pages), sum(len(p["text"]) for p in pages))
        return pages
    except Exception as exc:
        log.error("PDF parse failed for %s: %s", pdf_path, exc)
        print(f"  [ERROR] PDF parse failed: {exc}")
        return []


def chunk_text(full_text: str) -> list[dict]:
    """Section-aware sliding-window chunking."""
    chunk_words = int(CHUNK_SIZE_TOKENS * WORDS_PER_TOKEN)
    overlap_words = int(CHUNK_OVERLAP_TOKENS * WORDS_PER_TOKEN)

    sections: list[tuple[str, str]] = []
    last_end, current_header = 0, "preamble"
    for m in _SECTION_RE.finditer(full_text):
        if m.start() > last_end:
            sections.append((current_header, full_text[last_end:m.start()].strip()))
        current_header = m.group(0).strip()
        last_end = m.end()
    sections.append((current_header, full_text[last_end:].strip()))

    chunks, offset = [], 0
    for header, content in sections:
        if not content:
            continue
        words = content.split()
        start = 0
        while start < len(words):
            end = min(start + chunk_words, len(words))
            text = " ".join(words[start:end])
            if len(text.strip()) >= MIN_CHUNK_CHARS:
                chunks.append({
                    "chunk_text": text,
                    "section_header": header,
                    "char_start": offset,
                    "char_end": offset + len(text),
                })
            offset += len(text) + 1
            if end >= len(words):
                break
            start = end - overlap_words
        offset += 10
    return chunks


def ingest_source(row: dict, model: SentenceTransformer) -> list[dict]:
    """Parse, chunk, and embed a single source."""
    pdf_path = PROJECT_ROOT / row["raw_path"]
    if not pdf_path.exists():
        log.warning("[SKIP] %s - PDF not found at %s", row["source_id"], pdf_path)
        print(f"  [SKIP] {row['source_id']} - PDF not found")
        return []

    pages = extract_text(pdf_path)
    if not pages:
        return []

    full_text = "\n\n".join(p["text"] for p in pages)
    raw_chunks = chunk_text(full_text)

    records = []
    for idx, rc in enumerate(raw_chunks):
        records.append({
            "source_id": row["source_id"],
            "chunk_id": f"chunk_{idx:03d}",
            "full_id": f"{row['source_id']}_chunk_{idx:03d}",
            "chunk_text": rc["chunk_text"],
            "section_header": rc["section_header"],
            "char_start": rc["char_start"],
            "char_end": rc["char_end"],
            "title": row["title"],
            "authors": row["authors"],
            "year": row["year"],
            "source_type": row["source_type"],
            "venue": row["venue"],
            "url_or_doi": row["url_or_doi"],
        })

    if records:
        embs = model.encode(
            [r["chunk_text"] for r in records],
            show_progress_bar=False, batch_size=EMBED_BATCH_SIZE,
        )
        for r, e in zip(records, embs):
            r["embedding"] = e.tolist()
        log.info("%s: %d chunks (min %d chars, %d pages)",
                 row["source_id"], len(records), MIN_CHUNK_CHARS, len(pages))
        print(f"  {row['source_id']}: {len(records)} chunks")
    return records


def build_index(chunks: list[dict]) -> faiss.Index:
    """Build a FAISS inner-product index from normalised embeddings."""
    dim = len(chunks[0]["embedding"])
    idx = faiss.IndexFlatIP(dim)
    embs = np.array([c["embedding"] for c in chunks], dtype="float32")
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
    idx.add(embs / norms)
    return idx


def main() -> None:
    manifest = load_manifest()
    print(f"Loaded {len(manifest)} sources | chunk={CHUNK_SIZE_TOKENS}t overlap={CHUNK_OVERLAP_TOKENS}t | min_chars={MIN_CHUNK_CHARS}")
    log.info("Ingestion starting: %d sources", len(manifest))

    for old in PROCESSED_DIR.glob("*"):
        old.unlink()
        print(f"  [CLEAN] removed {old.name}")

    model = SentenceTransformer(EMBED_MODEL_NAME)
    all_chunks: list[dict] = []
    for row in manifest:
        all_chunks.extend(ingest_source(row, model))

    if not all_chunks:
        print("[ERROR] No chunks produced - ensure PDFs exist in data/raw/")
        print("        Run: make download")
        return

    index = build_index(all_chunks)

    store: list[dict] = []
    for c in all_chunks:
        c.pop("embedding", None)
        store.append(c)

    (PROCESSED_DIR / "chunk_store.json").write_text(
        json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    faiss.write_index(index, str(PROCESSED_DIR / "faiss_index.bin"))

    sources_ok = sorted(set(c["source_id"] for c in store))
    strategy = {
        "chunk_size_tokens": CHUNK_SIZE_TOKENS,
        "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
        "min_chunk_chars": MIN_CHUNK_CHARS,
        "embed_model": EMBED_MODEL_NAME,
        "embed_dim": index.d,
        "section_aware": True,
        "total_chunks": len(store),
        "total_sources": len(manifest),
        "sources_ingested": len(sources_ok),
        "sources_list": sources_ok,
    }
    (PROCESSED_DIR / "chunking_strategy.json").write_text(
        json.dumps(strategy, indent=2), encoding="utf-8"
    )
    print(f"\nIndexed {len(store)} chunks from {strategy['sources_ingested']}/{len(manifest)} sources")
    log.info("Ingestion complete: %d chunks, %d sources", len(store), len(sources_ok))


if __name__ == "__main__":
    main()
