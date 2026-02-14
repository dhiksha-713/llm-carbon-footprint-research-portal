"""
download_sources.py
-------------------
Downloads all PDFs listed in data/data_manifest.csv into data/raw/.
Handles arXiv links by converting to direct PDF URLs.
Run: python download_sources.py
"""

import csv
import os
import time
import urllib.request
import urllib.error
from pathlib import Path

MANIFEST_PATH = "data/data_manifest.csv"
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

ARXIV_PDF_TEMPLATE = "https://arxiv.org/pdf/{arxiv_id}.pdf"

def arxiv_id_from_url(url: str):
    """Extract arXiv ID from abstract or PDF URL."""
    url = url.rstrip("/")
    if "arxiv.org/abs/" in url:
        return url.split("arxiv.org/abs/")[-1]
    if "arxiv.org/pdf/" in url:
        return url.split("arxiv.org/pdf/")[-1].replace(".pdf", "")
    return None

def get_pdf_url(row: dict) -> str | None:
    """Return a direct PDF download URL from manifest row."""
    link = row["link_doi"].strip()
    arxiv_id = arxiv_id_from_url(link)
    if arxiv_id:
        return ARXIV_PDF_TEMPLATE.format(arxiv_id=arxiv_id)
    # For non-arXiv sources, return None (manual download needed)
    return None

def download_pdf(source_id: str, pdf_url: str, dest_path: Path) -> bool:
    """Download a single PDF. Returns True on success."""
    if dest_path.exists():
        print(f"  [SKIP] {source_id} already exists at {dest_path}")
        return True
    try:
        headers = {"User-Agent": "Mozilla/5.0 (research project; contact: student@university.edu)"}
        req = urllib.request.Request(pdf_url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read()
        with open(dest_path, "wb") as f:
            f.write(content)
        size_kb = len(content) / 1024
        print(f"  [OK]   {source_id} -> {dest_path.name} ({size_kb:.0f} KB)")
        return True
    except Exception as e:
        print(f"  [FAIL] {source_id}: {e}")
        return False

def main():
    print("=" * 60)
    print("Phase 2 — Source Downloader")
    print("=" * 60)

    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    success, failed, skipped = [], [], []

    for row in rows:
        source_id = row["source_id"]
        local_file = Path(row["local_file"])
        pdf_url = get_pdf_url(row)

        print(f"\n[{source_id}]")
        if pdf_url is None:
            print(f"  [MANUAL] No automatic download available for: {row['link_doi']}")
            print(f"           Please manually save PDF to: {local_file}")
            skipped.append(source_id)
            continue

        dest = RAW_DIR / local_file.name
        ok = download_pdf(source_id, pdf_url, dest)
        if ok:
            success.append(source_id)
        else:
            failed.append(source_id)

        time.sleep(2)  # Be polite to arXiv servers

    print("\n" + "=" * 60)
    print(f"Downloaded:      {len(success)}")
    print(f"Manual needed:   {len(skipped)}")
    print(f"Failed:          {len(failed)}")
    if failed:
        print(f"Failed sources:  {', '.join(failed)}")
    if skipped:
        print(f"\nManual download needed for:")
        for s in skipped:
            matching = next(r for r in rows if r["source_id"] == s)
            print(f"  {s}: {matching['link_doi']}")
    print("=" * 60)

if __name__ == "__main__":
    main()
