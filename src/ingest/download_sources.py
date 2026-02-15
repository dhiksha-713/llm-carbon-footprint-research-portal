"""Download PDFs for all corpus sources listed in the data manifest.

Supports arXiv, ACL Anthology, and direct-URL sources.
PDFs are stored locally in data/raw/ and are NOT committed to git.
"""

from __future__ import annotations

import csv
import time
import urllib.request
import urllib.error
from pathlib import Path

from src.config import MANIFEST_PATH, RAW_DIR, REQUEST_TIMEOUT, REQUEST_DELAY_S

_HEADERS = {"User-Agent": "Mozilla/5.0 (research-portal/2.0; academic project)"}

# ── URL resolvers ─────────────────────────────────────────────────────────

def _arxiv_id(url: str) -> str | None:
    """Extract arXiv ID from an arXiv URL."""
    url = url.rstrip("/")
    for prefix in ("arxiv.org/abs/", "arxiv.org/pdf/"):
        if prefix in url:
            return url.split(prefix)[-1].replace(".pdf", "")
    return None


def _resolve_pdf_url(url: str) -> str | None:
    """Convert a manifest URL to a direct PDF download link."""
    url = url.strip()

    # arXiv
    aid = _arxiv_id(url)
    if aid:
        return f"https://arxiv.org/pdf/{aid}.pdf"

    # ACL Anthology (e.g. https://aclanthology.org/2021.sustainlp-1.2)
    if "aclanthology.org/" in url:
        return url.rstrip("/") + ".pdf"

    # Direct PDF links
    if url.lower().endswith(".pdf"):
        return url

    return None


# ── Download ──────────────────────────────────────────────────────────────

def _download(source_id: str, pdf_url: str, dest: Path) -> bool:
    """Download a single PDF.  Skips if file already exists."""
    if dest.exists() and dest.stat().st_size > 1024:
        print(f"  [CACHED] {source_id} -- {dest.name} ({dest.stat().st_size // 1024} KB)")
        return True

    try:
        req = urllib.request.Request(pdf_url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            data = resp.read()
        if len(data) < 1024:
            print(f"  [WARN]  {source_id} -- response too small ({len(data)} bytes), skipping")
            return False
        dest.write_bytes(data)
        print(f"  [OK]    {source_id} -> {dest.name} ({len(data) // 1024} KB)")
        return True
    except Exception as exc:
        print(f"  [FAIL]  {source_id}: {exc}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    with open(MANIFEST_PATH, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    print(f"Manifest: {len(rows)} sources")
    ok, fail, manual = [], [], []

    for row in rows:
        sid = row["source_id"]
        fname = Path(row["raw_path"]).name
        pdf_url = _resolve_pdf_url(row["url_or_doi"])
        print(f"\n[{sid}]")

        if pdf_url is None:
            print(f"  [MANUAL] Cannot resolve URL. Save PDF manually to data/raw/{fname}")
            manual.append(sid)
            continue

        if _download(sid, pdf_url, RAW_DIR / fname):
            ok.append(sid)
        else:
            fail.append(sid)

        time.sleep(REQUEST_DELAY_S)

    print(f"\n{'=' * 50}")
    print(f"Downloaded: {len(ok)} | Cached: (included above) | Manual: {len(manual)} | Failed: {len(fail)}")
    if fail:
        print(f"  Failed: {', '.join(fail)}")
    if manual:
        print(f"  Manual: {', '.join(manual)}")
    print(f"PDFs stored in: {RAW_DIR}")


if __name__ == "__main__":
    main()
