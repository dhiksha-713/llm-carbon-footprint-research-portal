"""Download PDFs for all corpus sources listed in the data manifest.

Supports arXiv, ACL Anthology, DiVA portal, and direct-URL sources.
After downloading, validates each PDF by checking that the expected
paper title appears in the first page text.
PDFs are stored locally in data/raw/ and are NOT committed to git.
"""

from __future__ import annotations

import csv
import logging
import time
import urllib.request
import urllib.error
from pathlib import Path

import fitz

from src.config import MANIFEST_PATH, RAW_DIR, REQUEST_TIMEOUT, REQUEST_DELAY_S

log = logging.getLogger(__name__)
_HEADERS = {"User-Agent": "Mozilla/5.0 (research-portal/2.0; academic project)"}


def _arxiv_id(url: str) -> str | None:
    url = url.rstrip("/")
    for prefix in ("arxiv.org/abs/", "arxiv.org/pdf/"):
        if prefix in url:
            return url.split(prefix)[-1].replace(".pdf", "")
    return None


def _resolve_pdf_url(url: str) -> str | None:
    """Convert a manifest URL to a direct PDF download link."""
    url = url.strip()

    aid = _arxiv_id(url)
    if aid:
        return f"https://arxiv.org/pdf/{aid}.pdf"

    if "aclanthology.org/" in url:
        return url.rstrip("/") + ".pdf"

    if url.lower().endswith(".pdf"):
        return url

    return None


def _download(source_id: str, pdf_url: str, dest: Path, force: bool = False) -> bool:
    """Download a single PDF. Skips if file already exists unless force=True."""
    if not force and dest.exists() and dest.stat().st_size > 1024:
        log.info("[CACHED] %s -- %s (%d KB)", source_id, dest.name, dest.stat().st_size // 1024)
        print(f"  [CACHED] {source_id} -- {dest.name} ({dest.stat().st_size // 1024} KB)")
        return True

    try:
        req = urllib.request.Request(pdf_url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            data = resp.read()
        if len(data) < 1024:
            log.warning("[WARN] %s -- response too small (%d bytes)", source_id, len(data))
            print(f"  [WARN]  {source_id} -- response too small ({len(data)} bytes), skipping")
            return False
        dest.write_bytes(data)
        log.info("[OK] %s -> %s (%d KB)", source_id, dest.name, len(data) // 1024)
        print(f"  [OK]    {source_id} -> {dest.name} ({len(data) // 1024} KB)")
        return True
    except Exception as exc:
        log.error("[FAIL] %s: %s", source_id, exc)
        print(f"  [FAIL]  {source_id}: {exc}")
        return False


def validate_pdf(source_id: str, title: str, pdf_path: Path) -> bool:
    """Check that the PDF first-page text contains keywords from the expected title."""
    if not pdf_path.exists():
        return False
    try:
        doc = fitz.open(str(pdf_path))
        first_pages_text = ""
        for i in range(min(3, len(doc))):
            first_pages_text += doc[i].get_text("text")
        doc.close()

        first_pages_lower = first_pages_text.lower()

        title_words = [w.lower() for w in title.split() if len(w) > 3]
        matches = sum(1 for w in title_words if w in first_pages_lower)
        ratio = matches / len(title_words) if title_words else 0

        if ratio >= 0.4:
            log.info("[VALID] %s - title match %.0f%% (%d/%d words)",
                     source_id, ratio * 100, matches, len(title_words))
            return True
        else:
            log.warning("[MISMATCH] %s - title match only %.0f%% (%d/%d words). "
                        "Expected: '%s'. First 200 chars: '%s'",
                        source_id, ratio * 100, matches, len(title_words),
                        title, first_pages_text[:200].replace("\n", " "))
            print(f"  [MISMATCH] {source_id} - PDF does not match expected title!")
            print(f"    Expected: {title}")
            print(f"    Found:    {first_pages_text[:150].replace(chr(10), ' ')}")
            return False
    except Exception as exc:
        log.error("[VALIDATE-ERR] %s: %s", source_id, exc)
        return False


def main(force_redownload: bool = False) -> dict:
    """Download and validate all PDFs. Returns status dict."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    with open(MANIFEST_PATH, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    print(f"Manifest: {len(rows)} sources")
    log.info("Starting download: %d sources, force=%s", len(rows), force_redownload)
    ok, fail, manual, mismatch = [], [], [], []

    for row in rows:
        sid = row["source_id"]
        title = row["title"]
        fname = Path(row["raw_path"]).name
        dest = RAW_DIR / fname
        pdf_url = _resolve_pdf_url(row["url_or_doi"])
        print(f"\n[{sid}]")

        if pdf_url is None:
            print(f"  [MANUAL] Cannot resolve URL. Save PDF manually to data/raw/{fname}")
            manual.append(sid)
            continue

        if force_redownload and dest.exists():
            dest.unlink()
            print(f"  [REMOVED] Old PDF deleted for re-download")

        if not _download(sid, pdf_url, dest, force=force_redownload):
            fail.append(sid)
            continue

        if not validate_pdf(sid, title, dest):
            mismatch.append(sid)
            dest.unlink(missing_ok=True)
            print(f"  [DELETED] Mismatched PDF removed. Re-download will be attempted next run.")
            fail.append(sid)
            continue

        ok.append(sid)
        time.sleep(REQUEST_DELAY_S)

    print(f"\n{'=' * 50}")
    print(f"Valid: {len(ok)} | Failed: {len(fail)} | Manual: {len(manual)} | Mismatched: {len(mismatch)}")
    if fail:
        print(f"  Failed: {', '.join(fail)}")
    if manual:
        print(f"  Manual: {', '.join(manual)}")
    if mismatch:
        print(f"  Mismatched (wrong paper): {', '.join(mismatch)}")
    print(f"PDFs stored in: {RAW_DIR}")

    return {"ok": ok, "fail": fail, "manual": manual, "mismatch": mismatch}


if __name__ == "__main__":
    main()
