"""PDF downloader for corpus sources listed in the data manifest."""

import csv
import time
import urllib.request
import urllib.error
from pathlib import Path

from src.config import MANIFEST_PATH, RAW_DIR, REQUEST_TIMEOUT, REQUEST_DELAY_S

_ARXIV_PDF = "https://arxiv.org/pdf/{arxiv_id}.pdf"


def _arxiv_id_from_url(url: str) -> str | None:
    url = url.rstrip("/")
    for prefix in ("arxiv.org/abs/", "arxiv.org/pdf/"):
        if prefix in url:
            return url.split(prefix)[-1].replace(".pdf", "")
    return None


def _resolve_pdf_url(row: dict) -> str | None:
    aid = _arxiv_id_from_url(row["url_or_doi"].strip())
    return _ARXIV_PDF.format(arxiv_id=aid) if aid else None


def _download(source_id: str, pdf_url: str, dest: Path) -> bool:
    if dest.exists():
        print(f"  [SKIP] {source_id} — already at {dest.name}")
        return True
    try:
        req = urllib.request.Request(
            pdf_url,
            headers={"User-Agent": "Mozilla/5.0 (research project)"},
        )
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            data = resp.read()
        dest.write_bytes(data)
        print(f"  [OK]   {source_id} -> {dest.name} ({len(data)/1024:.0f} KB)")
        return True
    except Exception as exc:
        print(f"  [FAIL] {source_id}: {exc}")
        return False


def main() -> None:
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    ok, fail, manual = [], [], []
    for row in rows:
        sid = row["source_id"]
        fname = Path(row["raw_path"]).name
        url = _resolve_pdf_url(row)
        print(f"\n[{sid}]")
        if url is None:
            print(f"  [MANUAL] Save PDF to data/raw/{fname}")
            manual.append(sid)
            continue
        if _download(sid, url, RAW_DIR / fname):
            ok.append(sid)
        else:
            fail.append(sid)
        time.sleep(REQUEST_DELAY_S)

    print(f"\nDone: {len(ok)} downloaded, {len(manual)} manual, {len(fail)} failed")
    if fail:
        print(f"  Failed: {', '.join(fail)}")


if __name__ == "__main__":
    main()
