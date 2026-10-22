"""Download and catalog the bounded LawAI source manifest.

This intentionally downloads stable seed files and source portal captures.
Unbounded collections such as all Gazette notices and all Supreme Court
judgments are recorded as discovery pages for reviewed, incremental intake.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MANIFEST_PATH = DATA_DIR / "source_manifest.json"
USER_AGENT = "LawAI-source-ingester/1.0 (+https://github.com/Veer-Parikh/lawai-rag)"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_").lower()
    return value[:140] or "source"


def choose_filename(item: dict, content_type: str | None) -> str:
    if item.get("filename"):
        return item["filename"]
    suffix = ".pdf" if "pdf" in (content_type or "").lower() else ".html"
    return f"{slug(item['source_id'])}{suffix}"


def download(url: str) -> tuple[bytes, str | None]:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    with urlopen(request, timeout=60) as response:
        return response.read(), response.headers.get("Content-Type")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    items = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    results: list[dict] = []
    for item in items:
        relative_dir = Path(item["topic"]) / item["subtopic"]
        target_dir = DATA_DIR / relative_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / choose_filename(item, None)
        metadata_path = target.with_name(target.name + ".metadata.json")
        retrieved_at = utc_now()

        if target.exists() and not args.refresh:
            content = target.read_bytes()
            status = "skipped_existing"
            content_type = "application/pdf" if target.suffix == ".pdf" else "text/html"
            error = None
        else:
            try:
                content, content_type = download(item["url"])
                target = target_dir / choose_filename(item, content_type)
                target.write_bytes(content)
                status = "downloaded"
                error = None
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                content = b""
                content_type = None
                status = "failed"
                error = str(exc)

        record = {
            **item,
            "local_path": str(target.relative_to(ROOT)),
            "metadata_path": str(metadata_path.relative_to(ROOT)),
            "retrieved_at": retrieved_at,
            "content_type": content_type,
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest() if content else None,
            "download_status": status,
            "error": error,
        }
        metadata_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        results.append(record)
        print(f"{status:18} {item['source_id']}: {record['local_path']}")
        time.sleep(0.25)

    manifest = DATA_DIR / "download_manifest.json"
    manifest.write_text(json.dumps({
        "generated_at": utc_now(),
        "root": str(DATA_DIR.relative_to(ROOT)),
        "records": results,
    }, indent=2) + "\n", encoding="utf-8")
    failed = sum(1 for record in results if record["download_status"] == "failed")
    print(f"\nDownloaded/cataloged: {len(results) - failed}/{len(results)}; failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
