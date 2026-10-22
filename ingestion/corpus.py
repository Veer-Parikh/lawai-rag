"""Corpus inventory: files on disk joined with provenance sidecars and the ingestion ledger."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ingestion.ledger import IngestionLedger

# Folders an administrator may upload into. Paths drive doc-type
# classification in run_ingestion._classify, so keep them role-first.
UPLOAD_DESTINATIONS: dict[str, dict[str, str]] = {
    "primary_statutes/current/bns": {"label": "BNS, 2023", "authority": "india_code"},
    "primary_statutes/current/bnss": {"label": "BNSS, 2023", "authority": "india_code"},
    "primary_statutes/current/bsa": {"label": "BSA, 2023", "authority": "india_code"},
    "primary_statutes/historical/ipc": {"label": "IPC, 1860", "authority": "india_code"},
    "primary_statutes/historical/crpc": {"label": "CrPC, 1973", "authority": "india_code"},
    "primary_statutes/historical/iea": {"label": "Indian Evidence Act, 1872", "authority": "india_code"},
    "gazette/enactments": {"label": "Gazette: enactments", "authority": "gazette"},
    "gazette/commencement": {"label": "Gazette: commencement", "authority": "gazette"},
    "gazette/amendments": {"label": "Gazette: amendments", "authority": "gazette"},
    "gazette/rules_and_orders": {"label": "Gazette: rules and orders", "authority": "gazette"},
    "judgments/supreme_court": {"label": "Supreme Court judgments", "authority": "supreme_court"},
    "mappings": {"label": "Section crosswalks", "authority": "mha"},
}

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
_TRACKED_SUFFIXES = {".pdf", ".html", ".htm"}


def safe_filename(name: str) -> str:
    stem = Path(name).stem
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-").lower()[:120]
    if not cleaned:
        raise ValueError("file name has no usable characters")
    return f"{cleaned}.pdf"


def resolve_inside(data_dir: str | Path, relative: str) -> Path:
    """Resolve a user-supplied path and refuse anything outside the data root."""
    root = Path(data_dir).resolve()
    relative = relative.replace("\\", "/").strip()
    if relative.startswith(root.as_posix()):
        candidate = Path(relative).resolve()
    else:
        relative = relative.removeprefix(f"{Path(data_dir).as_posix()}/")
        candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("path must stay inside the data directory")
    return candidate


def _read_sidecar(path: Path) -> dict[str, Any]:
    sidecar = path.with_name(path.name + ".metadata.json")
    if not sidecar.exists():
        return {}
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def scan_corpus(data_dir: str | Path, ledger: IngestionLedger) -> dict[str, Any]:
    root = Path(data_dir)
    latest = ledger.latest_by_path()
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _TRACKED_SUFFIXES:
            continue
        relative = path.relative_to(root)
        meta = _read_sidecar(path)
        entry = latest.get(IngestionLedger.normalize(path.resolve()))
        stat = path.stat()
        is_pdf = path.suffix.lower() == ".pdf"
        empty = stat.st_size == 0
        files.append({
            "path": relative.as_posix(),
            "name": path.name,
            "folder": relative.parent.as_posix(),
            "topic": relative.parts[0] if len(relative.parts) > 1 else "",
            "kind": "pdf" if is_pdf else "html",
            "ingestible": is_pdf and not empty,
            "empty": empty,
            "size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "title": meta.get("title"),
            "publisher": meta.get("publisher"),
            "authority": meta.get("authority"),
            "source_url": meta.get("url"),
            "acquired_via": meta.get("download_status"),
            "retrieved_at": meta.get("retrieved_at"),
            "sha256": meta.get("sha256"),
            "ingestion": entry.to_dict() if entry else None,
        })

    unavailable = []
    manifest_path = root / "download_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for record in manifest.get("records", []):
                if record.get("download_status") == "failed":
                    unavailable.append({
                        "source_id": record.get("source_id"),
                        "title": record.get("title"),
                        "url": record.get("url"),
                        "authority": record.get("authority"),
                        "error": record.get("error"),
                        "retrieved_at": record.get("retrieved_at"),
                    })
        except (OSError, json.JSONDecodeError):
            pass

    pdfs = [f for f in files if f["kind"] == "pdf" and not f["empty"]]
    status_of = lambda f: (f["ingestion"] or {}).get("status")  # noqa: E731
    return {
        "summary": {
            "files": len(files),
            "pdfs": len(pdfs),
            "ingested": sum(1 for f in pdfs if status_of(f) == "ingested"),
            "failed": sum(1 for f in pdfs if status_of(f) == "failed"),
            "pending": sum(1 for f in pdfs if status_of(f) not in ("ingested", "failed")),
            "empty": sum(1 for f in files if f["empty"]),
            "chunks_uploaded": sum((f["ingestion"] or {}).get("chunks_uploaded", 0) for f in pdfs if status_of(f) == "ingested"),
            "bytes": sum(f["size_bytes"] for f in files),
            "unavailable_sources": len(unavailable),
        },
        "files": files,
        "unavailable": unavailable,
        "destinations": [
            {"folder": folder, **info} for folder, info in UPLOAD_DESTINATIONS.items()
        ],
    }


def store_upload(
    data_dir: str | Path,
    *,
    destination: str,
    filename: str,
    content: bytes,
    title: str | None,
    source_url: str | None,
    publisher: str | None,
    uploaded_by: str,
) -> Path:
    if destination not in UPLOAD_DESTINATIONS:
        raise ValueError("choose one of the listed corpus folders")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("file is larger than 50 MB")
    if not content.startswith(b"%PDF"):
        raise ValueError("only PDF files can be added to the corpus")
    if source_url and not source_url.startswith("https://"):
        raise ValueError("source URL must start with https://")

    folder = resolve_inside(data_dir, destination)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / safe_filename(filename)
    if target.exists():
        raise FileExistsError(f"{target.name} already exists in {destination}")

    digest = hashlib.sha256(content).hexdigest()
    target.write_bytes(content)
    sidecar = {
        "source_id": target.stem,
        "title": (title or "").strip() or target.stem.replace("_", " "),
        "url": source_url,
        "publisher": (publisher or "").strip() or None,
        "authority": UPLOAD_DESTINATIONS[destination]["authority"],
        "document_type": "pdf",
        "topic": destination.split("/")[0],
        "subtopic": "/".join(destination.split("/")[1:]),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "size_bytes": len(content),
        "sha256": digest,
        "download_status": "uploaded",
        "uploaded_by": uploaded_by,
    }
    target.with_name(target.name + ".metadata.json").write_text(
        json.dumps(sidecar, indent=2), encoding="utf-8"
    )
    return target
