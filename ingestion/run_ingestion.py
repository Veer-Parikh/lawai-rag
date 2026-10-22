"""
ingestion/run_ingestion.py — CLI orchestrator for the full ingestion pipeline.

Usage:
    python -m ingestion.run_ingestion --path data/statutes/criminal --doc-type statute
    python -m ingestion.run_ingestion --path data/ --doc-type all
    python -m ingestion.run_ingestion --path data/ --doc-type all --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import uuid
from pathlib import Path
from typing import Callable

import config

from ingestion.embedder import embed_texts
from ingestion.ledger import IngestionLedger
from ingestion.judgment_chunker import chunk_judgment
from ingestion.mapping_parser import build_cross_ref_index, parse_mapping_doc
from ingestion.metadata_builder import (
    build_judgment_metadata,
    build_mapping_metadata,
    build_statute_metadata,
)
from ingestion.pdf_loader import load_pdf
from ingestion.pinecone_uploader import ensure_index_exists, upsert_chunks
from ingestion.statute_chunker import chunk_statute
from utils.logger import get_logger
from utils.text_cleaner import clean_text

logger = get_logger(__name__)

# ─── Act Name Inference ───────────────────────────────────────────────────────

_FILENAME_TO_ACT = {
    "ipc": "Indian Penal Code",
    "bns": "Bharatiya Nyaya Sanhita",
    "bnss": "Bharatiya Nagarik Suraksha Sanhita",
    "crpc": "Code of Criminal Procedure",
    "evidence": "Indian Evidence Act",
    "contract": "Contract Act",
    "specific_relief": "Specific Relief Act",
    "transfer_of_property": "Transfer of Property Act",
    "consumer": "Consumer Protection Act",
    "it_act": "Information Technology Act",
    "hindu_marriage": "Hindu Marriage Act",
    "hindu_succession": "Hindu Succession Act",
    "special_marriage": "Special Marriage Act",
    "constitution": "Constitution of India",
}


def _infer_act_name(path: Path) -> str:
    name_lower = path.stem.lower().replace(" ", "_").replace("-", "_")
    for key, act in _FILENAME_TO_ACT.items():
        if key in name_lower:
            return act
    return path.stem.replace("_", " ").title()


def _load_source_metadata(path: Path) -> dict:
    """Load downloader provenance stored beside a corpus file."""
    metadata_path = path.with_name(path.name + ".metadata.json")
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"Failed to read source metadata {metadata_path}: {exc}")
        return {}


def stable_chunk_id(pdf_path: Path, index: int) -> str:
    """Deterministic chunk ID so re-ingesting a file overwrites its vectors."""
    try:
        key = pdf_path.resolve().relative_to(Path(config.DATA_DIR).resolve()).as_posix()
    except ValueError:
        key = pdf_path.name
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"lawai-corpus:{key}#{index}"))


# ─── Cross-reference Index (built once from mappings/) ────────────────────────

def _build_global_cross_ref_index(data_dir: Path) -> dict[str, list[str]]:
    mappings_dir = data_dir / "mappings"
    if not mappings_dir.exists():
        return {}
    all_pairs = []
    for pdf_path in mappings_dir.rglob("*.pdf"):
        try:
            pages = load_pdf(pdf_path)
            pairs = parse_mapping_doc(pages)
            all_pairs.extend(pairs)
        except Exception as exc:
            logger.warning(f"Failed to parse mapping doc {pdf_path.name}: {exc}")
    return build_cross_ref_index(all_pairs)


# ─── Per-file Ingestion ───────────────────────────────────────────────────────

def ingest_file(
    pdf_path: Path,
    doc_type: str,
    cross_ref_index: dict,
    dry_run: bool = False,
) -> dict:
    """
    Ingest a single PDF file through the full pipeline.
    Returns a result dict: {file, chunks_created, chunks_uploaded, skipped, error}
    """
    result = {
        "file": str(pdf_path),
        "chunks_created": 0,
        "chunks_uploaded": 0,
        "error": None,
    }

    try:
        pages = load_pdf(pdf_path)
        if not pages:
            result["error"] = "No pages extracted"
            return result
        source_meta = _load_source_metadata(pdf_path)
        metadata_kwargs = {
            "source_url": source_meta.get("url"),
            "publisher": source_meta.get("publisher"),
            "publication_date": source_meta.get("publication_date"),
            "effective_date": source_meta.get("effective_date"),
            "document_id": source_meta.get("source_id"),
            "document_hash": source_meta.get("sha256"),
            "retrieved_at": source_meta.get("retrieved_at"),
            "language": source_meta.get("language", "en"),
            "source_authority": source_meta.get("authority"),
        }

        # ── Route to chunker ──────────────────────────────────────────────────
        if doc_type == "statute":
            act_name = _infer_act_name(pdf_path)
            raw_chunks = chunk_statute(pages, act_name=act_name)
            metadata_list = [
                build_statute_metadata(
                    chunk=rc,
                    source_path=str(pdf_path),
                    total_chunks=len(raw_chunks),
                    act_name=act_name,
                    cross_ref_index=cross_ref_index,
                    **metadata_kwargs,
                )
                for rc in raw_chunks
            ]
            texts = [rc["text"] for rc in raw_chunks]
            namespace = metadata_list[0]["namespace"] if metadata_list else "misc"

        elif doc_type == "judgment":
            raw_chunks, case_name, citation = chunk_judgment(pages)
            year_match = re.search(r"20\d{2}", str(pdf_path))
            year = int(year_match.group(0)) if year_match else None
            metadata_list = [
                build_judgment_metadata(
                    chunk=rc,
                    source_path=str(pdf_path),
                    total_chunks=len(raw_chunks),
                    case_name=case_name,
                    citation=citation,
                    year=year,
                    **metadata_kwargs,
                )
                for rc in raw_chunks
            ]
            texts = [rc["text"] for rc in raw_chunks]
            namespace = config.NAMESPACE_JUDGMENTS

        elif doc_type == "mapping":
            pairs = parse_mapping_doc(pages)
            metadata_list = []
            texts = []
            for i, pair in enumerate(pairs):
                chunk_id = str(uuid.uuid4())
                text = (
                    f"{pair['ipc_ref']} corresponds to {pair['bns_ref']}. "
                    f"{pair['description']}"
                )
                texts.append(text)
                metadata_list.append(
                    build_mapping_metadata(
                        chunk_id=chunk_id,
                        source_path=str(pdf_path),
                        chunk_index=i,
                        total_chunks=len(pairs),
                        ipc_ref=pair["ipc_ref"],
                        bns_ref=pair["bns_ref"],
                        description=pair["description"],
                    )
                )
            namespace = config.NAMESPACE_MAPPINGS
        else:
            result["error"] = f"Unknown doc_type: {doc_type}"
            return result

        result["chunks_created"] = len(texts)

        if not texts:
            result["error"] = "No chunks produced"
            return result

        if dry_run:
            logger.info(f"[DRY RUN] {pdf_path.name}: would upload {len(texts)} chunks")
            return result

        for index, meta in enumerate(metadata_list):
            meta["chunk_id"] = stable_chunk_id(pdf_path, index)

        # ── Embed ─────────────────────────────────────────────────────────────
        vectors = embed_texts(texts)

        # Store text in metadata for retrieval
        for meta, text in zip(metadata_list, texts):
            meta["text"] = text

        # ── Upsert ────────────────────────────────────────────────────────────
        n_upserted = upsert_chunks(metadata_list, namespace, vectors)
        result["chunks_uploaded"] = n_upserted

    except Exception as exc:
        logger.error(f"Failed to ingest {pdf_path}: {exc}")
        result["error"] = str(exc)

    return result


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def run(
    path: str,
    doc_type: str = "all",
    dry_run: bool = False,
    *,
    job_id: str | None = None,
    skip_ingested: bool = False,
    ledger: IngestionLedger | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    """
    Orchestrate ingestion for a path (a directory or a single PDF).
    Every file attempt is recorded in the ingestion ledger.
    Returns summary stats.
    """
    ledger = ledger or IngestionLedger(config.INGESTION_LEDGER_PATH)
    if not dry_run:
        ensure_index_exists()

    data_dir = Path(config.DATA_DIR)
    target = Path(path)

    # Build cross-reference index from mappings/ (always)
    cross_ref_index = _build_global_cross_ref_index(data_dir)
    logger.info(f"Cross-reference index: {len(cross_ref_index)} entries")

    # Collect PDFs
    pdf_files: list[tuple[Path, str]] = []  # (path, doc_type)

    def _classify(p: Path) -> str:
        parts = [part.lower() for part in p.parts]
        if "mappings" in parts:
            return "mapping"
        if "supreme_court_judgments" in parts or "judgments" in parts:
            return "judgment"
        if "statutes" in parts or "primary_statutes" in parts or "gazette" in parts:
            return "statute"
        return "statute"  # default for misc

    candidates = [target] if target.is_file() else sorted(target.rglob("*.pdf"))
    for pdf_path in candidates:
        if pdf_path.suffix.lower() != ".pdf":
            continue
        pdf_files.append((pdf_path, _classify(pdf_path) if doc_type == "all" else doc_type))

    if skip_ingested:
        latest = ledger.latest_by_path()
        pdf_files = [
            (pdf_path, dtype)
            for pdf_path, dtype in pdf_files
            if getattr(latest.get(IngestionLedger.normalize(pdf_path.resolve())), "status", None) != "ingested"
        ]

    logger.info(f"Starting ingestion: {len(pdf_files)} PDFs, dry_run={dry_run}")

    stats = {
        "files_found": len(pdf_files),
        "files_processed": 0,
        "total_chunks_created": 0,
        "total_chunks_uploaded": 0,
        "errors": [],
    }

    for position, (pdf_path, dtype) in enumerate(pdf_files, start=1):
        if on_progress:
            on_progress(position - 1, len(pdf_files), pdf_path.name)
        result = ingest_file(pdf_path, dtype, cross_ref_index, dry_run=dry_run)
        try:
            ledger.record(
                path=pdf_path.resolve(),
                doc_type=dtype,
                chunks_created=result["chunks_created"],
                chunks_uploaded=result["chunks_uploaded"],
                error=result["error"],
                dry_run=dry_run,
                job_id=job_id,
                sha256=_load_source_metadata(pdf_path).get("sha256"),
            )
        except Exception as exc:
            logger.warning(f"Ingestion ledger write failed for {pdf_path}: {exc}")
        if result["error"]:
            stats["errors"].append(f"{pdf_path.name}: {result['error']}")
        else:
            stats["files_processed"] += 1
            stats["total_chunks_created"] += result["chunks_created"]
            stats["total_chunks_uploaded"] += result["chunks_uploaded"]

    if on_progress:
        on_progress(len(pdf_files), len(pdf_files), "")
    logger.info(f"Ingestion complete: {stats}")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bharat Law ingestion pipeline")
    parser.add_argument("--path", required=True, help="Path to ingest (file or directory)")
    parser.add_argument(
        "--doc-type",
        choices=["statute", "judgment", "mapping", "all"],
        default="all",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    summary = run(path=args.path, doc_type=args.doc_type, dry_run=args.dry_run)
    print(summary)
