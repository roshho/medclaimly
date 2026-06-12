#!/usr/bin/env python3
"""
Build a RAG-ready policy corpus from local CMS MCD CSV exports.

Inputs:
- rules/MCD/ncd/ncd_csv/ncd_trkg.csv
- rules/MCD/past-and-present-articles/all_article_csv/article.csv

Outputs:
- rules/policy_corpus/docs/*.txt (clean, chunked text files)
- rules/policy_corpus/manifest.csv (metadata per chunk)

Default topic filter focuses on respiratory/ventilator and short-stay review context.
"""

from __future__ import annotations

import argparse
import csv
import html
import re
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NCD_CSV = PROJECT_ROOT / "rules" / "MCD" / "ncd" / "ncd_csv" / "ncd_trkg.csv"
DEFAULT_ARTICLE_CSV = (
    PROJECT_ROOT
    / "rules"
    / "MCD"
    / "past-and-present-articles"
    / "all_article_csv"
    / "article.csv"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "rules" / "policy_corpus"

DEFAULT_TOPIC_TERMS = [
    "drg 207",
    "respiratory",
    "ventilator",
    "ventilation",
    "pneumonia",
    "pulmonary",
    "short-stay",
    "short stay",
    "two-midnight",
    "mechanical ventilation",
    "tracheostomy",
    "sepsis",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a clean text corpus from local CMS MCD policy exports."
    )
    parser.add_argument("--ncd-csv", type=Path, default=DEFAULT_NCD_CSV)
    parser.add_argument("--article-csv", type=Path, default=DEFAULT_ARTICLE_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--topic",
        nargs="+",
        default=DEFAULT_TOPIC_TERMS,
        help="Topic terms used for include filtering (OR semantics).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1400,
        help="Target max chunk size in characters.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=180,
        help="Approximate overlap in characters between consecutive chunks.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap on number of source records emitted per source table.",
    )
    return parser.parse_args()


def clean_html_text(text: str) -> str:
    if not text:
        return ""
    decoded = html.unescape(text)
    without_tags = re.sub(r"<[^>]+>", " ", decoded)
    without_controls = without_tags.replace("\x00", " ")
    normalized = re.sub(r"\s+", " ", without_controls).strip()
    return normalized


def safe_value(row: dict[str, str], key: str) -> str:
    value = row.get(key, "")
    return value if value is not None else ""


def contains_topic(text: str, topic_terms: Iterable[str]) -> bool:
    if not text:
        return False
    haystack = text.lower()
    return any(term.lower() in haystack for term in topic_terms)


def chunk_text(text: str, chunk_size: int, overlap_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start_idx = 0

    while start_idx < len(words):
        current_words: list[str] = []
        current_len = 0
        idx = start_idx

        while idx < len(words):
            next_word = words[idx]
            projected = current_len + len(next_word) + (1 if current_words else 0)
            if projected > chunk_size and current_words:
                break
            current_words.append(next_word)
            current_len = projected
            idx += 1

        chunk = " ".join(current_words).strip()
        if chunk:
            chunks.append(chunk)

        if idx >= len(words):
            break

        back_chars = 0
        back_words = 0
        reverse_idx = len(current_words) - 1
        while reverse_idx >= 0 and back_chars < overlap_chars:
            back_chars += len(current_words[reverse_idx]) + 1
            back_words += 1
            reverse_idx -= 1

        start_idx = max(start_idx + 1, idx - back_words)

    return chunks


def write_chunk(
    docs_dir: Path,
    source: str,
    source_id: str,
    version: str,
    title: str,
    chunk_number: int,
    total_chunks: int,
    body: str,
) -> Path:
    base_name = f"{source}_{source_id}_v{version}_part{chunk_number:03d}.txt"
    file_path = docs_dir / re.sub(r"[^a-zA-Z0-9._-]", "_", base_name)

    header = [
        f"source: {source}",
        f"source_id: {source_id}",
        f"version: {version}",
        f"title: {title}",
        f"chunk: {chunk_number}/{total_chunks}",
        "",
    ]
    file_path.write_text("\n".join(header) + body + "\n", encoding="utf-8")
    return file_path


def build_ncd_body(row: dict[str, str]) -> str:
    pieces = [
        safe_value(row, "itm_srvc_desc"),
        safe_value(row, "indctn_lmtn"),
        safe_value(row, "xref_txt"),
        safe_value(row, "othr_txt"),
        safe_value(row, "rev_hstry"),
    ]
    cleaned = [clean_html_text(piece) for piece in pieces if piece]
    return "\n\n".join(piece for piece in cleaned if piece)


def build_article_body(row: dict[str, str]) -> str:
    pieces = [
        safe_value(row, "description"),
        safe_value(row, "other_comments"),
        safe_value(row, "cms_cov_policy"),
    ]
    cleaned = [clean_html_text(piece) for piece in pieces if piece]
    return "\n\n".join(piece for piece in cleaned if piece)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else (PROJECT_ROOT / args.out_dir)
    docs_dir = out_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    if args.chunk_size <= 100:
        raise ValueError("--chunk-size must be > 100")
    if args.chunk_overlap < 0:
        raise ValueError("--chunk-overlap must be >= 0")

    csv.field_size_limit(min(10**9, 2**31 - 1))

    manifest_rows: list[dict[str, str | int]] = []
    emitted_counts = {"ncd": 0, "article": 0}

    # NCD source
    with args.ncd_csv.open("r", encoding="latin-1", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            title = clean_html_text(safe_value(row, "NCD_mnl_sect_title"))
            source_id = safe_value(row, "NCD_id") or "unknown"
            version = safe_value(row, "NCD_vrsn_num") or "0"
            keyword_text = clean_html_text(safe_value(row, "ncd_keyword"))
            body = build_ncd_body(row)

            if not body:
                continue

            index_text = " ".join([title, keyword_text, body[:1000]])
            if not contains_topic(index_text, args.topic):
                continue

            chunks = chunk_text(body, chunk_size=args.chunk_size, overlap_chars=args.chunk_overlap)
            if not chunks:
                continue

            for i, chunk in enumerate(chunks, start=1):
                file_path = write_chunk(
                    docs_dir=docs_dir,
                    source="ncd",
                    source_id=source_id,
                    version=version,
                    title=title,
                    chunk_number=i,
                    total_chunks=len(chunks),
                    body=chunk,
                )
                manifest_rows.append(
                    {
                        "source": "ncd",
                        "source_id": source_id,
                        "version": version,
                        "title": title,
                        "chunk_number": i,
                        "chunk_total": len(chunks),
                        "file_path": str(file_path.relative_to(PROJECT_ROOT)),
                    }
                )

            emitted_counts["ncd"] += 1
            if args.max_records is not None and emitted_counts["ncd"] >= args.max_records:
                break

    # Article source
    with args.article_csv.open("r", encoding="latin-1", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            title = clean_html_text(safe_value(row, "title"))
            source_id = safe_value(row, "article_id") or "unknown"
            version = safe_value(row, "article_version") or "0"
            keyword_text = clean_html_text(safe_value(row, "keywords"))
            body = build_article_body(row)

            if not body:
                continue

            index_text = " ".join([title, keyword_text, body[:1000]])
            if not contains_topic(index_text, args.topic):
                continue

            chunks = chunk_text(body, chunk_size=args.chunk_size, overlap_chars=args.chunk_overlap)
            if not chunks:
                continue

            for i, chunk in enumerate(chunks, start=1):
                file_path = write_chunk(
                    docs_dir=docs_dir,
                    source="article",
                    source_id=source_id,
                    version=version,
                    title=title,
                    chunk_number=i,
                    total_chunks=len(chunks),
                    body=chunk,
                )
                manifest_rows.append(
                    {
                        "source": "article",
                        "source_id": source_id,
                        "version": version,
                        "title": title,
                        "chunk_number": i,
                        "chunk_total": len(chunks),
                        "file_path": str(file_path.relative_to(PROJECT_ROOT)),
                    }
                )

            emitted_counts["article"] += 1
            if args.max_records is not None and emitted_counts["article"] >= args.max_records:
                break

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "source",
            "source_id",
            "version",
            "title",
            "chunk_number",
            "chunk_total",
            "file_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("=" * 80)
    print("Policy corpus build complete")
    print("=" * 80)
    print(f"NCD records emitted:     {emitted_counts['ncd']:,}")
    print(f"Article records emitted: {emitted_counts['article']:,}")
    print(f"Total chunks written:    {len(manifest_rows):,}")
    print(f"Corpus docs directory:   {docs_dir.relative_to(PROJECT_ROOT)}")
    print(f"Manifest:                {manifest_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
