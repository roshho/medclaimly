#!/usr/bin/env python3
"""TF-IDF RAG engine over the LCD/NCD policy corpus.

Loads all policy document chunks from the manifest + text files,
fits a TF-IDF vectorizer, and retrieves top-k relevant chunks
for a given query string.

Usage as standalone test:
    python src/TTS/policy_rag.py "ventilator support respiratory failure COPD"
"""
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_DIR = PROJECT_ROOT / "rules" / "policy_corpus"


class PolicyRAG:
    """TF-IDF retrieval over Medicare LCD/NCD policy chunks."""

    def __init__(
        self,
        corpus_dir: Path | None = None,
        corpus_dirs: list[Path] | None = None,
    ) -> None:
        # Backward compatibility: single corpus_dir still works.
        if corpus_dirs:
            self.corpus_dirs = [Path(p) for p in corpus_dirs]
        elif corpus_dir:
            self.corpus_dirs = [Path(corpus_dir)]
        else:
            self.corpus_dirs = [DEFAULT_CORPUS_DIR]

        self.chunks: list[dict[str, Any]] = []
        self.texts: list[str] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix = None

        self._load_corpus()
        self._fit_tfidf()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_corpus(self) -> None:
        """Read one or more manifest CSV files and load each chunk text once."""
        existing_manifests = [d / "manifest.csv" for d in self.corpus_dirs if (d / "manifest.csv").exists()]
        if not existing_manifests:
            raise FileNotFoundError(
                "No policy corpus manifest found in any configured path. "
                "Run the policy corpus builder first."
            )

        seen: set[str] = set()
        for manifest_path in existing_manifests:
            with manifest_path.open("r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    file_path = PROJECT_ROOT / row["file_path"]
                    if not file_path.exists():
                        continue

                    text = file_path.read_text(encoding="utf-8").strip()
                    if not text:
                        continue

                    # Strip metadata header lines when present.
                    body_lines: list[str] = []
                    past_header = False
                    for line in text.splitlines():
                        if past_header:
                            body_lines.append(line)
                        elif line.strip() == "" and not past_header:
                            past_header = True
                        elif not line.startswith(("source:", "source_id:", "version:", "title:", "chunk:")):
                            body_lines.append(line)
                            past_header = True

                    body = "\n".join(body_lines).strip()
                    if not body:
                        body = text

                    dedup_material = "|".join(
                        [
                            row.get("source", ""),
                            row.get("source_id", ""),
                            row.get("version", ""),
                            row.get("chunk_number", ""),
                            body,
                        ]
                    )
                    dedup_key = hashlib.sha1(dedup_material.encode("utf-8", errors="ignore")).hexdigest()
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                    self.chunks.append(
                        {
                            "source": row["source"],
                            "source_id": row["source_id"],
                            "version": row["version"],
                            "title": row["title"],
                            "chunk_number": row["chunk_number"],
                            "chunk_total": row["chunk_total"],
                            "file_path": str(file_path),
                        }
                    )
                    self.texts.append(body)

    def _fit_tfidf(self) -> None:
        """Fit TF-IDF vectorizer on the loaded corpus."""
        if not self.texts:
            return
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=20000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def corpus_size(self) -> int:
        return len(self.chunks)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Return top-k policy chunks most relevant to *query*.

        Each result dict contains:
            title, source, source_id, chunk_number, chunk_text, score
        """
        if self.vectorizer is None or self.tfidf_matrix is None:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # argsort descending
        ranked_indices = scores.argsort()[::-1][:top_k]

        results: list[dict[str, Any]] = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score <= 0.0:
                break
            chunk_meta = self.chunks[idx]
            results.append(
                {
                    "title": chunk_meta["title"],
                    "source": chunk_meta["source"],
                    "source_id": chunk_meta["source_id"],
                    "chunk_number": chunk_meta["chunk_number"],
                    "chunk_text": self.texts[idx][:2000],  # truncate for LLM context
                    "score": round(score, 4),
                }
            )
        return results

    def format_for_prompt(self, results: list[dict[str, Any]]) -> str:
        """Format retrieval results as a text block for injection into an LLM prompt."""
        if not results:
            return "(No relevant policy documents found.)"

        parts: list[str] = []
        for i, r in enumerate(results, 1):
            parts.append(
                f"--- Policy Excerpt {i} (score={r['score']}) ---\n"
                f"Source: {r['source'].upper()} {r['source_id']} — {r['title']}\n"
                f"{r['chunk_text']}\n"
            )
        return "\n".join(parts)


# ----------------------------------------------------------------------
# Standalone test
# ----------------------------------------------------------------------

def main() -> None:
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "ventilator support respiratory failure COPD length of stay"
    )
    print(f"Loading policy corpus from {DEFAULT_CORPUS_DIR} ...")
    rag = PolicyRAG()
    print(f"Corpus loaded: {rag.corpus_size} chunks indexed.\n")

    print(f"Query: {query!r}\n")
    results = rag.retrieve(query, top_k=5)
    for r in results:
        print(f"  [{r['score']:.4f}] {r['source'].upper()} {r['source_id']} — {r['title']} (chunk {r['chunk_number']})")
        print(f"           {r['chunk_text'][:200]}...")
        print()

    if not results:
        print("  No results found.")


if __name__ == "__main__":
    main()
