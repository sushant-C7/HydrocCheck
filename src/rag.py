"""Local retrieval-augmented generation (RAG) retrieval for HydroCheck.

Documents are curated Markdown/text notes stored in knowledge_base/. At startup,
TF-IDF indexes the local text; queries do not make web requests. The external
LLM only synthesizes retrieved passages and is instructed to cite supplied URLs.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge_base"
MAX_QUERY_LENGTH = 600
MAX_RESULTS = 5


def _parse_document(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    metadata: dict[str, str] = {}
    body = raw

    # A lightweight front-matter parser avoids requiring a YAML package.
    if raw.startswith("---\n"):
        parts = raw.split("---\n", 2)
        if len(parts) == 3:
            for line in parts[1].splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                metadata[key.strip().lower()] = value.strip().strip('"\'')
            body = parts[2]

    title = metadata.get("title") or path.stem.replace("_", " ").title()
    publisher = metadata.get("publisher", "Unspecified publisher")
    jurisdiction = metadata.get("jurisdiction", "Unspecified jurisdiction")
    source_url = metadata.get("source_url", "")

    # Preserve section headings so a retrieved chunk still has context.
    sections = re.split(r"(?m)(?=^#{1,3}\s+)", body)
    chunks: list[dict[str, Any]] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        heading_match = re.match(r"^#{1,3}\s+(.+?)\s*$", section, flags=re.M)
        heading = heading_match.group(1) if heading_match else title
        content = re.sub(r"^#{1,3}\s+.+?$", "", section, count=1, flags=re.M).strip()
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
        current = ""
        for paragraph in paragraphs:
            # Keep chunks short enough to fit comfortably into an LLM tool result.
            if current and len(current) + len(paragraph) + 2 > 1300:
                chunks.append(_make_chunk(title, publisher, jurisdiction, source_url, heading, current, path))
                current = ""
            if len(paragraph) > 1300:
                if current:
                    chunks.append(_make_chunk(title, publisher, jurisdiction, source_url, heading, current, path))
                    current = ""
                for start in range(0, len(paragraph), 1100):
                    chunks.append(_make_chunk(title, publisher, jurisdiction, source_url, heading, paragraph[start:start + 1100], path))
            else:
                current = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if current:
            chunks.append(_make_chunk(title, publisher, jurisdiction, source_url, heading, current, path))

    return {"title": title, "chunks": chunks}


def _make_chunk(title, publisher, jurisdiction, source_url, heading, text, path):
    return {
        "title": title,
        "publisher": publisher,
        "jurisdiction": jurisdiction,
        "source_url": source_url,
        "heading": heading,
        "text": text,
        "document": path.name,
    }


@lru_cache(maxsize=1)
def _load_index():
    """Load documents and fit the local vectorizer once per process."""
    paths = sorted(
        path for path in KNOWLEDGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}
        and path.name.lower() != "readme.md"
    ) if KNOWLEDGE_DIR.exists() else []

    chunks: list[dict[str, Any]] = []
    for path in paths:
        parsed = _parse_document(path)
        chunks.extend(parsed["chunks"])

    if not chunks:
        return [], None, None

    corpus = [
        f"{item['title']} {item['publisher']} {item['jurisdiction']} {item['heading']} {item['text']}"
        for item in chunks
    ]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=20000,
    )
    matrix = vectorizer.fit_transform(corpus)
    return chunks, vectorizer, matrix


def search_knowledge(query: str, top_k: int = 4) -> dict[str, Any]:
    """Return relevant local passages, scores and citable source metadata."""
    if not isinstance(query, str) or not query.strip():
        return {"query": "", "found": False, "passages": [], "sources": [], "message": "Enter a search query."}

    query = query.strip()[:MAX_QUERY_LENGTH]
    try:
        top_k = max(1, min(int(top_k), MAX_RESULTS))
    except (TypeError, ValueError):
        top_k = 4

    chunks, vectorizer, matrix = _load_index()
    if not chunks or vectorizer is None:
        return {
            "query": query, "found": False, "passages": [], "sources": [],
            "message": "No knowledge-base documents were found. Add .md or .txt files to knowledge_base/ and restart the application.",
        }

    query_vector = vectorizer.transform([query])
    if query_vector.nnz == 0:
        return {
            "query": query, "found": False, "passages": [], "sources": [],
            "message": "The query did not match terms in the local knowledge base. Try different wording or add a relevant source document.",
        }

    scores = cosine_similarity(query_vector, matrix).ravel()
    ranked_indices = scores.argsort()[::-1]
    passages: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    seen_sources: set[tuple[str, str]] = set()

    for idx in ranked_indices:
        score = float(scores[idx])
        if score < 0.025 or len(passages) >= top_k:
            break
        chunk = chunks[int(idx)]
        passages.append({
            "title": chunk["title"],
            "publisher": chunk["publisher"],
            "jurisdiction": chunk["jurisdiction"],
            "source_url": chunk["source_url"],
            "section": chunk["heading"],
            "relevance": round(score, 4),
            "excerpt": chunk["text"][:1400],
        })
        key = (chunk["title"], chunk["source_url"])
        if key not in seen_sources:
            sources.append({
                "title": chunk["title"],
                "publisher": chunk["publisher"],
                "jurisdiction": chunk["jurisdiction"],
                "source_url": chunk["source_url"],
            })
            seen_sources.add(key)

    return {
        "query": query,
        "found": bool(passages),
        "passages": passages,
        "sources": sources,
        "message": "Retrieved local knowledge-base passages." if passages else "No sufficiently relevant passage was found. Try a more specific query or add a trusted document.",
        "retrieval_method": "TF-IDF term and bigram similarity",
    }


def knowledge_base_status() -> dict[str, Any]:
    chunks, _, _ = _load_index()
    return {
        "documents_indexed": len({item["document"] for item in chunks}),
        "passages_indexed": len(chunks),
        "knowledge_directory": str(KNOWLEDGE_DIR),
        "retrieval_method": "TF-IDF term and bigram similarity",
    }
