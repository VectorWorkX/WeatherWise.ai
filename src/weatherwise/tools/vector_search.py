"""RAG retrieval tool — queries ChromaDB for atmospheric science knowledge.

Lightweight runtime query: fetches summary matches from ChromaDB, then
retrieves original parent documents from the local file store.
This module must NOT import any ingestion code (unstructured, etc.).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCSTORE_DIR = DATA_DIR / "docstore"
CHROMA_DIR = DATA_DIR / "chroma_db"
COLLECTION_NAME = "weatherwise_knowledge"

# Cosine distance threshold — lower = more relevant
RELEVANCE_THRESHOLD = 1.2
TOP_K_CANDIDATES = 8
TOP_K_RESULTS = 5

_vectorstore_cache = None


def _get_vectorstore():
    """Return a cached ChromaDB vectorstore for similarity search."""
    global _vectorstore_cache
    if _vectorstore_cache is None:
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.environ["OPENAI_API_KEY"],
        )
        _vectorstore_cache = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )
    return _vectorstore_cache


def _load_parent(doc_id: str) -> dict[str, Any] | None:
    """Fetch the original parent document from the file store."""
    filepath = DOCSTORE_DIR / f"{doc_id}.json"
    if filepath.exists():
        return json.loads(filepath.read_text())
    return None


def search_knowledge_base(query: str) -> list[str]:
    """Search the WeatherWise atmospheric science knowledge base.

    Queries ChromaDB for summary matches, filters by relevance score,
    then retrieves full parent documents from the docstore.

    Args:
        query: A natural-language question about atmospheric science.

    Returns:
        List of formatted knowledge strings with source citations.
        Returns a "no results" message if the knowledge base is empty
        or no relevant documents are found.
    """
    # Check if ChromaDB directory exists and has data
    if not CHROMA_DIR.exists():
        return [
            "The knowledge base has not been set up yet. "
            "Run `python scripts/build_knowledge.py --all` to ingest documents."
        ]

    try:
        vectorstore = _get_vectorstore()
    except Exception as e:
        return [f"Failed to connect to knowledge base: {e}"]

    if vectorstore._collection.count() == 0:
        return [
            "The knowledge base is currently empty. No atmospheric science "
            "documents have been ingested yet."
        ]

    # Retrieve candidates, filter by relevance
    raw_results = vectorstore.similarity_search_with_score(query, k=TOP_K_CANDIDATES)
    results = [(doc, score) for doc, score in raw_results if score < RELEVANCE_THRESHOLD]

    if not results:
        return [f"No relevant documents found in the knowledge base for: {query}"]

    results = results[:TOP_K_RESULTS]

    # Build response with parent documents
    sections: list[str] = []

    for rank, (doc, score) in enumerate(results, 1):
        meta = doc.metadata
        doc_id = meta.get("doc_id", "")
        source = meta.get("source_pdf", "Unknown source")
        elem_type = meta.get("element_type", "text")
        page = meta.get("page_number", "?")

        parent = _load_parent(doc_id)

        if parent:
            parent_content = parent["content"]

            if elem_type == "image":
                section = (
                    f"[Source #{rank}: {source}, p.{page} — Chart/Figure]\n"
                    f"Visual analysis: {parent_content}"
                )
            elif elem_type == "table":
                section = (
                    f"[Source #{rank}: {source}, p.{page} — Table]\n"
                    f"Table data: {parent_content}"
                )
            else:
                section = (
                    f"[Source #{rank}: {source}, p.{page}]\n"
                    f"{parent_content}"
                )
        else:
            section = (
                f"[Source #{rank}: {source}, p.{page}]\n"
                f"{doc.page_content}"
            )

        sections.append(section)

    return sections
