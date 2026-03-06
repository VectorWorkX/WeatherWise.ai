"""RAG retrieval tool — queries Pinecone for atmospheric science knowledge.

Lightweight runtime query: fetches summary matches from Pinecone, then
retrieves parent content from vector metadata.
This module must NOT import any ingestion code (unstructured, etc.).
"""

from __future__ import annotations

import os
from typing import Any

# Relevance threshold — Pinecone cosine returns similarity (0–1), higher = more relevant
RELEVANCE_THRESHOLD = 0.3
TOP_K_CANDIDATES = 8
TOP_K_RESULTS = 5

_vectorstore_cache = None


def _get_vectorstore():
    """Return a cached Pinecone vectorstore for similarity search."""
    global _vectorstore_cache
    if _vectorstore_cache is None:
        from langchain_openai import OpenAIEmbeddings
        from langchain_pinecone import PineconeVectorStore

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.environ["OPENAI_API_KEY"],
        )
        _vectorstore_cache = PineconeVectorStore(
            index_name=os.environ["PINECONE_INDEX_NAME"],
            embedding=embeddings,
            pinecone_api_key=os.environ["PINECONE_API_KEY"],
        )
    return _vectorstore_cache


def _get_index_stats() -> int:
    """Return total vector count from the Pinecone index."""
    from pinecone import Pinecone

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    idx = pc.Index(os.environ["PINECONE_INDEX_NAME"])
    stats = idx.describe_index_stats()
    return stats.get("total_vector_count", 0)


def search_knowledge_base(query: str) -> list[str]:
    """Search the WeatherWise atmospheric science knowledge base.

    Queries Pinecone for summary matches, filters by relevance score,
    then retrieves parent content from vector metadata.

    Args:
        query: A natural-language question about atmospheric science.

    Returns:
        List of formatted knowledge strings with source citations.
        Returns a "no results" message if the knowledge base is empty
        or no relevant documents are found.
    """
    try:
        total_vectors = _get_index_stats()
    except Exception as e:
        return [f"Failed to connect to knowledge base: {e}"]

    if total_vectors == 0:
        return [
            "The knowledge base is currently empty. No atmospheric science "
            "documents have been ingested yet."
        ]

    try:
        vectorstore = _get_vectorstore()
    except Exception as e:
        return [f"Failed to connect to knowledge base: {e}"]

    # Retrieve candidates, filter by relevance
    # Pinecone similarity_search_with_score returns (doc, score) where score is cosine similarity (0–1)
    raw_results = vectorstore.similarity_search_with_score(query, k=TOP_K_CANDIDATES)
    results = [(doc, score) for doc, score in raw_results if score >= RELEVANCE_THRESHOLD]

    if not results:
        return [f"No relevant documents found in the knowledge base for: {query}"]

    results = results[:TOP_K_RESULTS]

    # Build response from metadata
    sections: list[str] = []

    for rank, (doc, score) in enumerate(results, 1):
        meta = doc.metadata
        source = meta.get("source_pdf", "Unknown source")
        elem_type = meta.get("element_type", "text")
        page = meta.get("page_number", "?")
        parent_content = meta.get("parent_content", "")

        # Use parent content from metadata if available, else fall back to summary
        content = parent_content if parent_content else doc.page_content

        if elem_type == "image":
            section = (
                f"[Source #{rank}: {source}, p.{page} — Chart/Figure]\n"
                f"Visual analysis: {content}"
            )
        elif elem_type == "table":
            section = (
                f"[Source #{rank}: {source}, p.{page} — Table]\n"
                f"Table data: {content}"
            )
        else:
            section = (
                f"[Source #{rank}: {source}, p.{page}]\n"
                f"{content}"
            )

        sections.append(section)

    return sections
