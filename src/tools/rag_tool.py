"""RAG retrieval tool for the WeatherWise agent.

Lightweight runtime query against ChromaDB — fetches summary matches, then
retrieves original parent documents from the local file store.  This module
must NOT import any ingestion code (unstructured, etc.).

Used by the Bucket 5 (Explainability) path of the LangGraph agent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

# ---------------------------------------------------------------------------
# Paths (must match build_knowledge.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCSTORE_DIR = DATA_DIR / "docstore"
CHROMA_DIR = DATA_DIR / "chroma_db"
COLLECTION_NAME = "weatherwise_knowledge"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Public tool
# ---------------------------------------------------------------------------

@tool
def search_knowledge_base(query: str) -> str:
    """Search the WeatherWise atmospheric science knowledge base.

    Use this tool when the user asks 'why' about a weather phenomenon and you
    need scientific background from research papers, technical reports, or
    weather charts.

    Args:
        query: A natural-language question about atmospheric science
               (e.g., 'What causes a heat dome?', 'How does lake effect snow form?').

    Returns:
        Retrieved knowledge with source citations. Includes text excerpts,
        table descriptions, and chart/image analysis from ingested documents.
    """
    vectorstore = _get_vectorstore()

    # Check if collection has data
    if vectorstore._collection.count() == 0:
        return (
            "The knowledge base is currently empty. No atmospheric science "
            "documents have been ingested yet. Please answer using your "
            "general scientific knowledge."
        )

    # Retrieve candidate matches, filter by relevance, keep top 5
    raw_results = vectorstore.similarity_search_with_score(query, k=8)

    # Filter by cosine distance threshold (lower = more relevant)
    results = [(doc, score) for doc, score in raw_results if score < 1.2]

    if not results:
        return (
            "No relevant documents found in the knowledge base for this query. "
            "Please answer using your general scientific knowledge."
        )

    # Keep only the top 5 most relevant after filtering
    results = results[:5]

    # Build response with parent documents
    sections: list[str] = []

    for rank, (doc, score) in enumerate(results, 1):
        meta = doc.metadata
        doc_id = meta.get("doc_id", "")
        source = meta.get("source_pdf", "Unknown source")
        elem_type = meta.get("element_type", "text")
        page = meta.get("page_number", "?")

        # Fetch original parent document
        parent = _load_parent(doc_id)

        if parent:
            parent_content = parent["content"]
            parent_meta = parent.get("metadata", {})

            if elem_type == "image":
                # For images, the content IS the vision summary
                section = (
                    f"[Source #{rank}: {source}, p.{page} — Chart/Figure]\n"
                    f"Visual analysis: {parent_content}\n"
                )
            elif elem_type == "table":
                section = (
                    f"[Source #{rank}: {source}, p.{page} — Table]\n"
                    f"Table data: {parent_content}\n"
                )
            else:
                section = (
                    f"[Source #{rank}: {source}, p.{page}]\n"
                    f"{parent_content}\n"
                )
        else:
            # Fallback to summary if parent not found
            section = (
                f"[Source #{rank}: {source}, p.{page}]\n"
                f"{doc.page_content}\n"
            )

        sections.append(section)

    header = f"Found {len(sections)} relevant documents:\n\n"
    return header + "\n---\n".join(sections)
