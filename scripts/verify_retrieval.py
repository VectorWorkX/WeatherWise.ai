"""Verify the RAG retrieval pipeline against known queries.

Runs test queries against ChromaDB and validates that:
1. Relevant documents are retrieved
2. Metadata (doc_id, source_pdf, element_type) is intact
3. Parent documents can be fetched from the file store

Usage::

    python scripts/verify_retrieval.py
    python scripts/verify_retrieval.py --query "What causes a heat dome?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from weatherwise.tools.vector_search import (
    CHROMA_DIR,
    COLLECTION_NAME,
    _get_vectorstore,
    _load_parent,
)

# ---------------------------------------------------------------------------
# Test queries mapped to expected topics
# ---------------------------------------------------------------------------

TEST_QUERIES = [
    {
        "query": "What causes a heat dome and blocking pattern?",
        "expected_topics": ["heat", "dome", "blocking", "ridge", "subsidence"],
    },
    {
        "query": "How does lake effect snow form?",
        "expected_topics": ["lake", "snow", "evaporation", "convection"],
    },
    {
        "query": "What are the effects of wildfire smoke on air quality?",
        "expected_topics": ["smoke", "AQI", "PM2.5", "wildfire", "aerosol"],
    },
]


def verify_collection_stats() -> None:
    """Print basic statistics about the ChromaDB collection."""
    vectorstore = _get_vectorstore()
    collection = vectorstore._collection

    count = collection.count()
    print(f"Collection '{COLLECTION_NAME}': {count} vectors")

    if count == 0:
        print("[WARN] Collection is empty. Run build_knowledge.py first.")
        return

    sample = collection.peek(limit=3)
    print(f"\nSample metadata (first 3 entries):")
    for i, meta in enumerate(sample.get("metadatas", [])):
        print(f"  [{i}] {meta}")


def verify_query(query: str, top_k: int = 3) -> list[dict]:
    """Run a similarity search and verify results."""
    vectorstore = _get_vectorstore()

    print(f"\nQuery: \"{query}\"")
    print(f"  Searching for top {top_k} matches...")

    results = vectorstore.similarity_search_with_score(query, k=top_k)

    if not results:
        print("  [FAIL] No results returned!")
        return []

    output = []
    for rank, (doc, score) in enumerate(results, 1):
        meta = doc.metadata
        doc_id = meta.get("doc_id", "UNKNOWN")
        source = meta.get("source_pdf", "UNKNOWN")
        elem_type = meta.get("element_type", "UNKNOWN")
        page = meta.get("page_number", "?")

        print(f"\n  --- Result #{rank} (score: {score:.4f}) ---")
        print(f"  Source: {source} | Type: {elem_type} | Page: {page}")
        print(f"  Summary: {doc.page_content[:200]}...")

        parent = _load_parent(doc_id)
        if parent:
            parent_preview = parent["content"][:150]
            print(f"  Parent:  {parent_preview}...")
            print(f"  [OK] Parent document retrieved from docstore")
        else:
            print(f"  [FAIL] Parent document NOT found for doc_id={doc_id}")

        output.append({
            "rank": rank,
            "score": score,
            "summary": doc.page_content,
            "metadata": meta,
            "parent_found": parent is not None,
        })

    return output


def run_test_suite() -> None:
    """Run all test queries and report results."""
    print("=" * 60)
    print("WeatherWise RAG Retrieval Verification")
    print("=" * 60)

    verify_collection_stats()

    vectorstore = _get_vectorstore()
    if vectorstore._collection.count() == 0:
        print("\nCannot run test queries on empty collection. Exiting.")
        return

    passed = 0
    total = len(TEST_QUERIES)

    for test in TEST_QUERIES:
        results = verify_query(test["query"])

        if not results:
            print(f"\n  [FAIL] No results for: {test['query']}")
            continue

        all_text = " ".join(r["summary"].lower() for r in results)
        hits = [t for t in test["expected_topics"] if t.lower() in all_text]

        if hits:
            print(f"\n  [PASS] Topic hits: {hits}")
            passed += 1
        else:
            print(f"\n  [WARN] No expected topics found in results")
            print(f"         Expected: {test['expected_topics']}")

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed}/{total} test queries passed")
    print(f"{'='*60}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Verify WeatherWise RAG retrieval pipeline"
    )
    parser.add_argument("--query", type=str, help="Run a single custom query")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results")
    args = parser.parse_args()

    if args.query:
        verify_collection_stats()
        verify_query(args.query, top_k=args.top_k)
    else:
        run_test_suite()


if __name__ == "__main__":
    main()
