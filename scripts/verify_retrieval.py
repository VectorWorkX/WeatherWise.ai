"""Verify the RAG retrieval pipeline against known queries.

Runs test queries against Pinecone and validates that:
1. Relevant documents are retrieved
2. Metadata (doc_id, source_pdf, element_type, parent_content) is intact
3. Relevance scores are above threshold

Usage::

    python scripts/verify_retrieval.py
    python scripts/verify_retrieval.py --query "What causes a heat dome?"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from pinecone import Pinecone

from weatherwise.tools.vector_search import (
    RELEVANCE_THRESHOLD,
    _get_vectorstore,
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


def _get_index_vector_count() -> int:
    """Query Pinecone for total vector count (verification only)."""
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    idx = pc.Index(os.environ["PINECONE_INDEX_NAME"])
    return idx.describe_index_stats().get("total_vector_count", 0)


def verify_collection_stats() -> None:
    """Print basic statistics about the Pinecone index."""
    index_name = os.environ.get("PINECONE_INDEX_NAME", "unknown")
    total = _get_index_vector_count()
    print(f"Pinecone index '{index_name}': {total} vectors")

    if total == 0:
        print("[WARN] Index is empty. Run build_knowledge.py first.")


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
        has_parent = bool(meta.get("parent_content", ""))

        print(f"\n  --- Result #{rank} (score: {score:.4f}) ---")
        print(f"  Source: {source} | Type: {elem_type} | Page: {page}")
        print(f"  Summary: {doc.page_content[:200]}...")

        if has_parent:
            parent_preview = meta["parent_content"][:150]
            print(f"  Parent:  {parent_preview}...")
            print(f"  [OK] Parent content in metadata")
        else:
            print(f"  [WARN] No parent_content in metadata for doc_id={doc_id}")

        relevance_ok = score >= RELEVANCE_THRESHOLD
        print(f"  Relevance: {'[OK]' if relevance_ok else '[LOW]'} "
              f"(score={score:.4f}, threshold={RELEVANCE_THRESHOLD})")

        output.append({
            "rank": rank,
            "score": score,
            "summary": doc.page_content,
            "metadata": meta,
            "has_parent_content": has_parent,
        })

    return output


def run_test_suite() -> None:
    """Run all test queries and report results."""
    print("=" * 60)
    print("WeatherWise RAG Retrieval Verification (Pinecone)")
    print("=" * 60)

    verify_collection_stats()

    total_vectors = _get_index_vector_count()
    if total_vectors == 0:
        print("\nCannot run test queries on empty index. Exiting.")
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
        description="Verify WeatherWise RAG retrieval pipeline (Pinecone)"
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
