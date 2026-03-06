"""One-time migration: ChromaDB -> Pinecone.

Reads all vectors (embeddings + metadata + documents) from the local ChromaDB,
enriches metadata with parent content from the docstore, and upserts into Pinecone.

Usage::
    python scripts/migrate_to_pinecone.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import chromadb
from pinecone import Pinecone

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHROMA_DIR = Path("data/chroma_db")
DOCSTORE_DIR = Path("data/docstore")
COLLECTION_NAME = "weatherwise_knowledge"
UPSERT_BATCH = 100


def load_parent(doc_id: str) -> str:
    """Load parent content from the local docstore."""
    fp = DOCSTORE_DIR / f"{doc_id}.json"
    if fp.exists():
        data = json.loads(fp.read_text())
        content = data.get("content", "")
        # For images, docstore stores vision summary as content
        # Skip base64 data from metadata
        return content
    return ""


def main() -> None:
    # --- Read all data from ChromaDB ---
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    count = collection.count()
    print(f"ChromaDB: {count} vectors to migrate")

    all_ids: list[str] = []
    all_embeddings: list[list[float]] = []
    all_metadatas: list[dict] = []
    all_documents: list[str] = []

    for offset in range(0, count, UPSERT_BATCH):
        batch = collection.get(
            limit=UPSERT_BATCH,
            offset=offset,
            include=["embeddings", "metadatas", "documents"],
        )
        all_ids.extend(batch["ids"])
        all_embeddings.extend(batch["embeddings"])
        all_metadatas.extend(batch["metadatas"])
        all_documents.extend(batch["documents"])

    print(f"Fetched {len(all_ids)} vectors from ChromaDB")

    # --- Upsert to Pinecone ---
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

    total_upserted = 0

    for i in range(0, len(all_ids), UPSERT_BATCH):
        batch_ids = all_ids[i:i + UPSERT_BATCH]
        batch_embeddings = all_embeddings[i:i + UPSERT_BATCH]
        batch_metadatas = all_metadatas[i:i + UPSERT_BATCH]
        batch_documents = all_documents[i:i + UPSERT_BATCH]

        vectors = []
        for vid, emb, meta, doc in zip(
            batch_ids, batch_embeddings, batch_metadatas, batch_documents
        ):
            doc_id = meta.get("doc_id", vid)
            parent_content = load_parent(doc_id)

            # Truncate if over 36K chars (Pinecone 40KB metadata limit)
            if len(parent_content) > 36000:
                parent_content = parent_content[:36000]

            enriched_meta = {
                **meta,
                "parent_content": parent_content,
                "text": doc,  # langchain-pinecone expects 'text' field for page_content
            }

            vectors.append({
                "id": vid,
                "values": emb,
                "metadata": enriched_meta,
            })

        index.upsert(vectors=vectors)
        total_upserted += len(vectors)
        print(f"  Upserted batch {i // UPSERT_BATCH + 1}: "
              f"{total_upserted}/{len(all_ids)}")

    print(f"\nMigration complete: {total_upserted} vectors upserted to Pinecone")

    # Verify
    time.sleep(3)
    stats = index.describe_index_stats()
    print(f"Pinecone index stats: {stats['total_vector_count']} vectors")


if __name__ == "__main__":
    main()
