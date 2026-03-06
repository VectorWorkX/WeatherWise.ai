"""Offline ingestion pipeline for the WeatherWise RAG knowledge base.

Parses atmospheric-science PDFs and satellite images, generates summaries,
embeds them in Pinecone, and stores the original parent documents in a
local file store (backup).  Run this script whenever new documents are added
to ``data/raw_pdfs/`` or ``data/satellite_imagery/``.

Strategy: Multi-Vector Retrieval (Summary-to-Source)
  - Summaries -> Pinecone (what we SEARCH against)
  - Parent content -> Pinecone metadata (what we RETRIEVE and return)
  - Originals -> data/docstore/ (local backup)

Usage::

    python scripts/build_knowledge.py --all
    python scripts/build_knowledge.py --pdf data/raw_pdfs/heat_report.pdf
    python scripts/build_knowledge.py --image data/satellite_imagery/goes_heatdome.png
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_PDFS_DIR = DATA_DIR / "raw_pdfs"
SATELLITE_DIR = DATA_DIR / "satellite_imagery"
DOCSTORE_DIR = DATA_DIR / "docstore"
EXTRACTED_IMAGES_DIR = DATA_DIR / "extracted_images"

for d in [DOCSTORE_DIR, EXTRACTED_IMAGES_DIR, RAW_PDFS_DIR, SATELLITE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# LLM + Embedding helpers
# ---------------------------------------------------------------------------

def _get_summarizer():
    """Return a Claude Sonnet instance for text/table summarization."""
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        max_tokens=1024,
        temperature=0,
    )


def _get_vision_model():
    """Return a Claude Sonnet instance for image summarization."""
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        max_tokens=1024,
        temperature=0,
    )


def _get_embeddings():
    """Return the OpenAI embedding model."""
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )


def _get_pinecone_vectorstore():
    """Return a Pinecone vectorstore for upserting embeddings."""
    from langchain_pinecone import PineconeVectorStore

    return PineconeVectorStore(
        index_name=os.environ["PINECONE_INDEX_NAME"],
        embedding=_get_embeddings(),
        pinecone_api_key=os.environ["PINECONE_API_KEY"],
    )


# ---------------------------------------------------------------------------
# Docstore helpers (local backup)
# ---------------------------------------------------------------------------

def _save_to_docstore(doc_id: str, content: str, metadata: dict[str, Any]) -> None:
    """Persist a parent document to the local file store (backup)."""
    record = {"content": content, "metadata": metadata}
    filepath = DOCSTORE_DIR / f"{doc_id}.json"
    filepath.write_text(json.dumps(record, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Summarization prompts
# ---------------------------------------------------------------------------

TEXT_SUMMARY_PROMPT = """\
You are a scientific document analyst specializing in atmospheric science.
Summarize the following text section from a weather/climate document.

Rules:
- Keep the summary to 2-3 sentences.
- Preserve ALL scientific terminology, phenomena names, and causal relationships.
- Include any specific measurements, thresholds, or data values mentioned.
- Do NOT add information not present in the original text.

Text section:
{text}
"""

TABLE_SUMMARY_PROMPT = """\
You are a scientific document analyst specializing in atmospheric science.
Describe the following table extracted from a weather/climate document.

Rules:
- State what the columns and rows represent.
- Highlight key values, trends, ranges, or anomalies.
- Preserve units and any footnotes.
- Keep the description to 3-4 sentences.

Table content:
{table}
"""

IMAGE_SUMMARY_PROMPT = """\
You are a meteorological chart analyst. Describe this weather chart, \
satellite image, or scientific figure in detail.

Include:
- Chart type (e.g., 500mb height map, surface analysis, satellite IR, radar)
- Axis labels, units, and spatial/temporal extent
- Key patterns: ridges, troughs, fronts, anomalies, gradients
- Specific values or contour levels if visible
- What this tells us about current or historical atmospheric conditions

Keep the description to 4-6 sentences. Use precise meteorological terminology.
"""


def summarize_text(text: str) -> str:
    """Generate a concise summary of a text chunk using Claude."""
    llm = _get_summarizer()
    response = llm.invoke(TEXT_SUMMARY_PROMPT.format(text=text))
    return response.content


def summarize_table(table_text: str) -> str:
    """Generate a description of an extracted table using Claude."""
    llm = _get_summarizer()
    response = llm.invoke(TABLE_SUMMARY_PROMPT.format(table=table_text))
    return response.content


MAX_VISION_BYTES = 5 * 1024 * 1024  # 5 MB Claude Vision limit


def _resize_for_vision(image_path: str) -> tuple[str, str]:
    """Resize an image to fit within Claude Vision's size limit."""
    import io

    from PIL import Image

    file_size = os.path.getsize(image_path)

    if file_size <= MAX_VISION_BYTES:
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        suffix = Path(image_path).suffix.lower()
        media_types = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".gif": "image/gif", ".webp": "image/webp", ".tiff": "image/tiff",
        }
        return data, media_types.get(suffix, "image/png")

    print(f"  [INFO] Image is {file_size / 1024 / 1024:.1f}MB, resizing for Vision...")
    img = Image.open(image_path)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    quality = 85
    scale = 1.0
    for _ in range(5):
        buf = io.BytesIO()
        resized = img
        if scale < 1.0:
            new_size = (int(img.width * scale), int(img.height * scale))
            resized = img.resize(new_size, Image.LANCZOS)
        resized.save(buf, format="JPEG", quality=quality)
        buf_size = buf.tell()
        if buf_size <= MAX_VISION_BYTES:
            print(f"  [INFO] Resized to {buf_size / 1024 / 1024:.1f}MB "
                  f"(scale={scale:.2f}, quality={quality})")
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("utf-8"), "image/jpeg"
        quality = max(quality - 15, 40)
        scale *= 0.75

    buf = io.BytesIO()
    resized = img.resize((img.width // 3, img.height // 3), Image.LANCZOS)
    resized.save(buf, format="JPEG", quality=50)
    print(f"  [INFO] Aggressively resized to {buf.tell() / 1024 / 1024:.1f}MB")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8"), "image/jpeg"


def summarize_image(image_path: str) -> str:
    """Generate a technical description of a weather chart/image using Claude Vision."""
    from langchain_core.messages import HumanMessage

    llm = _get_vision_model()
    image_data, media_type = _resize_for_vision(image_path)

    message = HumanMessage(
        content=[
            {"type": "text", "text": IMAGE_SUMMARY_PROMPT},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{image_data}"},
            },
        ]
    )
    response = llm.invoke([message])
    return response.content


# ---------------------------------------------------------------------------
# Pinecone upsert helper
# ---------------------------------------------------------------------------

# Pinecone metadata limit is 40KB per vector. Text/table parent content is
# always under 5KB. For images, we store only the vision summary (not base64).
MAX_METADATA_CONTENT_CHARS = 36000  # leave room for other metadata fields


def _prepare_pinecone_batch(
    doc_ids: list[str],
    summaries: list[str],
    metadatas: list[dict[str, Any]],
    parent_contents: list[str],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Prepare batch data for Pinecone upsert, embedding parent_content in metadata."""
    final_metadatas = []
    for meta, parent_content in zip(metadatas, parent_contents):
        enriched = {**meta}
        # Truncate parent content if it would exceed metadata limit
        if len(parent_content) > MAX_METADATA_CONTENT_CHARS:
            enriched["parent_content"] = parent_content[:MAX_METADATA_CONTENT_CHARS]
        else:
            enriched["parent_content"] = parent_content
        final_metadatas.append(enriched)
    return doc_ids, summaries, final_metadatas


# ---------------------------------------------------------------------------
# PDF ingestion
# ---------------------------------------------------------------------------

def ingest_pdf(pdf_path: str | Path) -> list[str]:
    """Parse a PDF, chunk by title, summarize, embed, and store."""
    from unstructured.chunking.title import chunk_by_title
    from unstructured.partition.pdf import partition_pdf

    pdf_path = Path(pdf_path)
    pdf_name = pdf_path.name
    print(f"\n{'='*60}")
    print(f"Ingesting PDF: {pdf_name}")
    print(f"{'='*60}")

    print("[1/5] Parsing PDF with hi_res strategy...")
    elements = partition_pdf(
        filename=str(pdf_path),
        strategy="hi_res",
        extract_images_in_pdf=True,
        extract_image_block_output_dir=str(EXTRACTED_IMAGES_DIR),
        infer_table_structure=True,
        include_page_breaks=True,
    )
    print(f"  -> Extracted {len(elements)} raw elements")

    image_elements = [el for el in elements if el.category == "Image"]
    non_image_elements = [el for el in elements if el.category != "Image"]
    table_elements = [el for el in elements if el.category == "Table"]

    print("[2/5] Chunking by title (semantic grouping)...")
    chunked = chunk_by_title(
        non_image_elements,
        max_characters=4000,
        new_after_n_chars=3500,
        combine_text_under_n_chars=2000,
        multipage_sections=True,
    )
    text_chunks = [el for el in chunked if el.category != "Table"]
    print(f"  -> {len(text_chunks)} text chunks, {len(table_elements)} tables, "
          f"{len(image_elements)} images")

    doc_ids: list[str] = []
    summaries: list[str] = []
    metadatas: list[dict[str, Any]] = []
    parent_contents: list[str] = []

    print("[3/5] Summarizing text chunks...")
    for i, chunk in enumerate(text_chunks):
        original_text = str(chunk)
        if len(original_text.strip()) < 50:
            continue

        doc_id = f"{pdf_name}__text__{uuid.uuid4().hex[:8]}"
        page_num = getattr(chunk.metadata, "page_number", None)

        try:
            summary = summarize_text(original_text)
        except Exception as e:
            print(f"  [WARN] Failed to summarize text chunk {i}: {e}")
            summary = original_text[:500]

        # Save to local docstore (backup)
        _save_to_docstore(doc_id, original_text, {
            "source_pdf": pdf_name,
            "element_type": "text",
            "page_number": page_num,
        })

        doc_ids.append(doc_id)
        summaries.append(summary)
        metadatas.append({
            "doc_id": doc_id,
            "source_pdf": pdf_name,
            "element_type": "text",
            "page_number": page_num or 0,
        })
        parent_contents.append(original_text)
        print(f"  -> Text chunk {i+1}/{len(text_chunks)} summarized ({len(summary)} chars)")

    print("[4/5] Summarizing tables...")
    for i, table in enumerate(table_elements):
        table_text = str(table)
        if len(table_text.strip()) < 20:
            continue

        doc_id = f"{pdf_name}__table__{uuid.uuid4().hex[:8]}"
        page_num = getattr(table.metadata, "page_number", None)

        try:
            summary = summarize_table(table_text)
        except Exception as e:
            print(f"  [WARN] Failed to summarize table {i}: {e}")
            summary = table_text[:500]

        _save_to_docstore(doc_id, table_text, {
            "source_pdf": pdf_name,
            "element_type": "table",
            "page_number": page_num,
        })

        doc_ids.append(doc_id)
        summaries.append(summary)
        metadatas.append({
            "doc_id": doc_id,
            "source_pdf": pdf_name,
            "element_type": "table",
            "page_number": page_num or 0,
        })
        parent_contents.append(table_text)
        print(f"  -> Table {i+1}/{len(table_elements)} summarized")

    print("[5/5] Summarizing images via Claude Vision...")
    for i, img_el in enumerate(image_elements):
        image_path = getattr(img_el.metadata, "image_path", None)
        if not image_path or not Path(image_path).exists():
            print(f"  [SKIP] Image {i}: no file found at {image_path}")
            continue

        doc_id = f"{pdf_name}__image__{uuid.uuid4().hex[:8]}"
        page_num = getattr(img_el.metadata, "page_number", None)

        try:
            vision_summary = summarize_image(image_path)
        except Exception as e:
            print(f"  [WARN] Failed to summarize image {i}: {e}")
            vision_summary = f"Image from {pdf_name}, page {page_num}"

        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        # Save full data (including base64) to local docstore backup
        _save_to_docstore(doc_id, vision_summary, {
            "source_pdf": pdf_name,
            "element_type": "image",
            "page_number": page_num,
            "image_path": str(image_path),
            "image_base64": image_b64,
        })

        doc_ids.append(doc_id)
        summaries.append(vision_summary)
        metadatas.append({
            "doc_id": doc_id,
            "source_pdf": pdf_name,
            "element_type": "image",
            "page_number": page_num or 0,
        })
        # For images, parent_content is the vision summary (not base64)
        parent_contents.append(vision_summary)
        print(f"  -> Image {i+1}/{len(image_elements)} summarized")

    if summaries:
        print(f"\nUpserting {len(summaries)} vectors into Pinecone...")
        _, texts, enriched_metas = _prepare_pinecone_batch(
            doc_ids, summaries, metadatas, parent_contents
        )
        vectorstore = _get_pinecone_vectorstore()
        vectorstore.add_texts(
            texts=texts,
            metadatas=enriched_metas,
            ids=doc_ids,
        )
        print(f"  -> Stored {len(summaries)} vectors in Pinecone")
    else:
        print("\n[WARN] No content extracted from PDF.")

    return doc_ids


# ---------------------------------------------------------------------------
# Standalone image ingestion
# ---------------------------------------------------------------------------

def ingest_image(image_path: str | Path, source_label: str = "") -> str:
    """Ingest a standalone satellite image or weather chart."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    source = source_label or image_path.name
    print(f"\nIngesting image: {image_path.name}")

    doc_id = f"satellite__{uuid.uuid4().hex[:8]}"

    try:
        vision_summary = summarize_image(str(image_path))
    except Exception as e:
        print(f"  [WARN] Vision summarization failed: {e}")
        vision_summary = f"Satellite/weather image: {source}"

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    # Save full data to local docstore backup
    _save_to_docstore(doc_id, vision_summary, {
        "source_pdf": source,
        "element_type": "image",
        "page_number": 0,
        "image_path": str(image_path),
        "image_base64": image_b64,
    })

    vectorstore = _get_pinecone_vectorstore()
    vectorstore.add_texts(
        texts=[vision_summary],
        metadatas=[{
            "doc_id": doc_id,
            "source_pdf": source,
            "element_type": "image",
            "page_number": 0,
            "parent_content": vision_summary,
        }],
        ids=[doc_id],
    )
    print(f"  -> Stored as {doc_id}")
    return doc_id


# ---------------------------------------------------------------------------
# Batch ingestion
# ---------------------------------------------------------------------------

def ingest_all() -> None:
    """Ingest all PDFs in data/raw_pdfs/ and images in data/satellite_imagery/."""
    pdf_files = sorted(RAW_PDFS_DIR.glob("*.pdf"))
    image_files = sorted(
        f for f in SATELLITE_DIR.iterdir()
        if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".tiff", ".webp"}
    )

    print(f"Found {len(pdf_files)} PDFs and {len(image_files)} satellite images\n")

    total_docs = 0
    for pdf in pdf_files:
        ids = ingest_pdf(pdf)
        total_docs += len(ids)

    for img in image_files:
        ingest_image(img)
        total_docs += 1

    print(f"\n{'='*60}")
    print(f"INGESTION COMPLETE: {total_docs} documents indexed")
    print(f"  Pinecone index: {os.environ.get('PINECONE_INDEX_NAME', 'N/A')}")
    print(f"  Docstore backup: {DOCSTORE_DIR}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for the ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="WeatherWise RAG ingestion pipeline"
    )
    parser.add_argument("--pdf", type=str, help="Ingest a single PDF file")
    parser.add_argument("--image", type=str, help="Ingest a single satellite image")
    parser.add_argument("--all", action="store_true",
                        help="Ingest all PDFs and images from data/ directories")
    args = parser.parse_args()

    if args.pdf:
        ingest_pdf(args.pdf)
    elif args.image:
        ingest_image(args.image)
    elif args.all:
        ingest_all()
    else:
        print("No action specified. Use --pdf, --image, or --all.")
        print("Run with --help for usage details.")


if __name__ == "__main__":
    main()
