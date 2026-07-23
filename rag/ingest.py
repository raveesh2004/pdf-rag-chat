"""PDF ingestion: extract text per page, chunk with overlap, embed into ChromaDB.

Embeddings use ChromaDB's default embedding function (all-MiniLM-L6-v2 via
ONNX runtime) — local and free, no extra API key needed.
"""

import hashlib

import chromadb
from pypdf import PdfReader


def extract_pages(file) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] for pages that contain text."""
    reader = PdfReader(file)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages


def chunk_pages(
    pages: list[tuple[int, str]],
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[dict]:
    """Split page texts into overlapping character chunks, keeping page numbers.

    Overlap prevents answers from being lost at chunk boundaries.
    """
    chunks = []
    for page_num, text in pages:
        start = 0
        while start < len(text):
            piece = text[start : start + chunk_size]
            chunks.append({"text": piece, "page": page_num})
            if start + chunk_size >= len(text):
                break
            start += chunk_size - overlap
    return chunks


def build_index(chunks: list[dict], doc_name: str):
    """Embed chunks into an in-memory ChromaDB collection and return it."""
    client = chromadb.EphemeralClient()
    collection_name = "doc_" + hashlib.md5(doc_name.encode()).hexdigest()[:12]
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=[c["text"] for c in chunks],
        metadatas=[{"page": c["page"]} for c in chunks],
    )
    return collection


def retrieve(collection, query: str, k: int = 5) -> list[dict]:
    """Return the k most relevant chunks for a query."""
    results = collection.query(query_texts=[query], n_results=k)
    return [
        {"text": doc, "page": meta["page"]}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]
