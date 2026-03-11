# rag_core/rag_pipeline.py

import uuid
import tempfile
import os
from rag_core.pdf_loader import (
    extract_pages_from_pdf,
    chunk_pages,
    get_pdf_metadata
)
from rag_core.chunking import semantic_chunk_text
from rag_core.embeddings import embed_text, embed_batch
from rag_core.vector_store import reset_collection, insert_chunks, retrieve
from rag_core.reranker import rerank
from rag_core.llm import generate_response


def process_pdf(file_path: str, filename: str = "unknown") -> dict:
    """
    Full ingestion pipeline for a single PDF.

    Steps:
        1. Extract text page by page (preserves page numbers)
        2. Chunk each page's text (fixed chunking per page)
        3. Apply semantic chunking on top for better boundaries
        4. Batch embed all chunks
        5. Store in Qdrant with full metadata (doc_id, filename, page_num)

    Returns a summary dict for the UI dashboard.
    """
    doc_id = str(uuid.uuid4())

    # Step 1 — Extract page by page
    pages = extract_pages_from_pdf(file_path)
    if not pages:
        raise ValueError(
            f"❌ Could not extract any text from '{filename}'. "
            "Is it a scanned PDF?"
        )

    # Step 2 — Chunk pages (preserves page_num per chunk)
    page_chunks = chunk_pages(pages, chunk_size=600, overlap=60)
    if not page_chunks:
        raise ValueError("❌ Document produced no chunks after splitting.")

    # Step 3 — Apply semantic refinement on plain text chunks
    # We extract just the text, semantically re-chunk, then re-attach page nums
    all_text = " ".join(c["text"] for c in page_chunks)
    semantic_texts = semantic_chunk_text(all_text)

    # Re-attach page numbers by matching semantic chunks back to page chunks
    final_chunks = _attach_page_numbers(semantic_texts, page_chunks)

    # Step 4 — Batch embed
    texts_only = [c["text"] for c in final_chunks]
    vectors = embed_batch(texts_only)

    # Step 5 — Store
    reset_collection(doc_id=doc_id)
    insert_chunks(final_chunks, vectors, doc_id=doc_id, filename=filename)

    # Get metadata
    metadata = get_pdf_metadata(file_path)

    return {
        "doc_id": doc_id,
        "filename": filename,
        "num_chunks": len(final_chunks),
        "num_pages": metadata.get("num_pages", len(pages)),
        "char_count": len(all_text),
        "word_count": len(all_text.split()),
        "title": metadata.get("title", filename),
        "author": metadata.get("author", "Unknown")
    }


def ask_question(
    query: str,
    mode: str = "fast",
    model: str = "mistral",
    doc_id: str = None,
    chat_history: list = None
) -> dict:
    """
    Full RAG query pipeline with reranking and conversational memory.

    Steps:
        1. Embed the query
        2. Retrieve top-10 candidates from Qdrant
        3. Rerank to top-3 using cross-encoder
        4. Build prompt with chat history (conversational memory)
        5. Generate answer via local LLM

    Returns a dict with answer, sources (with page numbers), and confidence.
    """

    # Step 1 — Embed query
    query_vector = embed_text(query)

    # Step 2 — Retrieve top-10 candidates
    candidates = retrieve(query_vector, top_k=10, doc_id=doc_id)

    if not candidates:
        return {
            "answer": (
                "❌ No relevant content found. "
                "Please upload and process a document first."
            ),
            "sources": [],
            "confidence": 0.0,
            "mode": mode
        }

    # Step 3 — Rerank to top-3 using cross-encoder
    top_k = 5 if mode == "deep" else 3
    results = rerank(query, candidates, top_k=top_k)

    # Confidence = average cosine similarity score of reranked results
    confidence = round(
        sum(r["score"] for r in results) / len(results), 1
    )

    # Build context string with page citations
    retrieved_text = "\n\n---\n\n".join(
        f"[Source: {r['filename']} | Page {r['page_num']} | "
        f"Score: {r['score']}%]\n{r['text']}"
        for r in results
    )

    # Step 4 — Build conversational memory string
    history_text = ""
    if chat_history:
        history_lines = []
        # Take last 6 messages (3 exchanges) for context window efficiency
        for turn in chat_history[-6:]:
            role = "User" if turn["role"] == "user" else "Assistant"
            history_lines.append(f"{role}: {turn['content']}")
        history_text = "\n".join(history_lines)

    # Step 5 — Build prompt based on mode
    history_block = (
        f"Conversation History:\n{history_text}\n\n"
        if history_text else ""
    )

    if mode == "fast":
        prompt = f"""You are a helpful research assistant.
{history_block}Use the following document context to answer the question briefly and accurately.
If the answer is not in the context, say so honestly.

Context:
{retrieved_text}

Question: {query}

Answer concisely:"""

    else:  # deep mode
        prompt = f"""You are an expert technical research assistant.
{history_block}Using the context below, provide a comprehensive structured response.
If the answer is not fully covered in the context, say so honestly.

Context:
{retrieved_text}

Question: {query}

Respond using this exact format:
## Summary
## Key Insights
## Implementation Plan
## Risks
## Conclusion"""

    # Generate answer
    answer = generate_response(prompt, model=model, stream=False)

    return {
        "answer": answer,
        "sources": results,
        "confidence": confidence,
        "mode": mode
    }


def process_uploaded_file(uploaded_file) -> dict:
    """
    Safely handle Streamlit UploadedFile objects.
    Uses a unique temp file per upload — no collision between PDFs.
    """
    suffix = os.path.splitext(uploaded_file.name)[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        result = process_pdf(tmp_path, filename=uploaded_file.name)
    finally:
        os.unlink(tmp_path)  # always clean up

    return result


# ── Internal helpers ─────────────────────────────────────────────────────────

def _attach_page_numbers(
    semantic_texts: list[str],
    page_chunks: list[dict]
) -> list[dict]:
    """
    Re-attach page numbers to semantically chunked text.

    Strategy: for each semantic chunk, find which page chunk
    shares the most overlapping text and inherit its page number.
    Falls back to page 1 if no match found.
    """
    result = []

    for i, sem_text in enumerate(semantic_texts):
        best_page = 1
        best_overlap = 0

        # Compare first 100 chars of semantic chunk against all page chunks
        sample = sem_text[:100].lower()

        for pc in page_chunks:
            pc_text = pc["text"].lower()
            # Simple overlap: count shared words
            overlap = sum(1 for word in sample.split() if word in pc_text)
            if overlap > best_overlap:
                best_overlap = overlap
                best_page = pc["page_num"]

        result.append({
            "text": sem_text,
            "page_num": best_page,
            "chunk_index": i
        })

    return result