# rag_core/rag_pipeline.py

import uuid
import tempfile
import os
from rag_core.pdf_loader import extract_text_from_pdf
from rag_core.chunking import chunk_text
from rag_core.embeddings import embed_text, embed_batch
from rag_core.vector_store import reset_collection, insert_chunks, retrieve
from rag_core.llm import generate_response


def process_pdf(file_path: str, filename: str = "unknown") -> dict:
    """
    Full ingestion pipeline for a single PDF.

    Steps:
        1. Extract text from PDF
        2. Split into overlapping chunks
        3. Batch embed all chunks at once (fast)
        4. Store in Qdrant with doc_id + filename metadata

    Returns a summary dict for the UI dashboard.
    """

    # Generate a unique ID for this document — fixes the temp file collision bug
    # Previously all docs were saved as "temp.pdf", overwriting each other
    doc_id = str(uuid.uuid4())

    # Step 1 — Extract
    text = extract_text_from_pdf(file_path)
    if not text.strip():
        raise ValueError(f"❌ Could not extract any text from '{filename}'. Is it a scanned PDF?")

    # Step 2 — Chunk
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    if not chunks:
        raise ValueError("❌ Document produced no chunks after splitting.")

    # Step 3 — Batch embed (much faster than embedding one by one)
    vectors = embed_batch(chunks)

    # Step 4 — Store (selective reset: only removes previous version of THIS doc)
    reset_collection(doc_id=doc_id)
    insert_chunks(chunks, vectors, doc_id=doc_id, filename=filename)

    return {
        "doc_id": doc_id,
        "filename": filename,
        "num_chunks": len(chunks),
        "num_pages": _count_pages(file_path),
        "char_count": len(text),
        "word_count": len(text.split())
    }


def ask_question(
    query: str,
    mode: str = "fast",
    model: str = "mistral",
    doc_id: str = None,
    chat_history: list[dict] = None
) -> dict:
    """
    Full RAG query pipeline.

    Steps:
        1. Embed the query
        2. Retrieve relevant chunks (optionally filtered by doc_id)
        3. Build prompt with optional chat history for conversational memory
        4. Generate answer via local LLM

    Returns a dict with answer, retrieved sources, and real confidence score.
    """

    # Step 1 — Embed query
    query_vector = embed_text(query)

    # Step 2 — Retrieve (top_k=5 for deep mode, 3 for fast)
    top_k = 5 if mode == "deep" else 3
    results = retrieve(query_vector, top_k=top_k, doc_id=doc_id)

    if not results:
        return {
            "answer": "❌ No relevant content found. Please upload and process a document first.",
            "sources": [],
            "confidence": 0.0,
            "mode": mode
        }

    # Step 3 — Real confidence score: average of actual cosine similarity scores
    confidence = round(sum(r["score"] for r in results) / len(results), 1)

    # Build context string from retrieved chunks
    retrieved_text = "\n\n---\n\n".join(
        f"[Source: {r['filename']} | Chunk {r['chunk_index']} | Score: {r['score']}%]\n{r['text']}"
        for r in results
    )

    # Build conversation history string if provided
    history_text = ""
    if chat_history:
        history_lines = []
        for turn in chat_history[-6:]:  # last 3 exchanges (6 messages)
            role = "User" if turn["role"] == "user" else "Assistant"
            history_lines.append(f"{role}: {turn['content']}")
        history_text = "\n".join(history_lines)

    # Step 4 — Build prompt based on mode
    if mode == "fast":
        prompt = f"""Use the following document context to answer the question briefly and accurately.
{f'Conversation so far:{chr(10)}{history_text}{chr(10)}' if history_text else ''}
Context:
{retrieved_text}

Question: {query}

Answer concisely:"""

    else:  # deep mode
        prompt = f"""You are a technical research assistant with deep expertise.
{f'Conversation so far:{chr(10)}{history_text}{chr(10)}' if history_text else ''}
Using the context below, provide a comprehensive structured response.

Context:
{retrieved_text}

Question: {query}

Respond using this format:
## Summary
## Key Insights
## Implementation Plan
## Risks
## Conclusion"""

    # Generate answer (non-streaming here — streaming handled in app.py)
    answer = generate_response(prompt, model=model, stream=False)

    return {
        "answer": answer,
        "sources": results,
        "confidence": confidence,
        "mode": mode
    }


def process_uploaded_file(uploaded_file) -> dict:
    """
    Helper to safely handle Streamlit UploadedFile objects.

    Fixes the original bug where all files were saved as 'temp.pdf',
    meaning uploading a second PDF silently overwrote the first.

    Now uses a proper temp file with a unique name per upload.
    """
    suffix = os.path.splitext(uploaded_file.name)[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        result = process_pdf(tmp_path, filename=uploaded_file.name)
    finally:
        os.unlink(tmp_path)  # always clean up temp file after processing

    return result


# ── Internal helpers ────────────────────────────────────────────────────────

def _count_pages(file_path: str) -> int:
    """Safely count pages in a PDF without crashing if it fails."""
    try:
        from pypdf import PdfReader
        return len(PdfReader(file_path).pages)
    except Exception:
        return 0