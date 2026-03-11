# rag_core/chunking.py
from sentence_transformers import SentenceTransformer
import numpy as np

# ── Simple fixed chunking (kept as fallback) ─────────────────────────────────

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Original fixed-size chunking — kept as fallback for short documents.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# ── Semantic chunking ─────────────────────────────────────────────────────────

def semantic_chunk_text(
    text: str,
    model: SentenceTransformer = None,
    threshold: float = 0.45,
    min_chunk_size: int = 150,
    max_chunk_size: int = 1000
) -> list[str]:
    """
    Semantic chunking — splits text based on meaning shifts rather than
    fixed character counts.

    How it works:
        1. Split text into individual sentences
        2. Embed each sentence
        3. Compute cosine similarity between adjacent sentences
        4. When similarity drops below threshold → start a new chunk
        5. Merge tiny chunks and split oversized ones

    This preserves logical units (paragraphs, arguments, code blocks)
    instead of breaking mid-sentence like fixed chunking does.
    """
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")

    # Step 1 — Split into sentences
    sentences = _split_into_sentences(text)
    if len(sentences) <= 1:
        return [text]  # too short to chunk semantically

    # Step 2 — Embed all sentences in one batch
    embeddings = model.encode(sentences, batch_size=32, show_progress_bar=False)

    # Step 3 — Compute cosine similarity between adjacent sentences
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
        similarities.append(sim)

    # Step 4 — Group sentences into chunks based on similarity drops
    chunks = []
    current_chunk_sentences = [sentences[0]]

    for i, sim in enumerate(similarities):
        next_sentence = sentences[i + 1]

        if sim < threshold:
            # Similarity dropped — this is a semantic boundary, start new chunk
            chunk_text_str = " ".join(current_chunk_sentences).strip()
            if chunk_text_str:
                chunks.append(chunk_text_str)
            current_chunk_sentences = [next_sentence]
        else:
            current_chunk_sentences.append(next_sentence)

    # Don't forget the last chunk
    if current_chunk_sentences:
        chunk_text_str = " ".join(current_chunk_sentences).strip()
        if chunk_text_str:
            chunks.append(chunk_text_str)

    # Step 5 — Post-process: merge tiny chunks, split oversized ones
    chunks = _merge_small_chunks(chunks, min_chunk_size)
    chunks = _split_large_chunks(chunks, max_chunk_size)

    return [c for c in chunks if c.strip()]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences using punctuation heuristics.
    Handles common edge cases like abbreviations and decimal numbers.
    """
    import re
    # Split on sentence-ending punctuation followed by whitespace + capital
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    # Filter out empty strings and very short fragments
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return float(dot / norm) if norm > 0 else 0.0


def _merge_small_chunks(chunks: list[str], min_size: int) -> list[str]:
    """Merge chunks that are too small into their neighbour."""
    merged = []
    buffer = ""

    for chunk in chunks:
        buffer = (buffer + " " + chunk).strip() if buffer else chunk
        if len(buffer) >= min_size:
            merged.append(buffer)
            buffer = ""

    if buffer:
        if merged:
            merged[-1] = merged[-1] + " " + buffer  # attach leftover to last chunk
        else:
            merged.append(buffer)

    return merged


def _split_large_chunks(chunks: list[str], max_size: int) -> list[str]:
    """Split any chunks that are too large using fixed chunking as fallback."""
    result = []
    for chunk in chunks:
        if len(chunk) <= max_size:
            result.append(chunk)
        else:
            # Fall back to fixed chunking for oversized chunks
            sub_chunks = chunk_text(chunk, chunk_size=max_size, overlap=50)
            result.extend(sub_chunks)
    return result