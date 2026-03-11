# rag_core/reranker.py
import streamlit as st
from sentence_transformers import CrossEncoder

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@st.cache_resource(show_spinner="Loading reranker model...")
def load_reranker():
    """
    Load the cross-encoder reranker model once and cache it.

    Cross-encoder vs Bi-encoder (your current embeddings):
        - Bi-encoder: encodes query and chunk SEPARATELY then compares
          → fast but less accurate
        - Cross-encoder: encodes query + chunk TOGETHER as a pair
          → slower but dramatically more accurate

    Strategy: retrieve top-10 with fast bi-encoder,
    then rerank to top-3 with accurate cross-encoder.
    This gives you the best of both worlds.
    """
    return CrossEncoder(RERANKER_MODEL)


def rerank(query: str, chunks: list[dict], top_k: int = 3) -> list[dict]:
    """
    Rerank retrieved chunks using a cross-encoder model.

    Args:
        query:   the user's question
        chunks:  list of dicts from vector_store.retrieve()
                 each must have a 'text' key
        top_k:   how many top results to return after reranking

    Returns:
        Reranked list of chunk dicts, each with an added
        'rerank_score' field showing the cross-encoder confidence.

    Example:
        # Retrieve 10 candidates, rerank to top 3
        candidates = retrieve(query_vector, top_k=10)
        best = rerank(query, candidates, top_k=3)
    """
    if not chunks:
        return []

    # If fewer chunks than top_k just return them as-is
    if len(chunks) <= top_k:
        return chunks

    model = load_reranker()

    # Build (query, chunk_text) pairs for the cross-encoder
    pairs = [(query, chunk["text"]) for chunk in chunks]

    # Score all pairs — cross-encoder returns a raw logit score per pair
    scores = model.predict(pairs)

    # Attach rerank score to each chunk
    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = round(float(scores[i]), 4)

    # Sort by rerank score descending and return top_k
    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

    return reranked[:top_k]


def rerank_with_threshold(
    query: str,
    chunks: list[dict],
    top_k: int = 3,
    min_score: float = -5.0
) -> list[dict]:
    """
    Rerank and additionally filter out chunks below a minimum score.

    Cross-encoder scores are raw logits (can be negative).
    Typical ranges:
        > 5.0  → very high relevance
        0 - 5  → moderate relevance
        < 0    → likely irrelevant

    Use this when you want to avoid showing low-quality sources
    even if nothing better is available.
    """
    reranked = rerank(query, chunks, top_k=top_k)
    filtered = [c for c in reranked if c.get("rerank_score", 0) >= min_score]

    # Always return at least 1 result even if below threshold
    return filtered if filtered else reranked[:1]