# rag_core/embeddings.py
import streamlit as st
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    """
    Load the SentenceTransformer model once and cache it for the entire session.
    Without this, the model reloads on every interaction — causing slow startup
    and unnecessary memory usage.
    """
    return SentenceTransformer(MODEL_NAME)


def embed_text(text: str) -> list[float]:
    """
    Convert a string into a vector embedding.
    Uses the cached model — safe to call repeatedly with no performance penalty.
    """
    model = load_embedding_model()
    return model.encode(text).tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts in one batch — much faster than calling embed_text()
    in a loop for large document chunks.
    """
    model = load_embedding_model()
    return model.encode(texts, show_progress_bar=True, batch_size=32).tolist()