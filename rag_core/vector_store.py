# rag_core/vector_store.py
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)

COLLECTION_NAME = "rag_collection"
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 output dimension


def get_client() -> QdrantClient:
    """
    Returns a Qdrant client instance.
    Raises a clear error if Qdrant is not running.
    """
    try:
        client = QdrantClient(host="localhost", port=6333)
        client.get_collections()  # lightweight ping to verify connection
        return client
    except Exception:
        raise ConnectionError(
            "❌ Cannot connect to Qdrant. Make sure it is running:\n"
            "`docker run -p 6333:6333 qdrant/qdrant`"
        )


def reset_collection(doc_id: str = None):
    """
    Reset the entire collection OR delete chunks belonging to a specific doc_id.
    - If doc_id is None: wipes everything (original behaviour)
    - If doc_id is provided: only removes that document's chunks (multi-doc support)
    """
    client = get_client()

    if doc_id is None:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    else:
        existing = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
        else:
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                )
            )


def insert_chunks(chunks: list, vectors: list, doc_id: str = "default", filename: str = "unknown"):
    """
    Insert pre-computed vectors + chunks into Qdrant.
    Each point stores: text, doc_id, filename, chunk_index in its payload.
    """
    client = get_client()

    points = [
        PointStruct(
            id=abs(hash(f"{doc_id}_{i}")) % (2**63),
            vector=vectors[i],
            payload={
                "text": chunk,
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": i
            }
        )
        for i, chunk in enumerate(chunks)
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)


def retrieve(query_vector: list, top_k: int = 5, doc_id: str = None):
    """
    Retrieve the most relevant chunks for a query vector.

    Returns a list of dicts with:
        - text: the chunk content
        - score: real cosine similarity score (0.0 - 1.0)
        - filename: source document name
        - chunk_index: position in original document
        - doc_id: which document this came from

    If doc_id is specified, search is restricted to that document only.
    """
    client = get_client()

    query_filter = None
    if doc_id:
        query_filter = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        query_filter=query_filter
    )

    return [
        {
            "text": point.payload.get("text", ""),
            "score": round(point.score * 100, 1),
            "filename": point.payload.get("filename", "unknown"),
            "chunk_index": point.payload.get("chunk_index", 0),
            "doc_id": point.payload.get("doc_id", "default")
        }
        for point in results.points
    ]


def get_collection_stats() -> dict:
    """
    Returns basic stats about the current collection.
    Used by the document dashboard in the UI.
    """
    try:
        client = get_client()
        info = client.get_collection(COLLECTION_NAME)
        return {
            "total_chunks": info.points_count,
            "status": str(info.status)
        }
    except Exception:
        return {"total_chunks": 0, "status": "not found"}