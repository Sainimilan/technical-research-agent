from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

collection_name = "rag_collection"
client = QdrantClient(host="localhost", port=6333)

def reset_collection():
    try:
        client.delete_collection(collection_name)
    except:
        pass

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

def insert_chunks(chunks, embed_function):
    points = []

    for i, chunk in enumerate(chunks):
        vector = embed_function(chunk)
        points.append(
            PointStruct(
                id=i,
                vector=vector,
                payload={"text": chunk}
            )
        )

    client.upsert(collection_name=collection_name, points=points)

def retrieve(query_vector, top_k=3):
    return client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k
    )
