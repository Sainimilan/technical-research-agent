from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from rag_core.pdf_loader import extract_text_from_pdf
from rag_core.chunking import chunk_text
import requests

# ------------------------
# Setup
# ------------------------

client = QdrantClient(host="localhost", port=6333)
collection_name = "rag_collection"

# Create collection if not exists
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------
# Load and Chunk PDF
# ------------------------

pdf_path = "sample.pdf"
text = extract_text_from_pdf(pdf_path)
chunks = chunk_text(text, chunk_size=500, overlap=50)

print(f"Total chunks: {len(chunks)}")

# ------------------------
# Store Chunks in Qdrant
# ------------------------

points = []

for i, chunk in enumerate(chunks):
    vector = model.encode(chunk).tolist()

    points.append(
        PointStruct(
            id=i,
            vector=vector,
            payload={"text": chunk}
        )
    )

client.upsert(collection_name=collection_name, points=points)

print("All chunks stored successfully!")

# ------------------------
# RAG Query
# ------------------------

query = "Explain the main idea of this document"
mode = "deep"   # change to "fast" or "deep"

query_vector = model.encode(query).tolist()

search_result = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    limit=3
)

retrieved_text = "\n\n".join(
    [point.payload["text"] for point in search_result.points]
)

# ------------------------
# Build Prompt Based on Mode
# ------------------------

if mode == "fast":
    prompt = f"""
Use the following context to answer the question briefly.

Context:
{retrieved_text}

Question:
{query}

Give a concise answer.
"""

elif mode == "deep":
    prompt = f"""
You are a technical research assistant.

Using the following context, provide a structured technical response.

Context:
{retrieved_text}

Question:
{query}

Format the response as:

1. Summary
2. Key Insights
3. Implementation Plan
4. Risks or Limitations
5. Conclusion
"""

# ------------------------
# Call Local LLM
# ------------------------

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    }
)

print("\nFinal RAG Answer:\n")
print(response.json()["response"])
