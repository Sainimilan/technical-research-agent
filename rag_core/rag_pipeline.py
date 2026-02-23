from rag_core.pdf_loader import extract_text_from_pdf
from rag_core.chunking import chunk_text
from rag_core.embeddings import embed_text
from rag_core.vector_store import reset_collection, insert_chunks, retrieve
from rag_core.llm import generate_response


def process_pdf(file_path):

    reset_collection()

    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text, chunk_size=500, overlap=50)

    insert_chunks(chunks, embed_text)


def ask_question(query, mode="fast"):

    query_vector = embed_text(query)

    search_result = retrieve(query_vector, top_k=3)

    confidence = (len(search_result.points) / 3) * 100

    retrieved_text = "\n\n".join(
        [point.payload["text"] for point in search_result.points]
    )

    if mode == "fast":
        prompt = f"""
Use the following context to answer briefly.

Context:
{retrieved_text}

Question:
{query}
"""
    else:
        prompt = f"""
You are a technical research assistant.

Using the following context, provide a structured response.

Context:
{retrieved_text}

Question:
{query}

Format:
## Summary
## Key Insights
## Implementation Plan
## Risks
## Conclusion
"""

    final_answer = generate_response(prompt)

    return final_answer, retrieved_text, confidence
