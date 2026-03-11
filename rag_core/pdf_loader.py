# rag_core/pdf_loader.py
from pypdf import PdfReader


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract all text from a PDF as a single string.
    Kept for backward compatibility — used by simple pipelines.
    """
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"

    return text


def extract_pages_from_pdf(file_path: str) -> list[dict]:
    """
    Extract text page by page — returns a list of page dicts.

    Each dict contains:
        - page_num: 1-based page number
        - text: extracted text from that page

    This is the foundation for source citations — every chunk
    generated from this output will know exactly which page it came from.
    """
    reader = PdfReader(file_path)
    pages = []

    for i, page in enumerate(reader.pages):
        extracted = page.extract_text()
        if extracted and extracted.strip():
            pages.append({
                "page_num": i + 1,  # 1-based so it matches the PDF viewer
                "text": extracted.strip()
            })

    return pages


def get_pdf_metadata(file_path: str) -> dict:
    """
    Extract basic metadata from a PDF.
    Used by the document dashboard in the UI.
    """
    try:
        reader = PdfReader(file_path)
        meta = reader.metadata or {}

        return {
            "num_pages": len(reader.pages),
            "title": meta.get("/Title", "Unknown"),
            "author": meta.get("/Author", "Unknown"),
            "subject": meta.get("/Subject", ""),
            "creator": meta.get("/Creator", "")
        }
    except Exception:
        return {
            "num_pages": 0,
            "title": "Unknown",
            "author": "Unknown",
            "subject": "",
            "creator": ""
        }


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 500,
    overlap: int = 50
) -> list[dict]:
    """
    Split page-aware text into chunks while preserving page number metadata.

    Each chunk dict contains:
        - text: the chunk content
        - page_num: which page this chunk came from
        - chunk_index: position within the document

    When a chunk spans a page boundary, it is tagged with the page
    where it started.
    """
    chunks = []
    chunk_index = 0

    for page in pages:
        page_num = page["page_num"]
        text = page["text"]
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "page_num": page_num,
                    "chunk_index": chunk_index
                })
                chunk_index += 1

            start += chunk_size - overlap

    return chunks