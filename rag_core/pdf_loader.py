from pypdf import PdfReader

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts all text from a PDF file and returns it as a single string.
    """
    reader = PdfReader(file_path)
    full_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    return full_text
