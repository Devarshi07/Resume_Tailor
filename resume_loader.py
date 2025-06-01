import os
import docx2txt
import PyPDF2

# Load resume text from .pdf or .docx

def load_resume_text(file_path: str) -> str:
    """
    Given a path to a .pdf or .docx resume, extract its text while preserving line breaks.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return _load_pdf_text(file_path)
    elif ext == ".docx":
        return _load_docx_text(file_path)
    else:
        raise ValueError(f"Unsupported resume format: {ext}. Please provide a .pdf or .docx file.")


def _load_pdf_text(pdf_path: str) -> str:
    """Extract text from each PDF page, concatenated with newline separators."""
    text_pages = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text_pages.append(page.extract_text() or "")
    return "\n".join(text_pages)


def _load_docx_text(docx_path: str) -> str:
    """
    Use docx2txt to extract text from a .docx.
    docx2txt will insert line breaks where appropriate,
    including for bullet points.
    """
    return docx2txt.process(docx_path)
