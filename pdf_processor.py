import fitz  # PyMuPDF
import re


class PDFProcessor:
    """
    Handles PDF loading, text extraction and chunking.
    """

    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text(self, pdf_path):
        """Extract raw text from PDF with page tracking."""
        doc = fitz.open(pdf_path)
        pages = []

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            # Clean up whitespace
            text = re.sub(r'\n+', '\n', text).strip()
            if text:
                pages.append({
                    "page": page_num,
                    "text": text
                })

        doc.close()
        return pages

    def chunk_text(self, pages):
        """
        Split text into overlapping chunks for better retrieval.
        Each chunk tracks which page it came from.
        """
        chunks = []

        for page_data in pages:
            text = page_data["text"]
            page_num = page_data["page"]
            words = text.split()

            if not words:
                continue

            start = 0
            while start < len(words):
                end = min(start + self.chunk_size, len(words))
                chunk_text = " ".join(words[start:end])

                chunks.append({
                    "text": chunk_text,
                    "page": page_num,
                    "chunk_id": len(chunks)
                })

                # Move forward with overlap
                start += self.chunk_size - self.chunk_overlap

        return chunks

    def process(self, pdf_path):
        """Full pipeline: extract and chunk."""
        pages = self.extract_text(pdf_path)
        chunks = self.chunk_text(pages)
        return chunks, len(pages)