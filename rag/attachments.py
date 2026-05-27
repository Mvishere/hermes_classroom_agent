"""Local attachment text extraction."""

import logging
from pathlib import Path
from typing import List, Tuple

from docx import Document
from pypdf import PdfReader


class AttachmentTextExtractor:
    """Extracts text from supported attachment formats."""

    def __init__(self, base_dir: Path, max_chars: int = 6000):
        self.base_dir = Path(base_dir)
        self.max_chars = max_chars

    def extract(self, attachment_paths: List[str]) -> Tuple[str, List[str]]:
        """Return combined text and a list of detected formats."""
        formats = []
        text_chunks = []

        for rel_path in attachment_paths:
            path = self.base_dir / rel_path
            suffix = path.suffix.lower()

            if suffix:
                formats.append(suffix.lstrip("."))

            if suffix == ".pdf":
                text = self._extract_pdf(path)
                if text:
                    text_chunks.append(text)
            elif suffix == ".docx":
                text = self._extract_docx(path)
                if text:
                    text_chunks.append(text)

        combined = "\n".join(text_chunks)

        if len(combined) > self.max_chars:
            combined = combined[: self.max_chars]

        return combined, sorted(set(formats))

    def extract_chunks(
        self,
        attachment_paths: List[str],
        chunk_size: int = 2000,
        overlap: int = 200,
        max_chars: int | None = None,
    ) -> Tuple[List[str], List[str]]:
        """Return chunked text and detected formats."""

        combined_text, formats = self.extract(attachment_paths)

        if max_chars:
            combined_text = combined_text[:max_chars]

        chunks = []

        start = 0
        step = max(chunk_size - overlap, 1)

        while start < len(combined_text):
            end = start + chunk_size
            chunks.append(combined_text[start:end])
            start += step

        return chunks, formats

    def _extract_pdf(self, path: Path) -> str:
        if not path.exists():
            return ""

        try:
            reader = PdfReader(str(path))
            pages_text = []

            for page in reader.pages:
                page_text = page.extract_text() or ""

                if page_text:
                    pages_text.append(page_text)

            return "\n".join(pages_text)

        except Exception:
            logging.exception("Failed to extract PDF text: %s", path)
            return ""

    def _extract_docx(self, path: Path) -> str:
        if not path.exists():
            return ""

        try:
            document = Document(str(path))
            lines = []
            for paragraph in document.paragraphs:
                text = paragraph.text.strip()
                if text:
                    lines.append(text)
            return "\n".join(lines)
        except Exception:
            logging.exception("Failed to extract DOCX text: %s", path)
            return ""
