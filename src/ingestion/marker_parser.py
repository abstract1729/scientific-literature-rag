from __future__ import annotations

from pathlib import Path

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

from .base import PDFParser, ParsedDocument


class MarkerParser(PDFParser):
    """PDF parser implementation using Marker."""

    name = "marker"

    def __init__(self) -> None:
        self.converter = PdfConverter(artifact_dict=create_model_dict())

    def parse(self, pdf_path: str | Path) -> ParsedDocument:
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

        paper_id = pdf_path.stem
        rendered = self.converter(str(pdf_path))

        # Marker versions can expose the converted result differently.
        # Prefer Markdown when available.
        if hasattr(rendered, "markdown"):
            full_text = rendered.markdown

        elif isinstance(rendered, str):
            full_text = rendered

        else:
            full_text = str(rendered)

        return ParsedDocument(paper_id=paper_id,full_text=full_text,
                              metadata={"parser": self.name,"source_path": str(pdf_path)})