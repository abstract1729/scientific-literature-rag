from __future__ import annotations

from pathlib import Path

from docling.document_converter import DocumentConverter

from .base import PDFParser, ParsedDocument, ParsedPage


class DoclingParser(PDFParser):
    """PDF parser implementation using Docling."""

    name = "docling"

    def __init__(self) -> None:
        self.converter = DocumentConverter()

    def parse(self, pdf_path: str | Path) -> ParsedDocument:
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF file does not exist: {pdf_path}"
            )

        paper_id = pdf_path.stem

        result = self.converter.convert(str(pdf_path))
        document = result.document

        # Export the Docling representation as Markdown.
        # This preserves considerably more structure than plain text.
        full_text = document.export_to_markdown()

        pages: list[ParsedPage] = []

        # Keep the common page representation conservative for now.
        # The complete Docling document is retained in metadata so that
        # structural information can be inspected during experimentation.
        for page_number in range(1, len(document.pages) + 1):
            pages.append(
                ParsedPage(
                    page_number=page_number,
                    text="",
                )
            )

        return ParsedDocument(
            paper_id=paper_id,
            pages=pages,
            full_text=full_text,
            metadata={
                "parser": self.name,
                "source_path": str(pdf_path),
                "page_count": len(pages),
                "docling_document": document,
            },
        )