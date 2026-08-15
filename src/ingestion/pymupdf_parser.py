from __future__ import annotations

from pathlib import Path

import pymupdf

from .base import PDFParser, ParsedDocument, ParsedPage


class PyMuPDFParser(PDFParser):
    name = "pymupdf"

    def parse(self, pdf_path: str | Path) -> ParsedDocument:
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

        paper_id = pdf_path.stem
        pages = []
        text_parts = []

        with pymupdf.open(pdf_path) as document:
            for page_index, page in enumerate(document):
                text = page.get_text("text")

                pages.append(
                    ParsedPage(
                        page_number=page_index + 1,
                        text=text,
                    )
                )

                if text.strip():
                    text_parts.append(text)

        return ParsedDocument(
            paper_id=paper_id,
            pages=pages,
            full_text="\n\n".join(text_parts),
            metadata={
                "parser": self.name,
                "source_path": str(pdf_path),
                "page_count": len(pages),
            },
        )
