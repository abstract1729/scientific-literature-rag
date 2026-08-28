from pathlib import Path

import pymupdf 
from PIL import Image


class PDFPageRenderer:
    """
    Renders PDF pages into persistent PNG images.

    Output:
        data/processed/colpali/pages/
            <paper_id>/
                page_0001.png
                page_0002.png
                ...
    """

    def __init__(self,output_dir: str | Path = "data/processed/colpali/pages",dpi: int = 150):
        self.output_dir = Path(output_dir)
        self.dpi = dpi

    def render_pdf(self,pdf_path: str | Path,overwrite: bool = False) -> list[dict]:
        """
        Render all pages of a PDF.

        Returns:
            List of page metadata dictionaries.
        """

        pdf_path = Path(pdf_path)

        # IMPORTANT:
        # Path.stem preserves dots inside the filename.
        # 1312.5602.pdf -> 1312.5602
        paper_id = pdf_path.stem

        paper_output_dir = self.output_dir / paper_id
        paper_output_dir.mkdir(parents=True, exist_ok=True)

        document = pymupdf.open(pdf_path)

        page_metadata = []

        # PyMuPDF uses a scale factor relative to 72 DPI.
        zoom = self.dpi / 72
        matrix = pymupdf.Matrix(zoom, zoom)

        try:
            for page_idx in range(len(document)):
                page_number = page_idx + 1

                image_path = (paper_output_dir / f"page_{page_number:04d}.png")

                if image_path.exists() and not overwrite:
                    # Read dimensions without rerendering.
                    with Image.open(image_path) as image:
                        width, height = image.size

                else:
                    page = document[page_idx]

                    pixmap = page.get_pixmap(
                        matrix=matrix,
                        alpha=False,
                    )

                    pixmap.save(str(image_path))

                    width = pixmap.width
                    height = pixmap.height

                page_metadata.append(
                    {
                        "paper_id": paper_id,
                        "pdf_filename": pdf_path.name,
                        "page_number": page_number,
                        "page_id": (
                            f"{paper_id}_page_{page_number:04d}"
                        ),
                        "image_path": str(image_path),
                        "width": width,
                        "height": height,
                    }
                )

        finally:
            document.close()

        return page_metadata