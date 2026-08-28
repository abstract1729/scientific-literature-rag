import json
from pathlib import Path

from tqdm import tqdm

from .page_renderer import PDFPageRenderer


class CorpusRenderer:
    """
    Renders all PDFs in a corpus and maintains
    page-level metadata in JSONL format.
    """

    def __init__(self,pdf_dir: str | Path,output_dir: str | Path = "data/processed/colpali/pages",
                 metadata_path: str | Path = ("data/processed/colpali/metadata/pages.jsonl"),dpi: int = 150,):
        self.pdf_dir = Path(pdf_dir)
        self.metadata_path = Path(metadata_path)
        self.renderer = PDFPageRenderer(output_dir=output_dir,dpi=dpi)

    def render_corpus(self,overwrite: bool = False) -> int:
        """
        Render every PDF in the corpus.

        Returns:
            Total number of pages rendered/registered.
        """

        pdf_files = sorted(self.pdf_dir.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {self.pdf_dir}")
        self.metadata_path.parent.mkdir(parents=True,exist_ok=True)
        total_pages = 0

        with self.metadata_path.open("w",encoding="utf-8",) as metadata_file:
            progress = tqdm(pdf_files,desc="Rendering PDFs",unit="pdf",)
            for pdf_path in progress:
                progress.set_postfix(paper=pdf_path.stem,pages=total_pages)
                pages = self.renderer.render_pdf(pdf_path,overwrite=overwrite)
                for page in pages:
                    metadata_file.write(json.dumps(page) + "\n")
                total_pages += len(pages)
                progress.set_postfix(paper=pdf_path.stem,pages=total_pages)

        print()
        print("Rendering complete.")
        print(f"PDFs processed : {len(pdf_files)}")
        print(f"Total pages    : {total_pages}")
        print(f"Metadata       : {self.metadata_path}")

        return total_pages