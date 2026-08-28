from pathlib import Path


def find_pdfs(pdf_dir: str | Path) -> list[Path]:
    """
    Find all PDF files in a directory.

    Parameters
    ----------
    pdf_dir:
        Directory containing PDF files.

    Returns
    -------
    list[Path]
        Sorted list of PDF paths.
    """
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory does not exist: {pdf_dir}")
    
    if not pdf_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory: {pdf_dir}")

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in: {pdf_dir}")
    return pdfs