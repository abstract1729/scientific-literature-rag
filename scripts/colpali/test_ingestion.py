from pathlib import Path

import torch

from src.colpali import (ColPaliEncoder,PDFPageRenderer,find_pdfs)
PDF_DIR = Path("data/raw/papers")


def main() -> None:
    print("=" * 60)
    print("ColPali Ingestion Smoke Test")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Find PDFs
    # ---------------------------------------------------------
    pdfs = find_pdfs(PDF_DIR)

    print(f"PDFs found: {len(pdfs)}")
    print(f"Testing: {pdfs[0].name}")

    # ---------------------------------------------------------
    # 2. Render first page
    # ---------------------------------------------------------
    renderer = PDFPageRenderer(dpi=150)
    image = renderer.render_pdf(pdf_path=pdfs[0],page_number=0)
    print(f"Page image size: {image.size}")
    print(f"Image mode: {image.mode}")

    # ---------------------------------------------------------
    # 3. Load ColPali
    # ---------------------------------------------------------
    print("\nLoading ColPali...")
    encoder = ColPaliEncoder(model_name="vidore/colpali-v1.3",device="mps")
    print("ColPali loaded.")

    # ---------------------------------------------------------
    # 4. Encode page
    # ---------------------------------------------------------
    print("\nEncoding page...")

    embeddings = encoder.encode_images([image])

    print("Encoding successful.")
    print(f"Embedding shape: {tuple(embeddings.shape)}")
    print(f"Embedding dtype: {embeddings.dtype}")
    print(f"Embedding device: {embeddings.device}")

    # ---------------------------------------------------------
    # 5. Basic sanity checks
    # ---------------------------------------------------------
    assert embeddings.ndim == 3
    assert embeddings.shape[0] == 1

    print("\nSanity checks passed.")

    # Release MPS cache where supported.
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print("=" * 60)


if __name__ == "__main__":
    main()