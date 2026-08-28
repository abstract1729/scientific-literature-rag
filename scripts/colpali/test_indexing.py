from pathlib import Path

from src.colpali.ingestion import ColPaliEncoder
from src.colpali.indexing import ColPaliQdrantIndexer


IMAGE_PATH = Path("data/processed/colpali/pages/""1312.5602/page_0001.png")

def main():
    print("Loading ColPali encoder...")
    encoder = ColPaliEncoder(model_name="vidore/colpali-v1.3",device="mps")
    print("Creating Qdrant indexer...")

    indexer = ColPaliQdrantIndexer(encoder=encoder,collection_name="colpali_pages_test",
                                   qdrant_path=("data/processed/colpali/qdrant_test"))

    page_id = "1312.5602_page_0001"

    payload = {
        "paper_id": "1312.5602",
        "pdf_filename": "1312.5602.pdf",
        "page_number": 1,
        "page_id": page_id,
        "image_path": str(IMAGE_PATH),
    }

    print("\nEncoding and indexing page...")

    indexed = indexer.index_page(
        page_id=page_id,
        image_path=IMAGE_PATH,
        payload=payload,
    )

    print(
        f"\nNewly indexed: {indexed}"
    )

    print("\nRunning the same page again...")

    indexed_again = indexer.index_page(
        page_id=page_id,
        image_path=IMAGE_PATH,
        payload=payload,
    )

    print(
        f"Newly indexed: {indexed_again}"
    )


if __name__ == "__main__":
    main()