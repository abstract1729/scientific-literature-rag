from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from src.colpali.ingestion import ColPaliEncoder
from src.colpali.indexing import ColPaliQdrantIndexer


METADATA_PATH = Path( "data/processed/colpali/metadata/pages.jsonl")
QDRANT_PATH = ( "data/processed/colpali/qdrant" )
MAX_PAGES = 10
COLLECTION_NAME = "colpali_pages"


def load_page_metadata(path: Path):
    """
    Stream page metadata from the JSONL file.

    One dictionary is yielded at a time, so the entire
    corpus metadata is never loaded into memory.
    """

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def count_pages(path: Path) -> int:
    """
    Count the number of page records.

    Used only to provide a useful tqdm total.
    """
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def main():
    print("=" * 70)
    print("ColPali → Qdrant Corpus Indexing")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Validate metadata
    # ---------------------------------------------------------------

    if not METADATA_PATH.exists():

        raise FileNotFoundError(
            f"Metadata file not found:\n"
            f"{METADATA_PATH}"
        )

    total_pages = count_pages(METADATA_PATH)
    print(f"Page records found: {total_pages}")

    # ---------------------------------------------------------------
    # Load ColPali
    # ---------------------------------------------------------------

    print("\nLoading ColPali...")
    encoder = ColPaliEncoder(model_name="vidore/colpali-v1.3",device="mps",)
    print("ColPali loaded.")

    # ---------------------------------------------------------------
    # Create Qdrant indexer
    # ---------------------------------------------------------------

    print("\nInitializing Qdrant...")
    indexer = ColPaliQdrantIndexer(encoder=encoder,collection_name=COLLECTION_NAME,qdrant_path=QDRANT_PATH,)
    print(f"Collection: {COLLECTION_NAME}")

    # ---------------------------------------------------------------
    # Stream and index pages
    # ---------------------------------------------------------------

    print("\nStarting corpus indexing...\n")
    records = load_page_metadata(METADATA_PATH)
    indexed = 0
    skipped = 0
    failed = 0

    with tqdm(total=total_pages,desc="ColPali indexing",unit="page",) as progress:
        for record in records:
            try:
                was_indexed = indexer.index_page(page_id=record["page_id"],
                                                 image_path=record["image_path"],payload=record,)
                if was_indexed:
                    indexed += 1
                else:
                    skipped += 1

            except Exception as exc:

                failed += 1

                print(
                    f"\nERROR: "
                    f"{record['page_id']}"
                )
                print(exc)

            finally:

                progress.update(1)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("Corpus indexing complete")
    print("=" * 70)

    print(f"Total pages : {total_pages}")
    print(f"Indexed     : {indexed}")
    print(f"Skipped     : {skipped}")
    print(f"Failed      : {failed}")


if __name__ == "__main__":
    main()