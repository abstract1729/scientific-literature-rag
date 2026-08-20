"""
Incrementally download arXiv papers from corpus metadata.

Default input:
    data/raw/corpus_metadata.csv

Default output:
    data/raw/papers/

Behavior:
    - Downloads only PDFs that are not already present.
    - Uses the arXiv paper ID as the PDF filename.
    - Validates downloaded files as PDFs.
    - Uses temporary .part files during download.
    - Records failed downloads in:
        data/raw/papers/download_failures.csv
"""

import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


# ---------------------------------------------------------------------
# Default project paths
# ---------------------------------------------------------------------

DEFAULT_METADATA = Path("data/raw/corpus_metadata.csv")
DEFAULT_OUTPUT_DIR = Path("data/raw/papers")

# ---------------------------------------------------------------------
# Download configuration
# ---------------------------------------------------------------------

REQUEST_TIMEOUT = 60
DOWNLOAD_DELAY_SECONDS = 1.0


def normalize_paper_id(paper_id: str) -> str:
    """Normalize an arXiv ID and remove an optional version suffix."""

    paper_id = str(paper_id).strip()
    if "v" in paper_id:
        suffix = paper_id.rsplit("v", 1)[-1]
        if suffix.isdigit():
            paper_id = paper_id.rsplit("v", 1)[0]
    return paper_id


def is_valid_pdf(path: Path) -> bool:
    """Check whether a file exists and is a valid PDF."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with path.open("rb") as file:
            return file.read(4) == b"%PDF"
    except OSError:
        return False


def get_existing_ids(output_dir: Path) -> set[str]:
    """Return IDs corresponding to valid PDFs already downloaded."""
    if not output_dir.exists():
        return set()
    existing_ids = set()
    for pdf_path in output_dir.glob("*.pdf"):
        if is_valid_pdf(pdf_path):
            existing_ids.add(normalize_paper_id(pdf_path.stem))
    return existing_ids


def download_papers(metadata_path: Path = DEFAULT_METADATA,output_dir: Path = DEFAULT_OUTPUT_DIR):
    """Incrementally download missing papers."""

    # -----------------------------------------------------------------
    # Validate metadata
    # -----------------------------------------------------------------

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {metadata_path}")
    metadata = pd.read_csv(metadata_path,dtype=str,)
    required_columns = {"paper_id","title","pdf_url",}
    missing_columns = required_columns - set(metadata.columns)

    if missing_columns:
        raise ValueError(
            "Metadata CSV is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # -----------------------------------------------------------------
    # Clean metadata
    # -----------------------------------------------------------------

    metadata["paper_id"] = (metadata["paper_id"].dropna().map(normalize_paper_id))
    metadata = metadata.dropna(subset=["paper_id", "pdf_url"])
    metadata = metadata.drop_duplicates(subset="paper_id",keep="first")

    # -----------------------------------------------------------------
    # Prepare output directory
    # -----------------------------------------------------------------

    output_dir.mkdir( parents=True, exist_ok=True, )

    # -----------------------------------------------------------------
    # Find already downloaded papers
    # -----------------------------------------------------------------

    existing_ids = get_existing_ids(output_dir)
    papers_to_download = [ row for row in metadata.itertuples(index=False) if row.paper_id not in existing_ids ]

    print(f"Corpus papers      : {len(metadata)}")
    print(f"Already downloaded : {len(existing_ids)}")
    print(f"To download        : {len(papers_to_download)}")

    if not papers_to_download:
        print("\nNo new papers to download.")
        return

    # -----------------------------------------------------------------
    # HTTP session
    # -----------------------------------------------------------------

    session = requests.Session()

    session.headers.update({
        "User-Agent": (
            "ScientificLiteratureRAG/0.1 "
            "(research corpus downloader)"
        )
    })

    failures = []

    # -----------------------------------------------------------------
    # Download
    # -----------------------------------------------------------------

    with tqdm(total=len(papers_to_download),desc="Downloading papers",unit="paper",) as progress:
        for row in papers_to_download:
            paper_id = normalize_paper_id(row.paper_id)

            # Example:
            # 1706.03762.pdf
            pdf_path = output_dir / f"{paper_id}.pdf"

            # Temporary file used during download.
            temp_path = output_dir / f"{paper_id}.pdf.part"

            try:
                response = session.get(row.pdf_url,timeout=REQUEST_TIMEOUT,)
                response.raise_for_status()
                content = response.content

                # Basic PDF validation.
                if not content.startswith(b"%PDF"):
                    raise ValueError("Downloaded content is not a valid PDF.")
                
                # Write to temporary file first.
                temp_path.write_bytes(content)

                # Rename only after successful write.
                temp_path.replace(pdf_path)
                time.sleep(DOWNLOAD_DELAY_SECONDS)

            except Exception as exc:

                failures.append({
                    "paper_id": paper_id,
                    "title": row.title,
                    "pdf_url": row.pdf_url,
                    "error": str(exc),
                })

                # Remove incomplete download.
                if temp_path.exists():
                    temp_path.unlink()

            progress.update(1)

    # -----------------------------------------------------------------
    # Failure log
    # -----------------------------------------------------------------

    failure_path = output_dir / "download_failures.csv"

    if failures:
        pd.DataFrame(failures).to_csv(failure_path,index=False,)
        print(f"\nFailed downloads : {len(failures)}")
        print(f"Failure log      : {failure_path}")

    else:
        # Remove an old failure log if everything succeeded.
        if failure_path.exists():
            failure_path.unlink()
        print("\nAll requested papers downloaded successfully.")

    # -----------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------

    final_existing_ids = get_existing_ids(output_dir)
    print(f"PDFs available    : {len(final_existing_ids)}")


    