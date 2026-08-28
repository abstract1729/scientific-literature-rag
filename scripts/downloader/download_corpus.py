"""
Download the scientific-paper corpus.

Workflow:
    1. Verify candidate arXiv IDs and create corpus metadata.
    2. Download papers listed in the verified metadata.

This script is the executable entry point.
Reusable logic lives under utils/downloaders/.
"""

from pathlib import Path
from utils.downloaders import (verify_arxiv_ids,download_papers,)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

CANDIDATE_IDS_PATH = Path("data/raw/arxiv_candidate_ids.csv")
CORPUS_METADATA_PATH = Path("data/raw/corpus_metadata.csv")
VERIFICATION_FAILURES_PATH = Path("data/raw/arxiv_verification_failures.csv")
PAPERS_DIR = Path("data/raw/papers")

def main():
    """Run the complete corpus download workflow."""

    print("=" * 70)
    print("Scientific Literature RAG - Corpus Downloader")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Step 1: Verify arXiv IDs
    # -----------------------------------------------------------------

    print("\n[1/2] Verifying arXiv paper IDs")
    print("-" * 70)

    verify_arxiv_ids(input_path=CANDIDATE_IDS_PATH,output_path=CORPUS_METADATA_PATH,failures_path=VERIFICATION_FAILURES_PATH,)

    # -----------------------------------------------------------------
    # Step 2: Download verified papers
    # -----------------------------------------------------------------

    print("\n[2/2] Downloading papers")
    print("-" * 70)

    download_papers(metadata_path=CORPUS_METADATA_PATH,output_dir=PAPERS_DIR,)

    # -----------------------------------------------------------------
    # Complete
    # -----------------------------------------------------------------

    print("\n" + "=" * 70)
    print("Corpus download workflow complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()