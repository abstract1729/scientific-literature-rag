"""
Verify candidate arXiv IDs and incrementally build corpus metadata.

Input:
    data/raw/arxiv_candidate_ids.csv

Outputs:
    data/raw/corpus_metadata.csv
    data/raw/arxiv_verification_failures.csv

Behavior:
    - If corpus_metadata.csv and/or arxiv_verification_failures.csv exist,
      IDs already present in either file are skipped.
    - Only new candidate IDs are queried against the arXiv API.
    - Newly verified papers are appended to corpus_metadata.csv.
    - Newly failed IDs are appended to arxiv_verification_failures.csv.
    - A progress bar shows verification progress.

The arXiv API is the source of truth for:
    - paper title
    - PDF URL
"""


import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = { "atom": "http://www.w3.org/2005/Atom"}
DEFAULT_INPUT = Path("data/raw/arxiv_candidate_ids.csv")
DEFAULT_OUTPUT = Path("data/raw/corpus_metadata.csv")
DEFAULT_FAILURES = Path("data/raw/arxiv_verification_failures.csv")

BATCH_SIZE = 20
API_DELAY_SECONDS = 3


def normalize_paper_id(paper_id: str) -> str:
    """Remove whitespace and an optional arXiv version suffix."""

    paper_id = str(paper_id).strip()

    if "v" in paper_id:
        suffix = paper_id.rsplit("v", 1)[-1]

        if suffix.isdigit():
            paper_id = paper_id.rsplit("v", 1)[0]

    return paper_id


def load_existing_ids(output_path: Path, failures_path: Path) -> set[str]:
    """
    Load IDs that have already been processed.

    An ID is considered processed if it appears in either:
        - corpus_metadata.csv
        - arxiv_verification_failures.csv
    """

    processed_ids = set()

    if output_path.exists():
        metadata = pd.read_csv(output_path, dtype=str)
        if "paper_id" in metadata.columns:
            processed_ids.update(metadata["paper_id"].dropna().map(normalize_paper_id))

    if failures_path.exists():
        failures = pd.read_csv(failures_path, dtype=str)
        if "paper_id" in failures.columns:
            processed_ids.update(failures["paper_id"].dropna().map(normalize_paper_id))

    return processed_ids


def query_arxiv(session, paper_ids, batch_size=BATCH_SIZE):
    """Query arXiv API in batches and return verified metadata and failures."""

    verified = []
    failures = []

    total = len(paper_ids)

    with tqdm(total=total,desc="Verifying arXiv IDs",unit="paper",) as progress:
        for start in range(0, total, batch_size):
            batch = paper_ids[start:start + batch_size]
            search_query = " OR ".join(f"id:{paper_id}" for paper_id in batch)
            response = session.get(ARXIV_API_URL,params={"search_query": search_query,"start": 0,"max_results": len(batch),},timeout=60)
            response.raise_for_status()
            root = ET.fromstring(response.text)
            returned_papers = {}

            for entry in root.findall("atom:entry", ATOM_NS):
                entry_id = entry.findtext("atom:id",default="",namespaces=ATOM_NS,)
                returned_id = entry_id.rstrip("/").split("/")[-1]
                returned_id = normalize_paper_id(returned_id)
                title = entry.findtext("atom:title",default="",namespaces=ATOM_NS,)
                title = " ".join(title.split())

                pdf_url = None
                for link in entry.findall("atom:link", ATOM_NS):
                    if link.attrib.get("title") == "pdf":
                        pdf_url = link.attrib.get("href")
                        break

                returned_papers[returned_id] = {
                    "paper_id": returned_id,
                    "title": title,
                    "pdf_url": pdf_url,
                }

            for paper_id in batch:
                if paper_id not in returned_papers:
                    failures.append({"paper_id": paper_id,"reason": "arXiv ID not found",})
                    continue

                metadata = returned_papers[paper_id]
                if not metadata["title"]:
                    failures.append({"paper_id": paper_id,"reason": "Missing title in arXiv response",})
                    continue

                if not metadata["pdf_url"]:
                    failures.append({"paper_id": paper_id,"reason": "Missing PDF URL in arXiv response",})
                    continue

                verified.append(metadata)
            progress.update(len(batch))

            # Respect arXiv API rate limits.
            if start + batch_size < total:
                time.sleep(API_DELAY_SECONDS)
    return verified, failures


def append_metadata(output_path: Path, verified: list[dict]) -> int:
    """Append newly verified papers and remove duplicate IDs."""

    if not verified:
        return 0

    new_df = pd.DataFrame(verified,columns=["paper_id","title","pdf_url",],)
    if output_path.exists():
        existing_df = pd.read_csv(output_path,dtype=str)
        combined_df = pd.concat([existing_df, new_df],ignore_index=True)

    else:
        combined_df = new_df

    combined_df["paper_id"] = (combined_df["paper_id"].map(normalize_paper_id))
    combined_df = (combined_df.drop_duplicates(subset="paper_id",keep="first",).sort_values("paper_id").reset_index(drop=True))
    output_path.parent.mkdir(parents=True,exist_ok=True,)
    combined_df.to_csv(output_path,index=False,)
    return len(new_df)


def append_failures(failures_path: Path, failures: list[dict]) -> int:
    """Append newly failed IDs and remove duplicate IDs."""

    if not failures:
        return 0

    new_df = pd.DataFrame(failures,columns=["paper_id","reason"])
    if failures_path.exists():
        existing_df = pd.read_csv(failures_path,dtype=str)
        combined_df = pd.concat([existing_df, new_df],ignore_index=True)

    else:
        combined_df = new_df

    combined_df["paper_id"] = (combined_df["paper_id"].map(normalize_paper_id))
    combined_df = (combined_df.drop_duplicates(subset="paper_id",keep="first",).sort_values("paper_id").reset_index(drop=True))
    failures_path.parent.mkdir(parents=True,exist_ok=True,)
    combined_df.to_csv(failures_path,index=False,)
    
    return len(new_df)


def verify_arxiv_ids(input_path: Path,output_path: Path,failures_path: Path):
    """Incrementally verify only unprocessed candidate IDs."""

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    candidates = pd.read_csv(input_path,dtype=str,)
    if "paper_id" not in candidates.columns:
        raise ValueError("Input CSV must contain a 'paper_id' column.")
    
    candidate_ids = (candidates["paper_id"].dropna().map(normalize_paper_id).drop_duplicates().tolist())
    processed_ids = load_existing_ids(output_path=output_path,failures_path=failures_path,)
    paper_ids = [ paper_id for paper_id in candidate_ids if paper_id not in processed_ids ]

    print(f"Candidate IDs : {len(candidate_ids)}")
    print(f"Already known : {len(processed_ids)}")
    print(f"To verify     : {len(paper_ids)}")

    if not paper_ids:
        print("\nNo new IDs to verify.")
        return

    session = requests.Session()

    session.headers.update({
        "User-Agent": (
            "ScientificLiteratureRAG/0.1 "
            "(research corpus construction)"
        )
    })

    verified, failures = query_arxiv(session=session,paper_ids=paper_ids)
    append_metadata(output_path=output_path,verified=verified)
    append_failures(failures_path=failures_path,failures=failures)

    print("\nVerification complete.")
    print(f"Newly verified : {len(verified)}")
    print(f"Newly failed   : {len(failures)}")
    print(f"\nMetadata       : {output_path}")
    print(f"Failures       : {failures_path}")

