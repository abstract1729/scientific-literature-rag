from __future__ import annotations

import time
from typing import Any

import torch
from qdrant_client import QdrantClient

from src.colpali.ingestion.colpali_encoder import ColPaliEncoder
from src.retrieval_evaluation.results import (
    RetrievedPage,
    RetrievalResult,
    RetrievalTiming,
)


class ColPaliQdrantRetriever:
    """
    Vanilla ColPali retrieval using Qdrant multivector search.

    Pipeline:

        Text query
            ↓
        ColPali query encoder
            ↓
        Query multivector
            ↓
        Qdrant MAX_SIM
            ↓
        Top-k pages
            ↓
        RetrievalResult

    This class implements the common retrieval interface expected
    by the retrieval evaluation framework.
    """

    def __init__(
        self,
        encoder: ColPaliEncoder,
        collection_name: str = "colpali_pages",
        qdrant_path: str = "data/processed/colpali/qdrant",
    ):
        self.encoder = encoder
        self.collection_name = collection_name
        self.client = QdrantClient(path=qdrant_path)

    # ------------------------------------------------------------------
    # Public retrieval API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        """
        Retrieve the top-k pages for a natural-language query.

        Returns
        -------
        RetrievalResult
            Standardized retrieval result used by the evaluation
            pipeline.
        """

        self._validate_inputs(
            query=query,
            top_k=top_k,
        )

        start_total = time.perf_counter()

        # --------------------------------------------------------------
        # Query encoding
        # --------------------------------------------------------------

        start_encoding = time.perf_counter()

        query_embedding = self.encoder.encode_query(query)

        self._synchronize_mps()

        encoding_time = (
            time.perf_counter() - start_encoding
        )

        # --------------------------------------------------------------
        # Prepare query vectors
        # --------------------------------------------------------------

        query_vectors = self._prepare_query_vectors(
            query_embedding
        )

        # --------------------------------------------------------------
        # Qdrant search
        # --------------------------------------------------------------

        start_qdrant = time.perf_counter()

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vectors,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        qdrant_time = (
            time.perf_counter() - start_qdrant
        )

        # --------------------------------------------------------------
        # Convert Qdrant results
        # --------------------------------------------------------------

        pages = self._convert_results(
            response.points
        )

        total_time = (
            time.perf_counter() - start_total
        )

        # --------------------------------------------------------------
        # Cleanup
        # --------------------------------------------------------------

        del query_embedding
        del query_vectors

        self._cleanup_mps()

        timing = RetrievalTiming(
            encoding_ms=encoding_time * 1000,
            retrieval_ms=qdrant_time * 1000,
            total_ms=total_time * 1000,
        )

        return RetrievalResult(
            query=query,
            results=pages,
            timing=timing,
            metadata={
                "retriever": "ColPaliQdrantRetriever",
                "collection_name": self.collection_name,
                "top_k": top_k,
            },
        )

    # ------------------------------------------------------------------
    # Timed retrieval API
    # ------------------------------------------------------------------

    def retrieve_with_timing(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        """
        Retrieve pages while recording stage-level latency.

        This method intentionally returns the same RetrievalResult
        interface as retrieve(). The only difference is that timing
        information is populated.
        """

        return self.retrieve(
            query=query,
            top_k=top_k,
        )

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_inputs(
        query: str,
        top_k: int,
    ) -> None:

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string."
            )

        if not query.strip():
            raise ValueError(
                "query must not be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

    # ------------------------------------------------------------------
    # Query preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_query_vectors(
        query_embedding: torch.Tensor,
    ) -> list[list[float]]:
        """
        Convert the ColPali query embedding into the format expected
        by Qdrant.

        Expected ColPali output shape:

            [batch_size, num_query_tokens, embedding_dim]

        For a single query:

            [1, num_query_tokens, embedding_dim]

        Qdrant receives:

            [num_query_tokens, embedding_dim]
        """

        if query_embedding.ndim != 3:
            raise ValueError(
                "Unexpected query embedding shape: "
                f"{tuple(query_embedding.shape)}"
            )

        query_vectors = (
            query_embedding
            .squeeze(0)
            .tolist()
        )

        return query_vectors

    # ------------------------------------------------------------------
    # Result conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_results(
        points: Any,
    ) -> list[RetrievedPage]:
        """
        Convert Qdrant ScoredPoint objects into the standardized
        RetrievedPage representation.
        """

        results: list[RetrievedPage] = []

        for point in points:

            payload = point.payload or {}

            page_id = payload.get("page_id")

            if page_id is None:
                raise ValueError(
                    "Qdrant payload is missing 'page_id'. "
                    f"Qdrant point ID: {point.id}"
                )

            paper_id = payload.get("paper_id")

            if paper_id is None:
                raise ValueError(
                    "Qdrant payload is missing 'paper_id'. "
                    f"Page ID: {page_id}"
                )

            page_number = payload.get("page_number")

            if page_number is None:
                raise ValueError(
                    "Qdrant payload is missing 'page_number'. "
                    f"Page ID: {page_id}"
                )

            results.append(
                RetrievedPage(
                    page_id=str(page_id),
                    paper_id=str(paper_id),
                    page_number=int(page_number),
                    score=float(point.score),
                )
            )

        return results

    # ------------------------------------------------------------------
    # MPS helpers
    # ------------------------------------------------------------------

    def _synchronize_mps(self) -> None:
        """
        Synchronize MPS execution before measuring downstream stages.

        ColPali encoding is asynchronous on MPS, so synchronization is
        necessary for accurate encoding latency measurement.
        """

        if self.encoder.device == "mps":
            torch.mps.synchronize()

    def _cleanup_mps(self) -> None:
        """
        Release cached MPS memory after a query.
        """

        if self.encoder.device == "mps":
            torch.mps.empty_cache()

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the Qdrant client."""

        self.client.close()

    def __enter__(self) -> "ColPaliQdrantRetriever":
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()