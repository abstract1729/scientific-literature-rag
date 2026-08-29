from __future__ import annotations

import time
from typing import Any

import torch
from qdrant_client import QdrantClient
from src.colpali.ingestion.colpali_encoder import ColPaliEncoder
from src.retrieval_evaluation.results import (RetrievedPage,RetrievalResult,RetrievalTiming)


class ColPaliQdrantRetriever:
    """
    Vanilla ColPali retrieval using Qdrant multivector search.

    Query:
        text
        ↓
        ColPali query embedding
        ↓
        Qdrant MAX_SIM
        ↓
        top-k pages
        ↓
        RetrievalResult
    """

    def __init__(self,encoder: ColPaliEncoder,collection_name: str = "colpali_pages",qdrant_path: str = "data/processed/colpali/qdrant"):
        self.encoder = encoder
        self.collection_name = collection_name
        self.qdrant_path = qdrant_path
        self.client = QdrantClient(path=qdrant_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_query(self, query: str):
        """Encode a text query using ColPali."""

        query_embedding = self.encoder.encode_query(query)

        if query_embedding.ndim != 3:
            raise ValueError(
                "Unexpected query embedding shape: "
                f"{tuple(query_embedding.shape)}"
            )

        return query_embedding

    def _search_qdrant(self,query_embedding,top_k: int,):
        """Run Qdrant MaxSim search."""

        query_vectors = (query_embedding.squeeze(0).tolist())
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vectors,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        return response

    @staticmethod
    def _format_results(response) -> list[RetrievedPage]:
        """
        Convert Qdrant points into standardized RetrievedPage objects.
        """

        results: list[RetrievedPage] = []
        for point in response.points:
            payload: dict[str, Any] = point.payload or {}
            page_id = payload.get("page_id")
            if page_id is None:
                raise ValueError(
                    "Qdrant point is missing required 'page_id' "
                    "in payload."
                )

            paper_id = payload.get("paper_id")

            if paper_id is None:
                raise ValueError(
                    "Qdrant point is missing required 'paper_id' "
                    "in payload."
                )

            page_number = payload.get("page_number")

            if page_number is None:
                raise ValueError(
                    "Qdrant point is missing required 'page_number' "
                    "in payload."
                )

            results.append(
                RetrievedPage(
                    page_id=str(page_id),
                    paper_id=str(paper_id),
                    page_number=int(page_number),
                    score=float(point.score),
                    metadata=payload,
                )
            )

        return results

    def _cleanup(
        self,
        query_embedding,
        query_vectors=None,
    ) -> None:
        """Release temporary query tensors."""

        if query_embedding is not None:
            del query_embedding

        if query_vectors is not None:
            del query_vectors

        if self.encoder.device == "mps":
            torch.mps.empty_cache()

    # ------------------------------------------------------------------
    # Standard retrieval interface
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        """
        Retrieve top-k pages for a natural-language query.

        Returns
        -------
        RetrievalResult
            Standardized retrieval result used by the evaluation
            framework.
        """

        if not query or not query.strip():
            raise ValueError(
                "query must be a non-empty string."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        query_embedding = None

        try:
            # ----------------------------------------------------------
            # Query encoding
            # ----------------------------------------------------------

            query_embedding = self._encode_query(query)

            # ----------------------------------------------------------
            # Qdrant MaxSim retrieval
            # ----------------------------------------------------------

            response = self._search_qdrant(
                query_embedding=query_embedding,
                top_k=top_k,
            )

            # ----------------------------------------------------------
            # Standardize results
            # ----------------------------------------------------------

            results = self._format_results(response)

            return RetrievalResult(
                query=query,
                results=results,
                timing=None,
            )

        finally:
            self._cleanup(query_embedding)

    # ------------------------------------------------------------------
    # Retrieval with latency measurement
    # ------------------------------------------------------------------

    def retrieve_with_timing(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        """
        Retrieve top-k pages while measuring:

        - query encoding latency
        - Qdrant retrieval latency
        - total retrieval latency

        Returns
        -------
        RetrievalResult
            Standardized retrieval result containing pages and
            timing information.
        """

        if not query or not query.strip():
            raise ValueError(
                "query must be a non-empty string."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        query_embedding = None

        start_total = time.perf_counter()

        try:
            # ----------------------------------------------------------
            # Query encoding
            # ----------------------------------------------------------

            start_encoding = time.perf_counter()

            query_embedding = self._encode_query(query)

            # MPS operations are asynchronous.
            # Synchronize before recording the encoding latency.
            if self.encoder.device == "mps":
                torch.mps.synchronize()

            encoding_time = (
                time.perf_counter()
                - start_encoding
            )

            # ----------------------------------------------------------
            # Qdrant retrieval
            # ----------------------------------------------------------

            start_qdrant = time.perf_counter()

            response = self._search_qdrant(
                query_embedding=query_embedding,
                top_k=top_k,
            )

            qdrant_time = (
                time.perf_counter()
                - start_qdrant
            )

            # ----------------------------------------------------------
            # Format results
            # ----------------------------------------------------------

            results = self._format_results(response)

            total_time = (
                time.perf_counter()
                - start_total
            )

            timing = RetrievalTiming(
                encoding_ms=encoding_time * 1000.0,
                retrieval_ms=qdrant_time * 1000.0,
                total_ms=total_time * 1000.0,
            )

            return RetrievalResult(
                query=query,
                results=results,
                timing=timing,
            )

        finally:
            self._cleanup(query_embedding)

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the Qdrant client."""

        self.client.close()

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.close()