from __future__ import annotations

from typing import Any

import torch
import time
from qdrant_client import QdrantClient

from src.colpali.ingestion.colpali_encoder import (ColPaliEncoder)


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
    """

    def __init__(self,encoder: ColPaliEncoder,collection_name: str = "colpali_pages",
                 qdrant_path: str = ("data/processed/colpali/qdrant")):
        self.encoder = encoder
        self.collection_name = collection_name
        self.client = QdrantClient(path=qdrant_path)

    def retrieve(self,query: str,top_k: int = 5,) -> list[dict[str, Any]]:
        """
        Retrieve the top-k pages for a natural-language query.

        Returns:
            List of dictionaries containing:
                - qdrant_id
                - score
                - payload
        """

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        # -------------------------------------------------------------
        # Encode query
        # -------------------------------------------------------------

        query_embedding = ( self.encoder.encode_query(query))
        # Shape:
        # [1, num_query_tokens, 128]
        if query_embedding.ndim != 3:
            raise ValueError(
                "Unexpected query embedding shape: "
                f"{tuple(query_embedding.shape)}"
            )

        # Remove batch dimension.
        query_vectors = (
            query_embedding
            .squeeze(0)
            .tolist()
        )

        # -------------------------------------------------------------
        # Qdrant MaxSim retrieval
        # -------------------------------------------------------------

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vectors,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        # -------------------------------------------------------------
        # Format results
        # -------------------------------------------------------------

        results = []

        for point in response.points:

            results.append(
                {
                    "qdrant_id": str(point.id),
                    "score": float(point.score),
                    "payload": point.payload,
                }
            )

        # -------------------------------------------------------------
        # Cleanup
        # -------------------------------------------------------------

        del query_embedding
        del query_vectors

        if self.encoder.device == "mps":
            torch.mps.empty_cache()

        return results


    def retrieve_with_timing(self,query: str,top_k: int = 5,) -> tuple[list[dict], dict]:
        """
        Retrieve pages while measuring query encoding,
        Qdrant search, and total retrieval latency.
        """

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        # -------------------------------------------------------------
        # Query encoding
        # -------------------------------------------------------------

        start_total = time.perf_counter()
        start_encoding = time.perf_counter()
        query_embedding = (self.encoder.encode_query(query))

        if self.encoder.device == "mps":
            torch.mps.synchronize()
            
        encoding_time = (time.perf_counter()- start_encoding)

        # -------------------------------------------------------------
        # Prepare query vectors
        # -------------------------------------------------------------

        start_qdrant = time.perf_counter()

        query_vectors = (query_embedding.squeeze(0).tolist())
        response = self.client.query_points(collection_name=self.collection_name,
                                            query=query_vectors,limit=top_k,with_payload=True,with_vectors=False,)
        qdrant_time = (time.perf_counter()- start_qdrant)

        # -------------------------------------------------------------
        # Format results
        # -------------------------------------------------------------

        results = []
        for point in response.points:
            results.append(
                {
                    "qdrant_id": str(point.id),
                    "score": float(point.score),
                    "payload": point.payload,
                }
            )

        total_time = (time.perf_counter()- start_total)

        # -------------------------------------------------------------
        # Cleanup
        # -------------------------------------------------------------

        del query_embedding
        del query_vectors

        if self.encoder.device == "mps":
            torch.mps.empty_cache()

        timing = {
            "encoding_ms": encoding_time * 1000,
            "qdrant_ms": qdrant_time * 1000,
            "total_ms": total_time * 1000,
        }

        return results, timing