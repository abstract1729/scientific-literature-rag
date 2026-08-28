from __future__ import annotations

import uuid
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import (Distance,HnswConfigDiff,MultiVectorComparator,
                                  MultiVectorConfig,PointStruct,VectorParams)
from src.colpali.ingestion.colpali_encoder import (ColPaliEncoder)


class ColPaliQdrantIndexer:
    """
    Index ColPali page embeddings directly into Qdrant.

    One PDF page corresponds to one Qdrant point.
    """

    def __init__(
        self,
        encoder: ColPaliEncoder,
        collection_name: str = "colpali_pages",
        qdrant_path: str = (
            "data/processed/colpali/qdrant"
        ),
    ):
        self.encoder = encoder
        self.collection_name = collection_name

        self.client = QdrantClient(
            path=qdrant_path
        )

        self._ensure_collection()

    # ------------------------------------------------------------------
    # Qdrant ID
    # ------------------------------------------------------------------

    @staticmethod
    def _point_id(page_id: str) -> str:
        """
        Generate a deterministic UUID for a page.
        The same page_id always produces the same Qdrant ID.
        """
        return str(uuid.uuid5(uuid.NAMESPACE_URL,page_id))

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        """
        Create the ColPali multivector collection if it
        does not already exist.
        """
        collections = self.client.get_collections()
        existing_names = { collection.name for collection in collections.collections }
        if self.collection_name in existing_names:
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=128,distance=Distance.DOT,
                                        multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM),),
                                        hnsw_config=HnswConfigDiff(m=0))

    # ------------------------------------------------------------------
    # Incremental check
    # ------------------------------------------------------------------

    def _page_exists(
        self,
        page_id: str,
    ) -> bool:
        """
        Check whether a page has already been indexed.
        """

        point_id = self._point_id(page_id)

        result = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
            with_payload=False,
            with_vectors=False,
        )

        return len(result) > 0

    # ------------------------------------------------------------------
    # Tensor conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _tensor_to_vectors(embedding: torch.Tensor,) -> list[list[float]]:
        """
        Convert a single-page ColPali embedding to the
        list-of-vectors representation expected by Qdrant.

        Input:
            [1, num_tokens, embedding_dim]

        Output:
            [num_tokens, embedding_dim]
        """

        if embedding.ndim != 3:
            raise ValueError(
                "Expected embedding with shape "
                "[batch, num_tokens, embedding_dim]. "
                f"Got: {tuple(embedding.shape)}"
            )

        if embedding.shape[0] != 1:
            raise ValueError(
                "index_page() expects exactly one page."
            )

        embedding = embedding.squeeze(0)

        return embedding.tolist()

    # ------------------------------------------------------------------
    # Single page
    # ------------------------------------------------------------------

    def index_page(self,page_id: str,image_path: str | Path,payload: dict) -> bool:
        """
        Encode and index one page.

        Returns
        -------
        bool
            True  -> page was newly indexed
            False -> page was already present
        """

        if self._page_exists(page_id):
            return False

        image = Image.open(image_path).convert("RGB")
        embeddings = None
        vectors = None

        try:
            embeddings = self.encoder.encode_images([image])
            vectors = self._tensor_to_vectors(embeddings)
            point = PointStruct(id=self._point_id(page_id),vector=vectors,payload=payload)
            self.client.upsert(collection_name=self.collection_name,points=[point])
            return True

        finally:
            image.close()

            del image

            if embeddings is not None:
                del embeddings

            if vectors is not None:
                del vectors

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    # ------------------------------------------------------------------
    # Corpus
    # ------------------------------------------------------------------

    def index_pages(self,page_records: Iterable[dict]) -> None:
        """
        Incrementally index all pages.

        Already-indexed pages are skipped using their
        deterministic Qdrant point IDs.
        """

        from tqdm import tqdm

        indexed = 0
        skipped = 0

        for record in tqdm(
            page_records,
            desc="Indexing ColPali pages",
            unit="page",
        ):
            was_indexed = self.index_page(
                page_id=record["page_id"],
                image_path=record["image_path"],
                payload=record,
            )

            if was_indexed:
                indexed += 1
            else:
                skipped += 1

        print(
            f"\nIndexed: {indexed} pages"
        )
        print(
            f"Skipped: {skipped} pages"
        )