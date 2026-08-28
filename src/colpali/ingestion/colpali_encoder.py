from __future__ import annotations

from typing import Sequence

import torch
from PIL import Image
from colpali_engine.models import ColPali, ColPaliProcessor


class ColPaliEncoder:
    """
    ColPali image encoder for visual document pages.

    Responsible only for:

        PIL image(s) -> ColPali multi-vector embeddings

    The encoder does not know anything about Qdrant or indexing.
    """

    def __init__(
        self,
        model_name: str = "vidore/colpali-v1.3",
        device: str = "mps",
    ):
        self.model_name = model_name
        self.device = self._resolve_device(device)

        self.processor = (
            ColPaliProcessor.from_pretrained(
                self.model_name
            )
        )

        self.model = (
            ColPali.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
            )
            .eval()
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        """
        Validate the requested computation device.
        """

        if device == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError(
                    "MPS was requested but is not available."
                )

        elif device != "cpu":
            raise ValueError(
                f"Unsupported device: {device}. "
                "Use 'mps' or 'cpu'."
            )

        return device

    @torch.no_grad()
    def encode_images(
        self,
        images: Sequence[Image.Image],
    ) -> torch.Tensor:
        """
        Encode images using ColPali.

        Returns
        -------
        torch.Tensor
            CPU tensor with shape:

                [batch_size, num_tokens, embedding_dim]

        The model itself runs on the configured device
        (MPS on your Mac), but the returned embeddings are
        immediately moved to CPU memory.
        """

        if not images:
            raise ValueError("No images provided.")

        batch = (
            self.processor
            .process_images(list(images))
            .to(self.device)
        )

        embeddings = self.model(**batch)

        # We do not need gradients and do not want the
        # embedding occupying MPS memory after inference.
        embeddings = embeddings.detach().cpu()

        # Release model inputs from MPS memory.
        del batch

        if self.device == "mps":
            torch.mps.empty_cache()

        return embeddings


    @torch.no_grad()
    def encode_query(
        self,
        query: str,
    ) -> torch.Tensor:
        """
        Encode a natural-language query using ColPali.

        Returns:
            CPU tensor with shape:

                [1, num_query_tokens, embedding_dim]
        """

        if not query or not query.strip():
            raise ValueError(
                "Query must be a non-empty string."
            )

        batch = (
            self.processor
            .process_queries([query])
            .to(self.device)
        )

        embeddings = self.model(**batch)

        # Query embedding is also moved to CPU.
        embeddings = embeddings.detach().cpu()

        del batch

        if self.device == "mps":
            torch.mps.empty_cache()

        return embeddings