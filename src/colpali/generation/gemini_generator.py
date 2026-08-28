from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types


class GeminiGenerator:
    """
    Multimodal answer generator for the ColPali RAG pipeline.

    The generator receives:
        - the user's query
        - retrieved page metadata

    It sends the query and the retrieved page images
    to Gemini 2.5 Flash.
    """

    def __init__(
        self,
        model_name: str = "gemini-3.6-flash",
        api_key: str | None = None,
    ):
        load_dotenv()

        self.model_name = model_name

        api_key = api_key or os.getenv(
            "GEMINI_API_KEY"
        )

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY was not found. "
                "Set it in your .env file."
            )

        self.client = genai.Client(
            api_key=api_key
        )

    def _build_prompt(
        self,
        query: str,
        retrieved_results: list[dict[str, Any]],
    ) -> str:
        """
        Build the textual portion of the multimodal prompt.

        The actual page images are passed separately.
        """

        evidence_description = []

        for rank, result in enumerate(
            retrieved_results,
            start=1,
        ):
            payload = result.get(
                "payload",
                {},
            )

            evidence_description.append(
                f"""
Evidence {rank}:
- Paper ID: {payload.get("paper_id")}
- Page ID: {payload.get("page_id")}
- Page number: {payload.get("page_number")}
- Retrieval score: {result.get("score")}
"""
            )

        evidence_text = "\n".join(
            evidence_description
        )

        return f"""
You are answering a question about scientific research papers.

Answer the user's question using ONLY the information
contained in the provided retrieved page images.

User question:
{query}

The retrieved pages are listed below. Each image corresponds
to one evidence item in the same order.

{evidence_text}

Instructions:
1. Answer the user's question directly and clearly.
2. Use the retrieved page images as your primary evidence.
3. Do not invent information that is not supported by the
   retrieved pages.
4. If the retrieved pages do not contain enough information
   to answer the question, explicitly say so.
5. When useful, refer to the relevant paper or page number.
6. Preserve equations, technical terminology, and numerical
   information accurately.
7. For tables and figures, interpret the information from
   the provided page images rather than assuming their contents.
8. Gemini may only cite/use the retrieved evidence and must not refer to pages that were not provided.
"""

    @staticmethod
    def _load_image_part(
        image_path: str | Path,
    ) -> types.Part:
        """
        Load a local page image and convert it into a Gemini
        multimodal input part.
        """

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Retrieved image does not exist: "
                f"{image_path}"
            )

        image_bytes = image_path.read_bytes()

        suffix = image_path.suffix.lower()

        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }

        mime_type = mime_types.get(
            suffix
        )

        if mime_type is None:
            raise ValueError(
                f"Unsupported image format: "
                f"{suffix}"
            )

        return types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type,
        )

    def generate(
        self,
        query: str,
        retrieved_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Generate an answer from retrieved page images.

        Returns:
            {
                "query": ...,
                "answer": ...,
                "evidence": [...]
            }
        """

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if not retrieved_results:
            raise ValueError(
                "No retrieved results were provided."
            )

        prompt = self._build_prompt(
            query=query,
            retrieved_results=retrieved_results,
        )

        # -------------------------------------------------------------
        # Construct multimodal context
        # -------------------------------------------------------------

        contents = [prompt]

        for result in retrieved_results:

            payload = result.get(
                "payload",
                {},
            )

            image_path = payload.get(
                "image_path"
            )

            if not image_path:
                raise ValueError(
                    "Retrieved result is missing "
                    "'image_path'."
                )

            contents.append(
                self._load_image_part(
                    image_path
                )
            )

        # -------------------------------------------------------------
        # Generate answer
        # -------------------------------------------------------------

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
        )

        answer = response.text

        # -------------------------------------------------------------
        # Preserve evidence for evaluation
        # -------------------------------------------------------------

        evidence = []

        for rank, result in enumerate(
            retrieved_results,
            start=1,
        ):
            payload = result.get(
                "payload",
                {},
            )

            evidence.append(
                {
                    "rank": rank,
                    "score": result.get("score"),
                    "qdrant_id": result.get(
                        "qdrant_id"
                    ),
                    "paper_id": payload.get(
                        "paper_id"
                    ),
                    "page_id": payload.get(
                        "page_id"
                    ),
                    "page_number": payload.get(
                        "page_number"
                    ),
                    "image_path": payload.get(
                        "image_path"
                    ),
                }
            )

        return {
            "query": query,
            "answer": answer,
            "evidence": evidence,
        }