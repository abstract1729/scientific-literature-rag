from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievedPage:
    """
    A single page returned by a retrieval system.

    The page is the evaluation unit. ColPali's internal multi-vectors
    are not exposed here as independent retrieval results.
    """

    page_id: str
    paper_id: str
    page_number: int
    score: float

    # Optional metadata returned by the retriever.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalTiming:
    """
    Timing information for one retrieval operation.

    The common fields cover E0. Additional fields allow E1/E2 to
    report their individual retrieval stages without changing the
    evaluation interface.
    """

    encoding_ms: float | None = None
    retrieval_ms: float | None = None
    total_ms: float | None = None

    # E1 / E2 optional stages.
    metadata_filter_ms: float | None = None
    candidate_retrieval_ms: float | None = None
    reranking_ms: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        """Convert timing information into a serializable dictionary."""

        return {
            "encoding_ms": self.encoding_ms,
            "retrieval_ms": self.retrieval_ms,
            "total_ms": self.total_ms,
            "metadata_filter_ms": self.metadata_filter_ms,
            "candidate_retrieval_ms": self.candidate_retrieval_ms,
            "reranking_ms": self.reranking_ms,
        }


@dataclass
class RetrievalResult:
    """
    Standardized output from any retrieval experiment.

    Every retriever used by the evaluation framework should return
    this object.
    """

    query: str
    results: list[RetrievedPage]

    timing: RetrievalTiming | None = None

    # Optional experiment-specific information.
    metadata: dict[str, Any] = field(default_factory=dict)

    def page_ids(self) -> list[str]:
        """Return retrieved page IDs in ranking order."""

        return [
            result.page_id
            for result in self.results
        ]

    def scores(self) -> list[float]:
        """Return retrieval scores in ranking order."""

        return [
            result.score
            for result in self.results
        ]

    def top_score(self) -> float | None:
        """Return the highest retrieval score, if available."""

        if not self.results:
            return None

        return self.results[0].score

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the complete retrieval result into a JSON-serializable
        dictionary.
        """

        return {
            "query": self.query,
            "results": [
                {
                    "page_id": result.page_id,
                    "paper_id": result.paper_id,
                    "page_number": result.page_number,
                    "score": result.score,
                    "metadata": result.metadata,
                }
                for result in self.results
            ],
            "timing": (
                self.timing.to_dict()
                if self.timing is not None
                else None
            ),
            "metadata": self.metadata,
        }