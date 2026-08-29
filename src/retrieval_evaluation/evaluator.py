from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .metrics import (evaluate_retrieval,evaluate_unanswerable)
from .protocol import RetrieverProtocol
from .schemas import EvaluationQuestion
from .results import RetrievalResult


@dataclass
class EvaluationRecord:
    """
    Evaluation result for a single question.

    This preserves the original question metadata together with
    retrieval results and computed metrics.
    """

    question_id: str
    question: str
    question_type: str
    difficulty: str
    answerability: str

    paper_ids: list[str]
    gold_page_ids: list[str]
    supporting_page_ids: list[str]

    retrieved_page_ids: list[str]
    retrieved_paper_ids: list[str]
    retrieved_scores: list[float]

    metrics: dict[str, float]

    timing: dict[str, float | None] | None = None

    reference_answer: str | None = None

    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the evaluation record into a JSON-serializable dict."""

        return {
            "question_id": self.question_id,
            "question": self.question,
            "question_type": self.question_type,
            "difficulty": self.difficulty,
            "answerability": self.answerability,
            "paper_ids": self.paper_ids,
            "gold_page_ids": self.gold_page_ids,
            "supporting_page_ids": self.supporting_page_ids,
            "retrieved_page_ids": self.retrieved_page_ids,
            "retrieved_paper_ids": self.retrieved_paper_ids,
            "retrieved_scores": self.retrieved_scores,
            "metrics": self.metrics,
            "timing": self.timing,
            "reference_answer": self.reference_answer,
            "metadata": self.metadata,
        }


class RetrievalEvaluator:
    """
    Evaluates any retriever implementing RetrieverProtocol.

    The evaluator is intentionally independent of the retrieval
    implementation. Therefore the same evaluator can be used for:

        E0 -> Vanilla ColPali + Qdrant
        E1 -> Metadata-filtered ColPali
        E2 -> Metadata filtering + reranking
    """

    def __init__(
        self,
        retriever: RetrieverProtocol,
        ks: Sequence[int] = (1, 3, 5, 10),
        use_timing: bool = True,
    ):
        if not ks:
            raise ValueError("ks must contain at least one value.")

        if any(k <= 0 for k in ks):
            raise ValueError("All values in ks must be greater than 0.")

        self.retriever = retriever
        self.ks = tuple(sorted(set(ks)))
        self.use_timing = use_timing

    # ------------------------------------------------------------------
    # Single-question evaluation
    # ------------------------------------------------------------------

    def evaluate_question(
        self,
        question: EvaluationQuestion,
        top_k: int | None = None,
    ) -> EvaluationRecord:
        """
        Evaluate retrieval for a single question.

        Parameters
        ----------
        question:
            EvaluationQuestion containing the query and ground-truth
            page IDs.

        top_k:
            Number of pages to retrieve. If omitted, the largest K
            required by the configured metrics is used.

        Returns
        -------
        EvaluationRecord
            Retrieval results, metrics, metadata and optional timing.
        """

        if not question.question.strip():
            raise ValueError(
                f"Question {question.question_id} contains an empty query."
            )

        # We need at least the largest K requested by the metrics.
        retrieval_k = top_k if top_k is not None else max(self.ks)

        if retrieval_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        # --------------------------------------------------------------
        # Retrieval
        # --------------------------------------------------------------

        if self.use_timing:
            retrieval_result = self.retriever.retrieve_with_timing(
                query=question.question,
                top_k=retrieval_k,
            )
        else:
            retrieval_result = self.retriever.retrieve(
                query=question.question,
                top_k=retrieval_k,
            )

        if not isinstance(retrieval_result, RetrievalResult):
            raise TypeError(
                "Retriever must return a RetrievalResult. "
                f"Got: {type(retrieval_result).__name__}"
            )

        # --------------------------------------------------------------
        # Extract retrieved information
        # --------------------------------------------------------------

        retrieved_page_ids = retrieval_result.page_ids()

        retrieved_paper_ids = [
            page.paper_id
            for page in retrieval_result.results
        ]

        retrieved_scores = retrieval_result.scores()

        # --------------------------------------------------------------
        # Compute metrics
        # --------------------------------------------------------------

        if question.answerability == "answerable":
            metrics = evaluate_retrieval(retrieved_page_ids=retrieved_page_ids,gold_page_ids=question.gold_page_ids,
                                         retrieved_paper_ids=retrieved_paper_ids,gold_paper_ids=question.paper_ids,ks=self.ks)
            
        elif question.answerability == "unanswerable":
            metrics = evaluate_unanswerable(retrieved_scores=retrieved_scores)
            
        else:
            raise ValueError(
                f"Unknown answerability value "
                f"'{question.answerability}' for question "
                f"{question.question_id}."
            )

        # --------------------------------------------------------------
        # Timing
        # --------------------------------------------------------------

        timing = None

        if retrieval_result.timing is not None:
            timing = retrieval_result.timing.to_dict()

        # --------------------------------------------------------------
        # Construct evaluation record
        # --------------------------------------------------------------

        return EvaluationRecord(
            question_id=question.question_id,
            question=question.question,
            question_type=question.question_type,
            difficulty=question.difficulty,
            answerability=question.answerability,
            paper_ids=list(question.paper_ids),
            gold_page_ids=list(question.gold_page_ids),
            supporting_page_ids=list(
                question.supporting_page_ids
            ),
            retrieved_page_ids=retrieved_page_ids,
            retrieved_paper_ids=retrieved_paper_ids,
            retrieved_scores=retrieved_scores,
            metrics=metrics,
            timing=timing,
            reference_answer=question.reference_answer,
            metadata=retrieval_result.metadata,
        )

    # ------------------------------------------------------------------
    # Dataset evaluation
    # ------------------------------------------------------------------

    def evaluate_dataset(
        self,
        questions: Sequence[EvaluationQuestion],
        top_k: int | None = None,
        show_progress: bool = True,
    ) -> list[EvaluationRecord]:
        """
        Evaluate a collection of questions.

        Parameters
        ----------
        questions:
            Validated evaluation questions.

        top_k:
            Number of pages to retrieve for each question.

        show_progress:
            Whether to display a tqdm progress bar.

        Returns
        -------
        list[EvaluationRecord]
            One evaluation record per question.
        """

        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(
                questions,
                desc="Evaluating retrieval",
                unit="question",
            )
        else:
            iterator = questions

        records: list[EvaluationRecord] = []

        for question in iterator:
            record = self.evaluate_question(
                question=question,
                top_k=top_k,
            )

            records.append(record)

        return records

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    @staticmethod
    def records_to_dict(
        records: Sequence[EvaluationRecord],
    ) -> list[dict[str, Any]]:
        """Convert evaluation records to dictionaries."""

        return [
            record.to_dict()
            for record in records
        ]