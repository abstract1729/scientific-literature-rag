from dataclasses import dataclass
from typing import Literal


QuestionType = Literal["direct_factual","formula","table","figure","conceptual","multi_hop","cross_paper","unanswerable"]
Difficulty = Literal["easy","medium","hard"]
Answerability = Literal["answerable","unanswerable"]
ValidationStatus = Literal["pending","done"]


@dataclass(frozen=True)
class EvaluationQuestion:
    """
    Represents a single manually curated retrieval-evaluation question.

    The gold_page_ids define the pages that contain the required evidence.
    supporting_page_ids contain useful contextual pages that are not
    necessarily required to answer the question.
    """

    question_id: str
    question: str
    question_type: QuestionType
    paper_ids: list[str]
    difficulty: Difficulty
    gold_page_ids: list[str]
    supporting_page_ids: list[str]
    answerability: Answerability
    validation_status: ValidationStatus
    reference_answer: str | None = None

    @property
    def is_validated(self) -> bool:
        """Return whether the question has been manually validated."""
        return self.validation_status == "done"

    @property
    def is_answerable(self) -> bool:
        """Return whether the question has ground-truth evidence."""
        return self.answerability == "answerable"