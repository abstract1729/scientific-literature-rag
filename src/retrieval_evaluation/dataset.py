import json
from pathlib import Path

from .schemas import EvaluationQuestion


class EvaluationDataset:
    """
    Loads and validates retrieval evaluation questions from JSONL.

    Each line in the JSONL file must contain one evaluation question.
    """

    REQUIRED_FIELDS = {"question_id","question","question_type","paper_ids","difficulty","gold_page_ids",
                       "supporting_page_ids","answerability","validation_status","reference_answer"}
    VALID_QUESTION_TYPES = {"direct_factual","formula","table","figure","conceptual","multi_hop","cross_paper","unanswerable"}
    VALID_DIFFICULTIES = {"easy","medium","hard"}
    VALID_ANSWERABILITY = {"answerable","unanswerable"}
    VALIDATION_STATUSES = {"pending","done"}

    def __init__(self, questions: list[EvaluationQuestion]):
        self.questions = questions

    @classmethod
    def from_jsonl(cls,path: str | Path,only_validated: bool = True) -> "EvaluationDataset":
        """
        Load evaluation questions from a JSONL file.

        Parameters
        ----------
        path:
            Path to the JSONL evaluation file.

        only_validated:
            If True, only questions with validation_status == "done"
            are loaded.

        Returns
        -------
        EvaluationDataset
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found: {path}")

        if not path.is_file():
            raise ValueError(f"Evaluation dataset path is not a file: {path}")

        questions = []
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):

                line = line.strip()

                # Allow blank lines in the JSONL file.
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at line {line_number} "
                        f"in {path}: {exc}"
                    ) from exc

                cls._validate_record(record, line_number)

                question = EvaluationQuestion(
                    question_id=record["question_id"],
                    question=record["question"],
                    question_type=record["question_type"],
                    paper_ids=record["paper_ids"],
                    difficulty=record["difficulty"],
                    gold_page_ids=record["gold_page_ids"],
                    supporting_page_ids=record["supporting_page_ids"],
                    answerability=record["answerability"],
                    validation_status=record["validation_status"],
                    reference_answer=record.get("reference_answer"),
                )

                if only_validated and not question.is_validated:
                    continue

                questions.append(question)

        return cls(questions)

    @staticmethod
    def _validate_record(record: dict,line_number: int,) -> None:
        """Validate the structure and controlled values of one record."""

        if not isinstance(record, dict):
            raise ValueError(f"Line {line_number}: expected a JSON object.")
        missing = EvaluationDataset.REQUIRED_FIELDS - record.keys()

        if missing:
            raise ValueError(
                f"Line {line_number}: missing required fields: "
                f"{sorted(missing)}"
            )

        if not isinstance(record["question_id"], str):
            raise ValueError(
                f"Line {line_number}: question_id must be a string."
            )

        if not isinstance(record["question"], str):
            raise ValueError(
                f"Line {line_number}: question must be a string."
            )

        if record["question_type"] not in EvaluationDataset.VALID_QUESTION_TYPES:
            raise ValueError(
                f"Line {line_number}: invalid question_type "
                f"'{record['question_type']}'."
            )

        if record["difficulty"] not in EvaluationDataset.VALID_DIFFICULTIES:
            raise ValueError(
                f"Line {line_number}: invalid difficulty "
                f"'{record['difficulty']}'."
            )

        if record["answerability"] not in EvaluationDataset.VALID_ANSWERABILITY:
            raise ValueError(
                f"Line {line_number}: invalid answerability "
                f"'{record['answerability']}'."
            )

        if record["validation_status"] not in EvaluationDataset.VALIDATION_STATUSES:
            raise ValueError(
                f"Line {line_number}: invalid validation_status "
                f"'{record['validation_status']}'."
            )

        for field in (
            "paper_ids",
            "gold_page_ids",
            "supporting_page_ids",
        ):
            if not isinstance(record[field], list):
                raise ValueError(
                    f"Line {line_number}: {field} must be a list."
                )

            if not all(isinstance(value, str) for value in record[field]):
                raise ValueError(
                    f"Line {line_number}: all values in {field} "
                    f"must be strings."
                )

        if record["answerability"] == "answerable":
            if not record["gold_page_ids"]:
                raise ValueError(
                    f"Line {line_number}: answerable questions must "
                    f"contain at least one gold_page_id."
                )

        if record["answerability"] == "unanswerable":
            if record["gold_page_ids"]:
                raise ValueError(
                    f"Line {line_number}: unanswerable questions should "
                    f"not contain gold_page_ids."
                )

        reference_answer = record.get("reference_answer")

        if reference_answer is not None and not isinstance(
            reference_answer, str
        ):
            raise ValueError(
                f"Line {line_number}: reference_answer must be "
                f"a string or null."
            )

    def __len__(self) -> int:
        return len(self.questions)

    def __iter__(self):
        return iter(self.questions)

    def __getitem__(self, index: int) -> EvaluationQuestion:
        return self.questions[index]

    def summary(self) -> dict:
        """Return basic statistics about the loaded dataset."""

        question_types = {}

        for question in self.questions:
            question_types[question.question_type] = (
                question_types.get(question.question_type, 0) + 1
            )

        answerable = sum(
            question.is_answerable
            for question in self.questions
        )

        return {
            "total_questions": len(self.questions),
            "answerable": answerable,
            "unanswerable": len(self.questions) - answerable,
            "question_types": question_types,
        }