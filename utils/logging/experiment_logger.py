from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from src.retrieval_evaluation.evaluator import EvaluationRecord


class ExperimentLogger:
    """
    Persistent logger for ColPali experiments.

    Each execution receives its own timestamped directory:

        results/
        └── colpali/
            └── retrieval/
                └── E0_vanilla/
                    └── YYYYMMDD_HHMMSS/
                        ├── run_metadata.json
                        ├── per_question.jsonl
                        └── metrics.json

    The logger is responsible only for persistence.
    Metric computation and evaluation logic remain outside this class.
    """

    def __init__(
        self,
        experiment_name: str,
        experiment_id: str,
        results_root: str | Path = "results/colpali/retrieval",
    ):
        self.experiment_name = experiment_name
        self.experiment_id = experiment_id

        self.results_root = Path(results_root)

        # Timestamp generated once when this logger instance is created.
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.run_dir = (
            self.results_root
            / experiment_name
            / self.run_id
        )

        self.run_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        self.metadata_path = self.run_dir / "run_metadata.json"
        self.per_question_path = self.run_dir / "per_question.jsonl"
        self.metrics_path = self.run_dir / "metrics.json"

    # ------------------------------------------------------------------
    # Run metadata
    # ------------------------------------------------------------------

    def log_run_metadata(
        self,
        *,
        evaluation_dataset: str,
        dataset_version: str | None = None,
        num_questions: int | None = None,
        top_k: int | None = None,
        additional_metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Persist metadata describing the evaluation run.

        Parameters
        ----------
        evaluation_dataset:
            Path or identifier of the evaluation dataset.

        dataset_version:
            Version/identifier of the manually validated evaluation set.

        num_questions:
            Number of questions evaluated.

        top_k:
            Retrieval cutoff used during evaluation.

        additional_metadata:
            Optional experiment-specific metadata.
        """

        metadata: dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),

            "evaluation_dataset": evaluation_dataset,
            "dataset_version": dataset_version,
            "num_questions": num_questions,
            "top_k": top_k,
        }

        if additional_metadata:
            metadata["additional"] = additional_metadata

        self._write_json(
            self.metadata_path,
            metadata,
        )

    # ------------------------------------------------------------------
    # Per-question results
    # ------------------------------------------------------------------

    def log_question(
        self,
        record: EvaluationRecord,
    ) -> None:
        """
        Append one EvaluationRecord to per_question.jsonl.

        Each line represents exactly one evaluation question.
        """

        with self.per_question_path.open(
            "a",
            encoding="utf-8",
        ) as file:

            json.dump(
                record.to_dict(),
                file,
                ensure_ascii=False,
                allow_nan=False,
            )

            file.write("\n")

    def log_questions(
        self,
        records: Sequence[EvaluationRecord],
    ) -> None:
        """
        Persist multiple evaluation records.

        The file is written in one operation rather than repeatedly
        opening and closing the file.
        """

        with self.per_question_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            for record in records:

                json.dump(
                    record.to_dict(),
                    file,
                    ensure_ascii=False,
                    allow_nan=False,
                )

                file.write("\n")

    # ------------------------------------------------------------------
    # Aggregated metrics
    # ------------------------------------------------------------------

    def log_metrics(
        self,
        metrics: dict[str, Any],
    ) -> None:
        """
        Persist aggregated evaluation metrics.
        """

        self._write_json(
            self.metrics_path,
            metrics,
        )

    # ------------------------------------------------------------------
    # Complete evaluation run
    # ------------------------------------------------------------------

    def save_evaluation(
        self,
        records: Sequence[EvaluationRecord],
        metrics: dict[str, Any],
    ) -> None:
        """
        Persist all evaluation results.

        This is the preferred method when evaluation has already
        completed successfully.
        """

        self.log_questions(records)
        self.log_metrics(metrics)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def output_directory(self) -> Path:
        """Return the directory containing this experiment run."""

        return self.run_dir

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _write_json(
        path: Path,
        data: dict[str, Any],
    ) -> None:
        """Write a dictionary as formatted JSON."""

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

            file.write("\n")