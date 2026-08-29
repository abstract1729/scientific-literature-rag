from __future__ import annotations

import math
from statistics import mean, median
from typing import Any, Sequence

from .evaluator import EvaluationRecord


class RetrievalAggregator:
    """
    Aggregate per-question retrieval evaluation results.

    Aggregation is performed at multiple levels:

        answerable
        unanswerable
        by question type
        by difficulty
        by answerability

    The aggregator does not perform retrieval and does not persist
    results. It only converts EvaluationRecord objects into summary
    statistics.
    """

    def __init__(
        self,
        records: Sequence[EvaluationRecord],
    ):
        self.records = list(records)

        if not self.records:
            raise ValueError(
                "Cannot aggregate an empty set of evaluation records."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def aggregate(self) -> dict[str, Any]:
        """
        Generate the complete aggregation report.

        Answerable and unanswerable questions are aggregated
        separately because they use different evaluation metrics.
        """

        answerable_records = [
            record
            for record in self.records
            if record.answerability == "answerable"
        ]

        unanswerable_records = [
            record
            for record in self.records
            if record.answerability == "unanswerable"
        ]

        return {
            "summary": self._dataset_summary(),

            # ----------------------------------------------------------
            # Answerable retrieval quality
            # ----------------------------------------------------------

            "answerable": self._aggregate_records(
                answerable_records
            ),

            # ----------------------------------------------------------
            # Unanswerable retrieval behavior
            # ----------------------------------------------------------

            "unanswerable": self._aggregate_records(
                unanswerable_records
            ),

            # ----------------------------------------------------------
            # Fine-grained breakdowns
            # ----------------------------------------------------------

            "by_question_type": self._group_and_aggregate(
                key=lambda record: record.question_type
            ),

            "by_difficulty": self._group_and_aggregate(
                key=lambda record: record.difficulty
            ),

            "by_answerability": self._group_and_aggregate(
                key=lambda record: record.answerability
            ),
        }
   
    # ------------------------------------------------------------------
    # Dataset summary
    # ------------------------------------------------------------------

    def _dataset_summary(self) -> dict[str, Any]:
        """Return basic information about the evaluated dataset."""

        question_types: dict[str, int] = {}
        difficulties: dict[str, int] = {}
        answerability: dict[str, int] = {}

        for record in self.records:

            question_types[record.question_type] = (
                question_types.get(record.question_type, 0) + 1
            )

            difficulties[record.difficulty] = (
                difficulties.get(record.difficulty, 0) + 1
            )

            answerability[record.answerability] = (
                answerability.get(record.answerability, 0) + 1
            )

        return {
            "total_questions": len(self.records),
            "question_types": question_types,
            "difficulties": difficulties,
            "answerability": answerability,
        }

    # ------------------------------------------------------------------
    # Grouping
    # ------------------------------------------------------------------

    def _group_and_aggregate(
        self,
        key,
    ) -> dict[str, Any]:
        """
        Group records using `key` and aggregate each group.
        """

        groups: dict[str, list[EvaluationRecord]] = {}

        for record in self.records:

            group_name = str(key(record))

            groups.setdefault(
                group_name,
                [],
            ).append(record)

        return {
            group_name: self._aggregate_records(group_records)
            for group_name, group_records in sorted(groups.items())
        }

    # ------------------------------------------------------------------
    # Core aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_records(
        records: Sequence[EvaluationRecord],
    ) -> dict[str, Any]:
        """
        Aggregate metrics and latency for a group of records.
        """

        if not records:
            return {
                "num_questions": 0,
            }

        result: dict[str, Any] = {
            "num_questions": len(records),
        }

        # --------------------------------------------------------------
        # Retrieval metrics
        # --------------------------------------------------------------

        metric_names = sorted(
            {
                metric_name
                for record in records
                for metric_name in record.metrics
            }
        )

        metrics: dict[str, Any] = {}

        for metric_name in metric_names:

            values = [
                float(record.metrics[metric_name])
                for record in records
                if _is_finite(record.metrics.get(metric_name))
            ]

            metrics[metric_name] = {
                "mean": _safe_mean(values),
                "median": _safe_median(values),
                "count": len(values),
            }

        result["metrics"] = metrics

        # --------------------------------------------------------------
        # Latency
        # --------------------------------------------------------------

        latency = RetrievalAggregator._aggregate_latency(
            records
        )

        if latency:
            result["latency_ms"] = latency

        return result

    # ------------------------------------------------------------------
    # Latency aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_latency(
        records: Sequence[EvaluationRecord],
    ) -> dict[str, Any] | None:
        """
        Aggregate retrieval latency.

        Supported timing fields:

            encoding_ms
            retrieval_ms
            total_ms

        Additional fields can be added later for E1/E2 without
        changing the aggregation architecture.
        """

        timing_fields = (
            "encoding_ms",
            "retrieval_ms",
            "total_ms",
        )

        result: dict[str, Any] = {}

        for field in timing_fields:

            values = []

            for record in records:

                if record.timing is None:
                    continue

                value = record.timing.get(field)

                if _is_finite(value):
                    values.append(float(value))

            if values:
                result[field] = {
                    "mean": _safe_mean(values),
                    "median": _safe_median(values),
                    "p95": _percentile(values, 95),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }

        return result or None


# ======================================================================
# Helper functions
# ======================================================================


def _is_finite(value: Any) -> bool:
    """
    Check whether a value is a valid finite number.
    """

    if value is None:
        return False

    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _safe_mean(
    values: Sequence[float],
) -> float | None:
    """Return mean or None when no values are available."""

    if not values:
        return None

    return mean(values)


def _safe_median(
    values: Sequence[float],
) -> float | None:
    """Return median or None when no values are available."""

    if not values:
        return None

    return median(values)


def _percentile(
    values: Sequence[float],
    percentile: float,
) -> float:
    """
    Calculate a percentile using linear interpolation.

    This avoids requiring NumPy just for aggregation.
    """

    if not values:
        raise ValueError(
            "Cannot calculate percentile of empty data."
        )

    if not 0 <= percentile <= 100:
        raise ValueError(
            "percentile must be between 0 and 100."
        )

    sorted_values = sorted(values)

    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (
        (len(sorted_values) - 1)
        * percentile
        / 100.0
    )

    lower = int(math.floor(position))
    upper = int(math.ceil(position))

    if lower == upper:
        return sorted_values[lower]

    weight = position - lower

    return (
        sorted_values[lower]
        * (1.0 - weight)
        + sorted_values[upper]
        * weight
    )