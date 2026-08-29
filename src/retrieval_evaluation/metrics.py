from __future__ import annotations

import math
from typing import Sequence


def _validate_k(k: int) -> None:
    """Validate a retrieval cutoff."""
    if k <= 0:
        raise ValueError("k must be greater than 0.")


def _validate_page_ids(page_ids: Sequence[str]) -> None:
    """Validate a sequence of page IDs."""
    if not all(isinstance(page_id, str) for page_id in page_ids):
        raise TypeError("All page IDs must be strings.")


def recall_at_k(
    retrieved_page_ids: Sequence[str],
    gold_page_ids: Sequence[str],
    k: int,
) -> float:
    """
    Compute page-level Recall@K.

    Recall@K is defined here as:

        number of gold pages retrieved in top-K
        -------------------------------------
                    total gold pages

    This is particularly useful when a question has multiple
    required evidence pages, such as multi-hop questions.

    Parameters
    ----------
    retrieved_page_ids:
        Page IDs returned by the retriever in ranking order.

    gold_page_ids:
        Pages containing required evidence.

    k:
        Retrieval cutoff.

    Returns
    -------
    float
        Recall@K in [0, 1].

    Notes
    -----
    For an unanswerable question, there are no gold pages.
    Therefore Recall@K is undefined and returns NaN.
    """

    _validate_k(k)
    _validate_page_ids(retrieved_page_ids)
    _validate_page_ids(gold_page_ids)

    if not gold_page_ids:
        return math.nan

    gold = set(gold_page_ids)
    retrieved = set(retrieved_page_ids[:k])

    return len(gold & retrieved) / len(gold)


def hit_at_k(
    retrieved_page_ids: Sequence[str],
    gold_page_ids: Sequence[str],
    k: int,
) -> float:
    """
    Compute Hit@K.

    Returns 1.0 if at least one gold page appears in the
    top-K retrieved pages, otherwise 0.0.

    For unanswerable questions, returns NaN.
    """

    _validate_k(k)
    _validate_page_ids(retrieved_page_ids)
    _validate_page_ids(gold_page_ids)

    if not gold_page_ids:
        return math.nan

    gold = set(gold_page_ids)

    return float(
        any(page_id in gold for page_id in retrieved_page_ids[:k])
    )


def reciprocal_rank(
    retrieved_page_ids: Sequence[str],
    gold_page_ids: Sequence[str],
) -> float:
    """
    Compute Mean Reciprocal Rank contribution for one query.

    The reciprocal rank is the reciprocal of the rank of the
    first relevant page.

    Example
    -------
    If the first gold page occurs at rank 3:

        RR = 1 / 3

    If no gold page is retrieved:

        RR = 0

    For unanswerable questions, returns NaN.
    """

    _validate_page_ids(retrieved_page_ids)
    _validate_page_ids(gold_page_ids)

    if not gold_page_ids:
        return math.nan

    gold = set(gold_page_ids)

    for rank, page_id in enumerate(retrieved_page_ids, start=1):
        if page_id in gold:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    retrieved_page_ids: Sequence[str],
    gold_page_ids: Sequence[str],
    k: int,
) -> float:
    """
    Compute binary-relevance nDCG@K.

    A retrieved page receives relevance:

        1 -> page is a gold evidence page
        0 -> otherwise

    This metric is useful when multiple gold pages exist because
    it rewards retrieving relevant evidence earlier in the ranking.

    For unanswerable questions, returns NaN.
    """

    _validate_k(k)
    _validate_page_ids(retrieved_page_ids)
    _validate_page_ids(gold_page_ids)

    if not gold_page_ids:
        return math.nan

    gold = set(gold_page_ids)

    retrieved = retrieved_page_ids[:k]

    # DCG
    dcg = 0.0

    for rank, page_id in enumerate(retrieved, start=1):
        relevance = 1.0 if page_id in gold else 0.0

        dcg += relevance / math.log2(rank + 1)

    # Ideal DCG
    ideal_relevant = min(len(gold), k)

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_relevant + 1)
    )

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def gold_page_coverage(
    retrieved_page_ids: Sequence[str],
    gold_page_ids: Sequence[str],
) -> float:
    """
    Compute the fraction of required gold pages retrieved.

    This is mathematically equivalent to Recall over the complete
    retrieved ranking, but is given a separate semantic name because
    it is particularly useful for multi-hop questions.

    Example
    -------
    Gold pages:
        [A, B, C]

    Retrieved:
        [A, X, B, Y]

    Coverage:
        2 / 3 = 0.667
    """

    _validate_page_ids(retrieved_page_ids)
    _validate_page_ids(gold_page_ids)

    if not gold_page_ids:
        return math.nan

    gold = set(gold_page_ids)
    retrieved = set(retrieved_page_ids)

    return len(gold & retrieved) / len(gold)


def paper_coverage(
    retrieved_paper_ids: Sequence[str],
    gold_paper_ids: Sequence[str],
) -> float:
    """
    Compute the fraction of required papers represented in retrieval.

    This is intended primarily for cross-paper questions.

    Example
    -------
    Required papers:
        [PaperA, PaperB]

    Retrieved papers:
        [PaperA, PaperC]

    Coverage:
        1 / 2 = 0.5

    For questions with no required papers, returns NaN.
    """

    _validate_page_ids(retrieved_paper_ids)
    _validate_page_ids(gold_paper_ids)

    if not gold_paper_ids:
        return math.nan

    gold = set(gold_paper_ids)
    retrieved = set(retrieved_paper_ids)

    return len(gold & retrieved) / len(gold)


def maximum_retrieval_score(
    scores: Sequence[float],
) -> float | None:
    """
    Return the maximum retrieval score.

    Useful for analyzing unanswerable questions and calibrating
    score-based abstention thresholds.

    Returns None when no scores are available.
    """

    if not scores:
        return None

    return max(float(score) for score in scores)

def mean_retrieval_score(
    scores: Sequence[float],
) -> float | None:
    """
    Return the mean retrieval score across retrieved pages.

    This is primarily useful for analyzing unanswerable
    questions and understanding the score distribution of
    retrieved candidates.

    Returns None when no scores are available.
    """

    if not scores:
        return None

    return sum(float(score) for score in scores) / len(scores)


def evaluate_unanswerable(
    retrieved_scores: Sequence[float],
) -> dict[str, float]:
    """
    Evaluate retrieval behavior for an unanswerable question.

    Unanswerable questions do not have gold evidence pages,
    so Recall@K, Hit@K, MRR and nDCG are not meaningful.

    Instead, we record retrieval-score statistics that can
    later be used to study score-based abstention thresholds.

    Returns
    -------
    dict
        Unanswerable-specific retrieval metrics.
    """

    metrics: dict[str, float] = {}

    max_score = maximum_retrieval_score(
        retrieved_scores
    )

    mean_score = mean_retrieval_score(
        retrieved_scores
    )

    metrics["maximum_retrieval_score"] = (
        float(max_score)
        if max_score is not None
        else math.nan
    )

    metrics["mean_retrieval_score"] = (
        float(mean_score)
        if mean_score is not None
        else math.nan
    )

    return metrics

def evaluate_retrieval(
    retrieved_page_ids: Sequence[str],
    gold_page_ids: Sequence[str],
    retrieved_paper_ids: Sequence[str] | None = None,
    gold_paper_ids: Sequence[str] | None = None,
    ks: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    """
    Compute the complete set of retrieval metrics for one question.

    Parameters
    ----------
    retrieved_page_ids:
        Retrieved page IDs in ranking order.

    gold_page_ids:
        Required evidence page IDs.

    retrieved_paper_ids:
        Paper IDs corresponding to retrieved pages.
        Optional.

    gold_paper_ids:
        Required paper IDs.
        Optional.

    ks:
        Retrieval cutoffs for Recall@K, Hit@K and nDCG@K.

    Returns
    -------
    dict
        Metric name -> metric value.
    """

    _validate_page_ids(retrieved_page_ids)
    _validate_page_ids(gold_page_ids)

    if not ks:
        raise ValueError("ks must contain at least one value.")

    metrics: dict[str, float] = {}

    # -------------------------------------------------------------
    # Standard page-level metrics
    # -------------------------------------------------------------

    for k in ks:
        _validate_k(k)

        metrics[f"recall_at_{k}"] = recall_at_k(
            retrieved_page_ids=retrieved_page_ids,
            gold_page_ids=gold_page_ids,
            k=k,
        )

        metrics[f"hit_at_{k}"] = hit_at_k(
            retrieved_page_ids=retrieved_page_ids,
            gold_page_ids=gold_page_ids,
            k=k,
        )

        metrics[f"ndcg_at_{k}"] = ndcg_at_k(
            retrieved_page_ids=retrieved_page_ids,
            gold_page_ids=gold_page_ids,
            k=k,
        )

    # -------------------------------------------------------------
    # Ranking metric
    # -------------------------------------------------------------

    metrics["reciprocal_rank"] = reciprocal_rank(
        retrieved_page_ids=retrieved_page_ids,
        gold_page_ids=gold_page_ids,
    )

    # -------------------------------------------------------------
    # Complete evidence coverage
    # -------------------------------------------------------------

    metrics["gold_page_coverage"] = gold_page_coverage(
        retrieved_page_ids=retrieved_page_ids,
        gold_page_ids=gold_page_ids,
    )

    # -------------------------------------------------------------
    # Cross-paper coverage
    # -------------------------------------------------------------

    if retrieved_paper_ids is not None and gold_paper_ids is not None:
        metrics["paper_coverage"] = paper_coverage(
            retrieved_paper_ids=retrieved_paper_ids,
            gold_paper_ids=gold_paper_ids,
        )

    return metrics