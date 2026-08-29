from .schemas import EvaluationQuestion
from .dataset import EvaluationDataset

from .results import (
    RetrievedPage,
    RetrievalResult,
    RetrievalTiming,
)

from .protocol import RetrieverProtocol

from .metrics import (
    recall_at_k,
    hit_at_k,
    reciprocal_rank,
    ndcg_at_k,
    gold_page_coverage,
    paper_coverage,
    evaluate_retrieval,
)

from .evaluator import (
    EvaluationRecord,
    RetrievalEvaluator,
)

from .aggregation import (
    RetrievalAggregator,
)

__all__ = [
    # Dataset
    "EvaluationQuestion",
    "EvaluationDataset",

    # Retrieval protocol/results
    "RetrieverProtocol",
    "RetrievedPage",
    "RetrievalResult",
    "RetrievalTiming",

    # Metrics
    "recall_at_k",
    "hit_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
    "gold_page_coverage",
    "paper_coverage",
    "evaluate_retrieval",

    # Evaluation
    "EvaluationRecord",
    "RetrievalEvaluator",

    # Aggregation
    "RetrievalAggregator",
]