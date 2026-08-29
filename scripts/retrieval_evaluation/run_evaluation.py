from __future__ import annotations

from pathlib import Path

from src.colpali.ingestion.colpali_encoder import ColPaliEncoder
from src.colpali.retrieval.qdrant_retriever import ColPaliQdrantRetriever
from src.retrieval_evaluation import (
    EvaluationDataset,
    RetrievalAggregator,
    RetrievalEvaluator,
)
from utils.logging import ExperimentLogger


# ======================================================================
# Configuration
# ======================================================================

QUESTION_PATH = Path("data/raw/retrieval/questions.jsonl")
EVAL_DATASET_VERSION = "v0.1"
EXPERIMENT_ID = "E0"
EXPERIMENT_NAME = "E0_vanilla"
TOP_K = 10
COLLECTION_NAME = "colpali_pages"
QDRANT_PATH = ("data/processed/colpali/qdrant")
MODEL_NAME = "vidore/colpali-v1.3"
DEVICE = "mps"

METRICS_K = (1, 3, 5, 10)


# ======================================================================
# Main
# ======================================================================

def main() -> None:

    print("=" * 70)
    print("ColPali Retrieval Evaluation - E0 Vanilla")
    print("=" * 70)

    retriever = None

    try:

        # --------------------------------------------------------------
        # 1. Load evaluation dataset
        # --------------------------------------------------------------

        print("\nLoading evaluation dataset...")

        dataset = EvaluationDataset.from_jsonl(
            QUESTION_PATH,
            only_validated=True,
        )

        print(
            f"Questions loaded : {len(dataset)}"
        )

        # --------------------------------------------------------------
        # 2. Load ColPali
        # --------------------------------------------------------------

        print("\nLoading ColPali...")

        encoder = ColPaliEncoder(
            model_name=MODEL_NAME,
            device=DEVICE,
        )

        print(
            f"ColPali loaded on {encoder.device}."
        )

        # --------------------------------------------------------------
        # 3. Initialize retriever
        # --------------------------------------------------------------

        print("\nInitializing Qdrant retriever...")

        retriever = ColPaliQdrantRetriever(
            encoder=encoder,
            collection_name=COLLECTION_NAME,
            qdrant_path=QDRANT_PATH,
        )

        print("Retriever initialized.")

        # --------------------------------------------------------------
        # 4. Initialize evaluator
        # --------------------------------------------------------------

        evaluator = RetrievalEvaluator(
            retriever=retriever,
            ks=METRICS_K,
            use_timing=True,
        )

        # --------------------------------------------------------------
        # 5. Run retrieval evaluation
        # --------------------------------------------------------------

        print("\nRunning retrieval evaluation...")

        records = evaluator.evaluate_dataset(
            questions=dataset.questions,
            top_k=TOP_K,
            show_progress=True,
        )

        print(
            f"\nEvaluation complete: {len(records)} questions."
        )

        # --------------------------------------------------------------
        # 6. Aggregate metrics
        # --------------------------------------------------------------

        print("\nAggregating metrics...")

        aggregator = RetrievalAggregator(
            records
        )

        aggregated_metrics = aggregator.aggregate()

        # --------------------------------------------------------------
        # 7. Persist results
        # --------------------------------------------------------------

        logger = ExperimentLogger(
            experiment_name=EXPERIMENT_NAME,
            experiment_id=EXPERIMENT_ID,
        )

        logger.log_run_metadata(
            evaluation_dataset=str(
                QUESTION_PATH
            ),
            dataset_version=EVAL_DATASET_VERSION,
            num_questions=len(records),
            top_k=TOP_K,
            additional_metadata={
                "model": MODEL_NAME,
                "device": DEVICE,
                "qdrant_collection": COLLECTION_NAME,
                "qdrant_path": QDRANT_PATH,
                "metrics_k": list(METRICS_K),
            },
        )

        logger.save_evaluation(
            records=records,
            metrics=aggregated_metrics,
        )

        # --------------------------------------------------------------
        # 8. Display summary
        # --------------------------------------------------------------

        print("\n" + "=" * 70)
        print("E0 RETRIEVAL EVALUATION COMPLETE")
        print("=" * 70)

        print(
            f"Questions evaluated : {len(records)}"
        )

        # ==============================================================
        # Answerable questions
        # ==============================================================

        answerable = aggregated_metrics.get(
            "answerable",
            {}
        )

        print("\nAnswerable Questions:")

        print(
            f"Count                : "
            f"{answerable.get('num_questions', 0)}"
        )

        answerable_metrics = answerable.get(
            "metrics",
            {}
        )

        for metric_name in (
            "recall_at_1",
            "recall_at_3",
            "recall_at_5",
            "recall_at_10",
            "hit_at_5",
            "ndcg_at_5",
            "reciprocal_rank",
            "gold_page_coverage",
            "paper_coverage",
        ):

            if metric_name not in answerable_metrics:
                continue

            value = answerable_metrics[
                metric_name
            ]["mean"]

            if value is not None:
                print(
                    f"{metric_name:<20}: "
                    f"{value:.4f}"
                )

        # ==============================================================
        # Unanswerable questions
        # ==============================================================

        unanswerable = aggregated_metrics.get(
            "unanswerable",
            {}
        )

        print("\nUnanswerable Questions:")

        print(
            f"Count                : "
            f"{unanswerable.get('num_questions', 0)}"
        )

        unanswerable_metrics = unanswerable.get(
            "metrics",
            {}
        )

        for metric_name in (
            "maximum_retrieval_score",
            "mean_retrieval_score",
        ):

            if metric_name not in unanswerable_metrics:
                continue

            value = unanswerable_metrics[
                metric_name
            ]["mean"]

            if value is not None:
                print(
                    f"{metric_name:<25}: "
                    f"{value:.4f}"
                )

        # ==============================================================
        # Latency
        # ==============================================================

        # Latency is reported from the complete evaluated set.
        #
        # This intentionally includes both answerable and
        # unanswerable queries because retrieval latency is an
        # operational property of the retriever, not an answerability
        # metric.

        all_latency = aggregator._aggregate_latency(
            records
        )

        if all_latency:

            print("\nLatency:")

            for field in (
                "encoding_ms",
                "retrieval_ms",
                "total_ms",
            ):

                if field not in all_latency:
                    continue

                values = all_latency[field]

                print(
                    f"{field:<20}: "
                    f"mean={values['mean']:.2f} ms | "
                    f"median={values['median']:.2f} ms | "
                    f"p95={values['p95']:.2f} ms"
                )

        # --------------------------------------------------------------
        # Fine-grained aggregation information
        # --------------------------------------------------------------

        print("\nBreakdowns available:")
        print(
            "  - by_question_type"
        )
        print(
            "  - by_difficulty"
        )
        print(
            "  - by_answerability"
        )

        # --------------------------------------------------------------
        # Results location
        # --------------------------------------------------------------

        print(
            f"\nResults saved to:\n"
            f"{logger.output_directory}"
        )

    finally:

        # --------------------------------------------------------------
        # Cleanup
        # --------------------------------------------------------------

        if retriever is not None:
            retriever.close()


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    main()