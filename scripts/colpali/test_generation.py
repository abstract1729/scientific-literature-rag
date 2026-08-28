from src.colpali.ingestion import ColPaliEncoder
from src.colpali.retrieval import (ColPaliQdrantRetriever)
from src.colpali.generation import (GeminiGenerator)

COLLECTION_NAME = "colpali_pages"
QDRANT_PATH = ("data/processed/colpali/qdrant")


def main():

    print("=" * 70)
    print("ColPali RAG - Generation Smoke Test")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Load ColPali
    # ---------------------------------------------------------------

    print("\nLoading ColPali...")

    encoder = ColPaliEncoder(
        model_name="vidore/colpali-v1.3",
        device="mps",
    )

    print("ColPali loaded.")

    # ---------------------------------------------------------------
    # Retriever
    # ---------------------------------------------------------------

    retriever = ColPaliQdrantRetriever(
        encoder=encoder,
        collection_name=COLLECTION_NAME,
        qdrant_path=QDRANT_PATH,
    )

    # ---------------------------------------------------------------
    # Generator
    # ---------------------------------------------------------------

    generator = GeminiGenerator(
        model_name="gemini-3.5-flash"
    )

    # ---------------------------------------------------------------
    # Query
    # ---------------------------------------------------------------

    query = (
        "Explain LoRA: Low-Rank Adaptation of "
        "Large Language Models in simple words."
    )

    print(f"\nQuery:\n{query}")

    # ---------------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------------

    print("\nRetrieving pages...")

    retrieved_results = retriever.retrieve(
        query=query,
        top_k=5,
    )

    print(
        f"Retrieved {len(retrieved_results)} pages."
    )

    for rank, result in enumerate(
        retrieved_results,
        start=1,
    ):
        payload = result["payload"]

        print(
            f"  {rank}. "
            f"{payload.get('paper_id')} "
            f"page {payload.get('page_number')} "
            f"(score={result['score']:.4f})"
        )

    # ---------------------------------------------------------------
    # Generation
    # ---------------------------------------------------------------

    print("\nGenerating answer...")

    result = generator.generate(
        query=query,
        retrieved_results=retrieved_results,
    )

    # ---------------------------------------------------------------
    # Display
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(result["answer"])

    print("\n" + "=" * 70)
    print("EVIDENCE")
    print("=" * 70)

    for evidence in result["evidence"]:

        print(
            f"[{evidence['rank']}] "
            f"{evidence['paper_id']} - "
            f"page {evidence['page_number']} "
            f"(score={evidence['score']:.4f})"
        )


if __name__ == "__main__":
    main()