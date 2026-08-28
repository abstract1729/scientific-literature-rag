from src.colpali.ingestion import ColPaliEncoder
from src.colpali.retrieval import (ColPaliQdrantRetriever)


COLLECTION_NAME = "colpali_pages"
QDRANT_PATH = ("data/processed/colpali/qdrant")


def main():

    print("=" * 70)
    print("ColPali Retrieval Smoke Test")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Load ColPali
    # ---------------------------------------------------------------

    print("\nLoading ColPali...")

    encoder = ColPaliEncoder(model_name="vidore/colpali-v1.3",device="mps",)
    print("ColPali loaded.")

    # ---------------------------------------------------------------
    # Initialize retriever
    # ---------------------------------------------------------------
    retriever = ColPaliQdrantRetriever(encoder=encoder,collection_name=COLLECTION_NAME,qdrant_path=QDRANT_PATH)

    # ---------------------------------------------------------------
    # Query
    # ---------------------------------------------------------------

    query = ("Explain LoRA: Low-Rank Adaptation of Large Language Models in simple words")
    top_k = 5
    print(f"\nQuery: {query}")
    print(f"Retrieving top-{top_k} pages...\n")
    results = retriever.retrieve(query=query,top_k=top_k)

    # ---------------------------------------------------------------
    # Display results
    # ---------------------------------------------------------------

    for rank, result in enumerate(results,start=1,):
        payload = result["payload"]
        print(f"[Rank {rank}]")
        print(f"Score      : {result['score']:.4f}")
        print(
            f"Paper ID   : "
            f"{payload.get('paper_id')}"
        )
        print(
            f"Page       : "
            f"{payload.get('page_number')}"
        )
        print(
            f"Page ID    : "
            f"{payload.get('page_id')}"
        )
        print(
            f"Image      : "
            f"{payload.get('image_path')}"
        )
        print("-" * 70)


if __name__ == "__main__":
    main()