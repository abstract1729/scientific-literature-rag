from src.retrieval_evaluation import EvaluationDataset


QUESTIONS_PATH = "data/raw/retrieval/questions.jsonl"


def main():
    dataset = EvaluationDataset.from_jsonl(
        QUESTIONS_PATH,
        only_validated=True,
    )

    print("=" * 60)
    print("Retrieval Evaluation Dataset")
    print("=" * 60)

    print(f"Loaded questions : {len(dataset)}")

    print("\nSummary:")
    for key, value in dataset.summary().items():
        print(f"{key}: {value}")

    print("\nFirst question:")
    question = dataset[0]

    print(f"ID             : {question.question_id}")
    print(f"Type           : {question.question_type}")
    print(f"Difficulty     : {question.difficulty}")
    print(f"Answerability  : {question.answerability}")
    print(f"Gold pages     : {question.gold_page_ids}")
    print(f"Supporting     : {question.supporting_page_ids}")


if __name__ == "__main__":
    main()