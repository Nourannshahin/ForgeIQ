from retriever import load_vector_store, search_knowledge_base


TEST_QUESTIONS = {
    "Q009": (
        "Can a technician loosen a pressurized hydraulic connection "
        "to investigate a leak?"
    ),
    "Q011": (
        "What information should be collected when an abnormal "
        "machine condition is reported?"
    ),
}


def main():
    print("Loading existing ForgeIQ vector database...")

    model, collection = load_vector_store()

    print(f"Available chunks: {collection.count()}")

    for question_id, question in TEST_QUESTIONS.items():
        print("\n" + "=" * 75)
        print(f"{question_id}: {question}")
        print("=" * 75)

        results = search_knowledge_base(
            query=question,
            model=model,
            collection=collection,
            top_k=10,
        )

        for rank, result in enumerate(results, start=1):
            print(f"\nRank: {rank}")
            print(f"Source: {result['source']}")
            print(f"Section: {result['section']}")
            print(f"Distance: {result['distance']:.4f}")
            print(f"Text: {result['text'][:700]}")


if __name__ == "__main__":
    main()

