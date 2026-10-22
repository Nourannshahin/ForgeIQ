from pathlib import Path
import json
from collections import defaultdict

from retriever import load_vector_store, search_knowledge_base


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_FILE = PROJECT_ROOT / "evaluation" / "evaluation_dataset.json"
REPORT_FILE = PROJECT_ROOT / "evaluation" / "retrieval_report.json"

TOP_K = 3
DIAGNOSTIC_K = 10


def load_evaluation_dataset():
    """Load and validate the evaluation questions."""

    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_FILE}"
        )

    with DATASET_FILE.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list) or not dataset:
        raise ValueError(
            "The evaluation dataset must be a non-empty list."
        )

    question_ids = [item.get("id") for item in dataset]

    if (
        any(not question_id for question_id in question_ids)
        or len(question_ids) != len(set(question_ids))
    ):
        raise ValueError(
            "Every evaluation question must have a unique, non-empty ID."
        )

    for item in dataset:
        if not item.get("question", "").strip():
            raise ValueError(
                f"Question {item.get('id')} has no question text."
            )

        if not isinstance(item.get("expected_sources"), list):
            raise ValueError(
                f"Question {item['id']} has invalid expected_sources."
            )

        if not item.get("category"):
            raise ValueError(
                f"Question {item['id']} has no category."
            )

    return dataset


def evaluate_question(item, model, collection):
    """Evaluate strict retrieval and diagnose expected-source rankings."""

    question = item["question"]
    expected_sources = item["expected_sources"]

    # Retrieve enough results to evaluate the original top-k metric
    # and inspect relevant passages ranked below the top three.
    retrieved = search_knowledge_base(
        question,
        model,
        collection,
        top_k=DIAGNOSTIC_K,
    )

    expected_pairs = {
        (source["source"], source["section"])
        for source in expected_sources
    }

    # Record the first rank at which each expected source-section pair
    # appears in the diagnostic results.
    retrieved_ranks = {}

    for rank, result in enumerate(retrieved, start=1):
        pair = (result["source"], result["section"])
        retrieved_ranks.setdefault(pair, rank)

    top_k_pairs = {
        (result["source"], result["section"])
        for result in retrieved[:TOP_K]
    }

    matched_top_k = expected_pairs & top_k_pairs

    # Preserve the original strict metric.
    if expected_pairs:
        hit = bool(matched_top_k)
        coverage = len(matched_top_k) / len(expected_pairs)
    else:
        # An empty expected-source list is not a retrieval success.
        # Missing-information handling must be evaluated separately.
        hit = None
        coverage = None

    expected_source_details = []

    for source, section in sorted(expected_pairs):
        rank = retrieved_ranks.get((source, section))

        expected_source_details.append({
            "source": source,
            "section": section,
            "rank_within_diagnostic_k": rank,
            "retrieved_at_k": (
                rank is not None and rank <= TOP_K
            ),
            "found_within_diagnostic_k": rank is not None,
        })

    retrieved_results = []

    for rank, result in enumerate(retrieved, start=1):
        retrieved_results.append({
            "rank": rank,
            "source": result["source"],
            "section": result["section"],
            "document_id": result["document_id"],
            "document_type": result["document_type"],
            "distance": round(float(result["distance"]), 4),
        })

    return {
        "id": item["id"],
        "category": item["category"],
        "question": question,
        "expected_sources": expected_source_details,
        "retrieved_results": retrieved_results,
        "hit_at_3": hit,
        "expected_source_coverage_at_3": coverage,
        "expected_behavior": item["expected_behavior"],
    }


def calculate_metrics(results):
    """Calculate overall and category-level strict retrieval metrics."""

    answerable_results = [
        result
        for result in results
        if result["hit_at_3"] is not None
    ]

    scored_count = len(answerable_results)

    hit_rate = (
        sum(result["hit_at_3"] for result in answerable_results)
        / scored_count
        if scored_count
        else None
    )

    coverage_values = [
        result["expected_source_coverage_at_3"]
        for result in answerable_results
    ]

    mean_source_coverage = (
        sum(coverage_values) / len(coverage_values)
        if coverage_values
        else None
    )

    category_results = defaultdict(list)

    for result in answerable_results:
        category_results[result["category"]].append(result)

    category_metrics = {}

    for category, category_items in sorted(category_results.items()):
        category_hits = [
            item["hit_at_3"]
            for item in category_items
        ]

        category_coverages = [
            item["expected_source_coverage_at_3"]
            for item in category_items
        ]

        category_metrics[category] = {
            "questions_scored": len(category_items),
            "hit_rate_at_3": round(
                sum(category_hits) / len(category_hits), 4
            ),
            "mean_expected_source_coverage_at_3": round(
                sum(category_coverages) / len(category_coverages), 4
            ),
        }

    unscored_results = [
        result
        for result in results
        if result["hit_at_3"] is None
    ]

    return {
        "top_k": TOP_K,
        "diagnostic_k": DIAGNOSTIC_K,
        "total_questions": len(results),
        "answerable_questions_evaluated": scored_count,
        "unscored_questions": len(unscored_results),
        "hit_rate_at_3": (
            round(hit_rate, 4) if hit_rate is not None else None
        ),
        "mean_expected_source_coverage_at_3": (
            round(mean_source_coverage, 4)
            if mean_source_coverage is not None
            else None
        ),
        "category_metrics": category_metrics,
        "unscored_question_ids": [
            result["id"] for result in unscored_results
        ],
    }


def main():
    dataset = load_evaluation_dataset()

    print(f"Loaded {len(dataset)} evaluation questions.")
    print("Loading existing ForgeIQ vector database...")

    model, collection = load_vector_store()

    print(f"Indexed chunks available: {collection.count()}")
    print(
        f"Scoring at k={TOP_K}; "
        f"diagnosing expected-source ranks through k={DIAGNOSTIC_K}.\n"
    )

    results = []

    for index, item in enumerate(dataset, start=1):
        result = evaluate_question(item, model, collection)
        results.append(result)

        if result["hit_at_3"] is None:
            status = "NOT SCORED — no expected sources"
        else:
            status = "PASS" if result["hit_at_3"] else "MISS"

        print(f'{index:02d}. {item["id"]} | {status}')
        print(f'    {item["question"]}')

        for source in result["expected_sources"]:
            rank = source["rank_within_diagnostic_k"]

            if rank is None:
                rank_text = f"NOT FOUND IN TOP {DIAGNOSTIC_K}"
            elif rank <= TOP_K:
                rank_text = f"FOUND AT RANK {rank}"
            else:
                rank_text = f"RANK {rank} — OUTSIDE TOP {TOP_K}"

            print(
                f'    Expected: {source["source"]} | '
                f'{source["section"]} | {rank_text}'
            )

    metrics = calculate_metrics(results)

    report = {
        "project": "ForgeIQ",
        "evaluation_type": "strict_retrieval",
        "evaluation_notes": [
            (
                "A hit means at least one expected source-section pair "
                f"appears in the top {TOP_K} results."
            ),
            (
                "Expected-source coverage measures the fraction of "
                "annotated source-section pairs found in the top "
                f"{TOP_K} results."
            ),
            (
                f"Ranks up to {DIAGNOSTIC_K} are recorded for diagnosis "
                "but do not affect the strict hit rate."
            ),
            (
                "Questions with no expected sources are unscored; "
                "this does not prove successful missing-information handling."
            ),
        ],
        "metrics": metrics,
        "results": results,
    }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("FORGEIQ RETRIEVAL EVALUATION SUMMARY")
    print("=" * 60)

    print(
        f'Questions scored: '
        f'{metrics["answerable_questions_evaluated"]}'
    )
    print(
        f'Questions not scored: '
        f'{metrics["unscored_questions"]}'
    )

    hit_rate = metrics["hit_rate_at_3"]
    coverage = metrics["mean_expected_source_coverage_at_3"]

    if hit_rate is not None:
        print(f"Strict hit rate@{TOP_K}: {hit_rate:.1%}")

    if coverage is not None:
        print(f"Mean expected-source coverage@{TOP_K}: {coverage:.1%}")

    print("\nCategory metrics:")

    for category, category_data in metrics["category_metrics"].items():
        print(
            f'  {category}: '
            f'{category_data["hit_rate_at_3"]:.1%} hit rate, '
            f'{category_data["mean_expected_source_coverage_at_3"]:.1%} '
            "source coverage "
            f'({category_data["questions_scored"]} scored)'
        )

    print(f"\nDetailed report saved to:\n{REPORT_FILE}")


if __name__ == "__main__":
    main()
