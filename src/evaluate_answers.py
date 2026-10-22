import json
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from ask_forgeiq import (
    API_KEY,
    MODEL_NAME,
    generate_answer,
)
from retriever import load_vector_store, search_knowledge_base


# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

DATASET_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "answer_evaluation_dataset.json"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "answer_report.json"
)


# --------------------------------------------------
# 2. LOAD EVALUATION QUESTIONS
# --------------------------------------------------

def load_evaluation_dataset():
    """Load the answer-evaluation questions."""

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    # Support either a list of questions or an object
    # containing a "questions" list.
    if isinstance(dataset, dict):
        questions = dataset.get("questions")

        if not isinstance(questions, list):
            raise ValueError(
                "The evaluation dataset must contain a "
                "'questions' list."
            )

        return questions

    if isinstance(dataset, list):
        return dataset

    raise ValueError(
        "Unexpected evaluation dataset format."
    )


# --------------------------------------------------
# 3. EVALUATE ONE QUESTION
# --------------------------------------------------

def evaluate_question(
    item,
    question_index,
    model,
    collection,
    client,
):
    """Run retrieval, answer generation, and citation validation."""

    question_id = item.get(
        "id",
        item.get("question_id", f"A{question_index:03d}"),
    )

    question = item.get("question")

    if not question:
        raise ValueError(
            f"Question {question_id} has no question text."
        )

    print(f"Evaluating {question_id}: {question}")

    # Retrieve the same top three passages used by ForgeIQ.
    retrieved_chunks = search_knowledge_base(
        question,
        model,
        collection,
        top_k=3,
    )

    if not retrieved_chunks:
        return {
            "question_id": question_id,
            "question": question,
            "generation_status": "no_retrieved_passages",
            "answer": None,
            "retrieved_sources": [],
            "retrieved_passages": [],
            "citations": [],
            "invalid_citations": [],
            "citation_status": "no_retrieved_passages",
            "citation_report": {
                "total_citations": 0,
                "valid_citations": 0,
                "invalid_citation_count": 0,
                "all_valid": False,
            },
        }

    # Generate the structured answer.
    result = generate_answer(
        question,
        retrieved_chunks,
        client,
    )

    # Preserve the metadata for every retrieved passage.
    retrieved_sources = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        retrieved_sources.append({
            "source_id": f"S{index}",
            "source": chunk.get("source"),
            "document_id": chunk.get("document_id"),
            "document_type": chunk.get("document_type"),
            "revision": chunk.get("revision"),
            "section": chunk.get("section"),
            "chunk_id": chunk.get("chunk_id"),
        })

    # Store the passage text as well, so the report can be
    # inspected without rerunning retrieval.
    retrieved_passages = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        retrieved_passages.append({
            "source_id": f"S{index}",
            "source": chunk.get("source"),
            "section": chunk.get("section"),
            "text": chunk.get("text"),
        })

    citation_report = result["citation_report"]

    # Assign a clear citation status.
    if citation_report["total_citations"] == 0:
        citation_status = "no_detectable_citations"

    elif citation_report["invalid_citation_count"] > 0:
        citation_status = "invalid_citations"

    else:
        citation_status = "validated"

    return {
        "question_id": question_id,
        "question": question,
        "generation_status": "success",
        "answer": result["answer"],
        "insufficient_information": (
            result["insufficient_information"]
        ),
        "retrieved_sources": retrieved_sources,
        "retrieved_passages": retrieved_passages,
        "citations": result["citations"],
        "invalid_citations": result["invalid_citations"],
        "citation_status": citation_status,
        "citation_report": citation_report,
    }


# --------------------------------------------------
# 4. BUILD EVALUATION SUMMARY
# --------------------------------------------------

def build_summary(results):
    """Calculate generation and citation-traceability metrics."""

    total_questions = len(results)

    successful_results = [
        result
        for result in results
        if result["generation_status"] == "success"
    ]

    failed_results = [
        result
        for result in results
        if result["generation_status"] != "success"
    ]

    validated_questions = sum(
        1
        for result in successful_results
        if result["citation_status"] == "validated"
    )

    invalid_question_citations = sum(
        1
        for result in successful_results
        if result["citation_status"] == "invalid_citations"
    )

    questions_without_citations = sum(
        1
        for result in successful_results
        if result["citation_status"] == "no_detectable_citations"
    )

    total_citations = sum(
        result["citation_report"]["total_citations"]
        for result in successful_results
    )

    valid_citations = sum(
        result["citation_report"]["valid_citations"]
        for result in successful_results
    )

    invalid_citations = sum(
        result["citation_report"]["invalid_citation_count"]
        for result in successful_results
    )

    citation_validity_rate = (
        valid_citations / total_citations * 100
        if total_citations
        else 0.0
    )

    return {
        "total_questions": total_questions,
        "successful_generations": len(successful_results),
        "failed_generations": len(failed_results),
        "questions_with_all_citations_valid": validated_questions,
        "questions_with_invalid_citations": (
            invalid_question_citations
        ),
        "questions_without_detectable_citations": (
            questions_without_citations
        ),
        "total_extracted_citations": total_citations,
        "valid_extracted_citations": valid_citations,
        "invalid_extracted_citations": invalid_citations,
        "citation_validity_rate_percent": round(
            citation_validity_rate,
            2,
        ),
        "metric_definition": (
            "Citation validity measures whether a citation ID "
            "matches a passage retrieved for the same question. "
            "It does not measure answer correctness or prove "
            "that a passage entails the cited claim."
        ),
    }


# --------------------------------------------------
# 5. SAVE THE REPORT
# --------------------------------------------------

def save_report(report):
    """Save evaluation results as readable JSON."""

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )


# --------------------------------------------------
# 6. MAIN EVALUATION PIPELINE
# --------------------------------------------------

def main():
    if not API_KEY:
        raise ValueError(
            "GEMINI_API_KEY was not found. Check your .env file."
        )

    questions = load_evaluation_dataset()

    print(
        f"Loaded {len(questions)} answer-evaluation questions."
    )

    print(f"Using model: {MODEL_NAME}")

    print("\nLoading the existing vector database...")

    model, collection = load_vector_store()

    print(
        f"Loaded {collection.count()} indexed chunks."
    )

    client = genai.Client(api_key=API_KEY)

    results = []

    for index, item in enumerate(questions, start=1):
        try:
            result = evaluate_question(
                item,
                index,
                model,
                collection,
                client,
            )

        except Exception as error:
            question_id = item.get(
                "id",
                item.get("question_id", f"A{index:03d}"),
            )

            print(
                f"ERROR evaluating {question_id}: {error}"
            )

            result = {
                "question_id": question_id,
                "question": item.get("question"),
                "generation_status": "error",
                "error": str(error),
                "answer": None,
                "retrieved_sources": [],
                "retrieved_passages": [],
                "citations": [],
                "invalid_citations": [],
                "citation_status": "evaluation_error",
                "citation_report": {
                    "total_citations": 0,
                    "valid_citations": 0,
                    "invalid_citation_count": 0,
                    "all_valid": False,
                },
            }

        results.append(result)

        print(
            f"Status: {result['generation_status']} | "
            f"Citations: {result['citation_status']}\n"
        )

    summary = build_summary(results)

    report = {
        "project": "ForgeIQ",
        "model": MODEL_NAME,
        "evaluation_type": "structured_citation_traceability",
        "summary": summary,
        "results": results,
    }

    save_report(report)

    # Display summary.
    print("\n" + "=" * 55)
    print("FORGEIQ ANSWER EVALUATION")
    print("=" * 55)

    print(f"Questions checked: {summary['total_questions']}")
    print(
        "Successful generations: "
        f"{summary['successful_generations']}"
    )
    print(
        "Failed generations: "
        f"{summary['failed_generations']}"
    )

    print(
        "\nQuestions with all citations valid: "
        f"{summary['questions_with_all_citations_valid']}"
    )

    print(
        "Questions with invalid citations: "
        f"{summary['questions_with_invalid_citations']}"
    )

    print(
        "Questions without detectable citations: "
        f"{summary['questions_without_detectable_citations']}"
    )

    print(
        "\nTotal citations: "
        f"{summary['total_extracted_citations']}"
    )

    print(
        "Valid citations: "
        f"{summary['valid_extracted_citations']}"
    )

    print(
        "Invalid citations: "
        f"{summary['invalid_extracted_citations']}"
    )

    print(
        "Citation validity rate: "
        f"{summary['citation_validity_rate_percent']}%"
    )

    print("\nImportant:")
    print(summary["metric_definition"])

    print(f"\nFull report saved to:\n{REPORT_PATH}")


if __name__ == "__main__":
    main()