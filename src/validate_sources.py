import re


def normalize(value):
    """Normalize strings for consistent source and section comparison."""
    text = " ".join(str(value).strip().lower().split())

    # Standardize punctuation immediately after section numbers.
    text = re.sub(r"^(\d+(?:\.\d+)*)\s*[:.]\s*", r"\1 ", text)

    # Remove trailing punctuation.
    text = text.rstrip(" .:;")

    return text


def extract_citations(answer):
    """
    Extract source filenames and nearby section references
    from a generated ForgeIQ answer.

    This is a simple extractor and may not recognize every
    possible citation format.
    """
    citations = []

    filename_pattern = re.compile(
        r"\b([a-zA-Z0-9_-]+\.txt)\b",
        re.IGNORECASE,
    )

    filenames = list(filename_pattern.finditer(answer))

    for index, match in enumerate(filenames):
        source = match.group(1)

        # Find the text between this filename and the next filename.
        next_start = (
            filenames[index + 1].start()
            if index + 1 < len(filenames)
            else len(answer)
        )

        nearby_text = answer[match.end():next_start][:250]

        # Look for a section reference following the filename.
        section_match = re.search(
            r"(?:section|sec\.?)\s*[:#]?\s*"
            r"([0-9]+(?:\.[0-9]+)*\.?\s*[^.\n;]+)",
            nearby_text,
            re.IGNORECASE,
        )

        if section_match:
            citations.append(
                {
                    "source": source,
                    "section": section_match.group(1).strip(),
                }
            )

    return citations


def validate_source_citations(cited_sources, retrieved_chunks):
    """
    Validate extracted citations against retrieved passages.

    A citation is valid when its source filename and section
    match a source-section pair in the retrieved passages.

    This checks citation traceability, not factual support.
    """

    retrieved_pairs = {
        (
            normalize(chunk.get("source", "")),
            normalize(chunk.get("section", "")),
        )
        for chunk in retrieved_chunks
    }

    results = []

    for citation in cited_sources:
        source = citation.get("source", "")
        section = citation.get("section", "")

        source_norm = normalize(source)
        section_norm = normalize(section)

        valid = (source_norm, section_norm) in retrieved_pairs

        results.append(
            {
                "source": source,
                "section": section,
                "valid": valid,
                "reason": (
                    "Citation matches a retrieved source and section."
                    if valid
                    else (
                        "Citation does not match any retrieved "
                        "source-section pair."
                    )
                ),
            }
        )

    valid_count = sum(
        1 for result in results if result["valid"]
    )

    total_count = len(results)

    return {
        "total_citations": total_count,
        "valid_citations": valid_count,
        "invalid_citations": total_count - valid_count,
        "all_valid": bool(results) and valid_count == total_count,
        "citations": results,
    }


if __name__ == "__main__":
    # Example test data to verify the validator independently.

    retrieved_chunks = [
        {
            "source": "maintenance_manual.txt",
            "section": "4. Hydraulic System Monitoring",
        },
        {
            "source": "inspection_checklist.txt",
            "section": "4. Hydraulic System Observations",
        },
    ]

    answer = (
        "Monitor fluid condition and leakage. "
        "Source: maintenance_manual.txt, "
        "Section 4: Hydraulic System Monitoring"
    )

    extracted = extract_citations(answer)

    report = validate_source_citations(
        extracted,
        retrieved_chunks,
    )

    print("FORGEIQ SOURCE VALIDATION TEST")
    print("=" * 40)

    print("\nExtracted citations:")
    for citation in extracted:
        print(
            f"- {citation['source']} | "
            f"{citation['section']}"
        )

    print("\nValidation results:")
    print(f"Total citations: {report['total_citations']}")
    print(f"Valid citations: {report['valid_citations']}")
    print(f"Invalid citations: {report['invalid_citations']}")
    print(f"All citations valid: {report['all_valid']}")

    for citation in report["citations"]:
        print(f"\nSource: {citation['source']}")
        print(f"Section: {citation['section']}")
        print(f"Valid: {citation['valid']}")
        print(f"Reason: {citation['reason']}")