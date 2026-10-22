from pathlib import Path
import json
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DATA_DIR / "chunks.json"

CHUNK_SIZE = 180
CHUNK_OVERLAP = 30


def load_documents():
    """Load all raw text documents."""
    documents = []

    for file_path in sorted(RAW_DATA_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")

        documents.append({
            "source": file_path.name,
            "text": text
        })

    return documents


def clean_text(text):
    """Normalize whitespace without removing document structure."""
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_metadata(text, source):
    """Extract document metadata from its header."""

    metadata = {
        "document_id": "Unknown",
        "document_type": "general_document",
        "revision": "Unknown",
        "status": "Unspecified"
    }

    patterns = {
        "document_id": r"(?im)^Document ID:\s*(.+?)\s*$",
        "revision": r"(?im)^Revision:\s*(.+?)\s*$",
        "document_type": r"(?im)^Document Type:\s*(.+?)\s*$",
        "status": r"(?im)^Status:\s*(.+?)\s*$"
    }

    for field, pattern in patterns.items():
        match = re.search(pattern, text)

        if match:
            metadata[field] = match.group(1).strip()

    # Use a consistent machine-friendly document type.
    metadata["document_type"] = re.sub(
        r"[^a-z0-9]+",
        "_",
        metadata["document_type"].lower()
    ).strip("_")

    return metadata


def split_into_sections(text):
    """Split a document at Markdown level-two headings."""

    parts = re.split(r"(?m)^##\s+(.+?)\s*$", text)

    sections = []

    # Everything before the first section heading.
    introduction = parts[0].strip()

    if introduction:
        sections.append({
            "section": "Document Overview",
            "text": introduction
        })

    # Remaining parts alternate between heading and section body.
    for index in range(1, len(parts), 2):
        section_title = parts[index].strip()
        section_body = parts[index + 1].strip()

        if section_body:
            sections.append({
                "section": section_title,
                "text": f"{section_title}\n{section_body}"
            })

    return sections


def split_long_section(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):
    """Split long sections into overlapping word-based chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Overlap must be non-negative and smaller than chunk size."
        )

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))

        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break

        start = end - overlap

    return chunks


def build_chunks(documents):
    """Create chunks and attach document-level metadata."""

    all_chunks = []

    for document in documents:
        cleaned_text = clean_text(document["text"])

        metadata = extract_metadata(
            cleaned_text,
            document["source"]
        )

        sections = split_into_sections(cleaned_text)

        chunk_index = 0

        for section in sections:
            section_chunks = split_long_section(section["text"])

            for chunk_text in section_chunks:
                chunk = {
                    "chunk_id": (
                        f'{document["source"]}::chunk_{chunk_index:04d}'
                    ),
                    "source": document["source"],
                    **metadata,
                    "chunk_index": chunk_index,
                    "section": section["section"],
                    "text": chunk_text
                }

                all_chunks.append(chunk)
                chunk_index += 1

    return all_chunks


def save_chunks(chunks):
    """Save chunks as JSON."""

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=2, ensure_ascii=False)


def main():
    documents = load_documents()

    if not documents:
        raise FileNotFoundError(
            f"No .txt documents found in {RAW_DATA_DIR}"
        )

    chunks = build_chunks(documents)
    save_chunks(chunks)

    print("ForgeIQ metadata-aware ingestion completed!")
    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Output saved to: {OUTPUT_FILE}")

    print("\nDocument summary:\n")

    for document in documents:
        metadata = extract_metadata(
            clean_text(document["text"]),
            document["source"]
        )

        document_chunks = [
            chunk for chunk in chunks
            if chunk["source"] == document["source"]
        ]

        print(
            f'{document["source"]} | '
            f'ID: {metadata["document_id"]} | '
            f'Type: {metadata["document_type"]} | '
            f'Chunks: {len(document_chunks)}'
        )

    print("\nFirst chunk preview:\n")
    print(json.dumps(chunks[0], indent=2, ensure_ascii=False)[:1500])


if __name__ == "__main__":
    main()

