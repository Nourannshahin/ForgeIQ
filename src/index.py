from retriever import load_chunks, create_vector_store


def main():
    print("Loading processed documents...")

    chunks = load_chunks()

    print(f"Loaded {len(chunks)} chunks.")

    print("\nBuilding ForgeIQ vector database...")
    print("This may take a moment while embeddings are generated.\n")

    model, collection = create_vector_store(chunks)

    print("\nIndexing completed successfully!")
    print(f"Indexed chunks: {collection.count()}")
    print("Stored metadata:")
    print("- Source")
    print("- Document ID")
    print("- Document type")
    print("- Revision")
    print("- Status")
    print("- Section")
    print("- Chunk index")

    print("\nForgeIQ is ready to retrieve information.")


if __name__ == "__main__":
    main()

