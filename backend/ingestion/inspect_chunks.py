from backend.ingestion.loader import load_documents
from backend.ingestion.splitter import split_documents

def main():
    documents = load_documents()
    chunks = split_documents(documents)

    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print()

    for i, chunk in enumerate(chunks[:10]):
        print("=" * 80)
        print(f"Chunk {i + 1}")
        print("-" * 80)

        print("Metadata:")
        for key, value in chunk.metadata.items():
            print(f"  {key}: {value}")

        print()

        print(f"Length: {len(chunk.page_content)}")


if __name__ == "__main__":
    main()