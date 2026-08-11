from langchain_core.documents import Document

from backend.rag.chain import _build_sources


def test_sources_are_deduplicated():
    documents = [
        Document(
            page_content="chunk 1",
            metadata={
                "file_path": "nci/bladder_cancer/symptoms.md",
                "title": "Bladder Cancer",
                "topic": "symptoms",
            },
        ),
        Document(
            page_content="chunk 2",
            metadata={
                "file_path": "nci/bladder_cancer/symptoms.md",
                "title": "Bladder Cancer",
                "topic": "symptoms",
            },
        ),
        Document(
            page_content="chunk 3",
            metadata={
                "file_path": "nci/bladder_cancer/diagnosis.md",
                "title": "Bladder Cancer",
                "topic": "diagnosis",
            },
        ),
    ]

    sources = _build_sources(documents)

    assert len(sources) == 2


def test_source_ids_are_sequential():
    documents = [
        Document(
            page_content="chunk",
            metadata={
                "file_path": "nci/a.md",
            },
        ),
        Document(
            page_content="chunk",
            metadata={
                "file_path": "nci/b.md",
            },
        ),
    ]

    sources = _build_sources(documents)

    assert sources[0]["id"] == 1
    assert sources[1]["id"] == 2