from langchain_core.documents import Document

from backend.ingestion.splitter import split_documents


def test_split_documents():
    document = Document(
        page_content="A " * 2000,
        metadata={
            "source_id": "test",
            "cancer_type": "test_cancer",
        },
    )

    chunks = split_documents([document])

    assert len(chunks) > 1


def test_split_preserves_metadata():
    document = Document(
        page_content="A " * 2000,
        metadata={
            "source_id": "nci",
            "cancer_type": "lung_cancer",
            "topic": "treatment",
        },
    )

    chunks = split_documents([document])

    for chunk in chunks:
        assert chunk.metadata["source_id"] == "nci"
        assert chunk.metadata["cancer_type"] == "lung_cancer"
        assert chunk.metadata["topic"] == "treatment"


def test_split_content_not_empty():
    document = Document(
        page_content="A " * 2000,
        metadata={},
    )

    chunks = split_documents([document])

    assert all(chunk.page_content.strip() for chunk in chunks)


def test_markdown_is_split():
    document = Document(
        page_content="""
# Lung Cancer

## Symptoms

Lung cancer symptoms include coughing and chest pain.

## Diagnosis

Doctors may use imaging tests and biopsies.
""",
        metadata={
            "source_id": "nci",
            "cancer_type": "lung_cancer",
        },
    )

    chunks = split_documents([document])

    assert len(chunks) >= 2


def test_metadata_is_preserved():
    document = Document(
        page_content="""
# Lung Cancer

## Symptoms

Lung cancer symptoms include coughing and chest pain.
""",
        metadata={
            "source_id": "nci",
            "cancer_type": "lung_cancer",
            "topic": "symptoms",
        },
    )

    chunks = split_documents([document])

    assert len(chunks) > 0

    for chunk in chunks:
        assert chunk.metadata["source_id"] == "nci"
        assert chunk.metadata["cancer_type"] == "lung_cancer"
        assert chunk.metadata["topic"] == "symptoms"


def test_header_metadata_is_created():
    document = Document(
        page_content="""
# Lung Cancer

## Symptoms

Lung cancer symptoms include coughing.
""",
        metadata={},
    )

    chunks = split_documents([document])

    assert len(chunks) > 0

    assert any(
        chunk.metadata.get("header_2") == "Symptoms"
        for chunk in chunks
    )