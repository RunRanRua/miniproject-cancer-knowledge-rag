from langchain_core.documents import Document
from backend.ingestion.loader import load_documents


def test_load_documents():
    documents = load_documents()

    assert len(documents) > 0
    assert all(isinstance(doc, Document) for doc in documents)


def test_document_metadata():
    documents = load_documents()

    document = documents[0]

    assert "source_id" in document.metadata
    assert "cancer_type" in document.metadata
    assert "title" in document.metadata
    assert "topic" in document.metadata
    assert "url" in document.metadata
    assert "file_path" in document.metadata
    assert "language" in document.metadata


def test_document_content_not_empty():
    documents = load_documents()

    assert all(document.page_content.strip() for document in documents)