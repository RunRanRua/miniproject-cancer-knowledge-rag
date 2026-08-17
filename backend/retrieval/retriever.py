"""Retrieval Module

Provides document retrieval from the ChromaDB vector store.
Documents are retrieved based on semantic similarity to the query.
"""

from config import VECTOR_STORE_DIR, COLLECTION_NAME
from backend.retrieval.embeddings import get_embeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

def _get_vector_store() -> Chroma:
    """Internal helper to initialize the ChromaDB vector store.
    
    Returns:
        Chroma: Vector store instance with:
            - collection_name: Named collection for organizing embeddings
            - embedding_function: Ollama embeddings for encoding text
            - persist_directory: Local storage for embeddings
    """
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(VECTOR_STORE_DIR),
    )

def retrieve_documents(question: str, k: int = 4) -> list[tuple[Document, float]]:
    """Retrieve documents most similar to the query.
    
    Args:
        question: The query text to search for
        k: Number of top results to return (default: 4)
        
    Returns:
        List of (Document, similarity_score) tuples ordered by relevance.
        Each Document includes:
        - page_content: The text chunk
        - metadata: Source information (title, cancer_type, url, etc.)
    """
    vector_store = _get_vector_store()
    # similarity_search_with_score returns tuples of (doc, score)
    # Higher scores indicate better matches
    results = vector_store.similarity_search_with_score(question, k=k)
    return results