"""Vector Store Management Module

Provides initialization of the ChromaDB vector store with embedded documents.
This module is used during the data ingestion pipeline to populate the vector store.
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from config import VECTOR_STORE_DIR, COLLECTION_NAME
from backend.retrieval.embeddings import get_embeddings


def create_vector_store(documents: list[Document]) -> Chroma:
    """Initialize ChromaDB vector store and embed documents.
    
    Process:
    1. Initialize embedding model
    2. Create ChromaDB collection
    3. Embed all documents and store in vector database
    4. Persist to disk for later retrieval
    
    Args:
        documents: List of LangChain Documents to embed and store
        
    Returns:
        Chroma: Initialized vector store instance
        
    Note:
        This function should be called once during data ingestion.
        After initialization, use retriever.py functions for retrieval.
    """
    emb = get_embeddings()

    # Create vector store from documents
    # This will:
    # 1. Embed each document using the embedding model
    # 2. Store embeddings and metadata in ChromaDB
    # 3. Persist to VECTOR_STORE_DIR
    vector_store = Chroma.from_documents(
        documents = documents,
        embedding = emb,
        collection_name = COLLECTION_NAME,
        persist_directory = str(VECTOR_STORE_DIR)
    )

    return vector_store