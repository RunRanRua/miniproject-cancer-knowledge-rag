"""Embeddings Module

Provides access to text embeddings through Ollama.
Embeddings are used to convert documents and queries into
vector representations for semantic similarity search.
"""

from langchain_ollama import OllamaEmbeddings

from config import EMBEDDING_MODEL, OLLAMA_BASE_URL


def get_embeddings():
    """Initialize and return an OllamaEmbeddings instance.
    
    Returns:
        OllamaEmbeddings: An embedding model configured with:
            - model: Embedding model name from config (e.g., nomic-embed-text)
            - base_url: Ollama server URL
            
    The embeddings are used for:
    - Encoding documents during ingestion
    - Encoding queries during retrieval
    - Computing semantic similarity for RAG
    """
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )