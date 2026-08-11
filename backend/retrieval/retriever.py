from config import VECTOR_STORE_DIR, COLLECTION_NAME
from backend.retrieval.embeddings import get_embeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

def _get_vector_store():
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(VECTOR_STORE_DIR),
    )

def retrieve_documents(question, k = 4) -> list[tuple[Document, float]]:
    vector_store = _get_vector_store()
    results = vector_store.similarity_search_with_score(question, k=k)
    return results