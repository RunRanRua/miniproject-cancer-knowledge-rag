from config import VECTOR_STORE_DIR, COLLECTION_NAME
from backend.retrieval.embeddings import get_embeddings
from langchain_chroma import Chroma

def _get_vector_store():
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(VECTOR_STORE_DIR),
    )

def get_retriever(k = 4):
    vector_store = _get_vector_store()
    return vector_store.as_retriever(
        search_kwargs={"k":k},
    )