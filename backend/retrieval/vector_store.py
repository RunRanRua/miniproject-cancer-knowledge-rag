from langchain_chroma import Chroma
from langchain_core.documents import Document
from config import VECTOR_STORE_DIR, COLLECTION_NAME
from backend.retrieval.embeddings import get_embeddings


def create_vector_store(documents: list[Document]) -> Chroma:
    emb = get_embeddings()

    vector_store = Chroma.from_documents(
        documents = documents,
        embedding = emb,
        collection_name = COLLECTION_NAME,
        persist_directory = str(VECTOR_STORE_DIR)
    )

    return vector_store