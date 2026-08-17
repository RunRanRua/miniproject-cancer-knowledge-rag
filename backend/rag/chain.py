"""RAG Chain Module

Orchestrates the complete Retrieval-Augmented Generation (RAG) pipeline:
1. Retrieve relevant documents from vector store
2. Format documents with metadata
3. Generate LLM response using retrieved context
4. Extract and deduplicate source information

The RAG chain is the main orchestration point for answering user questions.
"""

from backend.retrieval.retriever import retrieve_documents
from backend.generation.llm import get_llm
from backend.generation.prompt import RAG_PROMPT

def _format_documents(documents) -> str:
    """Format retrieved documents for LLM consumption.
    
    Creates a readable format with numbered sources and metadata.
    
    Args:
        documents: List of LangChain Document objects from retrieval
        
    Returns:
        Formatted string with source information and content
    """
    formatted_documents =[]

    for i, document in enumerate(documents, start=1):
        metadata = document.metadata

        src = (
            f"[Source {i}]\n"
            f"Title: {metadata.get('title', 'Unknown')}\n"
            f"Cancer type: {metadata.get('cancer_type', 'Unknown')}\n"
            f"Topic: {metadata.get('topic', 'Unknown')}\n"
            f"Section: {metadata.get('header_2', 'Unknown')}\n"
            f"URL: {metadata.get('url', 'Unknown')}\n"
        )
        formatted_documents.append(
            f"{src}\n"
            f"Content:\n{document.page_content}"
        )

    return "\n\n".join(formatted_documents)

def _build_sources(documents):
    """Extract and deduplicate source information from documents.
    
    Args:
        documents: List of retrieved documents
        
    Returns:
        List of deduplicated source metadata dicts
    """
    sources = []
    seen = set()

    for i, document in enumerate(documents, start=1):
        metadata = document.metadata
        source_key = metadata.get("file_path")

        if source_key in seen:
            continue

        seen.add(source_key)

        sources.append({
            "id": len(sources) + 1,
            "title": metadata.get("title"),
            "cancer_type": metadata.get("cancer_type"),
            "topic": metadata.get("topic"),
            "url": metadata.get("url"),
            "file_path": metadata.get("file_path"),
        })

    return sources


def create_rag_chain():
    """Create and return the RAG chain function."""
    llm = get_llm()

    def rag(question: str):
        # 1. get relevant documents from the vector store
        results = retrieve_documents(question, k=4)
        documents = [document for document, score in results]

        # 2. get llm output
        context = _format_documents(documents)
        messages = RAG_PROMPT.invoke({
            "context": context,
            "question": question,
        })
        response = llm.invoke(messages)

        # 3. deduplicate sources & prepare the final output
        sources = _build_sources(documents)
        retrieval_results = []
        for document, score in results:
            retrieval_results.append({
                "score": score,
                "file_path": document.metadata.get("file_path"),
            })

        return {
            "answer": response.content,
            "sources": sources,
            "retrieval_results": retrieval_results,
        }


    return rag