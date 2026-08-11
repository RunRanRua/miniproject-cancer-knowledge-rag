from backend.retrieval.retriever import get_retriever
from backend.generation.llm import get_llm
from backend.generation.prompt import RAG_PROMPT

def _format_documents(documents) -> str:
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

def create_rag_chain():
    retriever = get_retriever(k=4)
    llm = get_llm()

    def rag(question: str):
        documents = retriever.invoke(question)

        context = _format_documents(documents)

        messages = RAG_PROMPT.invoke({
            "context": context,
            "question": question,
        })

        response = llm.invoke(messages)


        src = []
        for i, document in enumerate(documents, start=1):
            metadata = document.metadata

            src.append({
                "id": i,
                "title": metadata.get("title"),
                "cancer_type": metadata.get("cancer_type"),
                "topic": metadata.get("topic"),
                "url": metadata.get("url"),
                "file_path": metadata.get("file_path"),
            })

        return {
            "answer": response.content,
            "sources": src,
            "documents": documents,
        }

    return rag