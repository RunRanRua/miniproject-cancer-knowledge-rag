"""History-Aware Retrieval Module

Provides functions for context-aware document retrieval that leverages
conversation history to better understand user intent.

Key features:
- Formats conversation history for LLM processing
- Rewrites follow-up questions to standalone form
- Retrieves documents based on rewritten question
"""

from langchain_core.documents import Document

from backend.generation.question_rewriter import create_question_rewriter
from backend.memory.chat_history import get_session_history
from backend.retrieval.retriever import retrieve_documents


def format_history(messages) -> str:
    """Format chat messages into a readable conversation string.
    
    Args:
        messages: List of LangChain BaseMessage objects from chat history
        
    Returns:
        Formatted string with human/AI labels, suitable for LLM input
        
    Example output:
        Human: What are the symptoms of bladder cancer?
        AI: Bladder cancer may cause several symptoms.
    """
    lines = []

    for message in messages:
        # Message type is either "human" or "ai"
        role = "Human" if message.type == "human" else "AI"

        lines.append(
            f"{role}: {message.content}"
        )

    return "\n".join(lines)


def retrieve_with_history(
    session_id: str,
    question: str,
) -> tuple[str, list[Document]]:
    """Retrieve documents with history-aware question rewriting.
    
    Process:
    1. Load conversation history from session
    2. Format history for LLM processing
    3. Rewrite question to standalone form using LLM
    4. Retrieve documents based on rewritten question
    
    Args:
        session_id: Session ID for accessing conversation history
        question: User's current question (may reference history)
        
    Returns:
        Tuple of (rewritten_question, relevant_documents)
        - rewritten_question: Standalone version of the question
        - relevant_documents: Retrieved document chunks with scores
    """
    # Get conversation history
    history = get_session_history(session_id)

    # Format previous messages for LLM
    history_text = format_history(
        history.messages
    )

    # Create question rewriter chain
    rewriter = create_question_rewriter()

    # Rewrite question to standalone form
    standalone_question = rewriter.invoke(
        {
            "history": history_text,
            "question": question,
        }
    )

    # Retrieve documents using rewritten question
    documents = retrieve_documents(standalone_question)

    return standalone_question, documents