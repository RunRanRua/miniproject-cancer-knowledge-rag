"""LLM Provider Module

Provides access to the language model through LangChain's Ollama integration.
The LLM is used for:
- Rewriting questions for better retrieval
- Generating answers based on retrieved context
"""

from langchain_ollama import ChatOllama

from config import LLM_MODEL, OLLAMA_BASE_URL


def get_llm() -> ChatOllama:
    """Initialize and return a ChatOllama LLM instance.
    
    Returns:
        ChatOllama: A language model instance configured with:
            - model: Model name from config (e.g., minimax-m3:cloud)
            - base_url: Ollama server URL
            - temperature: Set to 0 for deterministic responses
    """
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,  # Deterministic responses for consistent RAG output
    )