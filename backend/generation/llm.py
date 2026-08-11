from langchain_ollama import ChatOllama

from config import LLM_MODEL, OLLAMA_BASE_URL


def get_llm() -> ChatOllama:
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )