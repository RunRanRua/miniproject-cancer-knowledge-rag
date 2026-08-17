"""RAG Prompt Templates

Defines the prompt structure used for the RAG pipeline.
The system prompt constrains the LLM to only use provided context,
while the human prompt provides the question and retrieved context.
"""

from langchain_core.prompts import ChatPromptTemplate

# System prompt: Instructs the LLM to act as a cancer information assistant
# Key constraints:
# - Only use provided context (grounding)
# - Don't use external knowledge
# - Cite sources when making claims
SYSTEM_PROMPT_CONTENT = """
You are a cancer information assistant.

Answer the user's question using only the provided context.

If the answer cannot be found in the context, say that the
information is not available in the knowledge base.

Do not use outside knowledge.

Use the source numbers provided in the context when making claims.
Cite sources using the format [Source 1], [Source 2], etc.
"""

# Human prompt: Provides the user's question and retrieved context
# Variables:
# - {question}: The user's question (may be rewritten)
# - {context}: Retrieved documents formatted for the LLM
HUMAN_PROMPT_CONTENT = """
Question: {question}

Context: {context}
"""

# Combined prompt template for RAG responses
# Follows LangChain's chat message format with system and human roles
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT_CONTENT),
        ("human", HUMAN_PROMPT_CONTENT),
    ]
)