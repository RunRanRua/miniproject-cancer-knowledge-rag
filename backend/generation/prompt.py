from langchain_core.prompts import ChatPromptTemplate


SYSTEM_PROMPT_CONTENT = """
You are a cancer information assistant.

Answer the user's question using only the provided context.

If the answer cannot be found in the context, say that the
information is not available in the knowledge base.

Do not use outside knowledge.

Use the source numbers provided in the context when making claims.
Cite sources using the format [Source 1], [Source 2], etc.
"""

HUMAN_PROMPT_CONTENT = """
Question: {question}

Context: {context}
"""


RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",SYSTEM_PROMPT_CONTENT),
        ("human",HUMAN_PROMPT_CONTENT),
    ]
)