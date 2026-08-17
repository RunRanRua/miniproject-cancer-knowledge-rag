"""Question Rewriter Module

Implements history-aware question rewriting for better retrieval.
When users ask follow-up questions in a conversation, this module
converts them to standalone questions that include necessary context.

Example:
- User: "What about diagnosis?"
- Rewritten: "What is the diagnosis process for breast cancer?"
"""

from langchain_core.prompts import ChatPromptTemplate
from backend.generation.llm import get_llm
from langchain_core.output_parsers import StrOutputParser

# Prompt for rewriting questions to standalone format
# Input variables:
#   - {history}: Previous conversation messages
#   - {question}: Current user question
# Output: Rewritten question that includes relevant context
QUESTION_REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """
Given the conversation history and the user's latest question,
rewrite the latest question into a standalone question.

The standalone question must preserve the meaning and relevant
context from the conversation history.

If the latest question is already standalone, return it unchanged.

Do not answer the question.
Conversation history:
{history}

Latest question:
{question}
"""
)

def create_question_rewriter():
    """Create a LangChain runnable chain for question rewriting.
    
    Returns:
        A LangChain runnable that takes a dict with 'history' and 'question' keys
        and returns a rewritten question as a string.
        
        Usage:
            rewriter = create_question_rewriter()
            result = rewriter.invoke({
                'history': 'Previous conversation...',
                'question': 'Follow-up question?'
            })
    """
    llm = get_llm()
    
    # Chain: Prompt -> LLM -> String output parser
    # This uses LangChain's LCEL (LangChain Expression Language) syntax
    return (QUESTION_REWRITE_PROMPT | llm | StrOutputParser())