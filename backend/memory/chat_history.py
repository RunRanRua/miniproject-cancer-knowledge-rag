"""Chat History Module

Manages conversation state using LangChain's InMemoryChatMessageHistory.

Each session maintains its own conversation history for context-aware
interactions.

Note:
    This uses in-memory storage and is suitable for single-process
    applications. For production with multiple workers, consider Redis
    or a database.
"""

from langchain_core.chat_history import InMemoryChatMessageHistory


# ============================================================================
# SESSION STORE
# ============================================================================

# Maps:
#
#     session_id -> InMemoryChatMessageHistory
#
# Example:
#
#     {
#         "session-uuid-1": InMemoryChatMessageHistory(...),
#         "session-uuid-2": InMemoryChatMessageHistory(...)
#     }
#
store: dict[str, InMemoryChatMessageHistory] = {}


# ============================================================================
# GET SESSION HISTORY
# ============================================================================

def get_session_history(
    session_id: str,
) -> InMemoryChatMessageHistory:
    """
    Get or create the chat history for a session.

    Args:
        session_id:
            Unique identifier of the conversation.

    Returns:
        InMemoryChatMessageHistory:
            The chat history associated with the session.

    Each session maintains its own isolated conversation history.
    """

    if session_id not in store:
        store[session_id] = (
            InMemoryChatMessageHistory()
        )

    return store[session_id]


# ============================================================================
# DELETE SESSION
# ============================================================================

def delete_session(
    session_id: str,
) -> bool:
    """
    Completely delete a conversation session.

    This removes the session and all of its messages from memory.

    Args:
        session_id:
            Unique identifier of the conversation.

    Returns:
        bool:
            True if the session existed and was deleted.
            False if the session did not exist.
    """

    if session_id not in store:
        return False

    del store[session_id]

    return True


# ============================================================================
# CHECK SESSION
# ============================================================================

def session_exists(
    session_id: str,
) -> bool:
    """
    Check whether a session currently exists.
    """

    return session_id in store