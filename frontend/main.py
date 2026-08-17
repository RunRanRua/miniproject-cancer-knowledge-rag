"""Cancer Knowledge RAG - Streamlit Frontend

Interactive web interface for querying cancer-related information using RAG.

Features:
- Chat interface for asking questions
- Session-based conversation management
- Create and delete conversations
- Source attribution with metadata
- Conversation history tracking
- Clean, intuitive UI

Usage:
    streamlit run frontend/main.py
"""

from pathlib import Path
import sys
import uuid

# ============================================================================
# PROJECT PATH
# ============================================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ============================================================================
# IMPORTS
# ============================================================================

import streamlit as st

from backend.rag.chain import create_rag_chain
from backend.memory.chat_history import get_session_history, delete_session


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Cancer Knowledge RAG",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# HEADER
# ============================================================================

st.markdown(
    """
    <div style="
        background: linear-gradient(90deg, #f5f5f5, #eef4ff);
        padding: 0.8rem 1rem;
        border-radius: 12px;
        border: 1px solid #e6eaf2;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 8px rgba(31,41,55,0.05);
    ">
        <h1 style="
            margin: 0;
            font-size: 1.8rem;
            color: #1f2937;
        ">
            🏥 Cancer Knowledge Assistant
        </h1>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Ask questions about cancer types, symptoms, diagnosis, and treatment options. "
    "The AI assistant will retrieve relevant information from the knowledge base "
    "and provide grounded answers."
)


# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown(
    """
    <style>

    /* =====================================================================
       SIDEBAR
       ===================================================================== */

    [data-testid="stSidebar"] {
        background: #f8fafc;
    }

    /* Conversation button */

    [data-testid="stSidebar"] .conversation-button button {
        border-radius: 10px;
        border: 1px solid #dfe7f5;
        background: white;
        color: #1f2937;
        text-align: left;
        padding: 0.65rem 0.7rem;
        margin: 0.15rem 0;
        width: 100%;
        font-weight: 500;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    [data-testid="stSidebar"] .conversation-button button:hover {
        border-color: #b7c8ef;
        background: #f3f7ff;
    }

    /* Delete button */

    [data-testid="stSidebar"] .delete-button button {
        border-radius: 10px;
        border: 1px solid #fecaca;
        background: #fee2e2;
        color: #dc2626;
        padding: 0.65rem 0.3rem;
        margin: 0.15rem 0;
        width: 100%;
        font-weight: 600;
        white-space: nowrap;
    }

    [data-testid="stSidebar"] .delete-button button:hover {
        border-color: #ef4444;
        background: #fecaca;
        color: #b91c1c;
    }

    [data-testid="stSidebar"] .delete-button button:focus {
        box-shadow: 0 0 0 0.15rem rgba(239, 68, 68, 0.2);
    }

    /* Sources */

    [data-testid="stExpander"] {
        border-radius: 10px;
        border: 1px solid #e5e7eb;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

# Internal session ID of the current conversation.
# The UUID is never displayed to the user.
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


# Ordered list of conversation session IDs.
#
# The index determines the UI label:
#
#     index 0 -> Conversation 1
#     index 1 -> Conversation 2
#     index 2 -> Conversation 3
#
if "known_sessions" not in st.session_state:
    st.session_state.known_sessions = [st.session_state.session_id]


# Messages used by the Streamlit UI.
if "messages" not in st.session_state:
    st.session_state.messages = []


# Create the RAG chain only once.
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = create_rag_chain()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_conversation_number(session_id: str):
    """Return the UI number of a conversation."""

    try:
        return st.session_state.known_sessions.index(session_id) + 1
    except ValueError:
        return None


def get_conversation_label(session_id: str):
    """Return the human-readable conversation label."""

    number = get_conversation_number(session_id)

    if number is None:
        return "Conversation"

    return f"Conversation {number}"


def load_ui_messages(session_id: str):
    """
    Load messages from LangChain chat history.

    Sources are retrieval metadata and are not stored in
    InMemoryChatMessageHistory.
    """

    history = get_session_history(session_id)
    messages = []

    for msg in history.messages:
        if msg.type == "human":
            messages.append({
                "role": "user",
                "content": msg.content,
                "sources": [],
            })

        elif msg.type == "ai":
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "sources": [],
            })

    return messages


def render_sources(sources):
    """Render retrieved sources in a collapsed expander."""

    if not sources:
        return

    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
        for i, source in enumerate(sources, start=1):
            title = source.get("title", "Untitled")
            cancer_type = source.get("cancer_type", "N/A")
            topic = source.get("topic", "N/A")
            url = source.get("url")

            st.markdown(f"**{i}. {title}**")
            st.caption(f"Cancer type: {cancer_type} | Topic: {topic}")

            if url:
                st.markdown(f"[🔗 View source]({url})")

            if i < len(sources):
                st.divider()


def render_message(message):
    """Render a single chat message."""

    role = message.get("role", "assistant")
    content = message.get("content", "")
    sources = message.get("sources", [])

    with st.chat_message(role):
        st.markdown(content)

        if role == "assistant" and sources:
            render_sources(sources)


def create_new_conversation():
    """Create a new conversation and make it active."""

    new_session_id = str(uuid.uuid4())

    st.session_state.known_sessions.append(new_session_id)
    st.session_state.session_id = new_session_id
    st.session_state.messages = []


def delete_conversation(session_id: str):
    """
    Completely delete a conversation.

    This removes:
    1. LangChain chat history
    2. Session ID from known_sessions
    3. Current UI messages if necessary
    """

    # Delete backend history.
    delete_session(session_id)

    # Remove the session from the UI session list.
    if session_id in st.session_state.known_sessions:
        st.session_state.known_sessions.remove(session_id)

    # If the deleted conversation was the current one,
    # switch to another conversation.
    if session_id == st.session_state.session_id:

        if st.session_state.known_sessions:
            new_current_session = st.session_state.known_sessions[-1]
            st.session_state.session_id = new_current_session
            st.session_state.messages = load_ui_messages(new_current_session)

        else:
            # If no conversation remains, create a fresh one.
            new_session_id = str(uuid.uuid4())

            st.session_state.known_sessions = [new_session_id]
            st.session_state.session_id = new_session_id
            st.session_state.messages = []


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:

    st.header("💬 Conversations")

    # ------------------------------------------------------------------------
    # Conversation list
    # ------------------------------------------------------------------------

    for session_id in list(st.session_state.known_sessions):

        conversation_label = get_conversation_label(session_id)

        if session_id == st.session_state.session_id:
            conversation_label = f"● {conversation_label}"

        # Conversation button + Delete button.
        #
        # The sidebar is narrow, so use a wider area for the conversation
        # and a dedicated smaller area for Delete.
        col1, col2 = st.columns([3, 1.5], gap="small")

        # --------------------------------------------------------------------
        # Conversation
        # --------------------------------------------------------------------

        with col1:
            st.markdown(
                '<div class="conversation-button">',
                unsafe_allow_html=True,
            )

            if st.button(
                conversation_label,
                key=f"session_{session_id}",
                use_container_width=True,
            ):
                st.session_state.session_id = session_id
                st.session_state.messages = load_ui_messages(session_id)
                st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

        # --------------------------------------------------------------------
        # Delete
        # --------------------------------------------------------------------

        with col2:
            st.markdown(
                '<div class="delete-button">',
                unsafe_allow_html=True,
            )

            if st.button(
                "Delete",
                key=f"delete_{session_id}",
                help="Delete this conversation",
                use_container_width=True,
            ):
                delete_conversation(session_id)
                st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

    # ------------------------------------------------------------------------
    # New conversation
    # ------------------------------------------------------------------------

    st.divider()

    if st.button(
        "New Conversation",
        key="new_conversation",
        use_container_width=True,
    ):
        create_new_conversation()
        st.rerun()

    # ------------------------------------------------------------------------
    # Current conversation information
    # ------------------------------------------------------------------------

    history = get_session_history(st.session_state.session_id)

    st.divider()

    st.caption(
        f"Messages in conversation: {len(history.messages)}"
    )


# ============================================================================
# MAIN CHAT INTERFACE
# ============================================================================

conversation_label = get_conversation_label(st.session_state.session_id)

st.markdown(f"## {conversation_label}")


# ============================================================================
# LOAD CURRENT HISTORY
# ============================================================================

history = get_session_history(st.session_state.session_id)

if not st.session_state.messages and history.messages:
    st.session_state.messages = load_ui_messages(
        st.session_state.session_id
    )


# ============================================================================
# RENDER EXISTING MESSAGES
# ============================================================================

if st.session_state.messages:

    for message in st.session_state.messages:
        render_message(message)

else:

    st.info(
        "Start a conversation by asking a question below."
    )


# ============================================================================
# CHAT INPUT
# ============================================================================

if prompt := st.chat_input(
    "Ask about cancer symptoms, diagnosis, or treatment..."
):

    history = get_session_history(st.session_state.session_id)

    # ------------------------------------------------------------------------
    # Add user message
    # ------------------------------------------------------------------------

    history.add_user_message(prompt)

    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "sources": [],
    })

    # ------------------------------------------------------------------------
    # Render user message
    # ------------------------------------------------------------------------

    with st.chat_message("user"):
        st.markdown(prompt)

    # ------------------------------------------------------------------------
    # Generate RAG response
    # ------------------------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Searching knowledge base..."):

            try:
                result = st.session_state.rag_chain(prompt)

                answer = result.get(
                    "answer",
                    "I couldn't generate an answer.",
                )

                sources = result.get("sources", [])

                # Save answer to LangChain history.
                history.add_ai_message(answer)

                # Save answer + sources to Streamlit UI state.
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

                # Render answer.
                st.markdown(answer)

                # Render collapsed sources.
                render_sources(sources)

            except Exception as e:

                st.error(
                    f"Error processing question: {str(e)}"
                )

                # Remove failed user message from backend history.
                if (
                    history.messages
                    and history.messages[-1].type == "human"
                ):
                    history.messages.pop()

                # Remove failed user message from UI state.
                if (
                    st.session_state.messages
                    and st.session_state.messages[-1].get("role") == "user"
                ):
                    st.session_state.messages.pop()