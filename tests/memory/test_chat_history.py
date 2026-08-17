from backend.memory.chat_history import (
    get_session_history,
    store,
)


def test_session_history_is_created():
    session_id = "test_user_001"

    history = get_session_history(session_id)

    assert history is not None
    assert session_id in store

    store.pop(session_id, None)


def test_messages_can_be_stored():
    session_id = "test_user_002"

    history = get_session_history(session_id)

    history.add_user_message(
        "What are the symptoms of bladder cancer?"
    )

    history.add_ai_message(
        "Bladder cancer may cause several symptoms."
    )

    messages = history.messages

    assert len(messages) == 2
    assert messages[0].type == "human"
    assert messages[1].type == "ai"

    store.pop(session_id, None)


def test_sessions_are_isolated():
    session_a = "test_user_a"
    session_b = "test_user_b"

    history_a = get_session_history(session_a)
    history_b = get_session_history(session_b)

    history_a.add_user_message("Question A")
    history_b.add_user_message("Question B")

    assert history_a.messages[0].content == "Question A"
    assert history_b.messages[0].content == "Question B"

    store.pop(session_a, None)
    store.pop(session_b, None)