from backend.memory.chat_history import (
    get_session_history,
    store,
)
from backend.retrieval.history_aware import (
    format_history,
)


def test_format_history():

    session_id = "test_history_format"

    history = get_session_history(session_id)

    history.add_user_message(
        "What are the symptoms of bladder cancer?"
    )

    history.add_ai_message(
        "Bladder cancer may cause several symptoms."
    )

    result = format_history(
        history.messages
    )

    assert "Human:" in result
    assert "AI:" in result

    assert "symptoms of bladder cancer" in result

    store.pop(session_id, None)
    