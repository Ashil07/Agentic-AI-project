from langchain_core.chat_history import InMemoryChatMessageHistory

_histories = {}


def get_memory(session_id="default"):
    if session_id not in _histories:
        _histories[session_id] = InMemoryChatMessageHistory()

    return _histories[session_id]


def clear_memory(session_id="default"):
    if session_id in _histories:
        _histories[session_id].clear()
