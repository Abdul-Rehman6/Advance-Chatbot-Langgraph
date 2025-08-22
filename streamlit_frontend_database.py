import re
import uuid
import streamlit as st
from langchain_core.messages import HumanMessage

from langgraph_database_backend import (
    chatbot,
    retrieve_all_threads,
    llm,
    save_thread_summary,
    get_thread_summary,
    load_thread_summaries,
)

# ============================ Utilities ============================

def generate_thread_id() -> str:
    # use string IDs everywhere for consistency
    return str(uuid.uuid4())

def add_thread(thread_id: str):
    tid = str(thread_id)
    if tid not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(tid)
    if tid not in st.session_state["thread_summaries"]:
        st.session_state["thread_summaries"][tid] = "New Conversation"

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []

def load_conversation(thread_id: str):
    config = {"configurable": {"thread_id": str(thread_id)}}
    state = chatbot.get_state(config)
    if state is None:
        return []
    return state.values.get("messages", [])

# --------- Title generation (ChatGPT-like, one-time) ---------

def _to_title_case(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().title()

def _heuristic_title(messages) -> str:
    # Fallback: first user message → first 8 words
    first_user = ""
    for m in messages:
        if isinstance(m, HumanMessage):
            first_user = m.content.strip()
            break
    if not first_user:
        return "New Conversation"
    words = re.findall(r"\w+[\w-]*", first_user)[:8]
    return _to_title_case(" ".join(words)) or "New Conversation"

def generate_summary(messages) -> str:
    """
    Generate a concise, ChatGPT-style title once per thread.
    - 3–8 words
    - Title Case
    - No quotes/emojis
    """
    if not messages:
        return "New Conversation"

    # small excerpt (first few turns only)
    convo_lines = []
    for msg in messages[:4]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        text = msg.content.replace("\n", " ").strip()
        if len(text) > 240:
            text = text[:240] + "..."
        convo_lines.append(f"{role}: {text}")
    excerpt = "\n".join(convo_lines)

    prompt = f"""
You create concise chat titles.

Rules:
- 3 to 8 words.
- Title Case.
- No punctuation at the end. No quotes, emojis, or numbering.
- Capture the main topic or intent.

Conversation:
{excerpt}

Return ONLY the title text.
""".strip()

    try:
        resp = llm.invoke(prompt)
        title = resp.content.strip()
        # sanitize
        title = re.sub(r'["“”\'`]+', "", title)
        title = re.sub(r"[•\u2022]+", "", title)
        title = re.sub(r"[\.\!\?،،؛，。]+$", "", title)  # strip trailing punctuation
        # clamp words
        words = re.findall(r"\w+[\w-]*", title)[:8]
        title = _to_title_case(" ".join(words))
        return title if title else _heuristic_title(messages)
    except Exception:
        return _heuristic_title(messages)

# ============================ Session Setup ============================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "thread_summaries" not in st.session_state:
    # Load all titles from DB once; fall back to placeholder
    db_summaries = load_thread_summaries()  # {thread_id: title}
    st.session_state["thread_summaries"] = {}
    for thread_id in st.session_state["chat_threads"]:
        tid = str(thread_id)
        st.session_state["thread_summaries"][tid] = db_summaries.get(tid, "New Conversation")

# Ensure current thread is tracked
add_thread(st.session_state["thread_id"])

# ============================ Sidebar ============================

st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")

# Show most recent first
for thread_id in st.session_state["chat_threads"][::-1]:
    tid = str(thread_id)
    summary = st.session_state["thread_summaries"].get(tid, "New Conversation")
    if st.sidebar.button(summary, key=f"btn_{tid}"):
        st.session_state["thread_id"] = tid
        messages = load_conversation(tid)

        temp_messages = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": msg.content})
        st.session_state["message_history"] = temp_messages

# ============================ Main UI ============================

# Render existing history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    # 1) append + render user
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) stream assistant via LangGraph with persisted thread_id
    CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}
    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            message_chunk.content
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            )
        )

    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})

    # 3) One-time: generate & persist title if not present
    tid = str(st.session_state["thread_id"])
    existing_title = get_thread_summary(tid)
    if not existing_title or existing_title == "New Conversation":
        # Use authoritative history from backend
        messages = load_conversation(tid)
        if messages:
            title = generate_summary(messages)
            save_thread_summary(tid, title)               # persist in DB
            st.session_state["thread_summaries"][tid] = title
