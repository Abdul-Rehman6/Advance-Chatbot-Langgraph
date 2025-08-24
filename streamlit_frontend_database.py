import re
import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

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

# --------- Title generation ---------

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
    - 3-8 words
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
    # When a thread button is clicked:
    if st.sidebar.button(summary, key=f"btn_{tid}"):
        st.session_state["thread_id"] = tid
        messages = load_conversation(tid)

        temp_messages = []
        pending_tools = []  # collect tools seen before the next AI message

        for msg in messages:
            if isinstance(msg, HumanMessage):
                temp_messages.append({"role": "user", "content": msg.content})

            elif isinstance(msg, ToolMessage):
                tool_name = getattr(msg, "name", "tool")
                pending_tools.append(tool_name)

            elif isinstance(msg, AIMessage):
                # attach the tools collected since the last assistant turn
                temp_messages.append({
                    "role": "assistant",
                    "content": msg.content if isinstance(msg.content, str) else str(msg.content),
                    "tools": pending_tools[:]  # copy
                })
                pending_tools.clear()

        # If tools exist at the end without a following AIMessage, you can choose to
        # attach them to the last assistant message (optional/edge case)
        if pending_tools and temp_messages and temp_messages[-1]["role"] == "assistant":
            temp_messages[-1].setdefault("tools", []).extend(pending_tools)

        st.session_state["message_history"] = temp_messages

# ============================ Main UI ============================

# Render existing history
def render_message(message: dict):
    with st.chat_message(message["role"]):
        if message.get("tools") and message["role"] == "assistant":
            for tool in message["tools"]:
                box = st.status(f"🔧 Used `{tool}`", expanded=False)
                box.update(state="complete")
        # NEW: show any persisted log lines under the status
        if message.get("tool_logs") and message["role"] == "assistant":
            st.markdown("\n".join(message["tool_logs"]))
        st.markdown(message["content"])

for message in st.session_state["message_history"]:
    render_message(message)


user_input = st.chat_input(placeholder="Ask me anything...")

if user_input:
    # append + render user
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # stream assistant via LangGraph with persisted thread_id
    # metadata and run_name for observability in LangSmith

    CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]},
              "metadata": {"thread_id": st.session_state["thread_id"]},
              "run_name": "Chat_turn"}
    
    # Assistant streaming block
    # Assistant streaming block
    with st.chat_message("assistant"):
        status_holder = {"box": None}
        log_area = st.empty()          # NEW: where we’ll print running lines
        final_text = []
        tools_used = []  # NEW: persist tools used in this assistant turn
        tool_logs = []                 # NEW: keep the lines so we can persist/replay


        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    tools_used.append(tool_name)

                    # NEW: write/update log lines
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                        tool_logs.append("- Calling API…")         # first line
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )
                        tool_logs.append("- Still working…")       # subsequent line(s)

                    # Re-render the full log each time (replaces previous content)
                    log_area.markdown("\n".join(tool_logs))

                if isinstance(message_chunk, AIMessage):
                    text_chunk = str(message_chunk.content)
                    final_text.append(text_chunk)
                    yield text_chunk

        st.write_stream(ai_only_stream)

        # After streaming finishes
        if status_holder["box"] is not None:
            tool_logs.append("- Done.")          # NEW
            log_area.markdown("\n".join(tool_logs))
            status_holder["box"].update(
                label="✅ Tool finished",
                state="complete",
                expanded=False,                   # collapse if you prefer; set True to keep open
            )

    # Save assistant message WITH the tools used
    st.session_state["message_history"].append({
        "role": "assistant",
        "content": "".join(final_text),
        "tools": tools_used,         # you already save this
        "tool_logs": tool_logs,      # NEW: persist the log lines
    })


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
