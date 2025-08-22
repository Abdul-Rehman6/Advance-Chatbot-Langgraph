from typing import TypedDict, Annotated, Optional
import sqlite3

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

# ---- LLM ----
llm = ChatOpenAI(model="gpt-4o-mini")

# ---- State ----
class ChatState(TypedDict):
    # reducer enforces appending messages instead of replacing
    messages: Annotated[list[BaseMessage], add_messages]

# ---- Node ----
def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# ---- SQLite + Checkpointer ----
# Allows DB connection across multiple threads (Streamlit)
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)

# Titles table (persist chat titles)
conn.execute(
    """
    CREATE TABLE IF NOT EXISTS thread_summaries (
        thread_id   TEXT PRIMARY KEY,
        title       TEXT NOT NULL,
        updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """
)
conn.commit()

checkpointer = SqliteSaver(conn=conn)

# ---- Graph ----
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
chatbot = graph.compile(checkpointer=checkpointer)

# ---- Threads API ----
def retrieve_all_threads() -> list[str]:
    """Return all unique thread_ids that have checkpoints."""
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        tid = str(checkpoint.config["configurable"]["thread_id"])
        all_threads.add(tid)
    return list(all_threads)

# ---- Summaries API ----
def save_thread_summary(thread_id: str, title: str) -> None:
    conn.execute(
        """
        INSERT INTO thread_summaries (thread_id, title, updated_at)
        VALUES (?, ?, datetime('now'))
        ON CONFLICT(thread_id) DO UPDATE SET
            title = excluded.title,
            updated_at = excluded.updated_at
        """,
        (str(thread_id), title),
    )
    conn.commit()

def get_thread_summary(thread_id: str) -> Optional[str]:
    row = conn.execute(
        "SELECT title FROM thread_summaries WHERE thread_id = ?",
        (str(thread_id),),
    ).fetchone()
    return row[0] if row else None

def load_thread_summaries() -> dict[str, str]:
    rows = conn.execute(
        "SELECT thread_id, title FROM thread_summaries ORDER BY updated_at DESC"
    ).fetchall()
    return {tid: title for (tid, title) in rows}
