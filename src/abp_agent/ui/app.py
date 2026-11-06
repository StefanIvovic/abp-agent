"""Chat page: questions go to the agent over the API, sources are shown under each answer."""

import json
import os

import httpx
import streamlit as st

API = os.environ["API_URL"]
LABELS = {"search": "Searching docs and code", "code_graph": "Reading the code graph"}

st.title("ABP Agent")
ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("thread_id", None)

with st.sidebar:
    if st.button("New chat"):
        ss.messages, ss.thread_id = [], None
        st.rerun()


def show_sources(citations: list[dict]) -> None:
    if citations:
        with st.expander(f"Sources ({len(citations)})"):
            for c in citations:
                st.markdown(f"**[{c['chunk_id']}]** [{c['title']}]({c['url']}) ({c['source']})")


def events(message: str):
    body = {"message": message, "thread_id": ss.thread_id}
    with httpx.stream("POST", f"{API}/agent/stream", json=body, timeout=600) as r:
        for line in r.iter_lines():
            if line.startswith("data: "):
                yield json.loads(line[6:])


for m in ss.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        show_sources(m.get("citations", []))

if question := st.chat_input("Ask about ABP"):
    ss.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        status = st.status("Thinking…", expanded=False)
        box = st.empty()
        for ev in events(question):
            if ev["type"] == "start":
                ss.thread_id = ev["thread_id"]
            elif ev["type"] == "tool_call":
                status.update(label=LABELS.get(ev["name"], ev["name"]) + "…")
                status.write(f"**{ev['name']}** `{ev['args']}`")
            elif ev["type"] == "tool_result":
                status.write(f"→ {ev['sources']} sources")
            elif ev["type"] == "error":
                status.update(label="Error", state="error")
                box.error(ev["message"])
            elif ev["type"] == "done":
                status.update(label=f"Done in {ev['latency_ms'] / 1000:.1f} s", state="complete")
                ss.messages.append({"role": "assistant", "content": ev["answer"],
                                    "citations": ev["citations"]})
                st.rerun()
