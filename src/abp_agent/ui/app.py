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


def send_rating(trace_id: str, thumbs_up: bool, comment: str | None = None) -> None:
    try:
        httpx.post(f"{API}/feedback", timeout=30, json={
            "trace_id": trace_id, "thumbs_up": thumbs_up, "comment": comment}).raise_for_status()
        st.toast("Thanks, saved")
    except httpx.HTTPError as e:
        st.toast(f"Could not save that: {e}")


def rate(trace_id: str, key: str) -> None:
    if ss.get(key) is not None:
        send_rating(trace_id, bool(ss[key]))


def comment(trace_id: str, key: str) -> None:
    if ss.get(key):
        send_rating(trace_id, False, ss[key])


def show_rating(i: int, m: dict) -> None:
    rating = f"rating_{i}"
    st.feedback("thumbs", key=rating, on_change=rate, args=(m["trace_id"], rating))
    if ss.get(rating) == 0:
        note = f"comment_{i}"
        st.text_input("What was wrong?", key=note, on_change=comment, args=(m["trace_id"], note))


for i, m in enumerate(ss.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        show_sources(m.get("citations", []))
        if m.get("trace_id"):
            show_rating(i, m)

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
                                    "citations": ev["citations"], "trace_id": ev["trace_id"]})
                st.rerun()
