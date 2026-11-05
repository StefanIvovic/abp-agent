"""Running agent turns: one event stream for the SSE endpoint, the invoke endpoint, the UI and the
evals, so all of them see the same behaviour.

Events (dicts with a "type"):
  start       {thread_id}
  token       {text}             answer text, chit-chat and off-topic turns only
  tool_call   {name, args}
  tool_result {name, sources}
  done        {answer, citations, steps, guard, latency_ms, thread_id}
  error       {message}

Research answers aren't streamed: the output guard can replace them, and a user must not see a
draft that's about to be withdrawn. They arrive whole in "done".
"""

import logging
import time
import uuid
from collections.abc import Iterator

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)

from abp_agent.agent.assistant import get_graph

log = logging.getLogger(__name__)


def may_stream(route: str | None) -> bool:
    """Only turns the output guard can't replace. Unknown route: hold it back."""
    return route in ("chitchat", "off_topic")


def stream_turn(message: str, thread_id: str | None = None,
                history: list[BaseMessage] | None = None) -> Iterator[dict]:
    """One turn. `history` seeds a new thread with earlier turns (eval cases with chat history)."""
    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}
    graph = get_graph()
    payload = {"messages": [*(history or []), HumanMessage(message)]}
    yield {"type": "start", "thread_id": thread_id}
    t0 = time.perf_counter()
    steps: list[dict] = []
    route = None
    try:
        for mode, data in graph.stream(payload, config, stream_mode=["messages", "updates"]):
            if mode == "messages":
                chunk, meta = data
                if (meta.get("langgraph_node") == "agent" and may_stream(route)
                        and isinstance(chunk, AIMessageChunk) and chunk.content):
                    yield {"type": "token", "text": chunk.content}
                continue
            for node, update in data.items():
                if not update:
                    continue
                if node == "route":
                    route = update["guard"]["route"]
                elif node == "agent":
                    for m in update.get("messages", []):
                        if isinstance(m, AIMessage):
                            for c in m.tool_calls:
                                steps.append({"name": c["name"], "args": c["args"]})
                                yield {"type": "tool_call", "name": c["name"], "args": c["args"]}
                elif node == "tools":
                    for m in update.get("messages", []):
                        if isinstance(m, ToolMessage):
                            n = len(m.artifact) if isinstance(m.artifact, list) else 0
                            for s in reversed(steps):
                                if s["name"] == m.name and "sources" not in s:
                                    s["sources"] = n
                                    break
                            yield {"type": "tool_result", "name": m.name, "sources": n}
    except Exception as e:  # noqa: BLE001 - the caller gets an event, not a broken stream
        log.exception("agent run failed")
        yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
        return
    state = graph.get_state(config).values
    yield {"type": "done", "thread_id": thread_id, "answer": state["messages"][-1].content,
           "citations": state.get("citations", []), "steps": steps,
           "guard": state.get("guard", {}),
           "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


def run_turn(message: str, thread_id: str | None = None,
             history: list[BaseMessage] | None = None) -> dict:
    """Non-streaming: the final "done" event, or raises on "error"."""
    for ev in stream_turn(message, thread_id, history):
        if ev["type"] == "error":
            raise RuntimeError(ev["message"])
        if ev["type"] == "done":
            return ev
    raise RuntimeError("agent stream ended without a result")
