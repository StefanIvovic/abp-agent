"""Running agent turns: one event stream for the SSE endpoint, the invoke endpoint, the UI and the
evals, so all of them see the same behaviour.

Events (dicts with a "type"):
  start       {thread_id}
  tool_call   {name, args}
  tool_result {name, sources}
  done        {answer, citations, steps, guard, latency_ms, thread_id, trace_id}
  error       {message}

Answers aren't streamed: the output guard can replace them, and a user must not see a draft
that's about to be withdrawn. They arrive whole in "done".
"""

import logging
import queue
import threading
import time
import uuid
from collections.abc import Iterator

import mlflow
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from mlflow.entities import SpanType

from abp_agent.agent.assistant import SYSTEM_PROMPT, get_graph
from abp_agent.agent.tools import TOOLS
from abp_agent.core.observability import setup_tracing, turn_tags, version_tags

log = logging.getLogger(__name__)

_END = object()
PREVIEW_CHARS = 1000    # MLflow stores previews as varchar(1000): a longer one fails the insert


def stream_turn(message: str, thread_id: str | None = None,
                history: list[BaseMessage] | None = None,
                environment: str = "live") -> Iterator[dict]:
    """One turn. `history` seeds a new thread with earlier turns (eval cases with chat history).

    The turn runs in its own thread, which holds the root span from start to end, and events reach
    the caller through a queue. The SSE endpoint pulls this generator from a thread pool, one item
    per call, so a span opened here would lose its context between items."""
    setup_tracing()
    thread_id = thread_id or str(uuid.uuid4())
    events: queue.Queue = queue.Queue()

    def work() -> None:
        try:
            for ev in _traced_turn(message, thread_id, history, environment):
                events.put(ev)
        except Exception as e:  # noqa: BLE001 - never leave the caller waiting
            log.exception("agent turn failed")
            events.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            events.put(_END)

    threading.Thread(target=work, name="agent-turn", daemon=True).start()
    while (ev := events.get()) is not _END:
        yield ev


def _traced_turn(message: str, thread_id: str, history: list[BaseMessage] | None,
                 environment: str) -> Iterator[dict]:
    """The turn inside a root span, so the trace list reads like a chat log. "done" is only sent
    after the span ends, so a caller that stops there can't cut the trace short."""
    final: dict | None = None
    with mlflow.start_span(name="abp", span_type=SpanType.AGENT) as span:
        span.set_inputs({"message": message, "thread_id": thread_id})
        for ev in _run_graph(message, thread_id, history, span.trace_id):
            if ev["type"] in ("done", "error"):
                final = ev
            else:
                yield ev
        done = final if final and final["type"] == "done" else None
        tags = {**version_tags(SYSTEM_PROMPT), "thread_id": thread_id}
        if done:
            tags |= turn_tags(done, {t.name for t in TOOLS}, environment)
            answer = done["answer"] or ""
            span.set_outputs({"answer": answer, "guard": done["guard"],
                              "citations": [c.get("chunk_id") for c in done["citations"]]})
        else:
            tags |= {"environment": environment, "status": "error"}
            answer = (final or {}).get("message", "")
            span.set_outputs({"error": answer})
        mlflow.update_current_trace(tags=tags, session_id=thread_id,
                                    request_preview=message[:PREVIEW_CHARS],
                                    response_preview=answer[:PREVIEW_CHARS],
                                    state="ERROR" if done is None else None)
    if final:
        yield final


def _run_graph(message: str, thread_id: str, history: list[BaseMessage] | None,
               trace_id: str | None) -> Iterator[dict]:
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}
    graph = get_graph()
    payload = {"messages": [*(history or []), HumanMessage(message)]}
    yield {"type": "start", "thread_id": thread_id}
    t0 = time.perf_counter()
    steps: list[dict] = []
    try:
        for data in graph.stream(payload, config, stream_mode="updates"):
            for node, update in data.items():
                if not update:
                    continue
                if node == "agent":
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
    yield {"type": "done", "thread_id": thread_id, "trace_id": trace_id,
           "answer": state["messages"][-1].content, "citations": state.get("citations", []),
           "steps": steps, "guard": state.get("guard", {}),
           "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


def run_turn(message: str, thread_id: str | None = None,
             history: list[BaseMessage] | None = None, environment: str = "live") -> dict:
    """Non-streaming: the final "done" event, or raises on "error"."""
    for ev in stream_turn(message, thread_id, history, environment):
        if ev["type"] == "error":
            raise RuntimeError(ev["message"])
        if ev["type"] == "done":
            return ev
    raise RuntimeError("agent stream ended without a result")
