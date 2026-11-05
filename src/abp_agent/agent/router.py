import json
from collections.abc import Iterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from abp_agent.agent import service

router = APIRouter(prefix="/agent", tags=["agent"])


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000,
                         examples=["What is the default type for the Id property?"])
    thread_id: str | None = Field(None, description="Omit to start a new conversation")


def _sse(events: Iterator[dict]) -> StreamingResponse:
    """Server-Sent Events: one `data: {json}` line per event. The sync generator runs in a
    threadpool, so a long LLM call doesn't block the event loop."""
    def body():
        for ev in events:
            yield f"data: {json.dumps(ev, default=str)}\n\n"
    return StreamingResponse(body(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/stream")
def stream(req: ChatRequest) -> StreamingResponse:
    """Ask a question; streams progress events, then the final answer."""
    return _sse(service.stream_turn(req.message, req.thread_id))


@router.post("/invoke")
def invoke(req: ChatRequest) -> dict:
    """Ask a question and wait for the full answer."""
    try:
        return service.run_turn(req.message, req.thread_id)
    except RuntimeError as e:
        raise HTTPException(502, str(e)) from e
