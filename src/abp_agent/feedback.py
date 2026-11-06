"""User ratings, stored on the MLflow trace of the answer they rate, so a bad rating leads straight
to the full run: prompt, tool calls, passages and versions. `evals.candidates` collects them.
"""

import mlflow
from fastapi import APIRouter, HTTPException
from mlflow.entities import AssessmentSource, AssessmentSourceType
from pydantic import BaseModel, Field

from abp_agent.core.observability import setup_tracing

RATING = "user_thumbs"

router = APIRouter(prefix="/feedback", tags=["feedback"])


class FeedbackRequest(BaseModel):
    trace_id: str = Field(min_length=1, description="From the `done` event of the answer")
    thumbs_up: bool
    comment: str | None = Field(None, max_length=2000, description="What was wrong, in your words")


@router.post("", status_code=201)
def feedback(req: FeedbackRequest) -> dict:
    """Rate an answer."""
    setup_tracing()
    try:
        mlflow.log_feedback(
            trace_id=req.trace_id, name=RATING, value=req.thumbs_up, rationale=req.comment,
            source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="chat-user"))
    except Exception as e:  # noqa: BLE001 - usually an unknown trace id
        raise HTTPException(404, f"no trace to rate: {req.trace_id}") from e
    return {"trace_id": req.trace_id, "recorded": True}
