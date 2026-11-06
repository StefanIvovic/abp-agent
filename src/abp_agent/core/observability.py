"""MLflow tracing: every agent turn is one trace, with a span for each node, model call and tool.
Tags hold what the agent decided and which versions decided it, so "every answer that was replaced"
is one filter, and a rating leads straight to the full run.
"""

import hashlib
import logging
from functools import cache

import mlflow

from abp_agent import __version__
from abp_agent.config import get_settings

log = logging.getLogger(__name__)

AGENT_EXPERIMENT = "agent"


@cache
def setup_tracing() -> bool:
    """Once per process. Tracing must never take the app down, so a failure only logs. Traces go
    to the active experiment, and evals switch to their own afterwards."""
    try:
        mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
        mlflow.set_experiment(AGENT_EXPERIMENT)
        mlflow.langchain.autolog()
        return True
    except Exception:  # noqa: BLE001
        log.exception("MLflow tracing disabled")
        return False


def prompt_version(prompt: str) -> str:
    return "p-" + hashlib.sha256(prompt.encode()).hexdigest()[:10]


def version_tags(prompt: str) -> dict[str, str]:
    return {"app_version": __version__, "prompt_version": prompt_version(prompt),
            "model": get_settings().llm_model}


def turn_tags(done: dict, known_tools: set[str], environment: str) -> dict[str, str]:
    """What the agent decided in a turn. Details stay in the spans; tags make them filterable."""
    guard = done.get("guard") or {}
    tools = [s["name"] for s in done.get("steps", [])]
    analysis = guard.get("analysis") or {}
    return {
        "environment": environment,
        "status": "done",
        "technology": analysis.get("technology") or "none",
        "platform": analysis.get("platform") or "none",
        "intent": analysis.get("intent") or "none",
        "identifiers": str(len(analysis.get("identifiers") or [])),
        "analysis_failed": "yes" if guard.get("analysis_failed") else "no",
        "route": guard.get("route") or "none",
        "input_blocked": guard.get("input_blocked") or "no",
        "router_failed": "yes" if guard.get("router_failed") else "no",
        "first_call_retried": "yes" if guard.get("first_call_retried") else "no",
        "answer_replaced": "yes" if guard.get("uncited_answer_replaced") else "no",
        "citations": str(len(done.get("citations") or [])),
        "citations_removed": str(len(guard.get("citations_removed") or [])),
        "unverified_names": str(len(guard.get("unverified_names") or [])),
        "unseen_calls": str(len(guard.get("unseen_calls") or [])),
        "tool_rounds": str(guard.get("tool_rounds", 0)),
        "tool_calls": str(len(tools)),
        "invalid_tools": ",".join(t for t in tools if t not in known_tools) or "none",
    }
