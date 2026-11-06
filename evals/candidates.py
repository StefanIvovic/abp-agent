"""Failures become eval cases: collects live turns worth a case into evals/results/candidates.jsonl.

A thumbs down is the strongest signal. Turns the guards had to act on are candidates too, because
users don't rate most answers. Only collecting is automatic: a candidate joins dev.jsonl once a
person has written its key and `evals.verify` has proved it (see evals/README.md).

    python -m evals.candidates
"""

import json
from pathlib import Path

import mlflow
from langchain_core.messages import AIMessage, HumanMessage

from abp_agent.agent.assistant import get_graph
from abp_agent.core.observability import AGENT_EXPERIMENT, setup_tracing
from abp_agent.feedback import RATING

OUT = Path(__file__).parent / "results" / "candidates.jsonl"
MAX_TRACES = 1000


def reasons(tags: dict, ratings: list[dict]) -> list[str]:
    found = ["thumbs_down"] if any(r["value"] is False for r in ratings) else []
    if tags.get("answer_replaced") == "yes":
        found.append("answer_replaced")
    if tags.get("router_failed") == "yes":
        found.append("router_failed")
    if tags.get("analysis_failed") == "yes":
        found.append("analysis_failed")
    if tags.get("status") == "error":
        found.append("error")
    found += [k for k in ("citations_removed", "unverified_names", "unseen_calls")
              if tags.get(k, "0") != "0"]
    return found


def turns_before(messages: list, question: str) -> list[dict]:
    """The chat before this question, in the eval case format, without tool traffic. Bugs that
    need history can only be reproduced with it."""
    texts = [m for m in messages if isinstance(m, HumanMessage)
             or (isinstance(m, AIMessage) and not m.tool_calls and m.content)]
    asked = [i for i, m in enumerate(texts) if isinstance(m, HumanMessage)
             and m.content == question]
    if not asked:
        return []
    return [{"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
            for m in texts[:asked[-1]]]


def ratings_of(trace) -> list[dict]:
    return [{"value": a.feedback.value, "comment": a.rationale} for a in trace.info.assessments
            if a.name == RATING and a.feedback is not None]


def candidate(trace, found: list[str]) -> dict:
    tags = trace.info.tags
    root = trace.data.spans[0]
    question = (root.inputs or {}).get("message") or trace.info.request_preview
    state = get_graph().get_state({"configurable": {"thread_id": tags.get("thread_id")}}).values
    outputs = root.outputs or {}
    return {"trace_id": trace.info.trace_id, "reasons": found, "question": question,
            "history": turns_before(state.get("messages", []), question),
            "answer": outputs.get("answer") or trace.info.response_preview,
            "comments": [r["comment"] for r in ratings_of(trace) if r["comment"]],
            "guard": outputs.get("guard", {}), "prompt_version": tags.get("prompt_version"),
            "model": tags.get("model")}


def seen_ids() -> set[str]:
    if not OUT.exists():
        return set()
    return {json.loads(line)["trace_id"] for line in OUT.read_text(encoding="utf-8").splitlines()}


def main() -> None:
    setup_tracing()
    experiment = mlflow.get_experiment_by_name(AGENT_EXPERIMENT)
    traces = mlflow.search_traces(locations=[experiment.experiment_id], max_results=MAX_TRACES,
                                  filter_string="tag.environment = 'live'", return_type="list")
    seen, added = seen_ids(), []
    for t in traces:
        found = reasons(t.info.tags, ratings_of(t))
        if found and t.info.trace_id not in seen:
            added.append(candidate(t, found))
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("a", encoding="utf-8") as f:
        for c in sorted(added, key=lambda c: "thumbs_down" not in c["reasons"]):
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
            print(f"{c['trace_id']} {','.join(c['reasons'])}: {c['question'][:70]}")
    print(f"{len(traces)} live turns read, {len(added)} new candidates in {OUT}")


if __name__ == "__main__":
    main()
