from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from abp_agent import feedback
from abp_agent.core import observability
from abp_agent.main import app
from evals import candidates


def done(**guard) -> dict:
    return {"guard": {"route": "research", **guard}, "citations": [{}, {}],
            "steps": [{"name": "search"}, {"name": "made_up"}]}


def test_tags_show_what_the_agent_decided():
    tags = observability.turn_tags(done(citations_removed=[9], unseen_calls=["A", "B"],
                                        first_call_retried=True, tool_rounds=2),
                                   {"search"}, "live")
    assert tags["route"] == "research" and tags["environment"] == "live"
    assert tags["citations"] == "2" and tags["citations_removed"] == "1"
    assert tags["unseen_calls"] == "2" and tags["first_call_retried"] == "yes"
    assert tags["invalid_tools"] == "made_up" and tags["tool_calls"] == "2"


def test_a_quiet_turn_has_nothing_to_flag():
    tags = observability.turn_tags(done(), {"search", "made_up"}, "live")
    assert candidates.reasons(tags, []) == []


def test_tag_values_are_all_strings():
    assert all(isinstance(v, str) for v in observability.turn_tags({}, set(), "eval").values())


def test_the_prompt_version_follows_the_prompt():
    assert observability.prompt_version("a") == observability.prompt_version("a")
    assert observability.prompt_version("a") != observability.prompt_version("b")


@pytest.mark.parametrize(("tags", "ratings", "expected"), [
    ({}, [{"value": False}], ["thumbs_down"]),
    ({}, [{"value": True}], []),
    ({"answer_replaced": "yes"}, [], ["answer_replaced"]),
    ({"router_failed": "yes"}, [], ["router_failed"]),
    ({"analysis_failed": "yes"}, [], ["analysis_failed"]),
    ({"status": "error"}, [], ["error"]),
    ({"citations_removed": "1", "unseen_calls": "2"}, [], ["citations_removed", "unseen_calls"]),
    ({"answer_replaced": "yes"}, [{"value": False}], ["thumbs_down", "answer_replaced"]),
])
def test_what_makes_a_turn_a_candidate(tags, ratings, expected):
    assert candidates.reasons(tags, ratings) == expected


def test_a_candidate_keeps_the_chat_before_it_without_tool_traffic():
    call = AIMessage("draft", tool_calls=[{"name": "search", "args": {}, "id": "1"}])
    messages = [HumanMessage("first"), call, AIMessage("answer"), HumanMessage("second"),
                AIMessage("later"), HumanMessage("first")]
    assert candidates.turns_before(messages[:5], "second") == [
        {"role": "user", "content": "first"}, {"role": "assistant", "content": "answer"}]
    assert len(candidates.turns_before(messages, "first")) == 4
    assert candidates.turns_before(messages, "never asked") == []


def test_ratings_are_read_from_the_trace():
    def assessment(name, value, rationale):
        return SimpleNamespace(name=name, feedback=SimpleNamespace(value=value),
                               rationale=rationale)
    info = SimpleNamespace(assessments=[assessment(feedback.RATING, False, "wrong class"),
                                        assessment("other", True, None)])
    assert candidates.ratings_of(SimpleNamespace(info=info)) == [
        {"value": False, "comment": "wrong class"}]


def test_a_rating_is_stored_on_the_trace(monkeypatch):
    saved = {}
    monkeypatch.setattr(feedback, "setup_tracing", lambda: True)
    monkeypatch.setattr(feedback.mlflow, "log_feedback", lambda **kw: saved.update(kw))
    r = TestClient(app).post("/feedback", json={"trace_id": "tr-1", "thumbs_up": False,
                                                "comment": "wrong class"})
    assert r.status_code == 201
    assert saved["trace_id"] == "tr-1" and saved["value"] is False
    assert saved["rationale"] == "wrong class" and saved["name"] == feedback.RATING


def test_rating_an_unknown_trace_is_a_404(monkeypatch):
    def boom(**_):
        raise ValueError("no such trace")
    monkeypatch.setattr(feedback, "setup_tracing", lambda: True)
    monkeypatch.setattr(feedback.mlflow, "log_feedback", boom)
    r = TestClient(app).post("/feedback", json={"trace_id": "nope", "thumbs_up": True})
    assert r.status_code == 404


def test_the_analysis_is_filterable_and_a_failure_shows():
    tags = observability.turn_tags(done(
        analysis={"technology": "javascript", "platform": "mvc", "intent": "why",
                  "identifiers": ["abp.ajax"]}), set(), "live")
    assert (tags["technology"], tags["platform"], tags["intent"]) == ("javascript", "mvc", "why")
    assert tags["identifiers"] == "1" and tags["analysis_failed"] == "no"
    failed = observability.turn_tags(done(analysis_failed=True), set(), "live")
    assert failed["analysis_failed"] == "yes" and failed["technology"] == "none"
