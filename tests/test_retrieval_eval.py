from types import SimpleNamespace

from abp_agent.retrieval.hybrid import interleave
from evals import retrieval_eval as ev
from evals.schema import Evidence

PROOF = Evidence(path="docs/en/framework/a.md", quote="the exact   text")
OTHER = Evidence(path="framework/src/B.cs", quote="class B : IThing")


def hit(path: str, text: str) -> SimpleNamespace:
    return SimpleNamespace(path=path, text=text)


def test_docs_paths_are_stored_without_their_prefix():
    assert ev.stored_path("docs/en/framework/a.md") == "framework/a.md"
    assert ev.stored_path("framework/src/B.cs") == "framework/src/B.cs"


def test_a_hit_needs_the_right_file_and_the_quote():
    assert ev.matches(PROOF, "framework/a.md", "has the exact\ntext in it")
    assert not ev.matches(PROOF, "framework/other.md", "the exact text")
    assert not ev.matches(PROOF, "framework/a.md", "something else")


def test_case_scores():
    hits = [hit("x.md", "no"), hit("framework/a.md", "the exact text")]
    out = ev.score_case([PROOF, OTHER], hits)
    assert out["hit@1"] == 0.0 and out["hit@5"] == 1.0
    assert out["mrr"] == 0.5 and out["recall@10"] == 0.5


def test_nothing_found_scores_zero():
    assert ev.score_case([PROOF], [hit("x.md", "no")])["mrr"] == 0.0


def test_interleave_alternates_and_keeps_leftovers():
    assert interleave(["d1", "d2", "d3"], ["c1"]) == ["d1", "c1", "d2", "d3"]
    assert interleave([], []) == []
