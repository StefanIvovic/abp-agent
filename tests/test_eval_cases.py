import pytest
from pydantic import ValidationError

from evals import verify
from evals.schema import Case

GOOD = {"id": "c001", "split": "dev", "kind": "code", "source": "forum", "url": "https://x",
        "question": "q", "answer": "a", "must_include": ["fact"],
        "evidence": [{"path": "src/A.cs", "quote": "class AbpThing"}]}


def write_splits(tmp_path, monkeypatch, test_rows=""):
    dev, test = tmp_path / "dev.jsonl", tmp_path / "test.jsonl"
    dev.write_text("")
    test.write_text(test_rows)
    monkeypatch.setattr(verify, "SPLITS", {"dev": dev, "test": test})
    monkeypatch.setattr(verify, "SEAL", tmp_path / "test.sha256")
    return test


@pytest.mark.parametrize("change", [
    {"evidence": []},                       # a key has to prove itself
    {"must_include": []},                   # a grader needs facts to check
    {"url": None},                          # a forum case links its thread
    {"id": "q1"},
])
def test_incomplete_keys_are_rejected(change):
    with pytest.raises(ValidationError):
        Case(**{**GOOD, **change})


def test_an_abstain_case_needs_no_evidence():
    assert Case(**{**GOOD, "kind": "abstain", "evidence": []}).kind == "abstain"


def test_a_quote_survives_line_wrapping_but_not_a_wrong_name(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "A.cs").write_text("public\n    class   AbpThing : IThing")
    assert verify.evidence_errors(Case(**GOOD), tmp_path) == []
    wrong = {**GOOD, "evidence": [{"path": "src/A.cs", "quote": "class AbpOther"}]}
    assert "quote not in" in verify.evidence_errors(Case(**wrong), tmp_path)[0]
    gone = {**GOOD, "evidence": [{"path": "src/B.cs", "quote": "class AbpThing"}]}
    assert "no such file" in verify.evidence_errors(Case(**gone), tmp_path)[0]


def test_changing_a_sealed_test_split_is_caught(tmp_path, monkeypatch):
    test = write_splits(tmp_path, monkeypatch, "row\n")
    assert verify.seal_errors() == []       # not sealed yet
    verify.SEAL.write_text(verify.digest(test) + "\n")
    assert verify.seal_errors() == []
    test.write_text("row changed\n")
    assert verify.seal_errors() == ["test split changed since it was sealed"]
