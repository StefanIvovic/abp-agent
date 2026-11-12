"""Embeddings come from LM Studio and must never make it load a model on demand."""

import pytest

from abp_agent.core import embeddings


class _Resp:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data

    def raise_for_status(self):
        pass


def _models(state: str | None):
    data = [{"id": "qwen3.8-27b", "state": "loaded"}]
    if state:
        data.append({"id": "text-embedding-bge-m3", "state": state})
    return _Resp({"data": data})


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(embeddings, "_checked_at", 0.0)


def test_native_api_sits_beside_v1():
    assert embeddings.native_api("http://h:1234/v1") == "http://h:1234/api/v0/models"
    assert embeddings.native_api("http://h:1234/v1/") == "http://h:1234/api/v0/models"


@pytest.mark.parametrize("state, expected", [("loaded", "loaded"), ("not-loaded", "not-loaded"),
                                             (None, "missing")])
def test_state_of_the_embedding_model(monkeypatch, state, expected):
    monkeypatch.setattr(embeddings.httpx, "get", lambda *a, **k: _models(state))
    assert embeddings.embed_model_state() == expected


def test_an_unloaded_model_is_never_called(monkeypatch):
    monkeypatch.setattr(embeddings.httpx, "get", lambda *a, **k: _models("not-loaded"))

    def no_post(*a, **k):
        raise AssertionError("an embedding request would make LM Studio load the model")

    monkeypatch.setattr(embeddings.httpx.Client, "post", no_post)
    with pytest.raises(RuntimeError, match="lms load text-embedding-bge-m3"):
        embeddings.embed_query("q")


def test_batches_keep_input_order_and_check_dimensions(monkeypatch):
    monkeypatch.setattr(embeddings.httpx, "get", lambda *a, **k: _models("loaded"))
    sent = []

    def post(self, url, headers, json):
        sent.append(len(json["input"]))
        rows = [{"index": i, "embedding": [float(len(t))] * embeddings.EMBED_DIM}
                for i, t in enumerate(json["input"])]
        return _Resp({"data": rows[::-1]})          # the server may return any order

    monkeypatch.setattr(embeddings.httpx.Client, "post", post)
    texts = ["x" * n for n in range(1, 41)]
    out = embeddings.embed_documents(texts)
    assert sent == [32, 8]
    assert [v[0] for v in out] == [float(n) for n in range(1, 41)]


def test_wrong_dimensions_fail(monkeypatch):
    monkeypatch.setattr(embeddings.httpx, "get", lambda *a, **k: _models("loaded"))
    monkeypatch.setattr(embeddings.httpx.Client, "post",
                        lambda self, url, headers, json: _Resp({"data": [
                            {"index": 0, "embedding": [0.0] * 768}]}))
    with pytest.raises(RuntimeError, match="768 dimensions"):
        embeddings.embed_query("q")
