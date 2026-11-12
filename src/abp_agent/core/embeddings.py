"""The one place that knows which embedding model we use: bge-m3, served by LM Studio on the host
next to the chat model, through its OpenAI-compatible /v1/embeddings endpoint (D41).

bge-m3 needs no task prefixes, so documents and queries are embedded the same way.

LM Studio loads a model on demand when a request names one that isn't loaded (JIT), and that can
unload the chat model. So every call first checks that the embedding model is already loaded and
fails with the load command otherwise. Load it once on the CPU:
    lms load text-embedding-bge-m3 --gpu off -c 2048
"""

import time

import httpx

from abp_agent.config import get_settings

EMBED_DIM = 1024  # bge-m3
BATCH = 32
_CHECK_EVERY = 60.0  # seconds between "is it loaded?" checks
_checked_at = 0.0


def native_api(base_url: str) -> str:
    """LM Studio's own API (model states) lives beside the OpenAI-compatible /v1."""
    return base_url.rstrip("/").removesuffix("/v1") + "/api/v0/models"


def embed_model_state(timeout: float = 5) -> str:
    """'loaded', 'not-loaded', or 'missing' (not downloaded) for the configured embedding model."""
    s = get_settings()
    models = httpx.get(native_api(s.embed_base_url), timeout=timeout).json()["data"]
    for m in models:
        if m["id"] == s.embed_model:
            return m.get("state", "not-loaded")
    return "missing"


def _require_loaded() -> None:
    global _checked_at
    if time.monotonic() - _checked_at < _CHECK_EVERY:
        return
    state = embed_model_state()
    if state != "loaded":
        model = get_settings().embed_model
        raise RuntimeError(f"embedding model {model} is {state} in LM Studio; load it first: "
                           f"lms load {model} --gpu off -c 2048")
    _checked_at = time.monotonic()


def _embed(texts: list[str]) -> list[list[float]]:
    _require_loaded()
    s = get_settings()
    out: list[list[float]] = []
    with httpx.Client(timeout=120) as c:
        for i in range(0, len(texts), BATCH):
            r = c.post(f"{s.embed_base_url.rstrip('/')}/embeddings",
                       headers={"Authorization": f"Bearer {s.llm_api_key}"},
                       json={"model": s.embed_model, "input": texts[i:i + BATCH]})
            r.raise_for_status()
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            out.extend(d["embedding"] for d in data)
    if out and len(out[0]) != EMBED_DIM:
        raise RuntimeError(
            f"{s.embed_model} returned {len(out[0])} dimensions, expected {EMBED_DIM}")
    return out


def embed_query(text: str) -> list[float]:
    return _embed([text])[0]


def embed_documents(texts: list[str]) -> list[list[float]]:
    return _embed(texts)
