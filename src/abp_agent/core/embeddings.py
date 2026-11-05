"""The one place that knows which embedding model we use.

nomic-embed-text is trained with task prefixes. Without them retrieval quality drops, so
documents get "search_document: " and queries get "search_query: "
(see the nomic-embed-text model card).
"""

from functools import lru_cache

from llama_index.embeddings.ollama import OllamaEmbedding

from abp_agent.config import get_settings

EMBED_DIM = 768  # nomic-embed-text v1.5


@lru_cache
def get_embed_model() -> OllamaEmbedding:
    s = get_settings()
    return OllamaEmbedding(
        model_name=s.embed_model,
        base_url=s.ollama_base_url,
        query_instruction="search_query: ",
        text_instruction="search_document: ",
        embed_batch_size=32,
        keep_alive="30m",
    )


def embed_query(text: str) -> list[float]:
    return get_embed_model().get_query_embedding(text)


def embed_documents(texts: list[str]) -> list[list[float]]:
    return get_embed_model().get_text_embedding_batch(texts)
