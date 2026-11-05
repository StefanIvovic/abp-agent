"""The chat model: any OpenAI-compatible server (LM Studio by default)."""

from langchain_openai import ChatOpenAI

from abp_agent.config import get_settings


def chat_model(*, max_tokens: int = 2048) -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(model=s.llm_model, base_url=s.llm_base_url, api_key=s.llm_api_key,
                      temperature=0.2, max_tokens=max_tokens, timeout=300, max_retries=1,
                      extra_body={"reasoning_effort": "none"})
