"""Checks on what an answer claims. A citation proves a passage was retrieved, not that it says
what the sentence says, so these read the answer itself:

- check_names: every code name must exist somewhere in ABP. Plain code, no LLM, almost free. Runs
  on live answers.
- check_claims: one LLM call compares the answer with the passages it cites. It finds real
  problems but flagged 59 of 92 answers, many wrongly, so it only lists suspects in evals (D23).
"""

import json
import logging
import re
from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import text

from abp_agent.agent.llm import chat_model
from abp_agent.core.db import new_session

log = logging.getLogger(__name__)

_IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b")
_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE = re.compile(r"`([^`\n]+)`")
_TYPE_NAME = re.compile(r"\b(?:[A-Z][a-z0-9]+){2,}[A-Za-z0-9]*\b|\bI[A-Z][a-z]+[A-Za-z0-9]*\b")
_DECLARED = re.compile(r"\b(?:class|interface|record|struct|enum)\s+(\w+)")

CLAIMS_PROMPT = """You check an answer about the ABP Framework against the passages it cites. List only claims the passages don't support or that they contradict: a class, flow or behaviour attributed to the wrong place, a cause the passages don't state, or a fact the cited passage doesn't contain. Don't list style, general .NET knowledge, or claims the passages do support. If everything is supported, return an empty list."""

CLAIMS_SCHEMA = {"type": "json_schema", "json_schema": {
    "name": "claims", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["unsupported"],
    "properties": {"unsupported": {"type": "array", "items": {"type": "string"}}}}}}


@lru_cache
def known_names() -> frozenset[str]:
    """Every identifier in the stored docs and code."""
    names: set[str] = set()
    with new_session() as db:
        for (body,) in db.execute(text("SELECT text FROM chunk")):
            names.update(_IDENTIFIER.findall(body))
    return frozenset(names)


def code_names(answer: str) -> set[str]:
    """Type-like names in `inline code`. Fenced samples are left out: they hold the reader's own
    example classes, and a name the answer declares there isn't a claim about ABP."""
    prose = _FENCE.sub(" ", answer)
    found = {n for span in _INLINE.findall(prose) for n in _TYPE_NAME.findall(span)}
    return found - set(_DECLARED.findall(answer))


def check_names(answer: str, question: str, known: frozenset[str] | set[str]) -> list[str]:
    """Names in the answer that exist nowhere in ABP and weren't in the question."""
    return sorted(code_names(answer) - set(known) - set(_IDENTIFIER.findall(question)))


def read_claims(content: str) -> list[str]:
    return [c for c in json.loads(content)["unsupported"] if c.strip()]


def check_claims(question: str, answer: str, passages: list[dict]) -> list[str]:
    cited = "\n\n".join(f"[{p['chunk_id']}] {p['title']}\n{p['text']}" for p in passages)
    reply = chat_model(max_tokens=800).bind(response_format=CLAIMS_SCHEMA).invoke([
        SystemMessage(CLAIMS_PROMPT),
        HumanMessage(f"Question:\n{question}\n\nAnswer:\n{answer}\n\nCited passages:\n{cited}")])
    return read_claims(reply.content)
