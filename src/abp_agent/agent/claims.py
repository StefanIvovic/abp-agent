"""Checks on what an answer claims. A citation proves a passage was retrieved, not that it says
what the sentence says, so these read the answer itself:

- check_names: every code name an answer claims must exist somewhere in ABP: names in inline code,
  and types that C# samples use. Plain code, no LLM, almost free. Runs on live answers.
- unseen_calls: calls in code samples must appear in what was retrieved this turn. Plain code too,
  and runs on live answers.
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

IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b")
_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE = re.compile(r"`([^`\n]+)`")
_TYPE_NAME = re.compile(r"\b(?:[A-Z][a-z0-9]+){2,}[A-Za-z0-9]*\b|\bI[A-Z][a-z]+[A-Za-z0-9]*\b")
# An exact API name in a question: abp.localization.getResource, IStringLocalizer. Not abp.io.
API_NAME = re.compile(rf"\babp\.(?!io\b)\w+(?:\.\w+)*|{_TYPE_NAME.pattern}")
_DECLARED = re.compile(r"\b(?:class|interface|record|struct|enum)\s+(\w+)")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
# A sentence saying a name doesn't exist isn't a claim that it does
_DENIAL = re.compile(r"no such|found nothing|(?:doesn't|does not) exist|there(?: is|'s) no"
                     r"|no built-in|(?:couldn't|could not) find|none of (?:those|these)|\bno `"
                     r"|not from `|(?:can't|cannot) (?:show|confirm|point)|(?:don't|do not) have",
                     re.I)
_SAMPLE = re.compile(r"```([\w#+-]*)[ \t]*\n(.*?)```", re.DOTALL)
_CSHARP = {"csharp", "cs", "c#", ""}
_CALL = re.compile(r"(?:^|[^\w])([A-Z]\w{3,})\s*(?:<[^<>()]*>)?\s*\(", re.MULTILINE)
# A method the sample itself defines: a type before the name, unlike "await Foo(" or "new Foo("
_SAMPLE_DECLARES = re.compile(r"\b(?!await\b|return\b|new\b|throw\b|typeof\b|nameof\b)"
                              r"[\w<>\[\]?,.]+\s+([A-Z]\w+)\s*(?:<[^<>()]*>)?\s*\(")
# Everyday .NET and DI calls a sample may use without a source showing them
COMMON_CALLS = frozenset({
    "AddScoped", "AddSingleton", "AddTransient", "Clear", "Configure", "Contains", "FromResult",
    "FromHours", "FromMinutes", "FromSeconds", "Insert", "MailAddress", "RemoveAll", "Replace", "Scoped", "Single", "Singleton",
    "ToDictionary", "Transient"})
_QUOTED = re.compile(r"\"[^\"]*\"")     # a string in a sample is the reader's value, not an API
_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
# A name used as a type in a sample: a generic argument, new, typeof, an attribute, a base type or
# static access. Configure<AbpDddOptions>(...) put an invented type past both checks.
_T = r"((?:[A-Z][a-z0-9]+){2,}[A-Za-z0-9]*|I[A-Z][a-z]+[A-Za-z0-9]*)"
_TYPE_USE = re.compile(rf"<\s*{_T}\b|,\s*{_T}\s*>|\bnew\s+{_T}\b|typeof\({_T}\)|\[{_T}\b"
                       rf"|:\s*{_T}\b|(?<![\w.]){_T}\.(?=[A-Z])")
_MEMBER_DECLARES = re.compile(r"\b(?:const|static|readonly)\s+\w+\s+(\w+)|\b\w+\s+(\w+)\s*\{\s*get")
_PLACEHOLDER = re.compile(r"(?:My|Your|Sample|Example)[A-Z]")   # the reader's names, not ABP's
_DOTNET_NAMESPACE = ("System.", "Microsoft.")
# .NET members answers name in prose. Not in ABP's text, so the name list can't know them.
DOTNET_NAMES = frozenset({"DbUpdateException", "ReplyTo", "ReplyToList"})

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
            names.update(IDENTIFIER.findall(body))
    return frozenset(names)


def code_names(answer: str) -> set[str]:
    """Type-like names the answer claims exist, from `inline code`. Fenced samples are left out:
    they hold the reader's own example classes. So are names it denies, the reader's placeholders
    and .NET names: warning about those teaches readers to ignore the warning."""
    found: set[str] = set()
    for sentence in _SENTENCE.split(_FENCE.sub(" ", answer)):
        if _DENIAL.search(sentence.replace("’", "'")):
            continue
        for span in _INLINE.findall(sentence):
            if not span.startswith(_DOTNET_NAMESPACE) and "*" not in span:   # * is a name pattern
                found.update(_TYPE_NAME.findall(_QUOTED.sub(" ", span)))
    return {n for n in found - set(_DECLARED.findall(answer)) - DOTNET_NAMES
            if not _PLACEHOLDER.match(n)}


def sample_types(answer: str) -> set[str]:
    """Names C# samples use as types, minus anything the answer declares in any sample: the
    reader's own classes, methods, constants and properties. Only fences marked as C#, since an
    unmarked one may be a connection string or config. Checked on 169 saved answers with code:
    4 of 5 flags were invented ABP types, the fifth the reader's example entity."""
    found: set[str] = set()
    for language, code in _SAMPLE.findall(answer):
        if language and language.lower() in _CSHARP:
            code = _QUOTED.sub(" ", _COMMENT.sub(" ", code))
            for groups in _TYPE_USE.findall(code):
                found.update(g for g in groups if g)
    declared = set(_DECLARED.findall(answer)) | set(_SAMPLE_DECLARES.findall(answer))
    for constant, prop in _MEMBER_DECLARES.findall(answer):
        declared |= {constant, prop}
    return {n for n in found - declared - DOTNET_NAMES if not _PLACEHOLDER.match(n)}


def check_names(answer: str, question: str, known: frozenset[str] | set[str]) -> list[str]:
    """Names in the answer that exist nowhere in ABP and weren't in the question: from inline code,
    and from the types C# samples use."""
    claimed = code_names(answer) | sample_types(answer)
    return sorted(claimed - set(known) - set(IDENTIFIER.findall(question)))


def unseen_calls(answer: str, seen: set[str]) -> list[str]:
    """Calls in C# samples that nothing retrieved this turn shows, so the model wrote them from
    memory. Whether a call exists somewhere isn't enough: ReplaceService does, on another type,
    and checking all of ABP found one real problem in 193 samples (D26)."""
    calls: set[str] = set()
    for language, code in _SAMPLE.findall(answer):
        if language.lower() in _CSHARP:
            calls |= set(_CALL.findall(code)) - set(_SAMPLE_DECLARES.findall(code))
            calls -= set(_DECLARED.findall(code))
    return sorted(c for c in calls - seen - COMMON_CALLS if not _PLACEHOLDER.match(c))


def read_claims(content: str) -> list[str]:
    return [c for c in json.loads(content)["unsupported"] if c.strip()]


def check_claims(question: str, answer: str, passages: list[dict]) -> list[str]:
    cited = "\n\n".join(f"[{p['chunk_id']}] {p['title']}\n{p['text']}" for p in passages)
    reply = chat_model(max_tokens=800).bind(response_format=CLAIMS_SCHEMA).invoke([
        SystemMessage(CLAIMS_PROMPT),
        HumanMessage(f"Question:\n{question}\n\nAnswer:\n{answer}\n\nCited passages:\n{cited}")])
    return read_claims(reply.content)
