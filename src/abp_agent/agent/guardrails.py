"""Deterministic guardrails around the agent (no LLM: fast, testable, can't be talked out of).

- Input: length limit and obvious prompt-injection phrases. Topic control is left to the
  system prompt; a keyword topic filter would block legitimate questions.
- Output: every [chunk_id] the answer cites must come from a tool result the model can still see.
  Invented ids are removed and counted, so the eval and the traces can measure it.
"""

import re
from dataclasses import dataclass, field

MAX_INPUT_CHARS = 2000

_INJECTION = re.compile(
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|rules)"
    r"|disregard (your|the) (instructions|rules|system prompt)"
    r"|(reveal|print|show) (me )?(your|the) (system )?prompt"
    r"|you are now (?!looking)",
    re.IGNORECASE,
)
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_CODE = re.compile(r"(`{3,}).*?(?:\1`*|\Z)|`[^`\n]+`", re.DOTALL)   # a fence, or inline code
# Tool-call syntax the model leaves after its answer. Only at the very end: the same tags can be
# legitimate inside an answer's code sample.
_TRAILING_TOOL_MARKUP = re.compile(r"(?:\s*</?(?:parameter|function|tool_call)\b[^>\n]*>)+\s*\Z")


def strip_tool_markup(answer: str) -> str:
    return _TRAILING_TOOL_MARKUP.sub("", answer)


@dataclass
class InputCheck:
    ok: bool
    reason: str | None = None
    reply: str | None = None


DEFAULT_REFUSAL = "I can't help with that request."


def check_input(message: str, refusal: str = DEFAULT_REFUSAL) -> InputCheck:
    """refusal: what the agent says when it blocks a prompt injection (each agent has its own
    scope, e.g. "only 10-K filings" vs "only ABP development")."""
    if not message.strip():
        return InputCheck(False, "empty", "Please ask a question.")
    if len(message) > MAX_INPUT_CHARS:
        return InputCheck(False, "too_long",
                          f"Please keep questions under {MAX_INPUT_CHARS} characters.")
    if _INJECTION.search(message):
        return InputCheck(False, "prompt_injection", refusal)
    return InputCheck(True)


@dataclass
class OutputCheck:
    answer: str
    cited: list[int] = field(default_factory=list)      # valid ids, in order of first use
    removed: list[int] = field(default_factory=list)    # cited but never retrieved


def check_citations(answer: str, retrieved: set[int]) -> OutputCheck:
    """Keep citations of retrieved chunks; drop the rest. "[12, 99]" keeps "[12]" if 99 is
    invented. Answers without any citation are allowed (e.g. "I don't know").

    Code is never changed: `data[0]` is an index, not a citation, and spacing matters there. A real
    id in code, such as a `// [123]` comment, still counts as cited. An invented one stays in the
    code uncounted, because it can't be told from an index."""
    out = OutputCheck(answer)

    def ids_in(m: re.Match) -> list[int]:
        return [int(x) for x in re.split(r"\s*,\s*", m.group(1))]

    def cite(ids: list[int]) -> None:
        out.cited += [i for i in dict.fromkeys(ids) if i not in out.cited]

    def fix(m: re.Match) -> str:
        ids = ids_in(m)
        keep = [i for i in ids if i in retrieved]
        out.removed += [i for i in ids if i not in retrieved]
        cite(keep)
        return f"[{', '.join(map(str, keep))}]" if keep else ""

    def prose(text: str) -> str:
        return re.sub(r"[ \t]+([.,;:])", r"\1", _CITATION.sub(fix, text))

    parts, at = [], 0
    for code in _CODE.finditer(answer):
        parts.append(prose(answer[at:code.start()]))
        for m in _CITATION.finditer(code.group()):
            cite([i for i in ids_in(m) if i in retrieved])
        parts.append(code.group())
        at = code.end()
    parts.append(prose(answer[at:]))
    out.answer = "".join(parts).strip()
    return out
