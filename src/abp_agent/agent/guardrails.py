"""Deterministic guardrails around the agent (no LLM: fast, testable, can't be talked out of).

- Input: length limit and obvious prompt-injection phrases. Topic control is left to the
  system prompt; a keyword topic filter would block legitimate questions.
- Output: every [chunk_id] the answer cites must come from a tool result of this turn.
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
    invented. Answers without any citation are allowed (e.g. "I don't know")."""
    out = OutputCheck(answer)

    def fix(m: re.Match) -> str:
        ids = [int(x) for x in re.split(r"\s*,\s*", m.group(1))]
        keep = [i for i in ids if i in retrieved]
        out.removed += [i for i in ids if i not in retrieved]
        for i in keep:
            if i not in out.cited:
                out.cited.append(i)
        return f"[{', '.join(map(str, keep))}]" if keep else ""

    out.answer = re.sub(r"[ \t]+([.,;:])", r"\1", _CITATION.sub(fix, answer)).strip()
    return out
