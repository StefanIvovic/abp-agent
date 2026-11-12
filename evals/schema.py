"""The eval case format. Every research case must prove its key: `evidence` holds quotes that
`evals.verify` looks up in the ABP repository at the pinned commit, so a key can't rest on memory.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ABP_COMMIT = "d1bf7f6f2b"       # every key is true of this commit; bump it only with a re-check

Kind = Literal[
    "docs",         # the docs answer it
    "code",         # only the source answers it
    "docs+code",    # needs both
    "graph",        # a closed list: dependencies, dependents, implementations
    "abstain",      # neither answers it: the agent must say so
    "off_topic",    # not about ABP
    "injection",    # tries to change or read the agent's instructions
]
RESEARCH = {"docs", "code", "docs+code", "graph", "abstain"}


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="file in the ABP repo, e.g. docs/en/framework/x.md")
    quote: str = Field(min_length=8, description="exact text from the file, spaces ignored")


class Turn(BaseModel):
    """An earlier turn of the conversation: bugs that need chat history need history in the case."""
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^c\d{3}$")
    split: Literal["dev", "test", "test2", "test3"]
    status: Literal["draft", "reviewed", "rejected"] = "draft"
    kind: Kind
    source: Literal["forum", "human", "claude"]
    url: str | None = Field(default=None, description="the forum thread it was paraphrased from")
    history: list[Turn] = []
    asked_before: list[str] = Field(
        default=[], description="questions really run first in the same chat, searches included: "
                                "`history` is only text, so it can't reproduce those bugs")
    question: str
    answer: str = Field(description="the expected answer in words: what a grader compares with")
    must_include: list[str] = Field(
        default=[], description="facts a correct answer states; a grader checks each one")
    must_not_say: list[str] = Field(
        default=[], description="wrong answers worth catching, e.g. an API that doesn't exist")
    must_match: list[str] = Field(
        default=[], description="regexes the answer must match: must_include a machine can check")
    must_not_match: list[str] = Field(
        default=[], description="regexes it must not match: must_not_say a machine can check")
    evidence: list[Evidence] = []
    note: str = ""

    @model_validator(mode="after")
    def _key_is_complete(self) -> "Case":
        if self.source == "forum" and not self.url:
            raise ValueError(f"{self.id}: a forum case needs its url")
        if self.kind in RESEARCH - {"abstain"} and not self.evidence:
            raise ValueError(f"{self.id}: a {self.kind} case needs evidence")
        if self.kind in RESEARCH and not self.must_include:
            raise ValueError(f"{self.id}: a {self.kind} case needs must_include")
        return self
