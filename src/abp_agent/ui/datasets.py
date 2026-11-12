"""Eval datasets page: read the .jsonl files as a table and as cases, without editing them.

Read-only on purpose: a test split is sealed by hash, so a change belongs in the file, made as a
visible step (evals/README.md).
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

DATASETS = Path(__file__).resolve().parents[3] / "evals" / "datasets"
SEALS = {"test": "test.sha256"}          # other splits use <split>.sha256


def load(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def seal_state(path: Path) -> str:
    seal = DATASETS / SEALS.get(path.stem, f"{path.stem}.sha256")
    if path.stem == "dev" or not path.stem.startswith("test"):
        return ""
    if not seal.exists():
        return "not sealed"
    same = seal.read_text().strip() == hashlib.sha256(path.read_bytes()).hexdigest()
    return "sealed" if same else "changed since it was sealed"


def bullets(title: str, items: list) -> None:
    if items:
        st.markdown(f"**{title}**")
        st.markdown("\n".join(f"- {x}" for x in items))


def show_case(c: dict) -> None:
    st.markdown(f"**Question.** {c['question']}")
    for turn in c.get("history", []):
        st.caption(f"Earlier {turn['role']}: {turn['content']}")
    for q in c.get("asked_before", []):
        st.caption(f"Asked before: {q}")
    if c.get("answer"):
        st.markdown(f"**Expected answer.** {c['answer']}")
    bullets("Must include", c.get("must_include", []))
    bullets("Must not say", c.get("must_not_say", []))
    bullets("Must match (regex)", [f"`{x}`" for x in c.get("must_match", [])])
    bullets("Must not match (regex)", [f"`{x}`" for x in c.get("must_not_match", [])])
    bullets("Expected", c.get("expected", []))
    if c.get("note"):
        st.markdown(f"**Note.** {c['note']}")
    if c.get("url"):
        st.markdown(f"**Forum thread.** {c['url']}")
    for e in c.get("evidence", []):
        st.markdown(f"**Evidence** · `{e['path']}`")
        st.code(e["quote"], language=None, wrap_lines=True)


st.title("Eval datasets")
files = sorted(DATASETS.glob("*.jsonl"))
if not files:
    st.info(f"No datasets in {DATASETS}")
    st.stop()

path = st.selectbox("Dataset", files, format_func=lambda p: p.stem)
cases = load(path)
state = seal_state(path)
st.caption(f"{len(cases)} cases" + (f" · {state}" if state else ""))

FIELDS = ("id", "status", "kind", "source", "relation", "question")
columns = [k for k in FIELDS if any(k in c for c in cases)]
table = pd.DataFrame([{k: c.get(k, "") for k in columns} for c in cases])

left, right, search = st.columns([1, 1, 2])
for col, name in ((left, "status"), (right, "kind")):
    if name in table:
        picked = col.multiselect(name.capitalize(), sorted(table[name].unique()))
        if picked:
            table = table[table[name].isin(picked)]
text = search.text_input("Search the questions")
if text:
    table = table[table["question"].str.contains(text, case=False, regex=False)]

st.dataframe(table, hide_index=True, width="stretch")

st.subheader(f"Cases ({len(table)})")
shown = set(table["id"]) if "id" in table else set()
for c in cases:
    if c.get("id") in shown:
        label = " · ".join(str(c[k]) for k in ("id", "kind", "status") if c.get(k))
        with st.expander(f"{label} — {c['question'][:90]}"):
            show_case(c)
