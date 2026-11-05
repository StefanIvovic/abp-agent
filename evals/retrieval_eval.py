"""Retrieval eval: does search find the passages that prove each key?

A chunk is relevant if it comes from an evidence file and contains that evidence's quote, so labels
survive re-chunking. Only quotes in indexed files count: `reachable` says how many that is.

    python -m evals.retrieval_eval             # dev only: test is for confirming a final choice
"""

import argparse
import statistics

import mlflow

from abp_agent.config import get_settings
from abp_agent.core.db import new_session
from abp_agent.documents.models import Document
from abp_agent.retrieval.hybrid import SearchFilters, search, search_both
from evals.schema import RESEARCH, Case, Evidence
from evals.verify import SPLITS, load, squeeze

DOCS_PREFIX = "docs/en/"
K = 10
CONFIGS = {
    "docs-only hybrid": ("hybrid", ["docs"]),
    "docs+code bm25": ("bm25", None),
    "docs+code vector": ("vector", None),
    "docs+code hybrid": ("hybrid", None),
    "docs+code balanced": ("balanced", None),
}


def stored_path(evidence_path: str) -> str:
    return evidence_path.removeprefix(DOCS_PREFIX)


def matches(evidence: Evidence, path: str, text: str) -> bool:
    return path == stored_path(evidence.path) and squeeze(evidence.quote) in squeeze(text)


def first_rank(evidence: Evidence, hits: list) -> int | None:
    return next((i + 1 for i, h in enumerate(hits) if matches(evidence, h.path, h.text)), None)


def score_case(evidence: list[Evidence], hits: list) -> dict:
    """Over the evidence that can be found. `any` metrics: at least one proof was retrieved."""
    ranks = [first_rank(e, hits) for e in evidence]
    found = [r for r in ranks if r]
    best = min(found) if found else None
    return {"hit@1": float(best == 1), "hit@5": float(bool(best and best <= 5)),
            "hit@10": float(bool(best and best <= K)), "mrr": 1 / best if best else 0.0,
            "recall@10": len(found) / len(evidence) if evidence else 0.0}


def run(cases: list[Case], indexed: set[str], mode: str, sources: list[str] | None) -> dict:
    rows = {}
    with new_session() as db:
        for c in cases:
            evidence = [e for e in c.evidence if stored_path(e.path) in indexed]
            if evidence:
                hits = (search_both(db, c.question, K) if mode == "balanced"
                        else search(db, c.question, mode, K, SearchFilters(sources)).hits)
                rows[c.id] = {"kind": c.kind, **score_case(evidence, hits)}
            db.rollback()       # SET LOCAL ends with the transaction
    return rows


def summary(rows: dict, kinds: set[str] | None = None) -> dict:
    part = [r for r in rows.values() if kinds is None or r["kind"] in kinds]
    if not part:
        return {}
    keys = ("hit@1", "hit@5", "hit@10", "mrr", "recall@10")
    return {"n": len(part), **{k: round(statistics.mean(r[k] for r in part), 3) for k in keys}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    args = ap.parse_args()
    cases = [c for c in load(SPLITS[args.split]) if c.kind in RESEARCH and c.evidence]
    with new_session() as db:
        indexed = set(db.scalars(Document.__table__.select().with_only_columns(Document.path)))
    total = sum(len(c.evidence) for c in cases)
    reachable = sum(stored_path(e.path) in indexed for c in cases for e in c.evidence)
    print(f"{len(cases)} cases, {reachable} of {total} proofs are in indexed files")
    for c in cases:
        for e in c.evidence:
            if stored_path(e.path) not in indexed:
                print(f"  not indexed: {c.id} {e.path}")
    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment("retrieval")
    for name, (mode, sources) in CONFIGS.items():
        rows = run(cases, indexed, mode, sources)
        groups = {"all": None, "code": {"code"}, "docs": {"docs"}, "both": {"docs+code"},
                  "graph": {"graph"}}
        result = {g: summary(rows, kinds) for g, kinds in groups.items()}
        with mlflow.start_run(run_name=f"{name} ({args.split})"):
            mlflow.log_params({"mode": mode, "sources": sources or "both", "split": args.split})
            mlflow.log_metrics({f"{g}_{k.replace('@', '_at_')}": v
                                for g, r in result.items() for k, v in r.items()})
            mlflow.log_dict(rows, "per_case.json")
        print(f"\n{name}")
        for g, r in result.items():
            print(f"  {g:5} {r}")


if __name__ == "__main__":
    main()
