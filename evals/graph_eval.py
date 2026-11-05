"""Graph eval: closed lists from Neo4j, compared with what search finds.

Expected lists in graph-lists.jsonl come from a plain text search of the repo, not from the parser
or Neo4j, so the graph isn't graded against itself. All are development lists: the locked test
split stays untouched. Search has no list of its own, so it's scored on recall: how many expected
names appear anywhere in its top 10 passages.

    python -m evals.graph_eval
"""

import json
import statistics
from pathlib import Path

import mlflow

from abp_agent.config import get_settings
from abp_agent.core.db import new_session
from abp_agent.graph import code_graph
from abp_agent.retrieval.hybrid import search_both

DATASET = Path(__file__).parent / "datasets" / "graph-lists.jsonl"


def score(found: set[str], expected: set[str]) -> dict:
    """Exact names: AbpCachingModule must not match AbpCachingStackExchangeRedisModule."""
    hits = len(found & expected)
    recall, precision = hits / len(expected), hits / len(found) if found else 0.0
    f1 = 2 * precision * recall / (precision + recall) if hits else 0.0
    return {"recall": recall, "precision": precision, "f1": f1}


def graph_answer(case: dict) -> set[str]:
    rows = code_graph.relation(case["name"], case["relation"])["rows"]
    return {r["package" if case["relation"] == "docs" else "name"] for r in rows}


def search_recall(db, case: dict) -> float:
    hits = search_both(db, case["question"], 10)
    text = " ".join(f"{h.path} {h.text}" for h in hits)
    return sum(e in text for e in case["expected"]) / len(case["expected"])


def main() -> None:
    cases = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line]
    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment("graph")
    rows = []
    with new_session() as db, mlflow.start_run(run_name="closed lists"):
        for c in cases:
            expected, found = set(c["expected"]), graph_answer(c)
            row = {"id": c["id"], "relation": c["relation"], **score(found, expected),
                   "search_recall": search_recall(db, c),
                   "missing": sorted(expected - found), "extra": sorted(found - expected)}
            db.rollback()
            rows.append(row)
            print(f"{c['id']} {c['relation']:15} {c['name'][:34]:34} F1 {row['f1']:.2f}  "
                  f"search recall {row['search_recall']:.2f}", flush=True)
            if row["missing"] or row["extra"]:
                print(f"     missing {row['missing'][:6]}  extra {row['extra'][:6]}")
        metrics = {k: round(statistics.fmean(r[k] for r in rows), 3)
                   for k in ("recall", "precision", "f1", "search_recall")}
        mlflow.log_metrics(metrics)
        mlflow.log_dict(rows, "per_case.json")
    print(json.dumps({"lists": len(rows), **metrics}))


if __name__ == "__main__":
    main()
