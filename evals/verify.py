"""Checks the eval datasets before anyone relies on them.

    python -m evals.verify            # all datasets: format, unique ids, evidence, the test seal
    python -m evals.verify --seal     # after a review: record the hash of the test split

A quote is compared with spaces collapsed, so line wrapping can't fail a true key. The test split
is sealed by hash: after you've seen results, changing it has to be a deliberate, visible step.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from abp_agent.config import get_settings
from evals.schema import ABP_COMMIT, Case

DATASETS = Path(__file__).parent / "datasets"
SPLITS = {"dev": DATASETS / "dev.jsonl", "test": DATASETS / "test.jsonl"}
SEAL = DATASETS / "test.sha256"


def load(path: Path) -> list[Case]:
    if not path.exists():
        return []
    return [Case(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def squeeze(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def evidence_errors(case: Case, repo: Path) -> list[str]:
    errors = []
    for e in case.evidence:
        file = repo / e.path
        if not file.is_file():
            errors.append(f"{case.id}: no such file {e.path}")
            continue
        text = file.read_text(encoding="utf-8-sig", errors="replace")
        if squeeze(e.quote) not in squeeze(text):
            errors.append(f"{case.id}: quote not in {e.path}: {e.quote[:60]!r}")
    return errors


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal_errors() -> list[str]:
    if not SEAL.exists():
        return []       # not sealed yet: the split is still being reviewed
    if SEAL.read_text().strip() != digest(SPLITS["test"]):
        return ["test split changed since it was sealed"]
    return []


def check(repo: Path) -> list[str]:
    errors, seen = [], set()
    for split, path in SPLITS.items():
        for case in load(path):
            if case.split != split:
                errors.append(f"{case.id}: split is {case.split}, file is {split}")
            if case.id in seen:
                errors.append(f"{case.id}: duplicate id")
            seen.add(case.id)
            errors += evidence_errors(case, repo)
    return errors + seal_errors()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seal", action="store_true")
    args = ap.parse_args()
    errors = check(Path(get_settings().abp_repo_dir))
    for e in errors:
        print(e)
    if errors and not args.seal:
        sys.exit(1)
    cases = {s: load(p) for s, p in SPLITS.items()}
    if args.seal:
        drafts = [c.id for c in cases["test"] if c.status == "draft"]
        hard = [e for e in errors if "sealed" not in e]
        if drafts or hard or not cases["test"]:
            sys.exit(f"can't seal: {len(hard)} errors, drafts {drafts}, "
                     f"{len(cases['test'])} test cases")
        SEAL.write_text(digest(SPLITS["test"]) + "\n")
        print("test split sealed")
    print(f"ok at ABP {ABP_COMMIT}: " + ", ".join(f"{len(c)} {s}" for s, c in cases.items()))


if __name__ == "__main__":
    main()
