# Evals

Datasets are in `datasets/`: `dev.jsonl` to develop against, and three sealed test sets,
`test.jsonl`, `test2.jsonl` and `test3.jsonl`. A new test set is written once the last one has been
read and diagnosed, so there is always a score on questions nobody has tuned against.
The format is in `schema.py`.

```powershell
docker compose run --rm -v "${PWD}:/app" api python -m evals.verify          # check everything
docker compose run --rm -v "${PWD}:/app" api python -m evals.verify --seal   # freeze the test split
docker compose run --rm -v "${PWD}:/app" api python -m evals.verify --seal test2   # freeze the second one
docker compose run --rm -v "${PWD}:/app" api python -m evals.verify --seal test3   # and the third
```

## Rules

- **Keys prove themselves.** Research cases carry quotes that `verify` finds in the ABP repo at the
  commit pinned in `schema.py`. Keys come from the docs and source, never from forum replies.
- **Drafts aren't keys.** A case counts once a person has reviewed it (`status: reviewed`). A reviewer may
  bring in a forum reply, but it only counts once it's checked against the source.
- **The test split is sealed by hash** after review. Changing it afterwards fails `verify`, so a
  change has to be a deliberate, visible step. Never tune on it.
- **Forum text is paraphrased** and linked. No usernames or emails, no copied answers, and it's
  used for testing only, not training (abp.io allows reference use, not AI training).

## From a failure to a case

1. `python -m evals.candidates` adds live turns to `results/candidates.jsonl`: thumbs down first,
   then turns the guards acted on. It keeps the question, chat history, answer, comments and trace
   id, and running it again adds only new turns.
2. Open the trace and find out what went wrong.
3. Write the key from the docs and source, with evidence quotes, and add the case to `dev.jsonl`
   with `status: reviewed`. A comment or rating is a lead, not a key. If the bug needs real searches
   on an earlier turn, use `asked_before`: `history` is only text.
4. Run `python -m evals.verify`, then the case with `python -m evals.agent_eval --cases <id>`.

Step 1 is automatic. Steps 2 and 3 stay with a person, because a key has to be proved.

## How forum questions are chosen

1. Take threads from `https://abp.io/sitemap-support.xml` last changed since 2024-11, leaving out
   the 14 threads the earlier prototype used. That's 2,690 threads.
2. Draw 220 at random: `shuf -n 220 --random-source=<(yes 20261105)`.
3. Keep threads asking how the open-source framework behaves or how to configure it. Skip
   licensing, Studio and Suite tooling, Pro-only modules and bug reports.
4. Write a key only if the source settles it. Record the threads dropped for that reason.

Steps 3 and 4 are human judgment, so they can make the set easier than real traffic.

`test2.jsonl` is a second test set, written after the fixes `test.jsonl` led to, so `test-v1` stays
comparable. Its forum questions follow the same steps with new inputs: threads changed since
2024-11, leaving out every thread used before (2,681 left), then
`shuf -n 120 --random-source=<(yes 20261106)`, read in that order. Candidates were judged from
their titles, then read. Thread 10707 was skipped as the same topic as dev case c014. Dropped
after reading: 10290, 8103, 9286, 8119, 9255, 9285 (no settled cause or no answer), 10377 (about
templates), 10084 (UI packages), 10228 (no clear question), 8951 (the Angular code isn't in the
repository).

`test3.jsonl` is a third test set, written after the rule C study on dev (2026-11-07), for a score
on questions nobody has tuned against. Forum questions follow the same steps: threads changed since
2024-11, leaving out every thread used before and the 14 threads the earlier prototype used (2,683 left), then
`shuf -n 120 --random-source=<(yes 20261107)`, read in that order. Candidates were judged from
their titles, then read. Kept: 10131, 9235, 8311, 10646, 8925, 10096, 10425, 10630. Dropped after
reading: 9412 (the page is in a Pro module), 9079 (the source doesn't show why their change
tracking settings didn't help), 9493 (the fix is only in a community article, which isn't
indexed), 9199 (microservice template and gateway setup), 8858, 9531, 8878, 10056, 9087, 8291,
9343, 8984, 10748 (no cause the source settles), 2768 (question deleted), 8833 (Serilog, not ABP),
10193 (SaaS is a Pro module). Seven cases are written by Claude for kinds the forum rarely
covers: two graph, two abstain, two exact signatures and one injection.

## Changes to sealed keys

Each one re-sealed with `verify --seal`, after `verify` passed at the pinned ABP commit.

- **2026-11-12, pin moved to ABP 10.7.1 (`6aeb69d831`) from `dev` `d1bf7f6f2b` (D42).** 7 of the
  124 quoted files differ between the two. Re-read each key against its diff:
  - c003, c069 (a conversion helper call), c068 (a .NET 11 helper removed), c064 (a bulk `SetAsync`
    overload removed) and c025 (the file-system provider retries with Polly): what the keys claim is
    unchanged, so they stay as they were.
  - test3 c066: same answer, but the Docs module uses the NEST client in 10.7.1, so two quotes now
    use its lines (`new ConnectionSettings(node)`, `settings.BasicAuthentication(...)`).
  - test3 c073: `docs/en/low-code/deleted-objects.md` isn't in 10.7.1, so its evidence and the
    sentence about the low-code "Purge for good" were removed. The key's point, that ABP has no
    built-in purge of soft-deleted rows, still holds; "purge" appears nowhere in 10.7.1 outside blog
    posts and community articles.

  test3 re-sealed; test and test2 didn't change.

- **2026-11-12, test3 c075:** `must_not_match` was `RequiresNew\s*=`, meant to catch the
  nonexistent `[UnitOfWork(RequiresNew = true)]`. Patterns match ignoring case, so it also failed a
  correct answer that quoted the real signature `Begin(AbpUnitOfWorkOptions options, bool
  requiresNew = false)` (test3-v2-bge run 0). Now `(?-i:RequiresNew\s*=)`, case-sensitive: ABP has
  no `RequiresNew` property or attribute argument at the pinned commit or in 10.7.1, only the
  `requiresNew` parameter. Checked on every saved c075 answer: test3-v1 unchanged (3 passes), the
  wrongly failed run now passes, and both wrong forms are still caught.

`must_match` and `must_not_match` hold regexes the automatic check applies, for what a machine can decide. Keep names out of `must_not_match` on abstain cases: a good answer may name a thing to say it does not exist.

Dropped so far because the source doesn't settle them: 9276 (validation order depends on
interceptor order), 8911 (no clear root cause).
