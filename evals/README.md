# Evals

Datasets are in `datasets/`: `dev.jsonl` to develop against, `test.jsonl` to confirm a change.
The format is in `schema.py`.

```powershell
docker compose run --rm -v "${PWD}:/app" api python -m evals.verify          # check everything
docker compose run --rm -v "${PWD}:/app" api python -m evals.verify --seal   # freeze the test split
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
   with `status: reviewed`. A comment or rating is a lead, not a key.
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

`must_match` and `must_not_match` hold regexes the automatic check applies, for what a machine can decide. Keep names out of `must_not_match` on abstain cases: a good answer may name a thing to say it does not exist.

Dropped so far because the source doesn't settle them: 9276 (validation order depends on
interceptor order), 8911 (no clear root cause).
