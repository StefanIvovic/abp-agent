# Experiments

Studies run on failures before changing the agent. Raw data stays in `evals/results/` (not in git).
They use dev, test2 after test2-v1 was graded, and new cross-technology questions. The first
`test` split appears only in the false details study (section 8), which counts errors already
recorded. test3 was written after all of them. Grading is by Claude in a session using my rules (D25), so the grader isn't
independent. Each section says whether grading was blind.

## 1. Why test2 answers failed (2026-11-06)

13 failed runs on test2-v1, grouped from the traces before changing any code:

| Group | Runs |
|---|---|
| The right text never came back | 3 |
| Text came back too little or too late; the model stayed on its first idea | 3 |
| Had the text, wrote something false or too strong | 4 |
| Stated something its results didn't show | 1 |
| Went along with a prompt attack | 2 |

## 2. Right passages only

The 7 failed questions, 3 runs each, given hand-picked passages instead of searching. Same model and
prompt. Passages were picked knowing the answers, so this is a best case, not a reachable score.

| | Real runs | Right passages only |
|---|---|---|
| Main point right, 6 research questions | 11 / 18 | 18 / 18 |
| No false detail, 7 questions | 8 / 21 | about 7 / 21 |

A follow-up gave each run the exact passages of one real run plus any hand-picked passage it had
missed, shuffled, graded blind: main point 17 / 18. So extra text barely hurt; missing passages did.

**Learned:** retrieval is the big lever for the main point. False details didn't change with clean
input, so those are partly a model limit.

## 3. Search measurement

From saved traces on test2. A "proof" is a chunk holding a quote from the key's evidence.

| | Result |
|---|---|
| Received at least one proof | 17 / 21 runs |
| Received every proof | 8 / 21 |
| Proof in the first search | 7 / 21 |

Code questions were weakest: proof in the first search for docs 13 / 18, docs+code 13 / 30, code
5 / 24 (dev and test2). And 774 of 5,947 C# types had no chunk showing their declaration, so the model
couldn't see what a large class inherits.

**Changed:** declarations on every code chunk (D34) and `analyze_query` (D35).

## 4. Why plain questions miss C# chunks

Exact ranks over all C# chunks, dev only. Proofs for 6 new cross-technology questions ranked median
931 (keyword) and 633 (vector); the agent's own later queries put the same chunks mostly in the top 3.
The words of a user's question and the code's names barely overlap (median 2 shared terms).

**Learned:** a vocabulary gap, not chunking or the candidate cut-off. A simple second-hop search
(code names from the first results) didn't bridge it: 1 of 6 new proofs.

## 5. Platform fix

On 6 cross-technology questions × 3, `analyze_query` guessed Angular in 5 of 18 runs with no Angular
in the question, and those runs searched for Angular. Code rule: `abp.*` is MVC unless the user names
Angular or Blazor. Rerun on the same, already seen, questions:

| | Before | After |
|---|---|---|
| Wrong Angular platform | 5 / 18 | 0 / 18 |
| Client side found | 17 / 18 | 18 / 18 |
| Median seconds | 36 | 30.5 |

## 6. HyDE: write a guess first, then search (dropped)

One fixed prompt, 2 guesses per question, 20 dev questions with C# proofs, 4 code slots.

| Proofs found | Plain | Guess alone | Question + guess |
|---|---|---|---|
| Existing, 62 | 29% | 32% | 37% |
| New cross-technology, 12 | 0% | 17% | 17% |

39% of the names in guesses don't exist in ABP (`AbpClockModule`, `AbpAuthService`). 4 of 6 new
questions still found nothing. The agent, reading results over several searches, already found the C#
side in 14 of 18 runs.

**Dropped:** the model doesn't know ABP's internal names, so a guess can't supply them.

## 7. LLM claim checker as a gate (dropped)

The existing `check_claims`, on 25 labelled answers (13 bad, 12 good), twice each. Not blind.

| | Bad (13) | Good (12) |
|---|---|---|
| Flagged anything, either run | 13 | 7 |
| Named the recorded error | 2 clearly, 2 partly | |

It answers "is this in the cited text?", which isn't "is this wrong?". It found 0 of 5 errors in code
samples clearly.

**Kept as a review aid in evals, not a live gate** (D23).

## 8. False details study

33 recorded false details from real answers (test-v1, dev-v5, test2-v1, live runs, cross-tech v1):

| Kind | Count |
|---|---|
| False behaviour | 12 |
| Wrong attribution | 7 |
| Broken code | 6 |
| Overclaim | 3 |
| Invented API | 2 |
| Wrong shape | 2 |
| Wrong signature | 1 |

22 in prose, 11 in code samples. 24 in the main answer, 9 in optional extras. Not clustered at the
end. In 5 of 10 traced cases the true fact was in what the model received.

## 9. Instruction test, one shot

10 failed runs, each with the passages it had, 3 instructions × 3 runs, 90 answers. Known false
details graded blind. One step, not the agent loop.

| 30 answers each | A current | B evidence first | C state the limit |
|---|---|---|---|
| Known false detail came back | 11 | 7 | 5 |
| Main point kept | 23 | 27 | 24 |

Too few seen items to separate B from C. C was taken to the real agent (section 10).

## 10. Rule C in the real agent (2026-11-07)

The decision rule was written before any answer was read
(`evals/results/grades-dev-v8-plan.md`): keep C only if false answers fall by at least 5 of 99, more
questions improve than get worse, and main-point and cautious answers stay within 3. Dev, 33
questions × 3, with and without the rule, same code, graded blind.

| Research answers | Without C | With C |
|---|---|---|
| With a false detail | 18 / 84 | 15 / 85 |
| Main point right | 80 | 80 |
| Too cautious | 1 | 1 |

A drop of 3 is below the 5 set in advance. **Inconclusive:** C stays as deployed (D39), and isn't
claimed as a win. The one-shot test (section 9) looked much bigger than the real loop.

## 11. Types in code samples (2026-11-12)

test3 c065#1 wrote `Configure<AbpDddOptions>`, a type that doesn't exist, and it passed both live
checks: the name check read only inline code, and the call check reads calls, not types. Measured on
169 saved answers with code before adding a type check: 5 flags, 4 of them invented ABP types, the
fifth the reader's own example entity (D40).
