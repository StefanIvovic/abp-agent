# Model trial (2026-11-05)

`python -m evals.model_trial`: 16 dev cases and 9 probes, 3 repeats, temperature 0.2, on one
RTX 3090. The test split wasn't used. Raw results stay in `evals/results/` (not in git).

| Check | qwen3.8-27b | qwen3.6-27b | Strata (qwen3.8-flash iq2_xs) |
|---|---|---|---|
| Router as forced tool call: correct (75 calls) | 0.93 | 1.00 | 1.00 |
| Router as JSON schema: correct (75 calls) | **1.00** | 1.00 | 0.96 (3 server errors) |
| Router median seconds, JSON schema | 0.84 | 1.03 | 1.51 |
| First tool: made a call (39 calls) | 1.00 | 1.00 | 1.00 |
| First tool: right tool | 0.69 | 0.69 | 0.15 |
| First tool: wrote text before the call | **0.00** | 0.31 | 0.00 |
| First tool median seconds | 1.40 | 1.51 | 3.20 |

**Choice: qwen3.8-27b with a JSON schema router.**

- The router fix works: the forced tool call failed 5 of 75 times on this model (no tool call at
  all), and JSON schema failed 0 of 75.
- qwen3.8-27b writes no draft before a tool call, unlike qwen3.6-27b, and it's slightly faster.
- Strata called `code_graph` on nearly everything.

**Limits.** This tests short single steps, not whole answers. The sets are small, so the 27B models
are close. Both 27B models missed the same 12 calls: 4 code cases, 3 repeats each, where they
searched the docs first. That's a reasonable move, since a model can't know the docs don't answer
it, so the agent has to reach code some other way (see D11).
