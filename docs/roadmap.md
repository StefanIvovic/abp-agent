# Roadmap

Answers ABP Framework questions from its docs **and** source code. Built phase by phase; each phase
runs and is tested before the next starts. The rule from the earlier prototype carries over: the model
makes judgment calls, code enforces what must always hold.

| Phase | Goal | Done when | Status |
|---|---|---|---|
| 0. Platform | The stack runs in Docker, beside the earlier prototype | `/health` is ok, tests and lint pass | Done |
| 1. Evals first | 30-50 real forum questions, keys checked against source, sealed test split, grading set | You reviewed the keys; the test split is frozen | Done |
| 2. Model test | Pick the agent model by measuring tool use and structured output | A short report; the choice is recorded | Done |
| 3. Sources | Ingest docs, and C# by class and method, with hybrid search | Retrieval eval beats docs-only on code questions | Done |
| 4. Graph | Modules, options, interfaces and implementations, and doc pages to classes; parsed, no LLM | Closed-list eval passes | Done |
| 5. Agent | One LangGraph agent, constrained router, code-side routing where the question type is clear | Agent eval runs, pass^3 reported | Done |
| 6. Guards | Answers held until the output guard passes; every code name checked against ABP; claim check as a review aid | Unit test: a replaced draft never reaches the client | Done |
| 7. Monitoring | MLflow traces with decision tags, feedback, failures become eval cases | The full loop can be demoed | Done |
| 8. MCP and CI | Read tools over MCP, with auth; tests and lint in CI | A client connects; CI runs on each push | Done; CI on GitHub Actions (D43) |

## Carried over from the earlier prototype

Hybrid search, guards, forced first search, pass^k, MLflow traces, Docker-only tooling, and the graph
parser. Each is copied when its phase starts and reviewed on the way in.

## Known problems from the earlier prototype, and how they were fixed

- `tool_choice="required"` isn't enforced by LM Studio, so the router could write prose and fall
  back to `research`. Fixed: the router uses JSON schema output, and failures are tagged (D10).
- Tokens streamed before the output guard, so a replaced draft flashed on screen. Fixed: research
  answers are held until the guard passes, proved by a unit test (D18, D24).
- A citation proves a passage was retrieved, not that it supports the sentence (q046). Partly
  addressed: code-name and code-call checks (D22, D26, D40) and a claim check as a review aid (D23).
  False details in prose are still the main open problem.
- Each eval case was a single turn. Fixed: cases can carry chat history and earlier real turns (D9, D30).

## After phase 8

Fresh test sets (test2, test3), failure studies (`docs/experiments.md`), code-chunk declarations,
`analyze_query`, ABP's JavaScript as a source, and rule C (D34 to D40).
