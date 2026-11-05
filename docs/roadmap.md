# Roadmap

Answers ABP Framework questions from its docs **and** source code. Built phase by phase; each phase
runs and is tested before the next starts. The rule from the earlier prototype carries over: the model
makes judgment calls, code enforces what must always hold.

| Phase | Goal | Done when |
|---|---|---|
| 0. Platform | The stack runs in Docker, beside the earlier prototype | `/health` is ok, tests and lint pass |
| 1. Evals first | 30-50 real forum questions, keys checked against source, sealed test split, grading set | You reviewed the keys; the test split is frozen |
| 2. Model test | Pick the agent model by measuring tool use and structured output | A short report; the choice is recorded |
| 3. Sources | Ingest docs, and C# by class and method, with hybrid search | Retrieval eval beats docs-only on code questions |
| 4. Graph | Modules, options, interfaces and implementations, and doc pages to classes; parsed, no LLM | Closed-list eval passes |
| 5. Agent | One LangGraph agent, constrained router, code-side routing where the question type is clear | Agent eval runs, pass^3 reported |
| 6. Guards | Answers held until the output guard passes; every code name checked against ABP; claim check as a review aid | Unit test: a replaced draft never reaches the client |
| 7. Monitoring | MLflow traces with decision tags, feedback, failures become eval cases | The full loop can be demoed |
| 8. MCP and CI | Read tools over MCP, with auth; tests and lint in CI | A client connects; CI runs on each push |

## Carried over from the earlier prototype

Hybrid search, guards, forced first search, pass^k, MLflow traces, Docker-only tooling, and the graph
parser. Each is copied when its phase starts and reviewed on the way in.

## Known problems to fix on the way in

- `tool_choice="required"` isn't enforced by LM Studio: the router can write prose and silently
  fall back to `research`. Use JSON schema output, and tag real failures.
- Tokens stream before the output guard, so a replaced draft flashes on screen.
- A citation proves a passage was retrieved, not that it supports the sentence (q046).
- Each eval case is a single turn, so bugs that need chat history can't show up. Add multi-turn cases.
