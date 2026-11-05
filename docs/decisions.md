# Decision log

| # | Decision | Why |
|---|---|---|
| D1 | New repo; reuse proven parts of the earlier prototype, reviewing each on the way in | Fresh structure without repeating solved problems. The earlier prototype stays as it is |
| D2 | Keep the stack: LangGraph, LlamaIndex, Postgres, Neo4j, MLflow, Prefect, FastAPI, Streamlit, MCP. DSPy only if a phase needs it | Proven in the earlier prototype. Query rewriting never passed its gate there, so it isn't carried over |
| D3 | One agent, ABP only | One domain removes the multi-corpus and two-agent plumbing |
| D4 | Claude grades answers in a session, calibrated against hand-checked labels; the grader stays swappable | Strata and the 27B can't share the GPU. Trade-off: graded evals can't run unattended, and Claude isn't independent on keys it wrote |
| D5 | Own Compose project and host ports (API 8011, UI 8502, MLflow 5001, Prefect 4201, Neo4j 7475/7688, Postgres 5435, Ollama 11436) | Runs beside the earlier prototype |
| D6 | Eval keys carry quotes checked by code against a pinned ABP commit (`d1bf7f6f2b`) | Wrong keys cost hours in the earlier prototype (q001, q030, q041). A key can no longer rest on memory |
| D7 | Forum questions are drawn by seeded random sample, with written selection rules | Hand-picking favours easy questions. See `evals/README.md` |
| D8 | The test split is sealed by hash after human review | The earlier prototype changed test keys after seeing results. A change now has to be deliberate and visible |
| D9 | Eval cases can carry chat history | The repeated-answer bug needed history, so single-turn cases couldn't catch it |
| D10 | Agent model is qwen3.8-27b; the router uses JSON schema output | Measured before writing the agent. See `docs/model-trial.md`: JSON schema 75 of 75, forced tool call 70 of 75 |
| D11 | Don't rely on the model to pick code over docs | Both 27B models searched the docs first on all 4 code-only cases. Phase 3 and 5 must make code reachable anyway, for example one search over both sources |
