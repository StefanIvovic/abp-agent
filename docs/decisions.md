# Decision log

| # | Decision | Why |
|---|---|---|
| D1 | New repo; reuse proven parts of the earlier prototype, reviewing each on the way in | Fresh structure without repeating solved problems. The earlier prototype stays as it is |
| D2 | Keep the stack: LangGraph, LlamaIndex, Postgres, Neo4j, MLflow, Prefect, FastAPI, Streamlit, MCP. DSPy only if a phase needs it | Proven in the earlier prototype. Query rewriting never passed its gate there, so it isn't carried over |
| D3 | One agent, ABP only | One domain removes the multi-corpus and two-agent plumbing |
| D4 | Claude grades answers in a session, calibrated against hand-checked labels; the grader stays swappable | Strata and the 27B can't share the GPU. Trade-off: graded evals can't run unattended, and Claude isn't independent on keys it wrote |
| D5 | Own Compose project and host ports (API 8011, UI 8502, MLflow 5001, Prefect 4201, Neo4j 7475/7688, Postgres 5435, Ollama 11436) | Runs beside the earlier prototype |
