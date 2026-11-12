# ABP Agent

Answers ABP Framework questions from its docs and source code, with citations. Built evals first:
real forum questions, answer keys proved against the ABP source, and sealed test sets.

| Doc | What's in it |
|---|---|
| [docs/decisions.md](docs/decisions.md) | Every design decision, with the measurement behind it |
| [docs/experiments.md](docs/experiments.md) | Studies run on failures: what was tried, kept and dropped |
| [docs/model-trial.md](docs/model-trial.md) | How the agent model was picked |
| [docs/roadmap.md](docs/roadmap.md) | The build phases |
| [evals/README.md](evals/README.md) | Eval rules, how questions are chosen, from a failure to an eval case |

## How it works

**Sources**, loaded by a Prefect flow into Postgres with BM25 (pg_search) and vectors (pgvector):
the ABP docs (4,842 chunks), the C# source parsed with tree-sitter by type and member (7,714
chunks), and ABP's own JavaScript. A **code graph** in Neo4j is parsed from the C#, with no LLM:
modules, dependencies, inheritance, interfaces and implementations, options, and the doc pages
naming each type.

Each turn is a LangGraph graph:

1. **Input guard** (code): length limit and obvious prompt-injection phrases.
2. **analyze_query**: code reads exact names; the model picks technology, platform and intent from
   fixed lists (JSON schema). The result is a hint for the turn.
3. **Router** (model, JSON schema): research, chit-chat or off-topic. Only research reaches the
   model; the others get fixed text.
4. **Agent loop** (qwen3.8-27b): calls `search` (docs and code ranked separately, then interleaved)
   and `code_graph` (exact relations) until it can answer with citations.
5. **Output guard** (code), before anything is shown: removes citations to passages the model wasn't
   given, flags code names that don't exist in ABP, and warns about calls in code samples that no
   retrieved passage showed. Research answers are held until it passes.

Every turn is an MLflow trace with decision tags. Thumbs down and guard actions are collected as
eval candidates. The read tools are also served over MCP.

## Results

Two scores, reported separately (D21, D25):

- **Automatic checks** prove behaviour: it searched, cited real passages, matched the key's
  patterns, and refused when ABP lacks the thing.
- **Read grades** check correctness: every answer is read in full against its key, and doubtful
  claims are checked in the ABP source. One false detail fails the answer. Grading is done by
  Claude in a session, using rules set from my marks on test answers (D25), so the grader isn't
  independent (D4). I haven't yet spot-checked the test2 and test3 grades myself.

Each question runs 3 times. Each test set was run once, after review and sealing.

| Test set | Questions | Automatic | Read: correct | Notes |
|---|---|---|---|---|
| test | 15 | 45 / 45 | 31 / 45 | 7 minor, 6 wrong, 1 no answer |
| test2 | 14 | 36 / 42 | 29 / 42 | forum 12 / 21, written 17 / 21 |
| test3 | 15 | 45 / 45 | 29 / 45 | main point right 42 / 45; 14 runs with a false detail |

Dev, latest (33 questions × 3): automatic pass^3 0.97, median 22 s per answer on one RTX 3090.

The gap between the two columns is the main finding: automatic checks can't see a false detail.
[docs/experiments.md](docs/experiments.md) covers where those come from and what was tried.

## Run

Requires Docker Desktop, a local model server and a local clone of
[abpframework/abp](https://github.com/abpframework/abp). Nothing else is installed on the host.

```powershell
copy .env.example .env      # then set passwords and ABP_REPO_DIR
docker compose up -d --build
docker compose ps           # wait until everything is healthy
```

| What | URL |
|---|---|
| UI | http://localhost:8502 |
| API docs | http://localhost:8011/docs |
| Health | http://localhost:8011/health |
| Prefect | http://localhost:4201 |
| MLflow | http://localhost:5001 |
| Neo4j Browser | http://localhost:7475 |

## Data and search

```powershell
# Load the ABP docs, C# and JavaScript (later runs skip unchanged files)
docker compose exec pipelines prefect deployment run "ingest/ingest" --watch

# Search: mode = bm25 | vector | hybrid, sources = docs, code (or try it at /docs)
$body = '{"query": "user name uniqueness validator", "sources": ["code"], "limit": 5}'
Invoke-RestMethod -Method Post http://localhost:8011/search -Body $body -ContentType application/json

# Retrieval eval on dev questions, logged to MLflow experiment "retrieval"
docker compose run --rm -v "${PWD}:/app" api python -m evals.retrieval_eval

# Build the graph (after ingest; no LLM, about a minute and a half)
docker compose exec pipelines prefect deployment run "build-graph/build" --watch

# Exact facts: kind = depends_on | all_dependencies | used_by | all_used_by | implementations |
# bases | options | module_of_options | docs
Invoke-RestMethod "http://localhost:8011/graph/relation?name=IEmailSender&kind=implementations"

# Closed-list eval, logged to MLflow experiment "graph"
docker compose run --rm -v "${PWD}:/app" api python -m evals.graph_eval
```

## Monitoring

Every chat turn is one trace in MLflow: http://localhost:5001/#/experiments/4/traces (experiment
`agent`). Eval traces go to `agent-eval`, so the two never mix.

| Question | Where |
|---|---|
| Volume, latency, errors, tokens | Overview |
| Which questions, slow or failed ones | Traces |
| Why an answer came out as it did | Open a trace: every step, passage and model call |
| Whole conversations | Sessions |
| Ratings and comments | Open a trace: Assessments |

Filter traces by tag: `route`, `answer_replaced`, `citations_removed`, `unverified_names`,
`unseen_calls`, `first_call_retried`, `router_failed`, `input_blocked`, `invalid_tools`,
`tool_rounds`, `tool_calls`, `citations`, `technology`, `platform`, `intent`, `identifiers`,
`analysis_failed`, `environment`, `status`, plus `model`, `prompt_version`
and `app_version`.

Rate answers with the thumbs on the chat page, or `POST /feedback`. A thumbs down asks what was
wrong. Then collect what's worth an eval case:

```powershell
docker compose run --rm -v "${PWD}:/app" api python -m evals.candidates
```

## MCP server

The agent's read tools for any MCP client, at `http://localhost:8011/mcp` (streamable HTTP):
`search` and `code_graph`. They need no LLM, and results list where each `[id]` points.

Every call needs the token from `MCP_TOKEN` in `.env`. With no token set, nobody gets in. Only
`/mcp` is protected: the rest of the API has no login, so keep it on localhost.

```powershell
claude mcp add --transport http abp-agent http://localhost:8011/mcp --header "Authorization: Bearer <MCP_TOKEN>"
```

## CI

`.github/workflows/ci.yml` builds the image, then runs lint and tests in it on each push. It is ready
but not connected: the repository is local for now, so it only runs once it has a GitHub remote.
To run the same steps locally:

```powershell
docker build -f docker/app.Dockerfile -t abp-agent-app:ci .
docker run --rm --network none abp-agent-app:ci ruff check .
docker run --rm --network none abp-agent-app:ci python -m pytest -q
```

`evals.verify` and the agent evals stay local: they need the ABP repository, the stack and a model.

## Common commands

```powershell
docker compose run --rm -v "${PWD}:/app" api pytest
docker compose run --rm -v "${PWD}:/app" api ruff check .
docker compose logs -f api
```

Update dependencies: edit `pyproject.toml`, then regenerate the lock file inside Docker:

```powershell
docker run --rm -v "${PWD}:/app" -w /app ghcr.io/astral-sh/uv:0.13.6-python3.12-trixie-slim uv lock
```
