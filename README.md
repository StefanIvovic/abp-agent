# ABP Agent

Answers ABP Framework questions from its docs and source code, with citations. See
[docs/roadmap.md](docs/roadmap.md) and [docs/decisions.md](docs/decisions.md).

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
# Load the ABP docs and C# source (later runs skip unchanged files)
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
