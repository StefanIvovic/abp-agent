# ABP Agent

Answers ABP Framework questions from its docs and source code, with citations. See
[docs/roadmap.md](docs/roadmap.md) and [docs/decisions.md](docs/decisions.md).

## Run

Requires Docker Desktop, an NVIDIA GPU and a local clone of
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
