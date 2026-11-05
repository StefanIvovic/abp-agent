# One image for api, ui, pipelines and migrate (different commands in compose.yaml).
# Pattern from the uv Docker guide: install dependencies first (cached layer), then the project.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# 1) Dependencies only: this layer is rebuilt only when pyproject.toml or uv.lock changes.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project

# 2) The project itself, installed in editable mode, so files that `compose watch` syncs into /app/src are picked up.
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

CMD ["uvicorn", "abp_agent.main:app", "--host", "0.0.0.0", "--port", "8000"]
