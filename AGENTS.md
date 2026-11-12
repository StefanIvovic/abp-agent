# Running this project (for coding agents)

Follow README.md for setup. Notes that matter when you drive it:

- Everything runs in Docker: `docker compose up -d --build`, then `docker compose ps` until healthy.
  Run Python through `docker compose run --rm -v "${PWD}:/app" api ...`, not on the host.
- `.env` is required: copy `.env.example`. Ask the user for passwords, `ABP_REPO_DIR` and
  `MCP_TOKEN`; don't copy them from other files or make them up. Never commit `.env`.
- Both models run in LM Studio on the host: qwen3.8-27b (chat) and text-embedding-bge-m3
  (embeddings, `lms load text-embedding-bge-m3 --gpu off -c 2048`). Check with `lms ps` and
  `/health`. Never load a model with a JIT request: it can unload the other one.
- Unit tests and lint after every change: `pytest` and `ruff check .` in the api container.
  They need no services.
- Agent evals need the stack, the ABP clone and the model, and take a long time. Ask before running
  them, and never run a test split (`test`, `test2`, `test3`) without being asked: each is sealed
  and its first run is the reported score.
- Never edit `evals/datasets/` to make an eval pass. Run `python -m evals.verify` after any change
  there; a sealed split that changed fails it.
- Record a design decision in `docs/decisions.md` with the measurement behind it, and a study in
  `docs/experiments.md`.
