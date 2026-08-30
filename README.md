# modal-labs

Serverless GPU deployments of open-weight generative models on
[Modal](https://modal.com). Each subdirectory is a self-contained project with
its own `README.md` and `AGENTS.md`; this repository holds the shared Python
environment and tooling.

## Projects

| Project | What it does |
| --- | --- |
| [`ideogram4/`](ideogram4/) | Ideogram 4 text-to-image on an H100, served as a ComfyUI API so a local ComfyUI can use it as a remote model endpoint |

## Setup

```bash
uv sync --all-groups
uv run modal setup        # once, to authenticate
```

Then follow the project's own README — for example
[`ideogram4/README.md`](ideogram4/README.md).

## Development

```bash
uv run ruff check .          # lint
uv run ruff format .         # format
uv run pytest -q             # tests
uv run pre-commit install    # once, to run the above on commit
```

CI runs the same three checks plus an import of each Modal app on every push to
`main` and every pull request. See [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Conventions

- Python is managed with `uv`; never `pip install` into the environment directly.
- `ruff` is the only linter and formatter.
- Model weights live in Modal Volumes, downloaded by an explicit one-off
  function. Nothing large is committed, and no project ships weights.
- Secrets live in the environment or in `~/.modal.toml`. `.env` is gitignored;
  each project ships a `.env.example` describing the variables it reads.
- Deployed endpoints require Modal proxy auth by default.
