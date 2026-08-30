# AGENTS.md — modal-labs

Repository-level context. Each project directory has its own `AGENTS.md` with
the design decisions specific to that deployment; read both.

## Shape of the repository

A single `uv` environment at the root serves every project. Projects are flat
directories, not installable packages:

```
modal-labs/
├── pyproject.toml        one environment, ruff + pytest config
├── .github/workflows/    lint, format, test, import-check every Modal app
└── <project>/
    ├── app.py            the Modal entrypoint — `modal deploy <project>/app.py`
    ├── README.md         how to run it
    └── AGENTS.md         why it is built that way
```

The root environment holds only what runs *locally* (`modal`, a CLI client's
dependencies, dev tools). Everything a container needs is declared in that
project's `modal.Image` and is never installed here. Container-only imports
therefore live inside function bodies or in modules that only the container
imports, so `modal deploy` works from a thin local environment.

## Conventions

**Modal apps are flat modules, not packages.** `app.py` sits beside its helper
modules and pulls them in with `image.add_local_python_source(...)`. Each app
inserts its own directory onto `sys.path` so `modal deploy` works from any
working directory.

**Separate wiring from logic.** `app.py` holds the Modal object graph — image,
volumes, GPU, endpoints — and nothing else. Request handling, model graphs and
CLI code live in plain modules that can be imported and tested without Modal.

**Deploy-time configuration through the environment.** Knobs are read at module
scope in `app.py` (GPU type, replica counts, auth) because Modal evaluates the
file at deploy time. Each project documents them in `.env.example`. Changing one
means re-deploying.

**Weights go in Volumes, never in the image.** A dedicated `download_models`
function populates the Volume; serving containers mount it read-only. This keeps
image builds fast and makes weight updates independent of code deploys.

**Endpoints are authenticated by default.** New web endpoints get
`requires_proxy_auth=True` unless there is a stated reason otherwise (browser
UIs cannot send the headers — those are meant for `modal serve`, not `deploy`).

**Pin versions.** Container dependencies, upstream git refs and pre-commit hooks
are all pinned. Unpinned upstreams have broken deployments silently before.

## Testing

Tests must run offline with no Modal credentials and no GPU. That means the
testable surface is the pure logic — model graphs, parameter resolution, request
mapping — while anything requiring the real service is a CLI subcommand a human
runs against a live deployment (see `ideogram4/client.py validate`).

CI additionally imports each `app.py`, which catches decorator and signature
mistakes without deploying anything.
