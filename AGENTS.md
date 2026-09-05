# AGENTS.md — modal-labs

Repository-level context. Each project directory has its own `AGENTS.md` with
the design decisions specific to that deployment; read both.

## Why a real ComfyUI, and not a sampling loop

Every service here runs a headless ComfyUI on the GPU with a thin HTTP layer in
front, rather than loading weights with `diffusers` and writing a sampling loop.

The reason is that the interesting parts of a modern image model — noise
schedules, guidance variants, quantised kernels, per-architecture text-encoder
taps — are all upstream code, fixed and tuned by people who work on it full
time. Reimplementing them buys nothing and drifts silently: the failure mode is
a plausible image that is subtly wrong, not an exception.

It also answers the requirement directly. Because ComfyUI *is* the server, the
deployed URL speaks the ComfyUI protocol with no translation layer, so any
ComfyUI client works against it unmodified.

The cost is a heavier container and a process to supervise. `comfyui_modal`
absorbs both.

## Shape of the repository

A single `uv` environment at the root serves every project. Projects are flat
directories, not installable packages:

```
modal-labs/
├── pyproject.toml        one environment, ruff + pytest config
├── comfyui_modal/        shared: image, supervisor, ASGI proxy, graph fragments,
│                         CLI, weight fetching, test doubles
├── comfy_node/           shared: one ComfyUI node package for every service
├── tests/                tests for the shared code
├── .github/workflows/    lint, format, test, import-check every Modal app
└── <project>/
    ├── app.py            the Modal entrypoint — `modal deploy <project>/app.py`
    ├── server.py         request model, resolver, service-specific routes
    ├── workflow.py       the model's graph, in ComfyUI API format
    ├── README.md         how to run it
    └── AGENTS.md         why it is built that way
```

**A service owns its model, not its plumbing.** Adding one means writing a
`workflow.py`, a weight table, and a request model — a few hundred lines. If you
find yourself copying the proxy, the supervisor or the progress mirror, extend
the shared package instead: those encode bugs found the hard way, and duplicated
copies lose the fixes one at a time.

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

Fetching itself lives in `comfyui_modal/weights.py`, which knows two sources:
Hugging Face (repo + path, optionally gated) and Civitai (version id, presigned
redirect, **published SHA256 verified before install**). Anything without an
upstream integrity guarantee must carry a digest — a substituted file then fails
closed regardless of which host served it. Do not add a third fetcher inline.

**Endpoints are authenticated by default.** New web endpoints get
`requires_proxy_auth=True` unless there is a stated reason otherwise (browser
UIs cannot send the headers — those are meant for `modal serve`, not `deploy`).

**Pin versions.** Container dependencies, upstream git refs and pre-commit hooks
are all pinned. Unpinned upstreams have broken deployments silently before.

## Testing

Tests must run offline with no Modal credentials and no GPU. That means the
testable surface is the pure logic — model graphs, parameter resolution, request
mapping — while anything requiring the real service is a CLI subcommand a human
runs against a live deployment (see any service's `client.py validate`).

CI additionally imports each `app.py` — in a separate process per project,
because every service defines modules named `app`, `server` and `workflow`.
Within pytest the same collision is handled by each test module clearing those
names from `sys.modules` before importing, plus a guard test asserting it bound
its own project's copy. Without it a suite silently tests the wrong project.

## Getting caller-supplied media into a graph

`LoadImage` names a file in ComfyUI's input directory, so bytes have to be there
before a graph can reference them. `ModelService.upload_fields` names the request
fields carrying base64; `run_generation` uploads each through ComfyUI's own
`/upload/image` and **replaces the field with the filenames ComfyUI returns**
before `resolve` runs. A field may be one image or a list of them, and its shape
is preserved.

Two rules, both learned the hard way rather than guessed:

* Reference the name ComfyUI reports back, never the one sent. It renames on
  collision, so the sent name may belong to someone else's file.
* Upload before the `duration_s` clock starts. It is transport, not render time,
  and a service with no `upload_fields` makes no extra call at all.

`comfyui_modal/graph.py` holds the graph fragments that are genuinely
architecture-independent — currently the img2img source chain, which six
services splice identically. Anything model-specific stays in the service.

## Guarding the shared layer

Two habits keep the extraction honest:

**Golden graphs.** Each service commits a reference graph under `workflows/` and
a test asserting `build_workflow` still emits exactly it. That file is what users
POST directly, so it must not drift — and it doubles as the equivalence check
whenever the shared code underneath is refactored.

**Pinned node contracts.** `tests/test_node_runtime.py` pins every ComfyUI node
id and its widget names *in order*. Those are what a saved workflow JSON
references, so renaming or reordering one silently breaks every workflow a user
has saved.
