# modal-labs

Serverless GPU deployments of open-weight generative models on
[Modal](https://modal.com). Each subdirectory is a self-contained project with
its own `README.md` and `AGENTS.md`; this repository holds the shared Python
environment and tooling.

## Projects

| Project | What it does |
| --- | --- |
| [`flux2klein/`](flux2klein/) | FLUX.2 [klein] 9B text-to-image, same pattern — natural-language prompts, base and 4-step distilled variants |
| [`ultra/`](ultra/) | ULTRA, a Krea 2 finetune fetched from Civitai with a verified digest — 8-step turbo sampling |
| [`zimageturbostableyogi/`](zimageturbostableyogi/) | Stable Yogi's Z-Image Turbo finetune — the cheapest of the set, a 6 GB fp8 model on a 24 GB L4 |
| [`finepornv4/`](finepornv4/) | FinePorn v4, a Krea 2 merge — bf16, the heaviest of the set, and the only one rendering above 1 MP by default |

Each exposes one URL that is simultaneously a real ComfyUI server and a typed
`/generate` contract, so a CPU-only ComfyUI can stay the UI while the GPU work
happens on Modal.

`ultra/` and `finepornv4/` serve the same Krea 2 base with different merges, and
share their text encoder and VAE — a test asserts those stay identical.

`finepornv4/` is the one project versioned in its own name: a later FinePorn
release gets a sibling directory and its own endpoint rather than replacing this
one, so the two can be compared and existing workflows keep their URL.

Retired services live in [`deprecated/`](deprecated/) — kept as reference, not
built or tested. See its README before reviving one.

## Shared code

| Directory | What it is |
| --- | --- |
| [`comfyui_modal/`](comfyui_modal/) | The container image, ComfyUI supervisor, ASGI proxy, resolution arithmetic, weight fetching, CLI transport and test doubles that every service uses |
| [`comfy_node/`](comfy_node/) | One ComfyUI custom-node package covering every deployment — install it once, get every model's node |

A service supplies only what is genuinely model-specific: its graph, its weight
table, and a handful of request fields.

## Setup

```bash
uv sync --all-groups
uv run modal setup        # once, to authenticate
```

Then follow the project's own README — for example
[`flux2klein/README.md`](flux2klein/README.md).

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
