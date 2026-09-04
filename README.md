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
| [`redgpt2gpt/`](redgpt2gpt/) | RedGPT2, GPT edition — a Krea 2 finetune, fp8, the lightest of the Krea 2 services |
| [`redcraftv3/`](redcraftv3/) | RedCraft v3 — a Krea 2 finetune, fp8, the only one whose author publishes a sampler recipe |
| [`darkbeastv3/`](darkbeastv3/) | Dark Beast v3 — a Krea 2 finetune, int8, from a listing whose headline product is a video model |

Each exposes one URL that is simultaneously a real ComfyUI server and a typed
`/generate` contract, so a CPU-only ComfyUI can stay the UI while the GPU work
happens on Modal.

**Five of them serve Krea 2** — `ultra/`, `finepornv4/`, `redgpt2gpt/`,
`redcraftv3/` and `darkbeastv3/` — with different finetunes over the same base,
sharing a text encoder and VAE. A test asserts those companions stay identical
across all five; divergence would change output without failing. All five also
inherit the Krea 2 Community License, whose free commercial use is capped by
company revenue.

Their sampler defaults deliberately differ, because their upstreams do: two
publish a recipe and three do not, so those three fall back to ComfyUI's
template. `GET /defaults` on each names its own source — do not sync settings
between them.

Four projects carry a version or edition in their name — `finepornv4/`,
`redgpt2gpt/`, `redcraftv3/` and `darkbeastv3/`. Their upstreams publish several
incompatible builds under one listing, so a later release gets a sibling
directory and its own endpoint rather than replacing what is deployed, letting
the two be compared while existing workflows keep their URL.

Those listings are also where most of the care goes: several mix unrelated base
models under one page, and several give every precision of a build the same
filename. Each service pins a `file_id` as well as a version id for that reason,
and its AGENTS.md records what was actually checked.

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

To talk to deployed services, set your workspace once rather than a URL per
service:

```bash
export MODAL_WORKSPACE=your-workspace
export MODAL_KEY=wk-...  MODAL_SECRET=ws-...
```

Both the CLI clients and the ComfyUI nodes derive each endpoint from that, so
adding a service needs no new variable. A per-service `<SLUG>_MODAL_URL` still
takes precedence — use it for a second deployment of one service, or an
ephemeral `modal serve` URL.

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
