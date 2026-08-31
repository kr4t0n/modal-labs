# Ideogram 4 on Modal, served as a ComfyUI API

Runs the open-weight [Ideogram 4](https://github.com/ideogram-oss/ideogram4)
text-to-image model on a Modal GPU and exposes it as a **ComfyUI server**. Point
your local ComfyUI at the resulting URL and render on an H100 without keeping
30 GB of weights or a big GPU on your own machine.

Ideogram 4 is a 9.3B single-stream diffusion transformer with a Qwen3-VL-8B text
encoder, native 2K output, and strong multilingual text rendering. It is
supported by ComfyUI core, so this deployment runs a real headless ComfyUI
rather than reimplementing the sampler — the schedule, the dual-branch CFG and
the fp8 kernels are all the upstream ones.

> **Licence.** The Ideogram 4 weights are released under the *Ideogram 4
> Non-Commercial* licence. Read
> [LICENSE.md](https://huggingface.co/ideogram-ai/ideogram-4-fp8/blob/main/LICENSE.md)
> before deploying. This repository ships no weights; it downloads them at your
> direction.

## What you get

One deployed URL that is simultaneously:

| Surface | Path | Use it for |
| --- | --- | --- |
| ComfyUI HTTP + WebSocket API | `/prompt`, `/history`, `/view`, `/object_info`, `/upload/image`, `/ws` | Any existing ComfyUI API client or script, unchanged |
| Typed generation contract | `POST /generate` | The bundled custom node, `client.py`, your own app |
| Prompt tooling | `GET /caption-template`, `GET /presets`, `POST /workflow` | Writing structured captions, inspecting the graph |
| ComfyUI web interface | `/` | Browsing the queue, dragging in workflows |

Plus a second, separate endpoint (`modal serve app.py`) that serves the raw
ComfyUI web UI with no proxy in front of it.

## Prerequisites

- A [Modal](https://modal.com) account with GPU access (`pip install modal && modal setup`)
- [uv](https://docs.astral.sh/uv/) for the local environment
- A local ComfyUI install, if you want the remote node (not required for the CLI)

## Setup

```bash
# from the repository root
uv sync --all-groups

cd ideogram4

# 1. Pull ~30 GB of weights into a Modal Volume. One-off; safe to re-run.
uv run modal run app.py::download_models

# 2. Deploy. Prints the endpoint URLs.
uv run modal deploy app.py
```

Deployment settings come from the environment — copy `.env.example` to `.env`,
edit, then:

```bash
set -a && source .env && set +a && uv run modal deploy app.py
```

`modal deploy` prints the endpoint URLs — copy them from its output rather than
constructing them, since Modal derives the label server-side. You want the one
for `Ideogram4.web` (the API); the `ui` one is the raw web interface. They look
roughly like:

```
https://<workspace>--ideogram4-comfyui-ideogram4-web.modal.run   # the API
https://<workspace>--ideogram4-comfyui-ui.modal.run              # the raw web UI
```

`modal app list` shows them again later.

### Authentication

The API endpoint requires [Modal proxy auth](https://modal.com/docs/guide/webhook-proxy-auth)
by default. Create a token in the Modal dashboard (Settings → Proxy Auth Tokens)
and send it as headers:

```bash
export IDEOGRAM4_MODAL_URL=https://<workspace>--ideogram4-comfyui-ideogram4-web.modal.run
export MODAL_KEY=wk-...
export MODAL_SECRET=ws-...
```

Set `IDEOGRAM4_REQUIRE_AUTH=0` before deploying to disable it. Do not do this on
a persistent deployment: an open ComfyUI endpoint lets anyone queue arbitrary
graphs on your GPU bill.

## Using it from your local ComfyUI

The nodes for every service live in one package at the repository root:

```bash
cp -r ../comfy_node /path/to/ComfyUI/custom_nodes/comfyui-modal-remote
cp ../comfy_node/.env.example /path/to/ComfyUI/custom_nodes/comfyui-modal-remote/.env
# edit that .env with your URL and Modal token, then restart ComfyUI
```

The node needs no GPU and no extra dependencies, so a CPU-only ComfyUI works as
a pure UI and orchestrator. For a clustered ComfyUI, wire the settings through a
Secret instead of the `.env` file — see
[`../comfy_node/README.md`](../comfy_node/README.md#install-on-a-kubernetes-comfyui).

Two nodes appear under **Ideogram 4 (Modal)**:

- **Ideogram 4 (Modal)** — prompt, preset, aspect ratio, seed, batch → `IMAGE`.
  It is an ordinary image source, so it feeds upscalers, `SaveImage`, ControlNet
  preprocessors and anything else in your local graph. The remote sampler's
  progress is mirrored onto the node's local progress bar.
- **Ideogram 4 Caption Template (Modal)** — emits the magic-prompt text to hand
  to an LLM (see *Prompting* below).

The node holds no credentials in the workflow JSON; it reads them from the
environment or from the `.env` beside it, so exported workflows stay shareable.

### Or point an existing ComfyUI client at it

The endpoint speaks the ComfyUI protocol verbatim, so anything that talks to a
local ComfyUI works — including
[`ComfyUI/script_examples/websockets_api_example.py`](https://github.com/comfyanonymous/ComfyUI/tree/master/script_examples):

```python
server = "<workspace>--ideogram4-comfyui-ideogram4-web.modal.run"
headers = {"Modal-Key": os.environ["MODAL_KEY"], "Modal-Secret": os.environ["MODAL_SECRET"]}
requests.post(f"https://{server}/prompt", json={"prompt": graph}, headers=headers)
```

`workflows/ideogram4_t2i_api.json` is a ready-made API-format graph to POST.

## Using it from the CLI

```bash
uv run python client.py health
uv run python client.py validate                     # graph vs. deployed node schemas
uv run python client.py generate "a neon ramen shop in the rain" --aspect-ratio 16:9
uv run python client.py generate "..." --preset Quality --megapixels 4 --seed 12345
uv run python client.py template "a neon ramen shop in the rain" > caption_prompt.txt
```

Or without the endpoint at all — `modal run` calls the container directly and
needs no proxy token:

```bash
uv run modal run app.py --prompt "a brutalist library at golden hour" --preset Turbo
```

## Prompting

Ideogram 4 is trained on **structured JSON captions**, not free text. Treat
this as required, not advisory: plain prompts are passed straight through, but
in practice they produce images with little relation to what you asked for —
the shorter the prompt, the worse. Weak conditioning makes the model drift
toward what it would produce unconditionally.

A caption looks like:

```json
{
  "aspect_ratio": "16:9",
  "high_level_description": "...",
  "compositional_deconstruction": {
    "background": "...",
    "elements": [{"type": "text", "bbox": [y1, x1, y2, x2], "text": "SATURN", "desc": "..."}]
  }
}
```

`GET /caption-template` returns the prompt that turns an idea into one of these
— feed it to any instruction-following LLM and post the JSON back as
`json_prompt`. `assets/example_json_prompt.json` is a worked example. The
custom node detects a caption pasted into the prompt box (anything starting with
`{`) and forwards it as `json_prompt` automatically.

## Sampling presets

| Preset | Steps | mu | std |
| --- | --- | --- | --- |
| `Turbo` | 12 | 0.5 | 1.75 |
| `Default` | 20 | 0.0 | 1.75 |
| `Quality` | 48 | 0.0 | 1.5 |

`mu` and `std` parameterise Ideogram 4's logit-normal noise schedule; they are
not the shift value used by Flux-style schedulers. Guidance is dual-branch:
`cfg` (default 7.0) applies throughout, then `late_cfg` (3.0) takes over from
70% of the way through the schedule.

Sides must be multiples of 16 in `[256, 2048]`; anything else is snapped up and
clamped.

## API reference

`POST /generate` — body fields, all optional except one of `prompt` /
`json_prompt`:

| Field | Default | Notes |
| --- | --- | --- |
| `prompt` | — | Plain text |
| `json_prompt` | — | Structured caption; wins over `prompt` |
| `width`, `height` | 1024 | 256–2048, snapped to /16 |
| `aspect_ratio` | — | `1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `9:16`, `16:9`, `21:9`; overrides width/height |
| `megapixels` | 1.0 | Pixel budget used with `aspect_ratio` |
| `preset` | `Default` | `Turbo` / `Default` / `Quality` |
| `steps`, `mu`, `std` | preset | Explicit overrides |
| `cfg`, `late_cfg`, `late_cfg_start` | 7.0 / 3.0 / 0.7 | Dual-branch guidance |
| `seed` | random | 0 – 2^64-1 |
| `batch_size` | 1 | 1–8 |
| `sampler_name` | `euler` | Any ComfyUI sampler |
| `timeout_s` | 900 | Server-side wait |
| `client_id` | generated | Forwarded to ComfyUI; subscribe to `/ws?clientId=<id>` for this render's progress |

Response: `{prompt_id, duration_s, params, images: [{filename, content_type, b64}]}`.
`POST /generate/image` takes the same body and returns the first image as raw
PNG bytes, with the seed in the `X-Seed` header and the queue id in `X-Prompt-Id`.

## Configuration

All read at deploy time. See `.env.example` for the annotated list.

| Variable | Default | Effect |
| --- | --- | --- |
| `IDEOGRAM4_GPU` | `H100` | Any Modal GPU string |
| `IDEOGRAM4_MIN_CONTAINERS` | `0` | Warm containers; removes cold starts, costs money idle |
| `IDEOGRAM4_MAX_CONTAINERS` | `1` | Keep at 1 unless clients submit and poll in one request |
| `IDEOGRAM4_SCALEDOWN_WINDOW` | `300` | Idle seconds before shutdown |
| `IDEOGRAM4_CONCURRENT_INPUTS` | `20` | Requests in flight per container |
| `IDEOGRAM4_REQUIRE_AUTH` | `1` | Proxy auth on the API endpoint |
| `IDEOGRAM4_UI_REQUIRE_AUTH` | `0` | Proxy auth on the raw UI endpoint |

Cost is driven by GPU seconds. A cold container pays for the container image
pull plus loading ~30 GB of weights from the Volume onto the GPU; after that,
containers stay warm for `IDEOGRAM4_SCALEDOWN_WINDOW` seconds.

### Choosing a GPU

The fp8 checkpoints need **fp8 tensor cores**, which means compute capability
8.9 or newer — Ada, Hopper or Blackwell. ComfyUI will still run on older cards
by computing in fp16, losing the speedup. `nvfp4` checkpoints additionally need
Blackwell (capability 10+).

| Modal `gpu=` | VRAM | fp8 | Verdict |
| --- | --- | --- | --- |
| `H100` | 80 GB | yes | Default. Fastest of the sane options. |
| `L40S` | 48 GB | yes | Holds the full stack at ~half H100's hourly rate. The one worth trying. |
| `RTX-PRO-6000` | 96 GB | yes (+fp4) | Cheaper per hour than H100; only card here that can run the `nvfp4` weights. |
| `A100-80GB` / `A100-40GB` | 80/40 GB | **no** | Dominated: pricier than L40S with no fp8. |
| `A10`, `L4`, `T4` | 24/24/16 GB | no/yes/no | Too small — a transformer swaps per step. |

**Compare cost per image, not per hour.** A cheaper card that is proportionally
slower costs more. L40S only wins if it is less than ~2x slower than H100 on
this model, which is unmeasured here — so measure it:

```bash
IDEOGRAM4_GPU=L40S uv run modal deploy app.py
uv run python client.py generate "benchmark" --preset Turbo   # cold, ignore
uv run python client.py generate "benchmark" --preset Turbo   # warm, compare duration_s
```

Multiply the warm `duration_s` by the card's per-second price and compare.

**Bigger levers than the GPU:**

- **Steps.** Cost is near-linear in them, so `Turbo` (12) is ~40% cheaper than
  `Default` (20) and ~75% cheaper than `Quality` (48). This dominates the GPU
  choice.
- **Resolution.** Attention is quadratic in token count, so 4 MP costs
  considerably more than 4x of 1 MP.
- **`IDEOGRAM4_SCALEDOWN_WINDOW`.** You pay for warm idle containers, but cold
  starts are billed too. Lower it for infrequent one-off renders; keep it high
  if you work in bursts.

## Project structure

```
ideogram4/
├── app.py         Modal object graph: weights, GPU class, endpoints
├── server.py      Request model, resolver and the two extra routes
├── workflow.py    The Ideogram 4 graph in ComfyUI API format
├── client.py      CLI: generate / template / health / validate
├── workflows/     Ready-to-POST API-format graph
├── assets/        Magic-prompt template, example JSON caption
└── tests/         Ideogram-4-specific assertions
```

Everything generic lives outside this directory: `../comfyui_modal` (container
image, ComfyUI supervisor, ASGI proxy, CLI plumbing) and `../comfy_node` (the
ComfyUI nodes for every service).

`AGENTS.md` covers the design decisions behind that layout.

## Testing

```bash
uv run pytest -q          # offline: graph structure, and the ASGI layer
                          # driven against a stubbed ComfyUI
uv run ruff check . && uv run ruff format --check .
uv run python client.py validate    # graph vs. the deployed ComfyUI's schemas
```

The offline suite covers the submit/poll/fetch sequence behind `/generate` and
whether the catch-all really is a transparent proxy. It cannot know whether the
node schemas still match a given ComfyUI build — that is what `validate` is for,
and it is the check that catches ComfyUI upgrades: a renamed node input still
builds a graph locally and only fails when it is queued.

## Troubleshooting

**`weights Volume is missing ...`** — run `modal run app.py::download_models`.

**First request takes minutes.** Cold start: image pull, ComfyUI boot, then ~30 GB
of weights read from the Volume. Set `IDEOGRAM4_MIN_CONTAINERS=1` if latency
matters more than idle cost.

**`401` / `403` from the endpoint.** Missing or wrong `Modal-Key` /
`Modal-Secret`. `modal curl <url>` signs a request for you.

**The image ignores the prompt.** Almost always a plain-text prompt. Confirm the
deployment itself is sound by rendering the bundled reference caption:

```bash
uv run python client.py generate --json-prompt assets/example_json_prompt.json \
  --aspect-ratio 9:16 --preset Quality
```

That should give a streetwear collage poster with "COMFY" in puffy 3D letters.
If it does, the pipeline is fine and you need a JSON caption — use
`client.py template "your idea"`, pass it through an LLM, and feed the result
back with `--json-prompt`. If it does not, the problem is below the graph.

**Node inputs mismatch after a ComfyUI bump.** `COMFYUI_REF` in `app.py` pins the
version. Run `client.py validate` after changing it.

**Out of memory on a smaller GPU.** Both transformers are resident during
sampling. Use `IDEOGRAM4_GPU=H100` or `A100-80GB`, or drop the resolution.
