# FLUX.2 [klein] 9B on Modal, served as a ComfyUI API

Runs Black Forest Labs' [FLUX.2 klein](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)
9B text-to-image model on a Modal GPU and exposes it as a **ComfyUI server**.
Same shape as the other services here: point a local or clustered
ComfyUI at the resulting URL and render remotely.

FLUX.2 klein is the compact member of the FLUX.2 family — a 9B transformer with
a Qwen3-8B text encoder — supported by ComfyUI core, so this deployment runs a
real headless ComfyUI rather than reimplementing the sampler.

Two checkpoints ship, and the choice matters:

| Variant | Steps | CFG | Use it for |
| --- | --- | --- | --- |
| `base` | 20 | 5.0 | Default. Undistilled, responds to CFG and negative prompts. |
| `distilled` | 4 | 1.0 | ~5x fewer steps. Guidance-distilled, so it ignores CFG and negative prompts. |
| `ponpoke-uncensored` | 20 | 5.0 | As `base`, with an abliterated text encoder — see below. |

The variant selects the transformer *and* its text encoder, because the two are
validated together. `ponpoke-uncensored` differs from `base` in the encoder alone:
same transformer, same schedule, so any difference in output comes from prompt
handling.

Unlike Ideogram 4 this takes ordinary natural-language prompts — no structured
caption required — and supports a real negative prompt.

> **Licence.** The transformers and both text encoders are under **FLUX.2
> non-commercial licences** and are **gated** on Hugging Face. You must accept
> each agreement with the account whose token you use. This repository ships no
> weights.

> **`ponpoke-uncensored`.** Its encoder is
> [`ponpoke/flux2-klein-9b-uncensored-text-encoder`](https://huggingface.co/ponpoke/flux2-klein-9b-uncensored-text-encoder),
> an *abliterated* Qwen3-8B: the refusal direction is orthogonalised out of the
> encoder's mid and late layers, so prompt-stage safety filtering no longer
> applies. Its gate carries its own terms of use, and what the deployment
> produces becomes entirely your responsibility. Skip this variant and its
> ~16 GB download if you do not want it — the other two are unaffected.

## What you get

One deployed URL that is simultaneously:

| Surface | Path | Use it for |
| --- | --- | --- |
| ComfyUI HTTP + WebSocket API | `/prompt`, `/history`, `/view`, `/object_info`, `/ws` | Any existing ComfyUI API client, unchanged |
| Typed generation contract | `POST /generate` | The bundled custom node, `client.py`, your own app |
| Introspection | `GET /variants`, `POST /workflow` | Sampler defaults, inspecting the graph |
| ComfyUI web interface | `/` | Browsing the queue, dragging in workflows |

## Setup

```bash
# from the repository root
uv sync --all-groups
cd flux2klein
```

**1. Accept the licence** for both transformers, signed in as the account that
owns your token:

- <https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8>
- <https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8>

**2. Give Modal the token** ([create one](https://huggingface.co/settings/tokens)):

```bash
uv run modal secret create huggingface-secret HF_TOKEN=hf_...
```

**3. Pull the weights and deploy:**

```bash
uv run modal run app.py::download_models   # ~45 GB into a Volume. One-off.
uv run modal deploy app.py
```

`download_models` runs on CPU — no GPU charge. It is idempotent by destination,
so re-running skips what is already there; pass `--force` to refetch.

Copy the endpoint URLs from the `modal deploy` output rather than constructing
them; Modal derives the label server-side. You want the one for `Flux2Klein.web`.

### Authentication

The API endpoint requires [Modal proxy auth](https://modal.com/docs/guide/webhook-proxy-auth)
by default. Create a token in the Modal dashboard (Settings → Proxy Auth Tokens):

```bash
export FLUX2KLEIN_MODAL_URL=https://<workspace>--flux2klein-comfyui-flux2klein-web.modal.run
export MODAL_KEY=wk-...
export MODAL_SECRET=ws-...
```

> Or set `MODAL_WORKSPACE=<workspace>` once and every service's URL is derived
> from it — see [`comfy_node/README.md`](../comfy_node/README.md). The explicit
> variable above still wins when set.

## LoRA adapters

An adapter can be layered onto the transformer for any variant. Omit `lora` and
the graph is exactly what it was before the feature existed.

| Name | Trigger words | What it is |
| --- | --- | --- |
| `snofs-v1.4` | — | [Ashen3 SNOFS v1.4](https://huggingface.co/Ashen3/SNOFS) — a LoKr adapter trained on klein 9B |
| `realstockings-v2` | `stockings`, `RealStockings` | [lajmar Stockings v2](https://civitai.com/models/2463208) — a standard LoRA trained on klein 9B |
| `realism-engine-v2` | — | [Realism Engine Klein v2](https://civitai.com/models/2374977) — a general nudity and anatomy finetune for klein 9B. Adult content |
| `nsfw-unlocked-v2` | `nude`, `naked`, `blow job`, `cum`, `ass`, `pussy` | [NSFW Unlocked v2](https://civitai.com/models/2063193?modelVersionId=3030169) — an explicit-content LoRA. Adult content |
| `naturalbeauty-v2` | `naked`, `topless`, `bottomless` | [NaturalBeauty v2](https://civitai.com/models/2532692?modelVersionId=2972296) — photorealistic female nudity. Adult content |

**Trigger words matter.** An adapter whose trigger is absent from the prompt
loads without error and simply is not invoked — the usual reason a LoRA appears
to do nothing. `GET /variants` lists them per adapter.

A dash in that column means the adapter has no trigger and applies to every
prompt — `snofs-v1.4` and `realism-engine-v2` are general finetunes rather than
concept adapters.

**Strength defaults per adapter.** Where an author publishes a recommended
band, the registry carries it and the server applies its midpoint when a request
omits `lora_strength`:

| Adapter | Published band | Applied by default |
| --- | --- | --- |
| `nsfw-unlocked-v2` | 0.5–0.9 | 0.7 |
| `realism-engine-v2` | 1.0–1.25 | 1.125 |
| everything else | — | 1.0 |

An explicit `lora_strength` always wins, including an explicit `1.0`. Omitting it
is what opts into the recommendation, so `--lora-strength` has no default at the
CLI. `GET /variants` reports both `recommended_strength` and the
`default_strength` that would be applied.

`nsfw-unlocked-v2` also suggests 20+ steps at cfg 3.5, so it pairs with `base`
rather than the 4-step `distilled` variant.

**Both of the last two ship builds for other architectures** under the same
Civitai model — Flux.1 D and Z-Image for one, Krea 2 for the other. The pinned
`model_version_id` in `app.py` is what selects the klein build, so it is not a
detail to update casually.

```bash
uv run python client.py generate "..." --lora snofs-v1.4 --lora-strength 0.8
```

`GET /variants` lists the registry alongside the variants. Adapters are applied
**model-only**: they patch the transformer, not the text encoder, so they compose
with `ponpoke-uncensored` as readily as with `base`.

Adding another means two lines — an entry in `LORAS` in `workflow.py` and a
matching weight file in `app.py`, so the weights are on the Volume before a
request can name them. A test asserts those two stay in step. Adapters may come
from Hugging Face or Civitai; Civitai ones pin a SHA256, since the CDN offers no
integrity guarantee of its own.

> **`snofs-v1.4` licence.** Ashen3 releases SNOFS under a *Model Personal Use
> License (No Service, No Derivatives, No Redistribution)*. It permits local use
> and selling the images, but prohibits running the model as a service — its
> definition of "Commercial Service Use" explicitly names APIs and hosted
> workflows, whether or not money changes hands. A private single-user
> deployment is a judgement call; anything multi-user or public needs a paid
> licence from the author. The other weights here are unaffected.

## Using it from ComfyUI

```bash
cp -r ../comfy_node /path/to/ComfyUI/custom_nodes/comfyui-modal-remote
```

Fill in the endpoint and token (see [`../comfy_node/README.md`](../comfy_node/README.md),
which also covers the Kubernetes Secret route), restart ComfyUI, and the
**FLUX.2 klein (Modal)** node appears. It returns an ordinary `IMAGE` tensor and
mirrors the remote sampler's progress onto its local progress bar.

The endpoint also speaks the ComfyUI protocol verbatim, so any existing ComfyUI
API client works against it. `workflows/flux2_klein_9b_t2i_api.json` is a
ready-to-POST graph.

## Using it from the CLI

```bash
uv run python client.py health
uv run python client.py variants
uv run python client.py validate                     # graph vs. deployed node schemas
uv run python client.py generate "a neon ramen shop in the rain" --aspect-ratio 16:9
uv run python client.py generate "a portrait" --variant distilled
uv run python client.py generate "a city" --negative "blurry, jpeg artifacts"
```

Or without the endpoint at all — `modal run` calls the container directly and
needs no proxy token:

```bash
uv run modal run app.py --prompt "a brutalist library at golden hour" --variant distilled
```

## API reference

`POST /generate` — all fields optional except `prompt`:

| Field | Default | Notes |
| --- | --- | --- |
| `prompt` | — | Natural language |
| `negative_prompt` | `""` | Base variant only |
| `variant` | `base` | `base`, `distilled` or `ponpoke-uncensored` |
| `lora` | — | Adapter name from `/variants`; omit for none |
| `lora_strength` | the adapter's own | Omit to use its recommended strength; ignored when no `lora` is named |
| `width`, `height` | 1024 | 256–2048, snapped to /16 |
| `aspect_ratio` | — | `1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `9:16`, `16:9`, `21:9`; overrides width/height |
| `megapixels` | 1.0 | Pixel budget used with `aspect_ratio` |
| `steps`, `cfg` | variant | Explicit overrides |
| `seed` | random | 0 – 2^64-1 |
| `batch_size` | 1 | 1–8 |
| `sampler_name` | `euler` | Any ComfyUI sampler |
| `timeout_s` | 900 | Server-side wait |
| `client_id` | generated | Subscribe to `/ws?clientId=<id>` for this render's progress |

Response: `{prompt_id, duration_s, params, images: [{filename, content_type, b64}]}`.
`POST /generate/image` takes the same body and returns raw PNG bytes.

## Configuration

All read at deploy time; see [`.env.example`](.env.example).

| Variable | Default | Effect |
| --- | --- | --- |
| `FLUX2KLEIN_HF_SECRET` | `huggingface-secret` | Modal Secret holding `HF_TOKEN` |
| `FLUX2KLEIN_GPU` | `H100` | Any Modal GPU string |
| `FLUX2KLEIN_MIN_CONTAINERS` | `0` | Warm containers |
| `FLUX2KLEIN_MAX_CONTAINERS` | `1` | Keep at 1 unless clients submit and poll in one request |
| `FLUX2KLEIN_SCALEDOWN_WINDOW` | `300` | Idle seconds before shutdown |
| `FLUX2KLEIN_CONCURRENT_INPUTS` | `20` | Requests in flight per container |
| `FLUX2KLEIN_REQUIRE_AUTH` | `1` | Proxy auth on the API endpoint |
| `FLUX2KLEIN_UI_REQUIRE_AUTH` | `0` | Proxy auth on the raw UI endpoint |

### Choosing a GPU

~44 GB of weights sit on disk, but only one transformer and one encoder load at
a time. The working set is about **18 GB** for `base` and `distilled`, and about
**26 GB** for `ponpoke-uncensored`, whose encoder is bf16 rather than fp8mixed.
Both are lighter than the ideogram4 service.

The fp8 checkpoints want fp8 tensor cores, meaning compute capability 8.9+
(Ada, Hopper, Blackwell). `L40S` (48 GB, roughly half H100's hourly rate) has
them and holds the working set comfortably. `A100` and `A10` do not, and will
fall back to fp16 compute.

**Compare cost per image, not per hour** — a card that is proportionally slower
costs more. Measure with two warm runs before switching:

```bash
FLUX2KLEIN_GPU=L40S uv run modal deploy app.py
uv run python client.py generate "benchmark" --variant distilled   # cold, ignore
uv run python client.py generate "benchmark" --variant distilled   # warm, compare duration_s
```

The `distilled` variant is the far bigger lever, and it cuts on two axes at
once: 4 steps against 20, *and* one transformer pass per step instead of two,
because ComfyUI skips the unconditional branch when cfg is 1. That is 4
transformer evaluations against 40 — a ~10x reduction in sampling work, which
dwarfs any GPU choice.

Wall-clock gain is smaller than 10x: text encoding and VAE decode happen once
per image regardless, so they become a proportionally larger share of a short
render. Measure before assuming a number.

## Project structure

```
flux2klein/
├── app.py         Modal object graph: weights, GPU class, endpoints
├── server.py      Request model, resolver and the /variants route
├── workflow.py    The FLUX.2 klein graph in ComfyUI API format
├── client.py      CLI: generate / variants / health / validate
├── workflows/     Ready-to-POST API-format graph
└── tests/         FLUX.2-klein-specific assertions
```

Everything generic lives outside this directory: `../comfyui_modal` (container
image, ComfyUI supervisor, ASGI proxy, CLI plumbing) and `../comfy_node` (the
ComfyUI nodes for every service).

## Testing

```bash
uv run pytest -q                   # offline, from the repository root
uv run python client.py validate   # graph vs. the deployed node schemas
```

## Troubleshooting

**`GatedRepoError` / 401 during download.** The token's account has not accepted
the FLUX.2 licence, or `HF_TOKEN` is missing from the Modal Secret.

**`weights Volume is missing ...`** — run `modal run app.py::download_models`.

**Distilled output looks washed out or over-contrasted.** You are probably
driving it with a CFG above 1. Turn `override_sampler` off, or set `cfg` to 1.

**Negative prompt appears to do nothing.** Expected on `distilled`; switch to
`base`.

**First request takes minutes.** Cold start: image pull, ComfyUI boot, then
18–26 GB onto the GPU. Set `FLUX2KLEIN_MIN_CONTAINERS=1` if latency matters more
than idle cost.
