# ULTRA (Krea 2) on Modal, served as a ComfyUI API

Runs [**ULTRA**](https://civitai.com/models/228525) by `AIA_civit` — a community
finetune of **Krea 2** — on a Modal GPU and exposes it as a **ComfyUI server**.
Same pattern as the other services here: point a local or clustered ComfyUI at
the URL and render remotely.

> **Source.** The checkpoint is fetched from **civitai.com** and verified against
> the SHA256 Civitai publishes. This model is also listed on a mirror domain
> whose certificate was days old when this was written; a pinned digest means a
> substituted file fails closed no matter which host serves the bytes. If you
> ever need to fetch through a mirror, change the host but keep the digest.

> **Licence.** Civitai's terms for this model permit derivatives and require no
> credit, but restrict commercial use to Civitai's own rent tiers — narrower
> than the WAI service. This repository ships no weights.

## What it needs

ULTRA is distributed as a **diffusion model only** — its safetensors carries
`model.diffusion_model.*` and nothing else — so the encoder and VAE come from
Comfy-Org's Krea 2 mirror. Nothing is gated; no token is required.

| File | Size | Source |
| --- | --- | --- |
| `ultra_v15.safetensors` (int8) | 13.80 GB | Civitai, digest-verified |
| `qwen3vl_4b_fp8_scaled.safetensors` | 5.24 GB | `Comfy-Org/Krea-2` |
| `qwen_image_vae.safetensors` | 0.25 GB | `Comfy-Org/Krea-2` |

Krea 2 pairs with the **4B** Qwen3-VL on a 12-layer tap and the **Qwen-Image**
autoencoder — neither is interchangeable with the 8B encoder or Flux-family VAE
the klein services use.

## Setup

```bash
# from the repository root
uv sync --all-groups
cd ultra

uv run modal run app.py::download_models   # ~19 GB into a Volume. One-off.
uv run modal deploy app.py
```

`download_models` runs on CPU — no GPU charge — and is idempotent by
destination; pass `--force` to refetch. Copy the endpoint URL from the deploy
output; you want the one for `Ultra.web`.

```bash
export ULTRA_MODAL_URL=https://...
export MODAL_KEY=wk-...  MODAL_SECRET=ws-...
```

> Or set `MODAL_WORKSPACE=your-workspace` once and every service's URL is
> derived from it — see [`comfy_node/README.md`](../comfy_node/README.md).
> The explicit variable above still wins when set.

## Using it

From ComfyUI, install the shared node package and the **ULTRA / Krea 2 (Modal)**
node appears:

```bash
cp -r ../comfy_node /path/to/ComfyUI/custom_nodes/comfyui-modal-remote
```

From the CLI:

```bash
uv run python client.py health
uv run python client.py defaults
uv run python client.py validate                     # graph vs. deployed node schemas
uv run python client.py generate "a cyclist on a rain-slick street at dusk" --aspect-ratio 16:9
```

Or without proxy tokens at all:

```bash
uv run modal run app.py --prompt "a neon-lit alley after rain"
```

## Sampler defaults, and one open question

Defaults come from ComfyUI's official **Krea 2 turbo** template — verified, not
inferred:

| Setting | Default |
| --- | --- |
| `steps` | 8 |
| `cfg` | 1.0 |
| `sampler_name` | `euler` |
| `scheduler` | `simple` |

**ULTRA's listing states no settings and does not say whether it descends from
the turbo or the raw branch of Krea 2.** Both are the same size in int8, so the
file cannot settle it either. The turbo defaults are the safe starting point —
if output looks undercooked or washed out, that is the signal it is raw-based,
and `--steps 20 --cfg 3` upward is the direction to explore. Worth establishing
once on a fixed seed.

At the default `cfg 1` the negative branch is never consulted, so the graph
zeroes it exactly as the reference template does. Supplying `negative_prompt`
swaps in a real encoder, which only does anything if you also raise `cfg`.

## img2img

Supply a source image and the sampler starts from it instead of an empty
latent. `denoise` then controls how much of the original survives — it is inert
without a source, which is why it only takes effect here.

```bash
uv run python client.py generate "make it dusk" --source photo.png --denoise 0.55
```

Or over the API, base64-encoded:

```json
{"prompt": "make it dusk", "source_image": "iVBORw0KGgo...", "denoise": 0.55}
```

In ComfyUI, connect anything producing an `IMAGE` to the node's optional
`source_image` input. Leave it unconnected and the node stays text-to-image, so
saved workflows are unaffected.

Three things change once a source is present:

- **The source sets the output size.** `width`, `height` and `aspect_ratio` are
  ignored; the image is scaled to `source_megapixels` and its dimensions drive
  the render. The response reports `width`/`height` as `null` and
  `is_img2img: true` rather than echoing values that were never applied.
- **`batch_size` must be 1.** One encoded source is one starting latent.
- **Steps are not reduced.** ComfyUI builds a `steps / denoise` schedule and
  keeps the last `steps + 1` sigmas, so 8 steps at `denoise 0.5` still samples
  8 times. No need to raise `steps` to compensate.

> **Unmeasured.** Krea 2 publishes no img2img recipe — the graph is standard
> ComfyUI img2img, but which `denoise` values look good on a turbo finetune at
> cfg 1 has not been measured here. Expect to find the usable band yourself; if
> you do, write it into this file.

## API reference

`POST /generate` — all fields optional except `prompt`:

| Field | Default | Notes |
| --- | --- | --- |
| `prompt` | — | Natural language |
| `negative_prompt` | `""` | Inactive at cfg 1 |
| `width`, `height` | 1024 | 256–2048, snapped to /16 |
| `aspect_ratio` | — | Overrides width/height; `megapixels` sets the budget |
| `steps` | 8 | |
| `cfg` | 1.0 | |
| `sampler_name` | `euler` | Any ComfyUI sampler |
| `scheduler` | `simple` | Any ComfyUI scheduler |
| `denoise` | 1.0 | Only meaningful with `source_image` |
| `source_image` | — | Base64 image to start from; see [img2img](#img2img) |
| `source_megapixels` | 1.0 | The source is scaled to this before encoding |
| `seed` | random | |
| `batch_size` | 1 | 1–8 |
| `client_id` | generated | Subscribe to `/ws?clientId=<id>` for progress |

Response: `{prompt_id, duration_s, params, images: [{filename, content_type, b64}]}`.

## Configuration

| Variable | Default | Effect |
| --- | --- | --- |
| `ULTRA_GPU` | `L40S` | Any Modal GPU string |
| `ULTRA_MIN_CONTAINERS` | `0` | Warm containers |
| `ULTRA_MAX_CONTAINERS` | `1` | Keep at 1 unless clients submit and poll in one request |
| `ULTRA_SCALEDOWN_WINDOW` | `300` | Idle seconds before shutdown |
| `ULTRA_CONCURRENT_INPUTS` | `20` | Requests in flight per container |
| `ULTRA_REQUIRE_AUTH` | `1` | Proxy auth on the API endpoint |
| `ULTRA_UI_REQUIRE_AUTH` | `0` | Proxy auth on the raw UI endpoint |

### Choosing a GPU

~19 GB of weights, but ComfyUI offloads the text encoder after encoding, so the
sampling working set is nearer **14 GB**.

`L40S` (48 GB) is the default: comfortable headroom, and Ada supports the int8
kernels this quantisation uses. **`A10` (24 GB) very likely fits and is roughly
half the price** — the reason to hesitate is only that 19 GB is tight if both
the encoder and the transformer stay resident. It is the first thing to try if
you want this cheaper, and 8-step sampling makes each test quick:

```bash
ULTRA_GPU=A10 uv run modal deploy app.py
uv run python client.py generate "benchmark"   # cold, ignore
uv run python client.py generate "benchmark"   # warm, compare duration_s
```

Compare cost per image, not per hour.

## Project structure

```
ultra/
├── app.py         Modal object graph: weight table, GPU class, endpoints
├── server.py      Request model, resolver and the /defaults route
├── workflow.py    The Krea 2 graph in ComfyUI API format
├── client.py      CLI: generate / defaults / health / validate
├── workflows/     Ready-to-POST API-format graph
└── tests/         Krea-2-specific assertions
```

Everything generic lives outside this directory: `../comfyui_modal` (container
image, ComfyUI supervisor, ASGI proxy, weight fetching, CLI plumbing) and
`../comfy_node` (the ComfyUI nodes for every service).

## Troubleshooting

**`checksum mismatch`.** The download did not match Civitai's published SHA256.
Re-run; if it persists the upstream file changed and the pin in `app.py` needs
updating deliberately.

**Output looks undercooked or washed out.** Most likely ULTRA is raw-based
rather than turbo-based. Raise `--steps` and `--cfg` together.

**`weights Volume is missing ...`** — run `modal run app.py::download_models`.
