# RedCraft v3 (Krea 2) on Modal, served as a ComfyUI API

Runs the **"赤佬 3.0 (Krea2)"** version of
[**RedCraft**](https://civitai.com/models/958009?modelVersionId=3139241) by
`AiMetatron` — a community finetune of **Krea 2 turbo** — on a Modal GPU and
exposes it as a **ComfyUI server**. Same pattern as the other services here:
point a local or clustered ComfyUI at the URL and render remotely.

> **The listing is a grab-bag; the version id is what matters.** That Civitai
> page carries twenty versions across half a dozen unrelated base models —
> Flux.1 D, SDXL, SD 1.5, Pony, Illustrious, Z-Image, MiniMax H3, LTX 2.5 and
> Krea 2. Its title mentions "LTX25 2K", which belongs to a different version
> entirely. Only `3139241` is the Krea 2 build served here.

> **Four precisions share one version id — and one filename.** See
> [Precisions](#precisions). The `file_id` is what selects which is fetched.

> **Source.** The checkpoint is fetched from **civitai.com** and verified
> against the SHA256 Civitai publishes, so a substituted file fails closed. This
> repository ships no weights.

## Licence

**Krea 2 Community License.** Everything derived from Krea 2 — this service,
`ultra/`, `finepornv4/` and `redgpt2gpt/` alike — inherits it. Free commercial
use is conditional on your **company-wide annual revenue being under $1,000,000
USD**, on a trailing twelve-month basis, counting affiliated entities under
common ownership. You own the outputs you generate provided you comply, and Krea
claims no ownership of them.

**AiMetatron access terms.** The model card describes a paid-access package
covering this and sibling models. The Krea 2 build deployed here downloads
without credentials, but check the card before any commercial use.

## What it needs

Like the other three Krea 2 services it is distributed as a **diffusion model
only**, so the encoder and VAE come from Comfy-Org's Krea 2 mirror — the same
two files, byte for byte. A test asserts they stay in step.

| File | Size | Source |
| --- | --- | --- |
| `redcraft_v3_krea2_fp8.safetensors` | 12.83 GB | Civitai, digest-verified |
| `qwen3vl_4b_fp8_scaled.safetensors` | 5.24 GB | `Comfy-Org/Krea-2` |
| `qwen_image_vae.safetensors` | 0.25 GB | `Comfy-Org/Krea-2` |

**No credentials needed.** Unlike `finepornv4/` and `redgpt2gpt/`, this listing
is not NSFW-flagged and a ranged GET against the real download URL returns `206`
anonymously — verified, not assumed. Don't add a `CIVITAI_TOKEN` Secret here on
the assumption the family should match; a test asserts none is wired.

Krea 2 pairs with the **4B** Qwen3-VL on a 12-layer tap and the **Qwen-Image**
autoencoder — neither is interchangeable with the 8B encoder or Flux-family VAE
the klein services use.

## Setup

```bash
# from the repository root
uv sync --all-groups
cd redcraft3

uv run modal run app.py::download_models   # ~18 GB into a Volume. One-off.
uv run modal deploy app.py
```

`download_models` runs on CPU — no GPU charge — and is idempotent by
destination; pass `--force` to refetch. Copy the endpoint URL from the deploy
output; you want the one for `RedCraft3.web`.

```bash
export REDCRAFT3_MODAL_URL=https://...
export MODAL_KEY=wk-...  MODAL_SECRET=ws-...
```

> Or set `MODAL_WORKSPACE=your-workspace` once and every service's URL is
> derived from it — see [`comfy_node/README.md`](../comfy_node/README.md).
> The explicit variable above still wins when set.

## Using it

From ComfyUI, install the shared node package and the **RedCraft v3 / Krea 2
(Modal)** node appears:

```bash
cp -r ../comfy_node /path/to/ComfyUI/custom_nodes/comfyui-modal-remote
```

From the CLI:

```bash
uv run python client.py generate "a rain-soaked neon alley at night" --aspect-ratio 16:9
uv run python client.py generate "..." --sampler er_sde --steps 12
uv run python client.py defaults
```

Or as a plain ComfyUI server — `/prompt`, `/history`, `/view`, `/object_info`
and `/ws` all work unmodified, and `workflows/redcraft3_krea2_t2i_api.json` is
a ready-to-POST graph.

## Sampler defaults

The version notes publish a recipe verbatim:

> ER_SDE / Euler | Simple | CFG = 1 | 8-12 Steps

| Setting | Value | Why |
| --- | --- | --- |
| `sampler_name` | `euler` | The card names it and `er_sde` interchangeably |
| `scheduler` | `simple` | From the card |
| `cfg` | 1.0 | From the card |
| `steps` | 10 | Midpoint of the published 8–12 |

**Why `euler` and not the `er_sde` the card lists first.** The two are offered
interchangeably, and `euler` is present in every ComfyUI build. Whether a given
deployment has `er_sde` is answerable only against a live one — run
`client.py validate` — and a default that failed to resolve would break every
request that omits a sampler. `er_sde` is offered in the node dropdown and
accepted by the API; `GET /defaults` reports it as `alternate_sampler`.

This is a real difference from `redgpt2gpt/`, the other service over this same
base, whose edition publishes **no** settings and therefore falls back to
ComfyUI's generic Krea 2 turbo template. `/defaults` on each says which it is,
so don't copy settings between them.

At cfg 1 the **negative prompt is inert**: the graph zeroes the conditioning
rather than encoding it. Supplying negative text swaps in a real encoder, which
only becomes useful alongside a raised cfg.

## Precisions

This one version publishes four builds and gives **all four the same filename**,
`redcraftHybridH3A2A_30Krea2.safetensors`. The version id identifies none of
them; only the `file_id` does.

| Build | Size | File id | Notes |
| --- | --- | --- | --- |
| fp8 | 12.83 GB | `3019490` | **Deployed.** The version's primary file |
| int8 | 12.53 GB | `3019607` | Marginally smaller |
| nvfp4 | 7.49 GB | `3064389` | Needs a Blackwell-class GPU for native nvfp4 |
| int4 | 6.27 GB | `3019523` | Smallest |

Switching means editing one `CivitaiFile` in `app.py` — the file id and the
digest, the version id stays — plus the `destination` filename, then re-running
`download_models`. Take the digest from the API rather than the page.

## API reference

`POST /generate`

| Field | Default | Notes |
| --- | --- | --- |
| `prompt` | required | Natural language |
| `negative_prompt` | `""` | Inert at cfg 1 |
| `width` / `height` | 1024 | Snapped to a multiple of 16, clamped to 256–2048 |
| `aspect_ratio` | — | Overrides width/height using `megapixels` as the budget |
| `megapixels` | 1.0 | |
| `steps` | 10 | Card publishes 8–12 |
| `cfg` | 1.0 | |
| `sampler_name` | `euler` | `er_sde` is the card's other choice |
| `scheduler` | `simple` | |
| `denoise` | 1.0 | |
| `batch_size` | 1 | |
| `seed` | random | |
| `client_id` | generated | Subscribe to `/ws?clientId=<id>` for progress |

`GET /defaults` returns the sampler conventions, the published step range, the
alternate sampler and the source of it all. `GET /health` proxies ComfyUI's
`/system_stats`.

## Configuration

Deploy-time settings come from the environment; see `.env.example`.

| Variable | Default | What it does |
| --- | --- | --- |
| `REDCRAFT3_GPU` | `L40S` | See below |
| `REDCRAFT3_MIN_CONTAINERS` | `0` | Warm containers |
| `REDCRAFT3_MAX_CONTAINERS` | `1` | Raise only if every client submits and polls in one request |
| `REDCRAFT3_SCALEDOWN_WINDOW` | `300` | Seconds warm after the last request |
| `REDCRAFT3_CONCURRENT_INPUTS` | `20` | Per container |
| `REDCRAFT3_REQUIRE_AUTH` | `1` | Proxy auth on the API |
| `REDCRAFT3_UI_REQUIRE_AUTH` | `0` | Browsers cannot attach the headers |

### Choosing a GPU

~18 GB of weights, but the text encoder offloads after encoding, so the sampling
working set is nearer 13 GB — the same shape as `ultra/` and `redgpt2gpt/`, and
much lighter than `finepornv4/`.

- **L40S (48 GB)** — the default, with plenty of headroom.
- **A10 (24 GB)** — about half the price and very likely fits.

Compare **cost per image, not per hour**. Measure with two warm runs:

```bash
uv run python client.py generate "benchmark"   # cold, ignore
uv run python client.py generate "benchmark"   # warm, compare duration_s
```

## Project structure

```
redcraft3/
├── app.py         Modal entrypoint: weights, container, endpoints
├── server.py      Request model, resolver, /defaults route
├── workflow.py    The Krea 2 graph in ComfyUI API format
├── client.py      CLI against a deployed endpoint
├── workflows/     Ready-to-POST API-format graph
└── tests/         Offline structural tests
```

## Troubleshooting

**First request takes minutes.** Cold start: image pull, ComfyUI boot, then the
weights onto the GPU. Set `REDCRAFT3_MIN_CONTAINERS=1` if latency matters more
than idle cost.

**`er_sde` is rejected at queue time.** That sampler is not in every ComfyUI
build. `uv run python client.py validate` checks the graph against the deployed
node schemas; stay on `euler`, which the card offers as an equal alternative.

**The wrong precision landed on the Volume.** All four builds share a filename
upstream, so this is easy to do by hand. The digest check refuses a mismatch, so
compare `file_id` against the API — not the download page.

**`/generate` reports a few seconds but the client waits far longer.** The
`duration_s` clock stops inside the container, before the response is serialised
and shipped. Images come back base64-encoded in JSON, so a large `batch_size` is
a multi-megabyte download. Separate the two with
`curl -w 'ttfb=%{time_starttransfer} total=%{time_total} bytes=%{size_download}'`.
