# FinePorn v4 (Krea 2) on Modal, served as a ComfyUI API

Runs [**FinePorn v4**](https://civitai.com/models/2762538?modelVersionId=3197873)
by `Reevo` — a community merge of **Krea 2 turbo** — on a Modal GPU and exposes
it as a **ComfyUI server**. Same pattern as the other services here: point a
local or clustered ComfyUI at the URL and render remotely.

This is the **bf16** build, the largest and most accurate of the four the author
publishes. See [Choosing a precision](#choosing-a-precision) before committing to
it — the fp8 and nvfp4 builds are a third to a half the size.

> **Adult content.** This merge targets explicit imagery. Nothing about the
> deployment restricts prompts; the endpoint defaults to requiring
> `Modal-Key` / `Modal-Secret`, and you should leave that on.

> **Source.** The checkpoint is fetched from **civitai.com** and verified against
> the SHA256 Civitai publishes, so a substituted file fails closed no matter
> which host serves the bytes. This repository ships no weights.

> **Not the author's own training.** The model card is explicit that FinePorn is
> a *merge*: a set of third-party LoRAs baked into Krea 2 at balanced weights,
> each credited on the card. Licence terms follow from those components, not
> from this repository.

## What it needs

Like ULTRA it is distributed as a **diffusion model only**, so the encoder and
VAE come from Comfy-Org's Krea 2 mirror — the same two files that service uses,
byte for byte. Nothing is gated; no token is required.

| File | Size | Source |
| --- | --- | --- |
| `fineporn_v4_bf16.safetensors` | 25.66 GB | Civitai, digest-verified |
| `qwen3vl_4b_fp8_scaled.safetensors` | 5.24 GB | `Comfy-Org/Krea-2` |
| `qwen_image_vae.safetensors` | 0.25 GB | `Comfy-Org/Krea-2` |

Krea 2 pairs with the **4B** Qwen3-VL on a 12-layer tap and the **Qwen-Image**
autoencoder — neither is interchangeable with the 8B encoder or Flux-family VAE
the klein services use.

The Civitai filename is `finepornV4INT8NVFP4BF16_v4Bf16.safetensors`, which
names all four precisions for a file that is only one of them. It is renamed on
the way into the Volume so the graph reads clearly.

## Setup

```bash
# from the repository root
uv sync --all-groups
cd finepornv4

uv run modal run app.py::download_models   # ~31 GB into a Volume. One-off.
uv run modal deploy app.py
```

`download_models` runs on CPU — no GPU charge — and is idempotent by
destination; pass `--force` to refetch. Copy the endpoint URL from the deploy
output; you want the one for `FinePornV4.web`.

```bash
export FINEPORNV4_MODAL_URL=https://...
export MODAL_KEY=wk-...  MODAL_SECRET=ws-...
```

## Using it

From ComfyUI, install the shared node package and the **FinePorn v4 / Krea 2
(Modal)** node appears:

```bash
cp -r ../comfy_node /path/to/ComfyUI/custom_nodes/comfyui-modal-remote
```

From the CLI:

```bash
uv run python client.py generate "this is a casual, low-quality photo of ..."
uv run python client.py generate "..." --aspect-ratio 3:4 --steps 12
uv run python client.py defaults
```

Or as a plain ComfyUI server — `/prompt`, `/history`, `/view`, `/object_info`
and `/ws` all work unmodified, and `workflows/finepornv4_krea2_t2i_api.json` is a
ready-to-POST graph.

## Prompting

The card is emphatic that this merge targets a *smartphone-snapshot* look and
renders flatter without being asked for it. Its author recommends opening
prompts with a phrase like:

> This is a casual, low-quality photo

> this is an amateur photo taken from smartphone, casual photo

This is **advice, not a trigger word** — there are no trained words on this
model, and the service never injects the phrase for you. `GET /defaults`
reports it under `prompt_guidance` so a client can surface it.

## Sampler and resolution defaults

Unlike the ULTRA service, whose defaults come from ComfyUI's generic Krea 2
turbo template, these come from **the model card's own V4 section**:

| Setting | Value | Why |
| --- | --- | --- |
| `sampler_name` | `euler` | Card lists `euler + beta` and `er_sde + simple` for v4 |
| `scheduler` | `beta` | The divergence from ULTRA, which uses `simple` |
| `cfg` | 1.0 | A turbo merge; every published version samples at 1 |
| `steps` | 10 | Card gives 8–10 for v1 and 8–12 for v2, and restates none for v4 |

**Resolution is the unusual part.** The author reports that standard Krea 2
sizes underperform on this merge and recommends scaling them up, so this service
defaults to **1280×1280** rather than the 1024×1024 every other service here
uses:

| Standard | Optimal (×1.25) | Recommended (×1.5) |
| --- | --- | --- |
| 832×1216 | 1040×1520 | 1248×1824 |
| 896×1152 | 1120×1440 | 1344×1728 |
| 1024×1024 | 1280×1280 | 1536×1536 |

Naming an `aspect_ratio` keeps the same raised pixel budget (1.64 MP) instead of
dropping back to 1 MP. Explicit `width`/`height` always win. Because the CLI and
the ComfyUI node always send a size, both carry these defaults too — a client
left at 1024 would silently override the server on every call.

At the default cfg 1 the **negative prompt is inert**: the graph zeroes the
conditioning rather than encoding it. Supplying negative text swaps in a real
encoder, which only becomes useful alongside a raised cfg.

## API reference

`POST /generate`

| Field | Default | Notes |
| --- | --- | --- |
| `prompt` | required | Natural language; see [Prompting](#prompting) |
| `negative_prompt` | `""` | Inert at cfg 1 |
| `width` / `height` | 1280 | Snapped to a multiple of 16, clamped to 256–2048 |
| `aspect_ratio` | — | Overrides width/height using `megapixels` as the budget |
| `megapixels` | 1.64 | Matched to the 1280×1280 default |
| `steps` | 10 | Card publishes 8–12 |
| `cfg` | 1.0 | Raising it is off-recipe for a turbo merge |
| `sampler_name` | `euler` | `er_sde` is the card's other v4 pairing |
| `scheduler` | `beta` | Pair `er_sde` with `simple` |
| `denoise` | 1.0 | |
| `batch_size` | 1 | |
| `seed` | random | |
| `client_id` | generated | Subscribe to `/ws?clientId=<id>` for progress |

`GET /defaults` returns everything above plus the recommended-resolution table
and the prompting guidance. `GET /health` proxies ComfyUI's `/system_stats`.

## Configuration

Deploy-time settings come from the environment; see `.env.example`.

| Variable | Default | What it does |
| --- | --- | --- |
| `FINEPORNV4_GPU` | `H100` | See below |
| `FINEPORNV4_MIN_CONTAINERS` | `0` | Warm containers |
| `FINEPORNV4_MAX_CONTAINERS` | `1` | Raise only if every client submits and polls in one request |
| `FINEPORNV4_SCALEDOWN_WINDOW` | `300` | Seconds warm after the last request |
| `FINEPORNV4_CONCURRENT_INPUTS` | `20` | Per container |
| `FINEPORNV4_REQUIRE_AUTH` | `1` | Proxy auth on the API |
| `FINEPORNV4_UI_REQUIRE_AUTH` | `0` | Browsers cannot attach the headers |

### Choosing a GPU

The sampling working set is the 25.7 GB transformer plus activations; the text
encoder offloads after encoding. Activations are larger here than in the other
services because the default render is 1.64 MP rather than 1.0.

- **H100 (80 GB)** — the default. Comfortable headroom at 1536×1536 and
  batch > 1, and much faster bf16 throughput than Ada.
- **L40S (48 GB)** — fits with room to spare and costs less per hour. The
  sensible choice if you render one 1280×1280 image at a time.
- Anything under 32 GB will not hold this build. Use a smaller precision.

Compare **cost per image, not per hour** — a proportionally slower card costs
more. Measure with two warm runs:

```bash
uv run python client.py generate "benchmark"   # cold, ignore
uv run python client.py generate "benchmark"   # warm, compare duration_s
```

### Choosing a precision

The author publishes four builds of v4 under one Civitai listing. This service
pins the bf16 one because that is what was asked for, but it is the most
expensive choice on both storage and VRAM:

| Build | Size | Version id | Notes |
| --- | --- | --- | --- |
| bf16 | 25.66 GB | `3197873` | **Deployed.** Native precision; the author calls it the most accurate and the slowest |
| int8 | 13.18 GB | `3187539` | Roughly half the size |
| fp8 | 12.83 GB | `3187539` | Same version as int8, different file id |
| nvfp4 | 7.98 GB | `3215452` | Smallest; needs a Blackwell-class GPU for native nvfp4 |

Switching means editing one `CivitaiFile` in `app.py` — the version id, the file
id and the digest — then re-running `download_models`. Take all three from the
API rather than the page, and note that the int8 and fp8 builds **share a
version id and a filename**, so only the file id tells them apart.

## Project structure

```
finepornv4/
├── app.py         Modal entrypoint: weights, container, endpoints
├── server.py      Request model, resolver, /defaults route
├── workflow.py    The Krea 2 graph in ComfyUI API format
├── client.py      CLI against a deployed endpoint
├── workflows/     Ready-to-POST API-format graph
└── tests/         Offline structural tests
```

## Troubleshooting

**First request takes minutes.** Cold start: image pull, ComfyUI boot, then
25.7 GB onto the GPU. This is the slowest cold start of any service here. Set
`FINEPORNV4_MIN_CONTAINERS=1` if latency matters more than idle cost.

**`/generate` reports a few seconds but the client waits far longer.** The
`duration_s` clock stops inside the container, before the response is
serialised and shipped. Images come back base64-encoded in JSON, so a large
`batch_size` or a 1536×1536 render is a multi-megabyte download. Time it with
`curl -w 'ttfb=%{time_starttransfer} total=%{time_total} bytes=%{size_download}'`
to separate render time from transfer.

**Output looks flat or over-polished.** Check the prompt opens with the card's
recommended phrasing, and confirm you are rendering at 1280 or above — a client
pinned to 1024 undoes the resolution default.

**`checksum mismatch ... refusing to install`.** Civitai served different bytes
than the digest pinned in `app.py`. Re-check the file id against the API; the
four precisions of v4 are easy to mix up.
