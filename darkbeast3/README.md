# Dark Beast v3 (Krea 2) on Modal, served as a ComfyUI API

Runs **"Dark Beast 3 黑兽3.0"** from
[**Dark Beast | H3 Director Edition**](https://civitai.com/models/2242173?modelVersionId=3173268)
by `AiMetatron` — a community finetune of **Krea 2 turbo** — on a Modal GPU and
exposes it as a **ComfyUI server**. Same pattern as the other services here:
point a local or clustered ComfyUI at the URL and render remotely.

> **This is a still-image model, despite the listing.** That page is titled "H3
> Director Edition" and its description is about a *video* pipeline: automated
> short-film production, 2K single-pass sampling, frame interpolation, 6-10 step
> guidance. All of that belongs to a **different version** on the same page
> (`3274224`), whose base model is MiniMax H3. The version deployed here is
> Krea 2 and produces stills. See [Versions](#versions).

> **Five precisions share one version id — and one filename.** The `file_id` is
> what selects which is fetched.

> **Adult content.** This finetune targets explicit, uncensored imagery. The
> endpoint defaults to requiring `Modal-Key` / `Modal-Secret`; leave that on.

> **Source.** The checkpoint is fetched from **civitai.com** and verified
> against the SHA256 Civitai publishes, so a substituted file fails closed. This
> repository ships no weights.

## Licence

**Krea 2 Community License.** Everything derived from Krea 2 — this service,
`ultra/`, `finepornv4/`, `redgpt2gpt/` and `redcraft3/` alike — inherits it.
Free commercial use is conditional on your **company-wide annual revenue being
under $1,000,000 USD**, on a trailing twelve-month basis, counting affiliated
entities under common ownership. You own the outputs you generate provided you
comply, and Krea claims no ownership of them.

**AiMetatron access terms.** The model card describes a paid-access package
covering this and sibling models. The Krea 2 build deployed here downloads
without credentials, but check the card before any commercial use.

## What it needs

Like the other four Krea 2 services it is distributed as a **diffusion model
only**, so the encoder and VAE come from Comfy-Org's Krea 2 mirror — the same
two files, byte for byte. A test asserts they stay in step.

| File | Size | Source |
| --- | --- | --- |
| `darkbeast_v3_krea2_int8.safetensors` | 13.80 GB | Civitai, digest-verified |
| `qwen3vl_4b_fp8_scaled.safetensors` | 5.24 GB | `Comfy-Org/Krea-2` |
| `qwen_image_vae.safetensors` | 0.25 GB | `Comfy-Org/Krea-2` |

**No credentials needed** — and that is worth stating precisely, because the
obvious heuristic is wrong here. This listing **is** NSFW-flagged, and it still
serves an anonymous ranged GET (`206`, real safetensors bytes, checked against
the live URL). Meanwhile `finepornv4/` and `redgpt2gpt/` are flagged and answer
`401`. The flag decides nothing; only a request does. A test asserts no
`CIVITAI_TOKEN` Secret is wired here.

Krea 2 pairs with the **4B** Qwen3-VL on a 12-layer tap and the **Qwen-Image**
autoencoder — neither is interchangeable with the 8B encoder or Flux-family VAE
the klein services use. That matters on this listing in particular: it also
publishes FLUX.2 klein versions.

## Setup

```bash
# from the repository root
uv sync --all-groups
cd darkbeast3

uv run modal run app.py::download_models   # ~19 GB into a Volume. One-off.
uv run modal deploy app.py
```

`download_models` runs on CPU — no GPU charge — and is idempotent by
destination; pass `--force` to refetch. Copy the endpoint URL from the deploy
output; you want the one for `DarkBeast3.web`.

```bash
export DARKBEAST3_MODAL_URL=https://...
export MODAL_KEY=wk-...  MODAL_SECRET=ws-...
```

> Or set `MODAL_WORKSPACE=your-workspace` once and every service's URL is
> derived from it — see [`comfy_node/README.md`](../comfy_node/README.md).
> The explicit variable above still wins when set.

## Using it

From ComfyUI, install the shared node package and the **Dark Beast v3 / Krea 2
(Modal)** node appears:

```bash
cp -r ../comfy_node /path/to/ComfyUI/custom_nodes/comfyui-modal-remote
```

From the CLI:

```bash
uv run python client.py generate "a close-up portrait in hard directional light"
uv run python client.py generate "..." --aspect-ratio 3:4 --steps 12
uv run python client.py defaults
```

Or as a plain ComfyUI server — `/prompt`, `/history`, `/view`, `/object_info`
and `/ws` all work unmodified, and `workflows/darkbeast3_krea2_t2i_api.json` is
a ready-to-POST graph.

## Sampler defaults, and where they come from

| Setting | Value |
| --- | --- |
| `steps` | 8 |
| `cfg` | 1.0 |
| `sampler_name` | `euler` |
| `scheduler` | `simple` |

**These are ComfyUI's Krea 2 turbo template values, not the author's.** This
version's notes are marketing copy and publish no sampler settings, so the
template is the honest fallback — the same choice `ultra/` and `redgpt2gpt/`
make. `GET /defaults` says so outright in its `source` field.

The model description does give **"6-10 steps"**, but that figure describes the
H3 *video* edition's single-pass sampling on a different base model. It is not
transferable, and adopting it here would be importing a number from another
model. Contrast `redcraft3/`, by the same author over the same base, whose
version notes *do* publish a recipe — `/defaults` on each names its own source,
so don't copy settings between them.

At the default cfg 1 the **negative prompt is inert**: the graph zeroes the
conditioning rather than encoding it. Supplying negative text swaps in a real
encoder, which only becomes useful alongside a raised cfg.

## Versions

The listing spans fifteen versions across MiniMax H3, Z-Image Turbo, FLUX.2
klein, SDXL and Krea 2. Only the Krea 2 ones are servable by this graph.

| Version | Version id | Base | Notes |
| --- | --- | --- | --- |
| Dark Beast 3 黑兽3.0 | `3173268` | Krea 2 | **Deployed** |
| Dark Beast KREA 2 黑兽 FP8 | `3078453` | Krea 2 | Earlier |
| KREA2 黑兽1.1 INT8 Convrot | `3091496` | Krea 2 | Earlier |
| Dark Beast H3 director | `3274224` | MiniMax H3 | Video; needs a different graph entirely |
| DBKleinV2 / DBK | `2740209`, … | FLUX.2 klein 9B | Different encoder and VAE |

### Precisions

The deployed version publishes five builds and gives **all five the same
filename**, `darkBeastH3Director_darkBeast330.safetensors`. The version id
identifies none of them; only the `file_id` does.

| Build | Size | File id | Notes |
| --- | --- | --- | --- |
| int8 | 13.80 GB | `3053854` | **Deployed.** The version's primary file |
| fp8 | 12.83 GB | `3064149` | |
| bf16 | 25.66 GB | `3054219` | Largest; needs the headroom `finepornv4/` uses |
| nvfp4 | 7.49 GB | `3064226` | Needs a Blackwell-class GPU for native nvfp4 |
| int4 | 6.74 GB | `3064214` | Smallest |

Switching means editing one `CivitaiFile` in `app.py` — the file id, the digest
and the `destination` filename, the version id stays — then re-running
`download_models`. Take the digest from the API rather than the page.

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

`POST /generate`

| Field | Default | Notes |
| --- | --- | --- |
| `prompt` | required | Natural language |
| `negative_prompt` | `""` | Inert at cfg 1 |
| `width` / `height` | 1024 | Snapped to a multiple of 16, clamped to 256–2048 |
| `aspect_ratio` | — | Overrides width/height using `megapixels` as the budget |
| `megapixels` | 1.0 | |
| `steps` | 8 | |
| `cfg` | 1.0 | |
| `sampler_name` | `euler` | |
| `scheduler` | `simple` | |
| `denoise` | 1.0 | Only meaningful with `source_image` |
| `source_image` | — | Base64 image to start from; see [img2img](#img2img) |
| `source_megapixels` | 1.0 | The source is scaled to this before encoding |
| `batch_size` | 1 | |
| `seed` | random | |
| `client_id` | generated | Subscribe to `/ws?clientId=<id>` for progress |

`GET /defaults` returns the sampler conventions and names their source.
`GET /health` proxies ComfyUI's `/system_stats`.

## Configuration

Deploy-time settings come from the environment; see `.env.example`.

| Variable | Default | What it does |
| --- | --- | --- |
| `DARKBEAST3_GPU` | `L40S` | See below |
| `DARKBEAST3_MIN_CONTAINERS` | `0` | Warm containers |
| `DARKBEAST3_MAX_CONTAINERS` | `1` | Raise only if every client submits and polls in one request |
| `DARKBEAST3_SCALEDOWN_WINDOW` | `300` | Seconds warm after the last request |
| `DARKBEAST3_CONCURRENT_INPUTS` | `20` | Per container |
| `DARKBEAST3_REQUIRE_AUTH` | `1` | Proxy auth on the API |
| `DARKBEAST3_UI_REQUIRE_AUTH` | `0` | Browsers cannot attach the headers |

### Choosing a GPU

~19 GB of weights, but the text encoder offloads after encoding, so the sampling
working set is nearer 14 GB — the same shape as `ultra/`, whose checkpoint is
also a 13.8 GB int8 build.

- **L40S (48 GB)** — the default, with plenty of headroom.
- **A10 (24 GB)** — about half the price and very likely fits.

Compare **cost per image, not per hour**. Measure with two warm runs:

```bash
uv run python client.py generate "benchmark"   # cold, ignore
uv run python client.py generate "benchmark"   # warm, compare duration_s
```

## Project structure

```
darkbeast3/
├── app.py         Modal entrypoint: weights, container, endpoints
├── server.py      Request model, resolver, /defaults route
├── workflow.py    The Krea 2 graph in ComfyUI API format
├── client.py      CLI against a deployed endpoint
├── workflows/     Ready-to-POST API-format graph
└── tests/         Offline structural tests
```

## Troubleshooting

**First request takes minutes.** Cold start: image pull, ComfyUI boot, then the
weights onto the GPU. Set `DARKBEAST3_MIN_CONTAINERS=1` if latency matters more
than idle cost.

**The wrong precision landed on the Volume.** All five builds share a filename
upstream, so this is easy to do by hand. The digest check refuses a mismatch, so
compare `file_id` against the API — not the download page.

**Output looks nothing like the listing's example videos.** Those are from the
H3 video edition, a different base model on the same page. This service renders
stills from the Krea 2 version.

**`/generate` reports a few seconds but the client waits far longer.** The
`duration_s` clock stops inside the container, before the response is serialised
and shipped. Images come back base64-encoded in JSON, so a large `batch_size` is
a multi-megabyte download. Separate the two with
`curl -w 'ttfb=%{time_starttransfer} total=%{time_total} bytes=%{size_download}'`.
