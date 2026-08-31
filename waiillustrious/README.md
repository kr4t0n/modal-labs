# WAI-illustrious-SDXL on Modal, served as a ComfyUI API

Runs [WAI-illustrious-SDXL](https://civitai.com/models/827184) — an Illustrious-XL
finetune with native Danbooru tag understanding — on a Modal GPU and exposes it
as a **ComfyUI server**. Same pattern as [`../ideogram4`](../ideogram4/) and
[`../flux2klein`](../flux2klein/): point a local or clustered ComfyUI at the URL
and render remotely.

This is the small, cheap one. A single ~6.8 GB fp16 SDXL checkpoint, so it runs
on a 24 GB **A10** rather than an H100, and prompts are Danbooru tags rather than
prose:

```
1girl, solo, silver hair, red eyes, city at night, masterpiece, best quality
```

> **Licence and content.** Governed by the model's
> [Civitai terms](https://civitai.com/models/827184): commercial *image* use,
> derivatives and relicensing are permitted, no credit required. The checkpoint
> is flagged NSFW-capable at source and applies no content filtering. This
> repository ships no weights.

## What you get

One deployed URL that is simultaneously:

| Surface | Path | Use it for |
| --- | --- | --- |
| ComfyUI HTTP + WebSocket API | `/prompt`, `/history`, `/view`, `/object_info`, `/ws` | Any existing ComfyUI API client, unchanged |
| Typed generation contract | `POST /generate` | The bundled custom node, `client.py`, your own app |
| Introspection | `GET /defaults`, `POST /workflow` | Sampler conventions, inspecting the graph |
| ComfyUI web interface | `/` | Browsing the queue, dragging in workflows |

## Setup

No credentials at all — unlike the other two services, this needs neither a
Hugging Face token nor a licence acceptance.

```bash
# from the repository root
uv sync --all-groups
cd waiillustrious

uv run modal run app.py::download_models   # ~6.8 GB into a Volume. One-off.
uv run modal deploy app.py
```

`download_models` runs on CPU — no GPU charge — and **verifies Civitai's
published SHA256** before installing. A mismatch aborts rather than serving a
corrupt or substituted checkpoint. Re-running skips an existing file; pass
`--force` to refetch.

Copy the endpoint URLs from the `modal deploy` output rather than constructing
them. You want the one for `WaiIllustrious.web`.

### Authentication

The API endpoint requires [Modal proxy auth](https://modal.com/docs/guide/webhook-proxy-auth)
by default. Create a token in the Modal dashboard (Settings → Proxy Auth Tokens):

```bash
export WAIILLUSTRIOUS_MODAL_URL=https://<workspace>--waiillustrious-comfyui-waiillustrious-web.modal.run
export MODAL_KEY=wk-...
export MODAL_SECRET=ws-...
```

## Using it from ComfyUI

```bash
cp -r ../comfy_node /path/to/ComfyUI/custom_nodes/comfyui-modal-remote
```

Fill in the endpoint and token (see [`../comfy_node/README.md`](../comfy_node/README.md)),
restart ComfyUI, and the **WAI-illustrious (Modal)** node appears. It returns an
ordinary `IMAGE` tensor and mirrors the remote sampler's progress onto its local
progress bar.

`workflows/wai_illustrious_sdxl_t2i_api.json` is a ready-to-POST graph for the
raw ComfyUI protocol.

## Using it from the CLI

```bash
uv run python client.py health
uv run python client.py defaults
uv run python client.py validate                      # graph vs. deployed node schemas
uv run python client.py generate "1girl, solo, masterpiece" --aspect-ratio 2:3
uv run python client.py generate "..." --cfg 7 --steps 32 --sampler dpmpp_2m --scheduler karras
uv run python client.py generate "..." --negative ""   # opt out of the default negative
```

Or without the endpoint — `modal run` calls the container directly and needs no
proxy token:

```bash
uv run modal run app.py --prompt "1girl, solo, masterpiece" --width 832 --height 1216
```

## Prompting and defaults

The author publishes no recommended settings, so these are the community
conventions for booru-tagged Illustrious finetunes. All are overridable per
request; none is required for correctness.

| Setting | Default | Why |
| --- | --- | --- |
| `steps` | 28 | Typical for SDXL anime finetunes |
| `cfg` | 5.0 | Illustrious derivatives prefer lower CFG than base SDXL |
| `sampler_name` | `euler_ancestral` | The usual choice for this family |
| `scheduler` | `normal` | — |
| `clip_skip` | `-2` | Booru-tagged SDXL finetunes are trained against the penultimate CLIP layer. Leaving it at `-1` does not error, it just degrades prompt adherence. |
| `negative_prompt` | standard Danbooru negative | Pass `""` to opt out |

`GET /defaults` returns exactly what the deployment applies.

**Resolution.** SDXL is trained around 1 megapixel. Sides are snapped to
multiples of 16 and clamped to 256–2048, but pushing much past ~1 MP total tends
to duplicate subjects rather than add detail. The usual portrait bucket is
832×1216; `--aspect-ratio 2:3` gets you there.

## API reference

`POST /generate` — all fields optional except `prompt`:

| Field | Default | Notes |
| --- | --- | --- |
| `prompt` | — | Danbooru tags |
| `negative_prompt` | standard negative | `""` to disable |
| `width`, `height` | 1024 | 256–2048, snapped to /16 |
| `aspect_ratio` | — | `1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `9:16`, `16:9`, `21:9`; overrides width/height |
| `megapixels` | 1.0 | Pixel budget used with `aspect_ratio` |
| `steps` | 28 | |
| `cfg` | 5.0 | |
| `sampler_name` | `euler_ancestral` | Any ComfyUI sampler |
| `scheduler` | `normal` | Any ComfyUI scheduler |
| `clip_skip` | -2 | −24 to −1 |
| `denoise` | 1.0 | |
| `seed` | random | 0 – 2^64-1 |
| `batch_size` | 1 | 1–8 |
| `timeout_s` | 900 | Server-side wait |
| `client_id` | generated | Subscribe to `/ws?clientId=<id>` for this render's progress |

Response: `{prompt_id, duration_s, params, images: [{filename, content_type, b64}]}`.
`POST /generate/image` takes the same body and returns raw PNG bytes.

## Configuration

All read at deploy time; see [`.env.example`](.env.example).

| Variable | Default | Effect |
| --- | --- | --- |
| `WAIILLUSTRIOUS_GPU` | `A10` | Any Modal GPU string |
| `WAIILLUSTRIOUS_MIN_CONTAINERS` | `0` | Warm containers |
| `WAIILLUSTRIOUS_MAX_CONTAINERS` | `1` | Keep at 1 unless clients submit and poll in one request |
| `WAIILLUSTRIOUS_SCALEDOWN_WINDOW` | `300` | Idle seconds before shutdown |
| `WAIILLUSTRIOUS_CONCURRENT_INPUTS` | `20` | Requests in flight per container |
| `WAIILLUSTRIOUS_REQUIRE_AUTH` | `1` | Proxy auth on the API endpoint |
| `WAIILLUSTRIOUS_UI_REQUIRE_AUTH` | `0` | Proxy auth on the raw UI endpoint |

### Choosing a GPU

The working set is about **7 GB** — the whole checkpoint — so this is by far the
cheapest of the three services to run.

`A10` (24 GB) is the default: ample headroom at 1 MP, and roughly a quarter of
H100's hourly rate. SDXL is fp16 throughout, so Ampere's lack of fp8 tensor
cores costs nothing here — the reason to avoid A100/A10 on the other two
services does not apply.

`L4` (24 GB) is cheaper still and worth benchmarking, but it has substantially
less compute, so it may cost *more* per image. **Compare cost per image, not per
hour:**

```bash
WAIILLUSTRIOUS_GPU=L4 uv run modal deploy app.py
uv run python client.py generate "benchmark"   # cold, ignore
uv run python client.py generate "benchmark"   # warm, compare duration_s
```

Multiply the warm `duration_s` by the card's per-second rate and compare.

## Project structure

```
waiillustrious/
├── app.py         Modal object graph: Civitai fetch, GPU class, endpoints
├── server.py      Request model, resolver and the /defaults route
├── workflow.py    The SDXL graph in ComfyUI API format
├── client.py      CLI: generate / defaults / health / validate
├── workflows/     Ready-to-POST API-format graph
└── tests/         SDXL-specific assertions
```

Everything generic lives outside this directory: `../comfyui_modal` (container
image, ComfyUI supervisor, ASGI proxy, CLI plumbing) and `../comfy_node` (the
ComfyUI nodes for every service).

## Troubleshooting

**`checksum mismatch from Civitai`.** The download did not match the published
SHA256. Re-run; if it persists, the upstream file changed and the pin in
`app.py` needs updating deliberately.

**`weights Volume is missing ...`** — run `modal run app.py::download_models`.

**Output looks washed out or ignores tags.** Check `clip_skip` is `-2`, and that
your prompt is tags rather than prose — this model is trained on Danbooru
captions, not natural language.

**Duplicated subjects or stretched anatomy.** Almost always resolution: you are
too far above ~1 MP. Drop to 832×1216 or 1024×1024.

**First request takes minutes.** Cold start: image pull, ComfyUI boot, then ~7 GB
onto the GPU. Set `WAIILLUSTRIOUS_MIN_CONTAINERS=1` if latency matters more than
idle cost.
