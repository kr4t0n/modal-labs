# RedGPT2 (Krea 2) on Modal, served as a ComfyUI API

Runs the **"KREA2 GPT 逼真版"** edition of
[**RedGPT2**](https://civitai.com/models/452459?modelVersionId=3123514) by
`AiMetatron` — a community finetune of **Krea 2 turbo** — on a Modal GPU and
exposes it as a **ComfyUI server**. Same pattern as the other services here:
point a local or clustered ComfyUI at the URL and render remotely.

> **This is the single-model edition, not the Alternating Evaluation one.**
> The Civitai listing is titled "Alternating Evaluation" and its card describes
> a scheme using *two* checkpoints — a high-noise and a low-noise model sampled
> alternately in a 4H+6L configuration. That is a **different version** on the
> same page (`3289607`), which ships two safetensors and a config. The edition
> deployed here (`3123514`) is one file and samples conventionally. See
> [Editions](#editions).

> **Adult content.** This finetune targets uncensored imagery. The endpoint
> defaults to requiring `Modal-Key` / `Modal-Secret`; leave that on.

> **Source.** The checkpoint is fetched from **civitai.com** and verified
> against the SHA256 Civitai publishes, so a substituted file fails closed. This
> repository ships no weights.

## Licence

Two layers apply, neither of them this repository's to grant:

**Krea 2 Community License.** Everything derived from Krea 2 — this service,
`ultra/` and `finepornv4/` alike — inherits it. Free commercial use is
conditional on your **company-wide annual revenue being under $1,000,000 USD**,
on a trailing twelve-month basis, counting affiliated entities under common
ownership. Above that, you need a separate arrangement with Krea. You own the
outputs you generate provided you comply, and Krea claims no ownership of them.

**AiMetatron access terms.** The model card states this finetune belongs to a
paid access package, and that buying it on the model page grants access to the
RedGPT2 Krea2 fine-tune files only. Its author also credits the underlying
training method as commercially licensed from a third party. Check the card
before any commercial use.

## What it needs

Like the other two Krea 2 services it is distributed as a **diffusion model
only**, so the encoder and VAE come from Comfy-Org's Krea 2 mirror — the same
two files, byte for byte. A test asserts they stay in step.

| File | Size | Source |
| --- | --- | --- |
| `redgpt2_krea2_gpt_fp8.safetensors` | 12.83 GB | Civitai, digest-verified |
| `qwen3vl_4b_fp8_scaled.safetensors` | 5.24 GB | `Comfy-Org/Krea-2` |
| `qwen_image_vae.safetensors` | 0.25 GB | `Comfy-Org/Krea-2` |

**The checkpoint download needs a Civitai API token.** This model is
NSFW-flagged and Civitai answers `401` to an unauthenticated request for it —
unlike `ultra/`, which downloads anonymously. Only the one-off `download_models`
needs the token; the serving containers read the Volume.

Krea 2 pairs with the **4B** Qwen3-VL on a 12-layer tap and the **Qwen-Image**
autoencoder — neither is interchangeable with the 8B encoder or Flux-family VAE
the klein services use.

## Setup

```bash
# from the repository root
uv sync --all-groups
cd redgpt2gpt

# One-off. Civitai -> Account settings -> API Keys.
uv run modal secret create civitai-secret CIVITAI_TOKEN=...

uv run modal run app.py::download_models   # ~18 GB into a Volume. One-off.
uv run modal deploy app.py
```

Reuse an existing `civitai-secret` if you made one for another service; nothing
in it is service-specific. Point `REDGPT2GPT_CIVITAI_SECRET` at a different name
if you keep more than one.

`download_models` runs on CPU — no GPU charge — and is idempotent by
destination; pass `--force` to refetch. Copy the endpoint URL from the deploy
output; you want the one for `RedGPT2GPT.web`.

```bash
export REDGPT2GPT_MODAL_URL=https://...
export MODAL_KEY=wk-...  MODAL_SECRET=ws-...
```

> Or set `MODAL_WORKSPACE=your-workspace` once and every service's URL is
> derived from it — see [`comfy_node/README.md`](../comfy_node/README.md).
> The explicit variable above still wins when set.

## Using it

From ComfyUI, install the shared node package and the **RedGPT2 GPT / Krea 2
(Modal)** node appears:

```bash
cp -r ../comfy_node /path/to/ComfyUI/custom_nodes/comfyui-modal-remote
```

From the CLI:

```bash
uv run python client.py generate "a portrait by a window" --aspect-ratio 3:4
uv run python client.py defaults
```

Or as a plain ComfyUI server — `/prompt`, `/history`, `/view`, `/object_info`
and `/ws` all work unmodified, and `workflows/redgpt2gpt_krea2_t2i_api.json` is
a ready-to-POST graph.

## Sampler defaults, and where they come from

| Setting | Value |
| --- | --- |
| `steps` | 8 |
| `cfg` | 1.0 |
| `sampler_name` | `euler` |
| `scheduler` | `simple` |

**These are ComfyUI's Krea 2 turbo template values, not the author's.** This
edition's notes cover training method and licensing but publish no sampler
settings, so the template is the honest fallback — the same choice `ultra/`
makes, and `GET /defaults` says so outright in its `source` field.

That is a real difference from `finepornv4/`, whose card *does* publish a recipe
(`euler`/`beta`, 10 steps) and whose service follows it. Do not copy those
settings here on the assumption that two Krea 2 services should match: it would
be inventing a recommendation nobody made. If you find settings that work
better, they belong in this file with a note saying they were measured.

The model card does mention a **4H + 6L** step split, but that describes the
Alternating Evaluation edition's two-model schedule and does not transfer to
this single-model build.

At the default cfg 1 the **negative prompt is inert**: the graph zeroes the
conditioning rather than encoding it. Supplying negative text swaps in a real
encoder, which only becomes useful alongside a raised cfg.

## Editions

The listing carries several builds. Editions on that page reuse filenames, so
the `file_id` is pinned alongside the version id.

| Edition | Version id | Files | Notes |
| --- | --- | --- | --- |
| KREA2 GPT 逼真版 | `3123514` | 1 × fp8, 12.83 GB | **Deployed.** Single model |
| GPT 逼真版 INT4/INT8 | `3131246` | nf4 6.74 GB, int8 12.84 GB | Quantizations of the same edition |
| KREA2RED AE 剧创 | `3289607` | 2 × int8 13.18 GB + config | Alternating Evaluation; needs a different graph |

Switching to a quantization means editing one `CivitaiFile` in `app.py` — the
version id, the file id and the digest — then re-running `download_models`. Take
all three from the API rather than the page.

Switching to the AE edition is **not** a config change: it needs two
`UNETLoader`s and an alternating sigma schedule, which this graph does not
implement. That would be a new service.

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
| `denoise` | 1.0 | |
| `batch_size` | 1 | |
| `seed` | random | |
| `client_id` | generated | Subscribe to `/ws?clientId=<id>` for progress |

`GET /defaults` returns the sampler conventions and names their source.
`GET /health` proxies ComfyUI's `/system_stats`.

## Configuration

Deploy-time settings come from the environment; see `.env.example`.

| Variable | Default | What it does |
| --- | --- | --- |
| `REDGPT2GPT_CIVITAI_SECRET` | `civitai-secret` | Modal Secret holding `CIVITAI_TOKEN` |
| `REDGPT2GPT_GPU` | `L40S` | See below |
| `REDGPT2GPT_MIN_CONTAINERS` | `0` | Warm containers |
| `REDGPT2GPT_MAX_CONTAINERS` | `1` | Raise only if every client submits and polls in one request |
| `REDGPT2GPT_SCALEDOWN_WINDOW` | `300` | Seconds warm after the last request |
| `REDGPT2GPT_CONCURRENT_INPUTS` | `20` | Per container |
| `REDGPT2GPT_REQUIRE_AUTH` | `1` | Proxy auth on the API |
| `REDGPT2GPT_UI_REQUIRE_AUTH` | `0` | Browsers cannot attach the headers |

### Choosing a GPU

~18 GB of weights, but the text encoder offloads after encoding, so the sampling
working set is nearer 13 GB — the same shape as `ultra/`, and much lighter than
`finepornv4/`.

- **L40S (48 GB)** — the default, with plenty of headroom.
- **A10 (24 GB)** — about half the price and very likely fits.

Compare **cost per image, not per hour**. Measure with two warm runs:

```bash
uv run python client.py generate "benchmark"   # cold, ignore
uv run python client.py generate "benchmark"   # warm, compare duration_s
```

## Project structure

```
redgpt2gpt/
├── app.py         Modal entrypoint: weights, container, endpoints
├── server.py      Request model, resolver, /defaults route
├── workflow.py    The Krea 2 graph in ComfyUI API format
├── client.py      CLI against a deployed endpoint
├── workflows/     Ready-to-POST API-format graph
└── tests/         Offline structural tests
```

## Troubleshooting

**First request takes minutes.** Cold start: image pull, ComfyUI boot, then the
weights onto the GPU. Set `REDGPT2GPT_MIN_CONTAINERS=1` if latency matters more
than idle cost.

**`download_models` fails, or the digest check rejects the file.** Almost always
the Civitai token. Without it Civitai may serve an HTML error page rather than
the safetensors, which the SHA256 check then refuses — so the failure is loud
but the message points at the checksum, not at auth. Verify the token with:

```bash
curl -sS -o /dev/null -L -r 0-1023 -w '%{http_code}\n' \
  -H "Authorization: Bearer $CIVITAI_TOKEN" \
  "https://civitai.com/api/download/models/3123514?fileId=3004003"
```

`206` means it works, `401` means it does not.

**`/generate` reports a few seconds but the client waits far longer.** The
`duration_s` clock stops inside the container, before the response is serialised
and shipped. Images come back base64-encoded in JSON, so a large `batch_size` is
a multi-megabyte download. Separate the two with
`curl -w 'ttfb=%{time_starttransfer} total=%{time_total} bytes=%{size_download}'`.
