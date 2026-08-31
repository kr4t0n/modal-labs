# Z-Image Turbo (Stable Yogi) on Modal, served as a ComfyUI API

Runs [**Zimage Turbo by Stable Yogi**](https://civitai.com/models/2221503) — a
finetune of Alibaba's Z-Image Turbo — on a Modal GPU and exposes it as a
**ComfyUI server**. The cheapest service here: a 6 GB fp8 diffusion model doing
8-step sampling on a 24 GB **L4**.

> **A Civitai API token is required.** Unlike the `ultra` service, every version
> of this model returns 401 anonymously. Create a key at civitai.com → Account
> settings → API Keys, then:
>
> ```bash
> modal secret create civitai-secret CIVITAI_TOKEN=<your-key>
> ```

> **Licence.** The strictest of the services here: Civitai's terms for this
> model set `allowDerivatives: false` and `allowNoCredit: false` — **credit is
> required** and derivatives are not permitted. Generated images may be used
> commercially. This repository ships no weights.

## What it needs

The finetune is distributed as a **diffusion model only**, so the encoder and
autoencoder come from Comfy-Org's Z-Image mirror. Only the first needs the token.

| File | Size | Source |
| --- | --- | --- |
| `zimageturbostableyogi.safetensors` | 6.01 GB | Civitai, token + digest-verified |
| `qwen_3_4b_fp8_mixed.safetensors` | 5.63 GB | `Comfy-Org/z_image_turbo` |
| `ae.safetensors` | 0.34 GB | `Comfy-Org/z_image_turbo` |

Z-Image conditions on a Qwen3-**4B** encoder, reached through ComfyUI's
`lumina2` CLIP type — its model class subclasses Lumina2. The 8B encoders the
klein services use are not interchangeable.

### Why the fp8 build

The model ships in several quantisations. `2603 NVFP4` (3.58 GB) is newest but
needs **Blackwell** — B200 upwards on Modal, far more card than a 6 GB turbo
model warrants. `2603 Fp8` is the same generation and runs on any Ada or Hopper
card. `2603 INT8-ConvRot` (6.15 GB) is the fallback if you want Ampere, which
has int8 but no fp8. The GGUF builds need the ComfyUI-GGUF custom node and are
not wired up.

## Setup

```bash
# from the repository root
uv sync --all-groups
cd zimageturbostableyogi

modal secret create civitai-secret CIVITAI_TOKEN=...   # if not already done
uv run modal run app.py::download_models               # ~12 GB into a Volume
uv run modal deploy app.py
```

```bash
export ZIMAGETURBOSTABLEYOGI_MODAL_URL=https://...   # the ZImageTurboStableYogi.web URL
export MODAL_KEY=wk-...  MODAL_SECRET=ws-...
```

## Using it

```bash
cp -r ../comfy_node /path/to/ComfyUI/custom_nodes/comfyui-modal-remote
```

The **Z-Image Turbo (Modal)** node appears after a restart. Or from the CLI:

```bash
uv run python client.py health
uv run python client.py defaults
uv run python client.py validate
uv run python client.py generate "a harbour at dawn, pastel houses" --aspect-ratio 16:9
uv run modal run app.py --prompt "..."      # no proxy token needed
```

## Sampler defaults

Taken verbatim from ComfyUI's official Z-Image Turbo template:

| Setting | Default |
| --- | --- |
| `steps` | 8 |
| `cfg` | 1.0 |
| `sampler_name` | `res_multistep` |
| `scheduler` | `simple` |
| `shift` | 3.0 |

**`shift` is not decorative.** The graph patches the model through
`ModelSamplingAuraFlow`, whose own node default is 1.73 while Z-Image declares
3.0. Omitting the patch, or leaving the node default, silently changes the noise
schedule rather than erroring.

At `cfg 1` the negative branch is never consulted, so the graph zeroes it exactly
as the reference does. Supplying `negative_prompt` swaps in a real encoder, which
only does anything if you also raise `cfg`.

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
| `shift` | 3.0 | ModelSamplingAuraFlow shift |
| `sampler_name` | `res_multistep` | Any ComfyUI sampler |
| `scheduler` | `simple` | Any ComfyUI scheduler |
| `denoise` | 1.0 | |
| `seed` | random | |
| `batch_size` | 1 | 1–8 |
| `client_id` | generated | Subscribe to `/ws?clientId=<id>` for progress |

## Configuration

| Variable | Default | Effect |
| --- | --- | --- |
| `ZIMAGETURBOSTABLEYOGI_CIVITAI_SECRET` | `civitai-secret` | Modal Secret holding `CIVITAI_TOKEN` |
| `ZIMAGETURBOSTABLEYOGI_GPU` | `L4` | Any Modal GPU string |
| `ZIMAGETURBOSTABLEYOGI_MIN_CONTAINERS` | `0` | Warm containers |
| `ZIMAGETURBOSTABLEYOGI_MAX_CONTAINERS` | `1` | Keep at 1 unless clients submit and poll in one request |
| `ZIMAGETURBOSTABLEYOGI_SCALEDOWN_WINDOW` | `300` | Idle seconds before shutdown |
| `ZIMAGETURBOSTABLEYOGI_CONCURRENT_INPUTS` | `20` | Requests in flight per container |
| `ZIMAGETURBOSTABLEYOGI_REQUIRE_AUTH` | `1` | Proxy auth on the API endpoint |
| `ZIMAGETURBOSTABLEYOGI_UI_REQUIRE_AUTH` | `0` | Proxy auth on the raw UI endpoint |

### Choosing a GPU

~12 GB of weights, and the encoder offloads after encoding, so the sampling
working set is nearer **7 GB**.

`L4` is the default and should be hard to beat: the `F8_E4M3` weights need fp8
tensor cores (compute capability ≥ 8.9), L4 is Ada so it qualifies, and at
roughly a fifth of an H100's hourly rate it is the cheapest card on Modal that
can run them. 8-step sampling is a light enough workload to suit its modest
compute.

If it turns out compute-bound, `L40S` is the next step up with the same fp8
support. Compare cost per image, not per hour — two warm renders settle it.

## Project structure

```
zimageturbostableyogi/
├── app.py         Modal object graph: weight table, GPU class, endpoints
├── server.py      Request model, resolver and the /defaults route
├── workflow.py    The Z-Image graph in ComfyUI API format
├── client.py      CLI: generate / defaults / health / validate
├── workflows/     Ready-to-POST API-format graph
└── tests/         Z-Image-specific assertions
```

Everything generic lives outside this directory: `../comfyui_modal` and
`../comfy_node`.

## Troubleshooting

**`401` during download.** The token is missing, wrong, or the secret is not
attached. `modal secret list` should show `civitai-secret`.

**`checksum mismatch`.** The bytes did not match Civitai's published SHA256.
Re-run; if it persists the upstream file changed and the pin needs updating
deliberately.

**Output looks off in a way steps do not fix.** Check `shift` is 3.0 — the node
default of 1.73 produces a different schedule.

**`weights Volume is missing ...`** — run `modal run app.py::download_models`.
