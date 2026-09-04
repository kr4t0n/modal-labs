# ComfyUI nodes for Modal-hosted models

One package, every deployment. Each node renders on a remote Modal endpoint and
returns an ordinary `IMAGE` tensor, so it drops into an existing local workflow
like any other image source. No weights, no GPU, no ComfyUI-side model
management — it runs unchanged on a CPU-only install.

| Node | Category |
| --- | --- |
| **FLUX.2 klein (Modal)** | FLUX.2 klein (Modal) |
| **ULTRA / Krea 2 (Modal)** | ULTRA / Krea 2 (Modal) |
| **Z-Image Turbo Stable Yogi (Modal)** | Z-Image Turbo Stable Yogi (Modal) |
| **FinePorn v4 / Krea 2 (Modal)** | FinePorn v4 / Krea 2 (Modal) |
| **RedGPT2 GPT / Krea 2 (Modal)** | RedGPT2 / Krea 2 (Modal) |

Nodes whose endpoint is unconfigured simply raise a clear error when run, so
installing the package without deploying every service is fine.

## Install

```bash
cp -r comfy_node /path/to/ComfyUI/custom_nodes/comfyui-modal-remote
cd /path/to/ComfyUI/custom_nodes/comfyui-modal-remote
cp .env.example .env    # then fill in the endpoint URLs and your Modal token
```

Restart ComfyUI — it only scans `custom_nodes/` at startup. Everything the
package imports (`torch`, `numpy`, `PIL`, `requests`, `aiohttp`) already ships
with ComfyUI.

Settings are read from the process environment first, then from `.env` in this
directory. Nothing is stored in the workflow JSON, so exported workflows are
safe to share.

### One variable for every service

Instead of a `<SERVICE>_MODAL_URL` per deployment, set your workspace once:

```
MODAL_WORKSPACE=your-workspace
MODAL_KEY=wk-...
MODAL_SECRET=ws-...
```

Modal composes endpoint hostnames deterministically, so each node derives its
own URL from that — including services added later, with no new variable.

A per-service `<SERVICE>_MODAL_URL` still wins when set, which is what you want
for a second deployment of one service, or to point a node at an ephemeral
`modal serve` URL.

Derivation is a convenience, not a guarantee: Modal truncates or hashes
hostnames past the DNS label limit, and a non-default Modal environment inserts
a suffix. The URL printed by `modal deploy` is authoritative — set the explicit
variable if they ever disagree.

### Migrating from the per-service packages

Earlier revisions shipped `comfyui-ideogram4-modal` and
`comfyui-flux2klein-modal` separately. Remove them when installing this one, or
ComfyUI will register the same node ids twice:

```bash
rm -rf /path/to/ComfyUI/custom_nodes/comfyui-ideogram4-modal
rm -rf /path/to/ComfyUI/custom_nodes/comfyui-flux2klein-modal
```

Node ids and widget names are unchanged, so **saved workflows keep working** —
`tests/test_node_runtime.py` pins them precisely so that stays true.

## Nodes

**Ideogram 4 (Modal)** → `IMAGE`, `INT` (seed), `STRING` (info)

Prompt, preset (`Turbo`/`Default`/`Quality`), aspect ratio or explicit
width/height, batch size, seed and CFG.

Ideogram 4 is trained on **structured JSON captions**, and this is not optional
in practice — plain text tends to produce an image with little relation to what
you asked for. Paste a caption into the prompt box and anything starting with
`{` is forwarded as a JSON prompt automatically. Use the caption-template node
to generate one.

**Ideogram 4 Caption Template (Modal)** → `STRING`

Fetches the magic-prompt template with your idea and target size filled in. Feed
it to any instruction-following LLM; paste the JSON it returns into the prompt
box of the generation node.

**FLUX.2 klein (Modal)** → `IMAGE`, `INT` (seed), `STRING` (info)

Natural-language prompt — no structured caption needed. A negative prompt is
genuinely supported because that graph encodes it separately.

| Variant | Steps | CFG | Notes |
| --- | --- | --- | --- |
| `base` | 20 | 5.0 | Undistilled. Responds to CFG and negative prompts. |
| `distilled` | 4 | 1.0 | Guidance-distilled. Ignores CFG and negative prompts. |
| `ponpoke-uncensored` | 20 | 5.0 | As `base`, but with an abliterated text encoder — no prompt-stage safety filtering. |

Leave `override_sampler` off and the server applies those defaults. Pushing the
distilled variant above cfg 1 degrades it rather than sharpening it.

`lora` layers an adapter onto the transformer — `none` for the plain variant.
Adapters are model-only, so they compose with any variant including
`ponpoke-uncensored`. The dropdown mirrors the deployment's registry; `GET
/variants` is authoritative if they diverge.

Leave `override_lora_strength` off and the server applies whatever strength the
adapter's author recommends, the same way `override_sampler` defers to the
variant. That matters because the recommendations run in both directions — one
adapter wants 0.5–0.9, another 1.0–1.25 — so the `lora_strength` widget's 1.0 is
not a safe universal default. Turn the toggle on to send the widget value
verbatim.

> Workflows saved before this toggle existed load with it **off**, so they now
> use the recommended strength rather than the 1.0 they used to send. Turn it on
> to restore the old behaviour exactly.

**WAI-illustrious (Modal)** → `IMAGE`, `INT` (seed), `STRING` (info)

Danbooru tags rather than prose: `1girl, solo, silver hair, masterpiece`. Full
SDXL sampler controls — `steps`, `cfg`, `sampler_name`, `scheduler` and
`clip_skip`. Leave `clip_skip` at `-2`: booru-tagged SDXL finetunes are trained
against the penultimate CLIP layer, and `-1` quietly degrades prompt adherence
rather than erroring. The negative prompt is pre-filled with the standard
Danbooru negative; clear it if you do not want it.

**ULTRA / Krea 2 (Modal)** → `IMAGE`, `INT` (seed), `STRING` (info)

Natural-language prompts. Defaults to Krea 2 turbo's 8 steps at cfg 1, where the
negative prompt has no effect — the node says so in its `info` output if you fill
one in anyway. Raise `cfg` to make it active.

**Z-Image Turbo Stable Yogi (Modal)** → `IMAGE`, `INT` (seed), `STRING` (info)

A community finetune, not Alibaba's stock Z-Image Turbo.

Natural-language prompts at 8 steps, cfg 1, `res_multistep`. The `shift` widget
drives `ModelSamplingAuraFlow`; leave it at 3.0 unless you know why you are
changing it — the underlying node's own default of 1.73 gives a different noise
schedule. As with ULTRA, the negative prompt is inert at cfg 1.

**FinePorn v4 / Krea 2 (Modal)** → `IMAGE`, `INT` (seed), `STRING` (info)

The same Krea 2 base as ULTRA, a different merge. Defaults to `euler` + `beta`
at 10 steps, cfg 1 — the pairing its model card names for v4, not the `simple`
scheduler ULTRA takes from ComfyUI's template. The negative prompt is inert at
cfg 1, as there.

Two widgets differ from every other node here. `width` and `height` default to
**1280**, not 1024, because the card reports standard Krea 2 resolutions
underperform on this merge; dropping them back to 1024 is a quality regression
with no error to notice. And its prompt wants a smartphone-snapshot opener —
"this is a casual, low-quality photo" or similar — which the node's default
prompt demonstrates and the tooltip explains.

**RedGPT2 GPT / Krea 2 (Modal)** → `IMAGE`, `INT` (seed), `STRING` (info)

The third Krea 2 finetune here, and the lightest. Defaults to the same turbo
template settings as ULTRA — `euler` + `simple`, 8 steps, cfg 1 — because its
upstream publishes no sampler recipe. Deliberately *not* FinePorn's
`euler`/`beta`: that pairing is its author's recommendation, not a Krea 2 one.
Negative prompt is inert at cfg 1, as with the other two.

This is the single-model edition. The Civitai listing is titled "Alternating
Evaluation" and a different version there uses two checkpoints sampled
alternately; that build would need its own node and endpoint.

For all render nodes, set `aspect_ratio` to `custom` to use the width/height
widgets, and use the optional `endpoint` widget to override the environment if
you run more than one deployment of a service.

## Progress

Every render node reports the remote sampler's progress on its own local
progress bar. The node sends a client id with the request and subscribes to the
deployment's websocket with the same id, so ComfyUI's per-step progress events
come back and drive `comfy.utils.ProgressBar`.

On a clustered install this also keeps the browser's websocket alive: each
update makes your ComfyUI push a frame to the tab, so the connection never sits
idle for the length of a render.

Progress is cosmetic and fails safe. If the websocket cannot be reached the
render proceeds normally, just without a moving bar.

## Install on a Kubernetes ComfyUI

Credentials come from the process environment, so wire them through a Secret
rather than the `.env` file:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: modal-endpoints
type: Opaque
stringData:
  IDEOGRAM4_MODAL_URL: https://<workspace>--ideogram4-comfyui-ideogram4-web.modal.run
  FLUX2KLEIN_MODAL_URL: https://<workspace>--flux2klein-comfyui-flux2klein-web.modal.run
  WAIILLUSTRIOUS_MODAL_URL: https://<workspace>--waiillustrious-comfyui-waiillustrious-web.modal.run
  MODAL_KEY: wk-...
  MODAL_SECRET: ws-...
---
# in the ComfyUI Deployment's container spec
envFrom:
  - secretRef:
      name: modal-endpoints
```

Get the package into `custom_nodes/` by whichever route matches your build:

- **Baked in** (preferred for immutable deploys) — `COPY comfy_node
  <comfyui>/custom_nodes/comfyui-modal-remote` in your Dockerfile.
- **On a volume** — if `custom_nodes/` is a PVC, copy the directory in once and
  restart the pods.

Two things the pods need from the cluster:

- **Egress to `*.modal.run` on 443.** The call is outbound from the ComfyUI pod,
  so an ingress timeout does not apply to it — but an egress policy or proxy does.
- **A generous websocket idle timeout on the ingress.** Progress mirroring keeps
  the browser socket busy, but confirm the timeout exceeds a single sampler step.

## Notes

- A cold deployment takes minutes to answer the first request. Raise `timeout_s`
  rather than assuming it hung.
- Errors from the deployment are surfaced verbatim in the ComfyUI error dialog.
