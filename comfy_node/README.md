# ComfyUI nodes for Modal-hosted models

One package, every deployment. Each node renders on a remote Modal endpoint and
returns an ordinary `IMAGE` tensor, so it drops into an existing local workflow
like any other image source. No weights, no GPU, no ComfyUI-side model
management — it runs unchanged on a CPU-only install.

| Node | Category |
| --- | --- |
| **Ideogram 4 (Modal)** | Ideogram 4 (Modal) |
| **Ideogram 4 Caption Template (Modal)** | Ideogram 4 (Modal) |
| **FLUX.2 klein (Modal)** | FLUX.2 klein (Modal) |
| **WAI-illustrious (Modal)** | WAI-illustrious (Modal) |

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

**WAI-illustrious (Modal)** → `IMAGE`, `INT` (seed), `STRING` (info)

Danbooru tags rather than prose: `1girl, solo, silver hair, masterpiece`. Full
SDXL sampler controls — `steps`, `cfg`, `sampler_name`, `scheduler` and
`clip_skip`. Leave `clip_skip` at `-2`: booru-tagged SDXL finetunes are trained
against the penultimate CLIP layer, and `-1` quietly degrades prompt adherence
rather than erroring. The negative prompt is pre-filled with the standard
Danbooru negative; clear it if you do not want it.

For all render nodes, set `aspect_ratio` to `custom` to use the width/height
widgets, and use the optional `endpoint` widget to override the environment if
you run more than one deployment of a service.

## Progress

Both render nodes report the remote sampler's progress on their own local
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
