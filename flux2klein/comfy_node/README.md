# FLUX.2 klein (Modal) — ComfyUI custom node

Renders on a remote FLUX.2 klein 9B deployment and returns an ordinary `IMAGE`
tensor, so it drops into an existing local workflow like any other image source.
No weights, no GPU, no ComfyUI-side model management.

## Install

```bash
cp -r comfy_node /path/to/ComfyUI/custom_nodes/comfyui-flux2klein-modal
cd /path/to/ComfyUI/custom_nodes/comfyui-flux2klein-modal
cp .env.example .env    # then fill in your endpoint URL and Modal token
```

Restart ComfyUI. Everything the node imports (`torch`, `numpy`, `PIL`,
`requests`) already ships with ComfyUI.

Settings are read from the process environment first, then from `.env` in this
directory. Nothing is stored in the workflow JSON, so exported workflows are
safe to share.

## Node

**FLUX.2 klein (Modal)** → `IMAGE`, `INT` (seed), `STRING` (info)

Natural-language prompt — no structured caption needed, unlike the Ideogram 4
service. A negative prompt is genuinely supported here because the graph encodes
it separately rather than zeroing out the positive conditioning.

`variant` picks the checkpoint and its tuned sampler settings:

| Variant | Steps | CFG | Notes |
| --- | --- | --- | --- |
| `base` | 20 | 5.0 | Undistilled. Responds to CFG and negative prompts. |
| `distilled` | 4 | 1.0 | Guidance-distilled. Ignores CFG and negative prompts. |

Leave `override_sampler` off and the server applies those defaults. Turn it on
to drive `steps` and `cfg` yourself — worth knowing that pushing the distilled
variant above cfg 1 degrades it rather than sharpening it.

Set `aspect_ratio` to `custom` to use the width/height widgets. The optional
`endpoint` widget overrides `FLUX2KLEIN_MODAL_URL` if you run more than one
deployment.

## Install on a Kubernetes ComfyUI

The node is pure orchestration — no GPU, no CUDA, no extra pip dependencies
beyond what ComfyUI already ships. It runs unchanged on a CPU-only pod.

Credentials come from the process environment, so wire them through a Secret
rather than the `.env` file:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: flux2klein-modal
type: Opaque
stringData:
  FLUX2KLEIN_MODAL_URL: https://<workspace>--flux2klein-comfyui-flux2klein-web.modal.run
  MODAL_KEY: wk-...
  MODAL_SECRET: ws-...
---
# in the ComfyUI Deployment's container spec
envFrom:
  - secretRef:
      name: flux2klein-modal
```

Get the node into `custom_nodes/` by whichever route matches how you build the
image:

- **Baked in** (preferred for immutable deploys) — `COPY comfy_node
  /root/ComfyUI/custom_nodes/comfyui-flux2klein-modal` in your ComfyUI Dockerfile.
- **On a volume** — if `custom_nodes/` is a PVC, copy the directory in once and
  restart the pods.

Two things the pods need from the cluster:

- **Egress to `*.modal.run` on 443.** The node's HTTP call is outbound from the
  ComfyUI pod, so an ingress timeout does not apply to it — but an egress policy
  or proxy does.
- **A generous websocket idle timeout on the ingress.** The node blocks for the
  whole render and emits nothing meanwhile, so the browser's ComfyUI websocket
  can sit silent for minutes. Ingresses that idle-timeout websockets at 60s will
  drop the tab mid-render even though the job completes fine on Modal.

## Progress

The node reports the remote sampler's progress on its own local progress bar. It
sends a client id with the request and subscribes to the deployment's websocket
with the same id, so ComfyUI's per-step progress events come back and drive
`comfy.utils.ProgressBar` — a render on a Modal GPU looks like a local one.

On a clustered install this is also what keeps the browser's websocket alive:
each update makes your ComfyUI push a frame to the tab, so the connection never
sits idle for the length of a render.

Progress is cosmetic and fails safe. If the websocket cannot be reached the
render proceeds normally, just without a moving bar.

## Notes

- A cold deployment takes minutes to answer the first request. Raise `timeout_s`
  rather than assuming it hung.
- Errors from the deployment are surfaced verbatim in the ComfyUI error dialog.
- Requires nothing beyond ComfyUI's own dependencies; the websocket client is
  `aiohttp`, which ComfyUI already uses for its server.
