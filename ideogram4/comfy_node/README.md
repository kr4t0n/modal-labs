# Ideogram 4 (Modal) — ComfyUI custom node

Renders on a remote Ideogram 4 deployment and returns an ordinary `IMAGE`
tensor, so it drops into an existing local workflow like any other image source.
No weights, no GPU, no ComfyUI-side model management.

## Install

```bash
cp -r comfy_node /path/to/ComfyUI/custom_nodes/comfyui-ideogram4-modal
cd /path/to/ComfyUI/custom_nodes/comfyui-ideogram4-modal
cp .env.example .env    # then fill in your endpoint URL and Modal token
```

Restart ComfyUI. Everything the node imports (`torch`, `numpy`, `PIL`,
`requests`) already ships with ComfyUI.

Settings are read from the process environment first, then from `.env` in this
directory. Nothing is stored in the workflow JSON, so exported workflows are
safe to share.

## Nodes

**Ideogram 4 (Modal)** → `IMAGE`, `INT` (seed), `STRING` (info)

Prompt, preset (`Turbo`/`Default`/`Quality`), aspect ratio or explicit
width/height, batch size, seed and CFG. Set `aspect_ratio` to `custom` to use
the width/height widgets.

**Paste a structured JSON caption into the prompt box** — anything starting with
`{` is forwarded as a JSON prompt automatically. This is not optional in
practice: Ideogram 4 is trained on that schema, and plain text tends to produce
an image unrelated to what you asked for. Use the caption-template node below to
generate one.

The optional `endpoint` widget overrides `IDEOGRAM4_MODAL_URL` if you run more
than one deployment.

**Ideogram 4 Caption Template (Modal)** → `STRING`

Fetches the magic-prompt template with your idea and target size filled in. Feed
it to any instruction-following LLM; paste the JSON it returns back into the
prompt box of the generation node.

## Install on a Kubernetes ComfyUI

The node is pure orchestration — no GPU, no CUDA, no extra pip dependencies
beyond what ComfyUI already ships. It runs unchanged on a CPU-only pod.

Credentials come from the process environment, so wire them through a Secret
rather than the `.env` file:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: ideogram4-modal
type: Opaque
stringData:
  IDEOGRAM4_MODAL_URL: https://<workspace>--ideogram4-comfyui-ideogram4-web.modal.run
  MODAL_KEY: wk-...
  MODAL_SECRET: ws-...
---
# in the ComfyUI Deployment's container spec
envFrom:
  - secretRef:
      name: ideogram4-modal
```

Get the node into `custom_nodes/` by whichever route matches how you build the
image:

- **Baked in** (preferred for immutable deploys) — `COPY comfy_node
  /opt/ComfyUI/custom_nodes/comfyui-ideogram4-modal` in your ComfyUI Dockerfile.
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
