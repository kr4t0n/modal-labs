# AGENTS.md — Z-Image Turbo on Modal

Context for anyone changing this deployment. The architectural reasoning lives in
the root [`AGENTS.md`](../AGENTS.md). This file covers what differs.

## Where the code lives

Only the model-specific parts are here: the weight table and Modal object graph
(`app.py`), the request model and `/defaults` route (`server.py`), the graph
(`workflow.py`) and the CLI's own arguments (`client.py`). Everything generic is
in `../comfyui_modal`; the ComfyUI nodes are in `../comfy_node`.

## Three graph details that are easy to get wrong

All three come from ComfyUI's `image_z_image_turbo` template rather than
inference, and each fails quietly rather than loudly:

**`ModelSamplingAuraFlow` is required.** The sampler reads the *patched* model,
not the loader. The node's own default shift is 1.73 while Z-Image's model class
declares 3.0, so omitting the patch — or leaving the node default — changes the
noise schedule without erroring. `shift` is therefore an explicit request field
with a validated default, and a test asserts the patch sits between loader and
sampler.

**The latent is `EmptySD3LatentImage`.** Sixteen channels at /8, not the
four-channel SD-style one. `EmptyLatentImage` would be accepted by the graph and
fail later at sample time.

**The CLIP type is `lumina2`.** Z-Image's model class subclasses Lumina2, and
ComfyUI routes a Qwen3-4B encoder to `z_image.te`/`ZImageTokenizer` for any clip
type that is not FLUX or FLUX2. `lumina2` is what the template uses; do not
"tidy" it to something that looks more specific.

The reference sampler is `res_multistep`, which is unusual enough to be worth
not substituting casually.

## Provenance and credentials

This is the only service whose weights **require** a token: every version of the
Civitai model returns 401 anonymously, verified across four version ids while
the `ultra` service's file still returned 206 from the same code path. So
`download_models` attaches `civitai-secret`, and `comfyui_modal/weights.py`
picks `CIVITAI_TOKEN` out of the environment — that fetcher was written with
this case in mind, so no code changed to support it.

The SHA256 is still pinned. A token proves who is asking, not what came back.

It was also handed over as a `civitai.red` link — the same lookalike mirror as
the `ultra` service. The fetch points at `civitai.com`; see
[`../ultra/AGENTS.md`](../ultra/AGENTS.md) for the reasoning.

**Licence is the strictest here:** `allowDerivatives: false` and
`allowNoCredit: false`. Credit is required and derivatives are not permitted, so
do not add a merge or fine-tune path to this service.

## Quantisation choice

The model ships in NVFP4, fp8, int8-convrot, bf16 and several GGUF builds. The
fp8 safetensors is wired up because NVFP4 needs Blackwell — B200 upwards, absurd
for a 6 GB turbo model — and GGUF needs a custom node pack. int8-convrot is the
Ampere fallback if fp8 hardware is ever unavailable; swapping means changing the
version id, file id, filename and SHA256 together, and the golden-graph test
will fail until `workflow.py` matches.

## Known gaps

- Text-to-image only.
- No LoRA support, though `Comfy-Org/z_image_turbo` ships a distill patch LoRA
  and `flux2klein` demonstrates the registry pattern.
- The pixel-space Z-Image variant (`ZImagePixelSpace`, no VAE) is a different
  model class and is not wired up.
