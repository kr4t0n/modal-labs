# AGENTS.md — FLUX.2 klein 9B on Modal

Context for anyone changing this deployment. It is a deliberate sibling of
[`../ideogram4`](../ideogram4/AGENTS.md); read that one first, since the
architectural reasoning (why a real ComfyUI runs in the container rather than a
hand-written sampler) applies here unchanged and is not repeated.

This file covers only what differs.

## Duplication is known and deliberate — for now

`app.py`, `server.py`, `client.py` and `comfy_node/nodes.py` are close copies of
the ideogram4 versions. That is a real maintenance hazard: the proxy header
handling, the progress mirror and the `_single_line` build guard all exist
because of bugs found once, and a fix applied to one copy will drift from the
other.

It was left duplicated so that adding this service could not destabilise a
running production deployment. With two services the extraction is now worth
doing: the model-specific surface is small — `workflow.py`, the weight table,
and the request model's extra fields — and everything else is shared. Anyone
adding a third service should extract first rather than copy again.

## What differs from ideogram4

**One transformer, not two.** `CFGGuider` replaces `DualModelGuider`, so the
working set during sampling is ~18.5 GB rather than ~18.9 GB across a pair, and
peak VRAM is bounded by a single 9.4 GB checkpoint plus the 8.7 GB encoder.

**A real negative prompt.** The graph encodes it with a second `CLIPTextEncode`
rather than `ConditioningZeroOut`, so `negative_prompt` is a genuine input —
on the `base` variant. The `distilled` checkpoint is guidance-distilled and
ignores both CFG and the negative branch; the node surfaces that in its `info`
output rather than failing, because the request is still valid.

**Natural-language prompts.** No structured-caption requirement, unlike Ideogram
4. There is no `/caption-template` endpoint here.

**Variants replace presets.** `variant` selects both the checkpoint *and* its
tuned `steps`/`cfg`, because those are not independent: base is 20 steps at cfg
5, distilled is 4 at cfg 1, and mixing them produces bad output rather than an
error. Both figures come from ComfyUI's official templates, not inference —
`image_flux2_text_to_image_9b.json` and
`image_flux2_klein_image_edit_9b_distilled.json`.

**The weights are gated.** The two transformers require an accepted FLUX.2
licence, so `download_models` carries a `modal.Secret` with `HF_TOKEN` and
translates `GatedRepoError` into an instruction rather than a stack trace. The
serving containers need no secret — they read the Volume.

**Explicit download destinations.** Ideogram 4's mirror happens to be laid out
exactly like ComfyUI's `models/`, so files could be downloaded in place. Here
four repos with three different internal layouts feed one directory, so every
entry in `MODEL_FILES` carries a `destination`. Downloads land in a staging
directory on the same Volume and are renamed into place — a rename, not a second
copy of 9 GB.

**A different VAE.** `full_encoder_small_decoder.safetensors`, not
`flux2-vae.safetensors`. The klein templates ship the small-decoder autoencoder;
using the flux2-dev VAE here is a silent quality regression, not an error.

## Cross-project gotcha: module names

Both services expose top-level modules named `workflow`, `server` and `app`, and
pytest collects every suite in one interpreter. Each test module therefore clears
those names from `sys.modules` before importing, and `test_workflow.py` in both
projects asserts that the module it bound actually lives in its own directory.

Without that reset the suites still *pass* — they just silently exercise
whichever project was collected first. That is why the guard test exists.
`pyproject.toml` also sets `--import-mode=importlib` so the two identically named
`test_workflow.py` files can coexist.

CI imports each `app.py` in a separate process for the same reason.

## Known gaps

- Text-to-image only. FLUX.2 klein also does image editing, and ComfyUI ships
  templates for it (`image_flux2_klein_image_edit_9b_*`); that needs
  `/upload/image` plumbing in the typed contract. The raw proxy already exposes
  the endpoint, so the graph is reachable today by posting to `/prompt` directly.
- The 4B variants are not wired up. They use a different text encoder
  (`qwen_3_4b`), which the pairing comment in `workflow.py` deliberately fixes
  to the 8B one — mixing encoder and transformer scales degrades output silently.
- No LoRA loading.
