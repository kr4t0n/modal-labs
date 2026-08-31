# Retired services

Services that were tried and set aside. Kept as reference, **not** built, tested
or deployed from here.

| Service | Model | Retired | Weights source |
| --- | --- | --- | --- |
| [`ideogram4/`](ideogram4/) | Ideogram 4 — 9.3B DiT, dual-branch CFG, Qwen3-VL-8B encoder | 2026-08-31 | Hugging Face, ungated |
| [`waiillustrious/`](waiillustrious/) | WAI-illustrious-SDXL — Illustrious-XL anime finetune, Danbooru tags | 2026-08-31 | Civitai, anonymous |

Both worked; neither earned its keep against the alternatives in day-to-day use.

## What is preserved, and what actually survives

Each directory is intact — graph, weight table, server adapter, CLI, tests and
notes. Two things are worth knowing about their shelf life:

**`workflows/*.json` is the durable artifact.** It is a verified, API-format
graph that can be POSTed to any ComfyUI with the right weights present. It has
no dependency on this repository's Python and cannot rot. If you only read one
file when reviving a model, read that one.

**The Python will drift.** These are excluded from `pytest` and CI, so a change
to `comfyui_modal` can break them without anything failing. That is deliberate:
keeping retired services green would make every future refactor of the shared
layer pay a tax for code nobody runs. They stay under `ruff`, so they remain
readable and correctly formatted, but treat them as documentation of a working
configuration rather than working code.

Their ComfyUI nodes were removed from `comfy_node/`'s registry so they stop
appearing in the node list, and moved here alongside the service —
`nodes_<service>.py`, previously `comfy_node/nodes_<service>.py`.

## Reviving one

1. Move the directory back to the repository root.
2. Revert its `sys.path` shim in `app.py`: retired services use
   `HERE.parent.parent` because they sit a level deeper; a live one uses
   `HERE.parent`.
3. Move `nodes_<service>.py` back into `comfy_node/` and re-add it to the
   imports and both mappings in `comfy_node/__init__.py`.
4. Add `<service>/tests` to `testpaths` in `pyproject.toml` and the service to
   the import loop in `.github/workflows/ci.yml`.
5. Add its node back to the pinned contract in `tests/test_node_runtime.py`.
6. Run `pytest` — the golden-graph test will tell you immediately whether the
   graph still matches what was committed.

## Modal-side cleanup

Retiring the code does **not** stop the Modal side. Compute already scales to
zero, so the only ongoing cost is Volume storage, billed per GB-month whether
or not anything reads it.

```bash
modal app stop ideogram4-comfyui
modal app stop waiillustrious-comfyui

# The actual saving — roughly 30 GB and 7 GB respectively.
modal volume delete ideogram4-models
modal volume delete waiillustrious-models
```

Stopping the apps also closes two authenticated public endpoints, which is worth
doing on its own account.

Deleting the Volumes is safe for both: `ideogram4`'s weights are ungated on
Hugging Face, and WAI-illustrious is among the most-downloaded checkpoints on
Civitai. Re-running `download_models` restores either in minutes. The general
caution about Civitai models disappearing applies more to obscure uploads than
to these.
