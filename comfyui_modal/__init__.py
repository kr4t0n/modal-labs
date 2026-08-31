"""Shared machinery for serving a ComfyUI-backed model on Modal.

Each service under this repository supplies only what is genuinely
model-specific — the graph, the weight table, and a handful of request fields.
Everything else (the container image, the ComfyUI supervisor, the ASGI proxy,
the CLI, the custom-node runtime) lives here so a fix lands once.
"""
