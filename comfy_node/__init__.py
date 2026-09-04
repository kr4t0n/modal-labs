"""ComfyUI custom nodes for remote model deployments on Modal.

One package, one install, one copy of the transport and progress-mirror code.
Each service contributes its own node; they share `_runtime`.
"""

from . import (
    nodes_darkbeast3,
    nodes_finepornv4,
    nodes_flux2klein,
    nodes_redcraft3,
    nodes_redgpt2gpt,
    nodes_ultra,
    nodes_zimageturbostableyogi,
)

NODE_CLASS_MAPPINGS = {
    **nodes_flux2klein.NODE_CLASS_MAPPINGS,
    **nodes_ultra.NODE_CLASS_MAPPINGS,
    **nodes_zimageturbostableyogi.NODE_CLASS_MAPPINGS,
    **nodes_finepornv4.NODE_CLASS_MAPPINGS,
    **nodes_redgpt2gpt.NODE_CLASS_MAPPINGS,
    **nodes_redcraft3.NODE_CLASS_MAPPINGS,
    **nodes_darkbeast3.NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **nodes_flux2klein.NODE_DISPLAY_NAME_MAPPINGS,
    **nodes_ultra.NODE_DISPLAY_NAME_MAPPINGS,
    **nodes_zimageturbostableyogi.NODE_DISPLAY_NAME_MAPPINGS,
    **nodes_finepornv4.NODE_DISPLAY_NAME_MAPPINGS,
    **nodes_redgpt2gpt.NODE_DISPLAY_NAME_MAPPINGS,
    **nodes_redcraft3.NODE_DISPLAY_NAME_MAPPINGS,
    **nodes_darkbeast3.NODE_DISPLAY_NAME_MAPPINGS,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
