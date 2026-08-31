"""ComfyUI custom nodes for remote model deployments on Modal.

One package, one install, one copy of the transport and progress-mirror code.
Each service contributes its own node; they share `_runtime`.
"""

from . import nodes_flux2klein, nodes_ideogram4

NODE_CLASS_MAPPINGS = {
    **nodes_ideogram4.NODE_CLASS_MAPPINGS,
    **nodes_flux2klein.NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **nodes_ideogram4.NODE_DISPLAY_NAME_MAPPINGS,
    **nodes_flux2klein.NODE_DISPLAY_NAME_MAPPINGS,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
