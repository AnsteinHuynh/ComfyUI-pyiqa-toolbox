"""Utility nodes for the pyiqa toolbox.

Currently one node: a passthrough that releases the pyiqa metric cache so its
VRAM goes back to the card. This is the manual counterpart of the score nodes'
``memory_management`` input - place it wherever you want the metrics freed
(e.g. on the image wire between the decode and the score nodes, or at the end of
a pipeline before the next generation).

It deliberately touches ONLY this pack's metrics. ComfyUI's own models are
managed by ComfyUI (and by that pack's ``make_room_for_metric`` before a big
metric loads); Edit > Unload Models offloads everything, including ours.
"""

import logging

import torch

from ..utility.pyiqa_utils import offload_metric_cache

logger = logging.getLogger(__name__)

try:  # ComfyUI's wildcard type; plain "*" is accepted as a fallback
    from comfy.comfy_types import AnyType

    _ANY = AnyType("*")
except Exception:
    _ANY = "*"


class UnloadPyiqaModels:
    """Pass any value through, releasing the cached pyiqa metrics to system RAM."""

    CATEGORY = "pyiqa/Utilities"
    DESCRIPTION = (
        "Releases the pyiqa metric cache: every loaded metric is parked in "
        "system RAM and its VRAM returned to the allocator, freeing the card "
        "for the next generation. The value on the input wire is passed through "
        "unchanged. Only this pack's metrics are touched - ComfyUI's own models "
        "are left to ComfyUI. Metrics reload automatically (lazily) the next "
        "time a score node runs."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unload_pyiqa_models": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Park every cached pyiqa metric in system RAM and "
                            "free its VRAM. Turn off to use this node purely as "
                            "a passthrough."
                        ),
                    },
                ),
                "empty_cuda_cache": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "After unloading, return PyTorch's reserved-but-unused "
                            "GPU memory to the driver (torch.cuda.empty_cache())."
                        ),
                    },
                ),
            },
            "optional": {
                "passthrough": (
                    _ANY,
                    {
                        "tooltip": (
                            "Any wire to pass through unchanged (image, latent, "
                            "etc). Can also be left unconnected if you only want "
                            "to trigger the unload."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = (_ANY,)
    RETURN_NAMES = ("passthrough",)
    FUNCTION = "run"
    OUTPUT_TOOLTIPS = ("The input value, returned unchanged.",)

    def run(self, unload_pyiqa_models=True, empty_cuda_cache=True, passthrough=None):
        if unload_pyiqa_models:
            offload_metric_cache()
        if empty_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return (passthrough,)
