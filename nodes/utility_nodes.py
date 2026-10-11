"""Utility nodes for the pyiqa toolbox.

Currently one node: a passthrough that fully unloads the pyiqa metric cache.
This is the manual memory-management counterpart of the score nodes'
``memory_management`` combo - place it wherever you want every metric gone
from VRAM and RAM (e.g. on the image wire between the decode and the score
nodes, or at the end of a pipeline before the next generation). Metrics
reload from disk on their next score.

It deliberately touches ONLY this pack's metrics. ComfyUI's own models are
managed by ComfyUI (and by the pack's ``make_room_for_metric`` before a big
metric loads).
"""

import logging

import torch

from ..utility.pyiqa_utils import evict_metric_cache

logger = logging.getLogger(__name__)

try:  # ComfyUI's wildcard type; plain "*" is accepted as a fallback
    from comfy.comfy_types import AnyType

    _ANY = AnyType("*")
except Exception:
    _ANY = "*"


class UnloadPyiqaModels:
    """Pass any value through, fully unloading the cached pyiqa metrics."""

    CATEGORY = "pyiqa/Utilities"
    DESCRIPTION = (
        "Fully unloads every cached pyiqa metric: gone from VRAM and RAM, "
        "nothing left but the files on disk (they reload on the next score - "
        "seconds for small models, ~30s for Q-ReAlign Pro). The value on the "
        "input wire is passed through unchanged, so this node can sit "
        "anywhere in a pipeline. Only this pack's metrics are touched - "
        "ComfyUI's own models are left to ComfyUI."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
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

    def run(self, empty_cuda_cache=True, passthrough=None):
        evict_metric_cache()
        if empty_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return (passthrough,)
