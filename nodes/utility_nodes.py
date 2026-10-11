"""Utility nodes for the pyiqa toolbox.

Currently one node: a passthrough that releases the pyiqa metric cache. This
is the manual memory-management counterpart of the score nodes'
``memory_management`` combo - place it wherever you want the metrics released
(e.g. on the image wire between the decode and the score nodes, or at the end
of a pipeline before the next generation), and choose HOW released they
should be: fully unloaded (gone from VRAM and RAM) or parked in system RAM.

It deliberately touches ONLY this pack's metrics. ComfyUI's own models are
managed by ComfyUI (and by the pack's ``make_room_for_metric`` before a big
metric loads); Edit > Unload Models offloads everything, including ours.
"""

import logging

import torch

from ..utility.pyiqa_utils import evict_metric_cache

logger = logging.getLogger(__name__)

EVICT_FULL = "fully unload (free VRAM + RAM)"
EVICT_PARK = "park in RAM (free VRAM only)"

try:  # ComfyUI's wildcard type; plain "*" is accepted as a fallback
    from comfy.comfy_types import AnyType

    _ANY = AnyType("*")
except Exception:
    _ANY = "*"


class UnloadPyiqaModels:
    """Pass any value through, releasing the cached pyiqa metrics."""

    CATEGORY = "pyiqa/Utilities"
    DESCRIPTION = (
        "Releases every cached pyiqa metric. 'fully unload' drops them from "
        "VRAM and RAM (nothing left but the files on disk; they reload on the "
        "next score). 'park in RAM' frees the VRAM and keeps them warm in "
        "system RAM for a fast restore. The value on the input wire is "
        "passed through unchanged, so this node can sit anywhere in a "
        "pipeline. Only this pack's metrics are touched - ComfyUI's own "
        "models are left to ComfyUI."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (
                    [EVICT_FULL, EVICT_PARK],
                    {
                        "default": EVICT_FULL,
                        "tooltip": (
                            "fully unload: metrics gone from VRAM and RAM "
                            "(reload from disk next score - seconds for small "
                            "models, ~30s for Q-ReAlign Pro). park in RAM: "
                            "VRAM freed, models warm in system RAM (~2s "
                            "restore). int8_convrot models can never be "
                            "parked, so they are fully unloaded either way."
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
                            "to trigger the release."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = (_ANY,)
    RETURN_NAMES = ("passthrough",)
    FUNCTION = "run"
    OUTPUT_TOOLTIPS = ("The input value, returned unchanged.",)

    def run(self, mode=EVICT_FULL, empty_cuda_cache=True, passthrough=None):
        evict_metric_cache(fully=mode == EVICT_FULL)
        if empty_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return (passthrough,)
