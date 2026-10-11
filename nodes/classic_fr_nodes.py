"""Classic full-reference metric nodes: ssim, ssimc, ms_ssim, fsim, psnr,
psnry, mad — plus this pack's own L1 node.

All classic metrics are wrapped via ``pyiqa.create_metric(name)`` with the
registry's own default options (ssim/ms_ssim use the Y channel, fsim is
chromatic, psnry is Y-channel PSNR), so scores match the library exactly.
These metrics carry no weights, so the loader is instant.

L1 is NOT a pyiqa metric (the registry has no l1; the nearest is mad) — the
L1 Score node is this pack's own convenience node and is marked as such.
"""

import torch

import pyiqa
from pyiqa.losses.losses import L1Loss

from ..utility.pyiqa_utils import (
    ORDER_SYMMETRIC,
    TARGET_TOOLTIP,
    comfy_image_to_pyiqa,
    default_device,
    ensure_on_device,
    get_cached_metric,
    reference_tooltip,
)

from ._common import (
    with_memory_management,
    MEMORY_MANAGEMENT_INPUT,
    LAZY_MODEL_OPT,
    LazyScoreMixin,
    lazy_image_opt,
)

CLASSIC_FR_METRICS = ["ssim", "ssimc", "ms_ssim", "fsim", "psnr", "psnry", "mad"]
LOWER_BETTER = {"mad"}

# pyiqa ships L1 as a loss (pyiqa.losses.losses.L1Loss), not as a registry
# metric — the L1 node uses that implementation directly.
_L1_LOSS = L1Loss(reduction="none")


class ClassicFRLoader:
    """Loads one of pyiqa's classic full-reference metrics."""

    CATEGORY = "pyiqa/Classic FR"
    DESCRIPTION = (
        "Loads one of pyiqa's classic full-reference metrics "
        "(SSIM-family / PSNR / FSIM / MAD). No weights, instant."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "metric": (
                    CLASSIC_FR_METRICS,
                    {
                        "default": "ssim",
                        "tooltip": (
                            "Weightless FR metrics, exactly as pyiqa registers them: "
                            "ssim (Y channel), ssimc (RGB), ms_ssim (Y channel, prod pooling), "
                            "fsim (chromatic on), psnr (RGB) / psnry (Y channel), mad. "
                            "Higher is better for all except mad (lower is better)."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("PYIQA_CLASSIC_FR",)
    RETURN_NAMES = ("model",)
    OUTPUT_TOOLTIPS = (
        "The loaded classic FR metric. Wire it into Classic FR Score.",
    )
    FUNCTION = "load"

    def load(self, metric):
        device = default_device()
        m = get_cached_metric(
            "classic_fr",
            (metric, str(device)),
            lambda: pyiqa.create_metric(metric, device=device),
        )
        ensure_on_device(m, device)
        return (m,)


class ClassicFRScore(LazyScoreMixin):
    """Scores the target against the reference with a classic FR metric."""

    CATEGORY = "pyiqa/Classic FR"
    DESCRIPTION = (
        "Scores the target against the reference with a classic FR metric "
        "(ssim/ssimc/ms_ssim/fsim/psnr/psnry/mad), using pyiqa's registered "
        "defaults."
    )
    LAZY_INPUTS = ("target", "reference", "model")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "memory_management": MEMORY_MANAGEMENT_INPUT,
                "model": ("PYIQA_CLASSIC_FR", dict(LAZY_MODEL_OPT)),
                "target": ("IMAGE", lazy_image_opt(TARGET_TOOLTIP)),
                "reference": (
                    "IMAGE",
                    lazy_image_opt(reference_tooltip(ORDER_SYMMETRIC)),
                ),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("score",)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_TOOLTIPS = (
        "Per-pair similarity/error between the target (the produced/distorted "
        "image being judged) and its reference (the pristine original). "
        "ssim/ssimc/ms_ssim/fsim/psnr are higher-is-better (identical images "
        "give 1.0 for the similarity metrics); mad is lower-is-better "
        "(identical images give 0).",
    )
    FUNCTION = "score"

    @with_memory_management
    def score(self, model=None, target=None, reference=None, **kwargs):
        if self._skipped(model, target, reference):
            return ([],)
        if target.shape[0] != reference.shape[0]:
            raise ValueError(
                f"target and reference must have the same batch size, got "
                f"{target.shape[0]} and {reference.shape[0]}"
            )
        dev = ensure_on_device(model, default_device())
        x = comfy_image_to_pyiqa(target, dev)
        y = comfy_image_to_pyiqa(reference, dev)
        out = model(x, y)
        return ([float(v) for v in out.flatten().tolist()],)


class L1Score(LazyScoreMixin):
    """Mean absolute error between two images, via pyiqa's own L1Loss."""

    CATEGORY = "pyiqa/Classic FR"
    DESCRIPTION = (
        "Mean absolute error per image pair, computed with pyiqa's own "
        "pyiqa.losses.losses.L1Loss. Note pyiqa implements L1 as a LOSS, not a "
        "registry metric (the metric registry's nearest entry is mad). "
        "Lower is better."
    )
    LAZY_INPUTS = ("target", "reference")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target": ("IMAGE", lazy_image_opt(TARGET_TOOLTIP)),
                "reference": (
                    "IMAGE",
                    lazy_image_opt(reference_tooltip(ORDER_SYMMETRIC)),
                ),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("score",)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_TOOLTIPS = (
        "Per-pair mean absolute error between target and reference "
        "(pyiqa.losses.L1Loss). LOWER is better; identical images give 0.",
    )
    FUNCTION = "score"

    def score(self, target=None, reference=None):
        if self._skipped(target, reference):
            return ([],)
        if target.shape[0] != reference.shape[0]:
            raise ValueError(
                f"target and reference must have the same batch size, got "
                f"{target.shape[0]} and {reference.shape[0]}"
            )
        scores = []
        for i in range(target.shape[0]):
            x = target[i : i + 1].float().permute(0, 3, 1, 2).contiguous()
            y = reference[i : i + 1].float().permute(0, 3, 1, 2).contiguous()
            scores.append(float(_L1_LOSS(x, y).mean()))
        return (scores,)
