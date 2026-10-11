"""LPIPS family nodes (learned perceptual full-reference distances).

Registry variants (all via ``pyiqa.create_metric``, all FR, lower is better,
range 0-1):
- lpips / lpips-vgg:     LPIPS v0.1 on AlexNet / VGG
- lpips+ / lpips-vgg+:   LPIPS+ (semantic_weight_layer=2) on AlexNet / VGG
- stlpips / stlpips-vgg: shift-tolerant LPIPS on AlexNet / VGG

Measured order behaviour: lpips, lpips-vgg, stlpips, stlpips-vgg are exactly
order-independent; the lpips+ variants are nominally symmetric with a ~0.1%
order dependence from the semantic weighting.

Custom checkpoints: ``*lpips*.pth`` in a registered pyiqa folder appears in the
custom dropdown; the base-metric combo selects the architecture it loads onto.
"""

from ..utility.pyiqa_utils import ORDER_SYMMETRIC
from ._common import FamilyLoaderBase, FRScoreBase


class LPIPSLoader(FamilyLoaderBase):
    """Loads an LPIPS-family metric via pyiqa.create_metric."""

    CATEGORY = "pyiqa/LPIPS"
    FAMILY_KEY = "lpips"
    MODEL_TYPE = "PYIQA_LPIPS"
    VARIANT_MAP = {
        "lpips (AlexNet)": "lpips",
        "lpips (VGG)": "lpips-vgg",
        "lpips+ (AlexNet)": "lpips+",
        "lpips+ (VGG)": "lpips-vgg+",
        "stlpips (AlexNet)": "stlpips",
        "stlpips (VGG)": "stlpips-vgg",
    }
    METRIC_TOOLTIP = (
        "lpips (classic v0.1) / lpips+ (semantic-weighted) differ in "
        "weighting; AlexNet vs VGG is the feature backbone. stlpips is the "
        "shift-tolerant variant. All LOWER is better."
    )
    CUSTOM_PATTERNS = ("*lpips*.pth",)
    # exclude pyiqa's own LPIPS AND STLPIPS default .pth files from customs
    DEFAULT_WEIGHTS = ("pyiqa.archs.lpips_arch", "pyiqa.archs.stlpips_arch")
    DESCRIPTION = (
        "Loads a learned perceptual distance: LPIPS (AlexNet/VGG), LPIPS+ "
        "(semantic-weighted), or shift-tolerant LPIPS. All are LOWER is better, "
        "range 0-1. Backbone weights auto-download on first use."
    )
    OUTPUT_TOOLTIPS = (
        "The loaded LPIPS-family metric. Wire it into LPIPS FR Score.",
    )


class LPIPSFrScore(FRScoreBase):
    """LPIPS-family score of target against reference. LOWER is better."""

    CATEGORY = "pyiqa/LPIPS"
    MODEL_TYPE = "PYIQA_LPIPS"
    # lpips/stlpips are exactly symmetric; lpips+ carries a ~0.1% order
    # dependence - close enough that one tooltip serves the family
    ORDER_HINT = ORDER_SYMMETRIC
    DESCRIPTION = (
        "Perceptual distance between target and reference - LPIPS / LPIPS+ / "
        "ST-LPIPS. LOWER is better, range 0-1. The model is loaded lazily, "
        "after the image exists."
    )
    OUTPUT_TOOLTIPS = (
        "Per-pair perceptual distance between the target (the "
        "produced/distorted image being judged) and its reference (the "
        "pristine original) - LPIPS / LPIPS+ / ST-LPIPS. LOWER is better, "
        "range 0-1; identical images give 0.",
    )
