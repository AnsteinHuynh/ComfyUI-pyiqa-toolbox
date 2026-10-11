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

import os

import pyiqa

from ..utility.pyiqa_utils import (
    ORDER_NEAR_SYMMETRIC,
    ORDER_SYMMETRIC,
    TARGET_TOOLTIP,
    comfy_image_to_pyiqa,
    default_device,
    ensure_on_device,
    get_cached_metric,
    list_pyiqa_checkpoints,
    reference_tooltip,
)

from ._common import (
    with_memory_management,
    MEMORY_MANAGEMENT_INPUT,
    LAZY_MODEL_OPT,
    LazyScoreMixin,
    lazy_image_opt,
)

# dropdown label -> (registry name, order hint)
VARIANTS = {
    "lpips (AlexNet)": ("lpips", ORDER_SYMMETRIC),
    "lpips (VGG)": ("lpips-vgg", ORDER_SYMMETRIC),
    "lpips+ (AlexNet)": ("lpips+", ORDER_NEAR_SYMMETRIC),
    "lpips+ (VGG)": ("lpips-vgg+", ORDER_NEAR_SYMMETRIC),
    "stlpips (AlexNet)": ("stlpips", ORDER_SYMMETRIC),
    "stlpips (VGG)": ("stlpips-vgg", ORDER_SYMMETRIC),
}


def _default_weight_basenames():
    """Basenames of pyiqa's OWN default LPIPS/STLPIPS weights (from its URL
    tables), so the cache's default .pth files never show as customs."""
    names = set()
    try:
        from pyiqa.archs import lpips_arch, stlpips_arch

        for module in (lpips_arch, stlpips_arch):
            for url in getattr(module, "default_model_urls", {}).values():
                names.add(os.path.basename(str(url).split("?")[0]))
    except Exception:
        pass
    return names


def _custom_options():
    return [
        (label, path)
        for label, path in list_pyiqa_checkpoints(
            "*lpips*.pth", exclude_names=_default_weight_basenames()
        )
    ]


class LPIPSBundle:
    """A loaded LPIPS metric plus its label and order hint."""

    def __init__(self, label, metric, order_hint):
        self.label = label
        self.metric = metric
        self.order_hint = order_hint

    @property
    def device(self):
        return self.metric.device

    @property
    def dummy_param(self):
        return self.metric.dummy_param

    def to(self, device):
        self.metric.to(device)
        return self


class LPIPSLoader:
    """Loads an LPIPS-family metric via pyiqa.create_metric."""

    CATEGORY = "pyiqa/LPIPS"
    DESCRIPTION = (
        "Loads a learned perceptual distance: LPIPS (AlexNet/VGG), LPIPS+ "
        "(semantic-weighted), or shift-tolerant LPIPS. All are LOWER is better, "
        "range 0-1. Backbone weights auto-download on first use."
    )

    @classmethod
    def INPUT_TYPES(cls):
        customs = _custom_options()
        return {
            "required": {
                "metric": (
                    list(VARIANTS.keys()),
                    {
                        "default": "lpips (AlexNet)",
                        "tooltip": (
                            "lpips (classic v0.1) / lpips+ (semantic-weighted) differ in "
                            "weighting; AlexNet vs VGG is the feature backbone. "
                            "stlpips is the shift-tolerant variant. All LOWER is better."
                        ),
                    },
                ),
                "custom_checkpoint": (
                    ["(none)"] + [label for label, _ in customs],
                    {
                        "default": "(none)",
                        "tooltip": (
                            "Custom .pth from models/pyiqa (files named *lpips*). When "
                            "selected it loads onto the base metric below."
                        ),
                    },
                ),
                "custom_base_metric": (
                    list(VARIANTS.keys()),
                    {
                        "default": "lpips (AlexNet)",
                        "tooltip": (
                            "Which architecture a custom checkpoint loads onto; a file "
                            "name cannot say which backbone/variant it was trained with."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("PYIQA_LPIPS",)
    RETURN_NAMES = ("model",)
    OUTPUT_TOOLTIPS = (
        "The loaded LPIPS-family metric. Wire it into LPIPS FR Score.",
    )
    FUNCTION = "load"

    def load(self, metric, custom_checkpoint, custom_base_metric):
        device = default_device()
        customs = dict(_custom_options())
        if custom_checkpoint != "(none)":
            base, hint = VARIANTS[custom_base_metric]
            path = customs[custom_checkpoint]
            key = (path, base, str(device))
            label = custom_checkpoint
        else:
            base, hint = VARIANTS[metric]
            path = None
            key = (base, str(device))
            label = metric

        def factory():
            kwargs = {"pretrained_model_path": path} if path else {}
            return LPIPSBundle(
                label, pyiqa.create_metric(base, device=device, **kwargs), hint
            )

        m = get_cached_metric("lpips", key, factory)
        ensure_on_device(m, device)
        return (m,)


class LPIPSFrScore(LazyScoreMixin):
    """LPIPS-family score of target against reference. LOWER is better."""

    CATEGORY = "pyiqa/LPIPS"
    DESCRIPTION = (
        "Perceptual distance between target and reference - LPIPS / LPIPS+ / "
        "ST-LPIPS. LOWER is better, range 0-1. The model is loaded lazily, "
        "after the image exists."
    )
    LAZY_INPUTS = ("target", "reference", "model")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "memory_management": MEMORY_MANAGEMENT_INPUT,
                "model": ("PYIQA_LPIPS", dict(LAZY_MODEL_OPT)),
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
        "Per-pair perceptual distance between the target (the "
        "produced/distorted image being judged) and its reference (the "
        "pristine original) - LPIPS / LPIPS+ / ST-LPIPS. LOWER is better, "
        "range 0-1; identical images give 0.",
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
        out = model.metric(x, y)
        return ([float(v) for v in out.flatten().tolist()],)
