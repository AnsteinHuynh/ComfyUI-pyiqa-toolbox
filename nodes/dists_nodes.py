"""DISTS family nodes (hand-crafted deep structure/texture FR distances).

Registry variants (via ``pyiqa.create_metric``, both FR, LOWER is better):
- dists: DISTS (Ding et al.) — deep image structure and texture similarity,
  range 0-1
- msswd: MS-SWD (learned multi-scale Sliced-Wasserstein distance), range 0-~10

Both measured exactly order-independent.
"""

import os

import pyiqa

from ..utility.pyiqa_utils import (
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

VARIANTS = {
    "dists": "dists",
    "msswd (MS-SWD learned)": "msswd",
}


def _default_weight_basenames():
    names = set()
    try:
        from pyiqa.archs import dists_arch

        for url in getattr(dists_arch, "default_model_urls", {}).values():
            names.add(os.path.basename(str(url).split("?")[0]))
    except Exception:
        pass
    return names


def _custom_options():
    found = list_pyiqa_checkpoints(
        "*dists*.pth", exclude_names=_default_weight_basenames()
    ) + list_pyiqa_checkpoints(
        "*msswd*.pth", exclude_names=_default_weight_basenames()
    )
    return found


class DISTSBundle:
    def __init__(self, label, metric):
        self.label = label
        self.metric = metric

    @property
    def device(self):
        return self.metric.device

    @property
    def dummy_param(self):
        return self.metric.dummy_param

    def to(self, device):
        self.metric.to(device)
        return self


class DISTSLoader:
    """Loads DISTS or MS-SWD via pyiqa.create_metric."""

    CATEGORY = "pyiqa/DISTS"
    DESCRIPTION = (
        "Loads DISTS (deep structure/texture distance, range 0-1) or MS-SWD "
        "(learned multi-scale Sliced-Wasserstein, 0-~10). Both FR, LOWER is "
        "better."
    )

    @classmethod
    def INPUT_TYPES(cls):
        customs = _custom_options()
        return {
            "required": {
                "metric": (
                    list(VARIANTS.keys()),
                    {
                        "default": "dists",
                        "tooltip": (
                            "dists: deep structure+texture similarity (0-1). msswd: "
                            "learned multi-scale Sliced-Wasserstein distance (0-~10). "
                            "Both LOWER is better and order-independent."
                        ),
                    },
                ),
                "custom_checkpoint": (
                    ["(none)"] + [label for label, _ in customs],
                    {
                        "default": "(none)",
                        "tooltip": (
                            "Custom .pth from models/pyiqa (files named *dists* or "
                            "*msswd*). When selected it loads onto the base metric below."
                        ),
                    },
                ),
                "custom_base_metric": (
                    list(VARIANTS.keys()),
                    {
                        "default": "dists",
                        "tooltip": "Which architecture a custom checkpoint loads onto.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("PYIQA_DISTS",)
    RETURN_NAMES = ("model",)
    OUTPUT_TOOLTIPS = (
        "The loaded DISTS / MS-SWD metric. Wire it into DISTS FR Score.",
    )
    FUNCTION = "load"

    def load(self, metric, custom_checkpoint, custom_base_metric):
        device = default_device()
        customs = dict(_custom_options())
        if custom_checkpoint != "(none)":
            base = VARIANTS[custom_base_metric]
            path = customs[custom_checkpoint]
            key = (path, base, str(device))
            label = custom_checkpoint
        else:
            base = VARIANTS[metric]
            path = None
            key = (base, str(device))
            label = metric

        def factory():
            kwargs = {"pretrained_model_path": path} if path else {}
            return DISTSBundle(
                label, pyiqa.create_metric(base, device=device, **kwargs)
            )

        m = get_cached_metric("dists", key, factory)
        ensure_on_device(m, device)
        return (m,)


class DISTSFrScore(LazyScoreMixin):
    """DISTS / MS-SWD score of target against reference. LOWER is better."""

    CATEGORY = "pyiqa/DISTS"
    DESCRIPTION = (
        "DISTS / MS-SWD distance between target and reference. LOWER is better. "
        "Order-independent. Lazy model load: the model is fetched only after the image(s) exist."
    )
    LAZY_INPUTS = ("target", "reference", "model")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "memory_management": MEMORY_MANAGEMENT_INPUT,
                "model": ("PYIQA_DISTS", dict(LAZY_MODEL_OPT)),
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
        "Per-pair structure/texture distance between the target (the "
        "produced/distorted image being judged) and its reference (the "
        "pristine original). LOWER is better; dists is 0-1, msswd 0-~10; "
        "identical images give 0.",
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
