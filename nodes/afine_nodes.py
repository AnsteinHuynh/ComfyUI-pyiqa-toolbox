"""A-FINE nodes — fine-grained IQA head on a CLIP backbone.

Registry variants (both via ``pyiqa.create_metric``):
- afine_fr: FULL-reference (compares target against reference)
- afine_nr: NO-reference

UNUSUAL DIRECTION, faithfully inherited from pyiqa: lower is better, score
range about -2..2 (the registry marks both entries lower_better=True).

Custom checkpoints: a ``.pth`` matching ``*afine*`` dropped into a registered
``pyiqa`` folder appears in the loader's custom dropdown; the base variant
combo selects which head (fr / nr) the checkpoint loads onto.

Score nodes: A-FINE needs sides divisible by 32 (CLIP ViT-B/32 patches) and
its forward assumes batch size 1 (squeeze/unsqueeze in its normalize), so both
score nodes resize per the ``resolution`` combo and score one image at a time.
"""

from ..utility.pyiqa_utils import (
    ORDER_SYMMETRIC,
    RESOLUTION_MODES,
    apply_resolution,
    comfy_image_to_pyiqa,
    default_device,
    ensure_on_device,
    find_pyiqa_model,
)
from ._common import (
    FamilyLoaderBase,
    FRScoreBase,
    MEMORY_MANAGEMENT_INPUT,
    NRScoreBase,
    with_memory_management,
)

BUILTIN_VARIANTS = ["afine_fr", "afine_nr"]


def _afine_forward(metric, x, y=None):
    """Score one pair at a time.

    AFINE's forward wraps its normalize in squeeze(0)/unsqueeze(0), which
    assumes batch size 1 — with a larger batch the unsqueeze ADDS a dimension
    ((1, B, 3, H, W)) and every downstream op breaks. Per-image calls are the
    only mode the arch actually supports, so that is what we feed it.
    """
    if y is None:
        return metric(x)
    return metric(x, y)


class AFINELoader(FamilyLoaderBase):
    """Loads an A-FINE scorer via pyiqa.create_metric."""

    CATEGORY = "pyiqa/A-FINE"
    FAMILY_KEY = "afine"
    MODEL_TYPE = "PYIQA_AFINE"
    VARIANT_MAP = {name: name for name in BUILTIN_VARIANTS}
    METRIC_INPUT_NAME = "model_name"  # shipped name - keep saved workflows
    CUSTOM_BASE_INPUT_NAME = "custom_base_variant"
    METRIC_TOOLTIP = (
        "afine_fr is FULL-reference, afine_nr NO-reference. UNUSUAL DIRECTION: "
        "lower is better, range about -2..2 (faithful to pyiqa's lower_better "
        "flag). Weights auto-download on first use."
    )
    CUSTOM_PATTERNS = ("*afine*.pth",)
    DEFAULT_WEIGHTS = "pyiqa.archs.afine_arch"
    DESCRIPTION = (
        "Loads an A-FINE fine-grained IQA model (CLIP backbone). afine_fr is "
        "full-reference, afine_nr no-reference. Lower is better (score about "
        "-2..2), faithful to pyiqa's lower_better flag."
    )
    OUTPUT_TOOLTIPS = (
        "The loaded A-FINE metric. Wire it into A-FINE FR Score or "
        "A-FINE NR Score; the score node checks the mode.",
    )
    FREE_COMFY_MEMORY = True

    @classmethod
    def _custom_exclude_paths(cls):
        # the resolved default afine.pth is not a custom checkpoint either
        default_path = find_pyiqa_model("afine.pth")
        return (default_path,) if default_path is not None else ()


_RESOLUTION_TOOLTIP = (
    "AFINE needs sides divisible by 32 (CLIP ViT-B/32 patches). 'auto' "
    "bicubic-resizes up to the next multiple of 32 — the graceful handling "
    "AFINE's own code intends. 'native' passes through untouched; fixed "
    "sizes resize exactly. Applied identically to both images of each pair."
)


class AFINEFRScore(FRScoreBase):
    """A-FINE full-reference score (lower is better)."""

    CATEGORY = "pyiqa/A-FINE"
    MODEL_TYPE = "PYIQA_AFINE"
    ORDER_HINT = ORDER_SYMMETRIC
    DESCRIPTION = (
        "A-FINE full-reference score of target against reference. LOWER is "
        "better, range about -2..2. Measured order-independent (swapping "
        "target and reference gives the same score). The model is loaded "
        "lazily, after the image exists."
    )
    OUTPUT_TOOLTIPS = (
        "Per-pair A-FINE score between the target (the produced/distorted "
        "image being judged) and its reference (the pristine original). LOWER "
        "is better, range about -2..2 (higher = target deviates more from "
        "reference).",
    )

    @classmethod
    def INPUT_TYPES(cls):
        types = super().INPUT_TYPES()
        types["required"]["resolution"] = (
            RESOLUTION_MODES,
            {"default": "auto (up to multiple of 32)", "tooltip": _RESOLUTION_TOOLTIP},
        )
        return types

    @with_memory_management
    def score(self, model=None, target=None, reference=None, resolution="auto (up to multiple of 32)", **kwargs):
        if self._skipped(model, target, reference):
            return ([],)
        if target.shape[0] != reference.shape[0]:
            raise ValueError(
                f"target and reference must have the same batch size, got "
                f"{target.shape[0]} and {reference.shape[0]}"
            )
        if getattr(model, "mode", None) not in (None, self.MODE_CHECK):
            raise ValueError(
                f"A-FINE variant '{model.label}' is {model.mode}-reference — "
                f"use the A-FINE {'NR' if self.MODE_CHECK == 'FR' else 'FR'} "
                f"Score node for it."
            )
        dev = ensure_on_device(model, default_device())
        scores = []
        for i in range(target.shape[0]):
            x = apply_resolution(
                comfy_image_to_pyiqa(target[i : i + 1], dev), resolution
            )
            y = apply_resolution(
                comfy_image_to_pyiqa(reference[i : i + 1], dev), resolution
            )
            scores.append(float(_afine_forward(model.metric, x, y).flatten()[0]))
        return (scores,)


class AFINENRScore(NRScoreBase):
    """A-FINE no-reference score (lower is better)."""

    CATEGORY = "pyiqa/A-FINE"
    MODEL_TYPE = "PYIQA_AFINE"
    DESCRIPTION = (
        "A-FINE no-reference quality score, single images. LOWER is better, "
        "range about -2..2. Lazy model load: the model is fetched only after "
        "the image(s) exist."
    )
    OUTPUT_TOOLTIPS = (
        "Per-image A-FINE no-reference score. LOWER is better, range about -2..2.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        types = super().INPUT_TYPES()
        types["required"]["resolution"] = (
            RESOLUTION_MODES,
            {"default": "auto (up to multiple of 32)", "tooltip": _RESOLUTION_TOOLTIP},
        )
        return types

    @with_memory_management
    def score(self, model=None, image=None, resolution="auto (up to multiple of 32)", **kwargs):
        if self._skipped(model, image):
            return ([],)
        if getattr(model, "mode", None) not in (None, self.MODE_CHECK):
            raise ValueError(
                f"A-FINE variant '{model.label}' is {model.mode}-reference — "
                f"use the A-FINE {'FR' if self.MODE_CHECK == 'NR' else 'NR'} "
                f"Score node for it."
            )
        dev = ensure_on_device(model, default_device())
        scores = []
        for i in range(image.shape[0]):
            x = apply_resolution(
                comfy_image_to_pyiqa(image[i : i + 1], dev), resolution
            )
            scores.append(float(_afine_forward(model.metric, x).flatten()[0]))
        return (scores,)
