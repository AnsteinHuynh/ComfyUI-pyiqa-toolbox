"""A-FINE nodes — fine-grained IQA head on a CLIP backbone.

Registry variants (both via ``pyiqa.create_metric``):
- afine_fr: FULL-reference (compares target against reference)
- afine_nr: NO-reference

UNUSUAL DIRECTION, faithfully inherited from pyiqa: lower is better, score
range about -2..2 (the registry marks both entries lower_better=True).

Custom checkpoints: a ``.pth`` matching ``*afine*`` dropped into a registered
``pyiqa`` folder appears in the loader's custom dropdown; the base variant
combo selects which head (fr / nr) the checkpoint loads onto.
"""

import os

import pyiqa

from ..utility.pyiqa_utils import (
    ORDER_SYMMETRIC,
    RESOLUTION_MODES,
    TARGET_TOOLTIP,
    apply_resolution,
    comfy_image_to_pyiqa,
    default_device,
    ensure_on_device,
    find_pyiqa_model,
    get_cached_metric,
    list_pyiqa_checkpoints,
    make_room_for_metric,
    reference_tooltip,
)

from ._common import (
    LAZY_MODEL_OPT,
    MEMORY_MANAGEMENT_INPUT,
    LazyScoreMixin,
    lazy_image_opt,
    with_memory_management,
)

BUILTIN_VARIANTS = ["afine_fr", "afine_nr"]


def _default_weight_basenames():
    """Basenames of pyiqa's OWN default A-FINE weights, taken from the arch's
    own URL table when it exposes one; derive nothing when it does not."""
    try:
        from pyiqa.archs.afine_arch import default_model_urls

        return {os.path.basename(url) for url in default_model_urls.values()}
    except Exception:
        return set()


def _custom_options():
    """Custom *afine*.pth fine-tunes: dropdown label -> path.

    Excludes pyiqa's own default weights (by basename, from the arch table) and
    the currently-resolved default main checkpoint (by path)."""
    default_path = find_pyiqa_model("afine.pth")
    return list_pyiqa_checkpoints(
        "*afine*.pth",
        exclude_names=_default_weight_basenames(),
        exclude_paths=[default_path] if default_path is not None else (),
    )


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


class AFINEBundle:
    """A loaded A-FINE metric plus its reference mode."""

    def __init__(self, label, metric):
        self.label = label
        self.metric = metric
        self.mode = metric.metric_mode  # 'FR' or 'NR', as pyiqa registered it

    @property
    def device(self):
        return self.metric.device

    @property
    def dummy_param(self):
        return self.metric.dummy_param

    def to(self, device):
        self.metric.to(device)
        return self


class AFINELoader:
    """Loads an A-FINE scorer via pyiqa.create_metric."""

    CATEGORY = "pyiqa/A-FINE"
    DESCRIPTION = (
        "Loads an A-FINE fine-grained IQA model (CLIP backbone). afine_fr is "
        "full-reference, afine_nr no-reference. Lower is better (score about "
        "-2..2), faithful to pyiqa's lower_better flag."
    )

    @classmethod
    def INPUT_TYPES(cls):
        customs = _custom_options()
        return {
            "required": {
                "model_name": (
                    BUILTIN_VARIANTS,
                    {
                        "default": "afine_fr",
                        "tooltip": (
                            "afine_fr is FULL-reference, afine_nr NO-reference. "
                            "UNUSUAL DIRECTION: lower is better, range about -2..2 "
                            "(faithful to pyiqa's lower_better flag). "
                            "Weights auto-download on first use."
                        ),
                    },
                ),
                "custom_checkpoint": (
                    ["(none)"] + [label for label, _ in customs],
                    {
                        "default": "(none)",
                        "tooltip": (
                            "Custom .pth from models/pyiqa (files named *afine*). "
                            "When selected, it loads onto the base variant below."
                        ),
                    },
                ),
                "custom_base_variant": (
                    BUILTIN_VARIANTS,
                    {
                        "default": "afine_fr",
                        "tooltip": (
                            "Which head (fr / nr) a custom checkpoint loads onto; "
                            "a file name cannot say which head it was trained with."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("PYIQA_AFINE",)
    RETURN_NAMES = ("model",)
    OUTPUT_TOOLTIPS = (
        "The loaded A-FINE metric. Wire it into A-FINE FR Score or "
        "A-FINE NR Score; the score node checks the mode.",
    )
    FREE_COMFY_MEMORY = True
    FUNCTION = "load"

    def load(self, model_name, custom_checkpoint, custom_base_variant):
        device = default_device()
        customs = dict(_custom_options())
        if custom_checkpoint != "(none)":
            base = custom_base_variant
            path = customs[custom_checkpoint]
            key = (path, base, str(device))
            label = custom_checkpoint
        else:
            base = model_name
            path = None
            key = (base, str(device))
            label = base

        def factory():
            make_room_for_metric(device)
            kwargs = {"pretrained_model_path": path} if path else {}
            return AFINEBundle(
                label, pyiqa.create_metric(base, device=device, **kwargs)
            )

        metric = get_cached_metric("afine", key, factory)
        ensure_on_device(metric, device)
        return (metric,)


class AFINEFRScore(LazyScoreMixin):
    """A-FINE full-reference score (lower is better)."""

    CATEGORY = "pyiqa/A-FINE"
    DESCRIPTION = (
        "A-FINE full-reference score of target against reference. LOWER is "
        "better, range about -2..2. Measured order-independent (swapping target "
        "and reference gives the same score). The model is loaded lazily, "
        "after the image exists."
    )
    LAZY_INPUTS = ("target", "reference", "model")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PYIQA_AFINE", dict(LAZY_MODEL_OPT)),
                "target": ("IMAGE", lazy_image_opt(TARGET_TOOLTIP)),
                "reference": (
                    "IMAGE",
                    lazy_image_opt(reference_tooltip(ORDER_SYMMETRIC)),
                ),
                "resolution": (
                    RESOLUTION_MODES,
                    {
                        "default": "auto (up to multiple of 32)",
                        "tooltip": (
                            "AFINE needs sides divisible by 32 (CLIP ViT-B/32 patches). "
                            "'auto' bicubic-resizes up to the next multiple of 32 — the "
                            "graceful handling AFINE's own code intends. 'native' passes "
                            "through untouched; fixed sizes resize exactly. Applied "
                            "identically to both images of each pair."
                        ),
                    },
                ),
                "memory_management": MEMORY_MANAGEMENT_INPUT,
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("score",)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_TOOLTIPS = (
        "Per-pair A-FINE score between the target (the produced/distorted "
        "image being judged) and its reference (the pristine original). LOWER "
        "is better, range about -2..2 (higher = target deviates more from "
        "reference).",
    )
    FUNCTION = "score"

    @with_memory_management
    def score(
        self,
        model=None,
        target=None,
        reference=None,
        resolution="auto (up to multiple of 32)",
        **kwargs,
    ):
        if self._skipped(model, target, reference):
            return ([],)
        if target.shape[0] != reference.shape[0]:
            raise ValueError(
                f"target and reference must have the same batch size, got "
                f"{target.shape[0]} and {reference.shape[0]}"
            )
        if model.mode != "FR":
            raise ValueError(
                f"A-FINE variant '{model.label}' is NO-reference — use the "
                f"A-FINE NR Score node for it."
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


class AFINENRScore(LazyScoreMixin):
    """A-FINE no-reference score (lower is better)."""

    CATEGORY = "pyiqa/A-FINE"
    DESCRIPTION = (
        "A-FINE no-reference quality score, single images. LOWER is better, "
        "range about -2..2. Lazy model load: the model is fetched only after the image(s) exist."
    )
    LAZY_INPUTS = ("image", "model")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PYIQA_AFINE", dict(LAZY_MODEL_OPT)),
                "image": (
                    "IMAGE",
                    lazy_image_opt(
                        "Image(s) to score; scored per-image. Lazy: evaluated "
                        "before the model loads."
                    ),
                ),
                "resolution": (
                    RESOLUTION_MODES,
                    {
                        "default": "auto (up to multiple of 32)",
                        "tooltip": (
                            "AFINE needs sides divisible by 32 (CLIP ViT-B/32 patches). "
                            "'auto' bicubic-resizes up to the next multiple of 32 — the "
                            "graceful handling AFINE's own code intends. 'native' passes "
                            "through untouched; fixed sizes resize exactly."
                        ),
                    },
                ),
                "memory_management": MEMORY_MANAGEMENT_INPUT,
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("score",)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_TOOLTIPS = (
        "Per-image A-FINE no-reference score. LOWER is better, range about -2..2.",
    )
    FUNCTION = "score"

    @with_memory_management
    def score(self, model=None, image=None, resolution="auto (up to multiple of 32)", **kwargs):
        if self._skipped(model, image):
            return ([],)
        if model.mode != "NR":
            raise ValueError(
                f"A-FINE variant '{model.label}' is FULL-reference — use the "
                f"A-FINE FR Score node (it needs a reference image) for it."
            )
        dev = ensure_on_device(model, default_device())
        scores = []
        for i in range(image.shape[0]):
            x = apply_resolution(
                comfy_image_to_pyiqa(image[i : i + 1], dev), resolution
            )
            scores.append(float(_afine_forward(model.metric, x).flatten()[0]))
        return (scores,)
