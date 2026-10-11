"""TOPIQ (CFANet) nodes — banker-training-free fine-grained IQA.

Registry variants (all via ``pyiqa.create_metric``):
- topiq_fr / topiq_fr-pipal: FULL-reference (KADID-10k / PIPAL trained)
- topiq_nr / topiq_nr-flive / topiq_nr-spaq: NO-reference (KonIQ / FLIVE / SPAQ)

Custom checkpoints: a ``.pth`` matching ``*topiq*`` or ``cfanet*`` dropped into
a registered ``pyiqa`` folder appears in the loader's custom dropdown; the base
variant combo selects which architecture (FR or NR) the checkpoint loads onto,
because a file name cannot say which head it was trained with.
"""

import os

import pyiqa

from pyiqa.archs.topiq_arch import default_model_urls as _topiq_default_urls

from ..utility.pyiqa_utils import (
    ORDER_ASYMMETRIC,
    TARGET_TOOLTIP,
    comfy_image_to_pyiqa,
    default_device,
    ensure_on_device,
    get_cached_metric,
    list_pyiqa_checkpoints,
    make_room_for_metric,
    reference_tooltip,
)

from ._common import (
    with_memory_management,
    MEMORY_MANAGEMENT_INPUT,
    LAZY_MODEL_OPT,
    LazyScoreMixin,
    lazy_image_opt,
)

BUILTIN_VARIANTS = [
    "topiq_fr",
    "topiq_fr-pipal",
    "topiq_nr",
    "topiq_nr-flive",
    "topiq_nr-spaq",
    "topiq_nr-face",
    "topiq_nr_swin-face",
    "topiq_nr-face-v1",
]


def _default_weight_basenames():
    """Basenames of pyiqa's OWN default TOPIQ/CFANet weights (the cfanet_*.pth
    and topiq_nr_*.pth files) — never customs."""
    return {os.path.basename(url) for url in _topiq_default_urls.values()}


def _custom_options():
    """Custom TOPIQ/CFANet checkpoints: dropdown label -> path.

    Excludes pyiqa's own default cfanet_*.pth / topiq_nr_*.pth weights by
    basename, so the family's own cached files never show as customs.
    """
    exclude_names = _default_weight_basenames()
    options = []
    for label, path in list_pyiqa_checkpoints(
        "*topiq*.pth", exclude_names=exclude_names
    ):
        options.append((label, path))
    for label, path in list_pyiqa_checkpoints(
        "cfanet*.pth", exclude_names=exclude_names
    ):
        options.append((label, path))
    return options


class TOPIQBundle:
    """A loaded TOPIQ metric plus its reference mode."""

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


class TOPIQLoader:
    """Loads a TOPIQ/CFANet scorer via pyiqa.create_metric."""

    CATEGORY = "pyiqa/TOPIQ"
    DESCRIPTION = (
        "Loads a TOPIQ (CFANet) quality model. FR variants (topiq_fr, "
        "topiq_fr-pipal) need a reference image; NR variants (topiq_nr, "
        "-flive, -spaq) score single images."
    )

    @classmethod
    def INPUT_TYPES(cls):
        customs = _custom_options()
        return {
            "required": {
                "model_name": (
                    BUILTIN_VARIANTS,
                    {
                        "default": "topiq_fr",
                        "tooltip": (
                            "topiq_fr (KADID-10k) and topiq_fr-pipal are FULL-reference "
                            "(and order-sensitive); topiq_nr (KonIQ), -flive, -spaq are "
                            "NO-reference. topiq_nr-face / -swin-face / -face-v1 are "
                            "GFIQA face models (run face alignment internally; expect "
                            "face crops). Range ~0-1, higher is better. Weights "
                            "auto-download on first use."
                        ),
                    },
                ),
                "custom_checkpoint": (
                    ["(none)"] + [label for label, _ in customs],
                    {
                        "default": "(none)",
                        "tooltip": (
                            "Custom .pth from models/pyiqa (files named *topiq* or cfanet*). "
                            "When selected, it loads onto the base variant below instead of "
                            "the built-in weights."
                        ),
                    },
                ),
                "custom_base_variant": (
                    ["topiq_fr", "topiq_nr"],
                    {
                        "default": "topiq_fr",
                        "tooltip": (
                            "Which architecture (FR or NR head) a custom checkpoint loads "
                            "onto; a file name cannot say which head it was trained with."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("PYIQA_TOPIQ",)
    RETURN_NAMES = ("model",)
    OUTPUT_TOOLTIPS = (
        "The loaded TOPIQ metric. Wire it into TOPIQ FR Score (for topiq_fr "
        "variants) or TOPIQ NR Score (for the nr variants); the score node "
        "checks the mode and tells you if you picked the wrong one.",
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
            return TOPIQBundle(
                label, pyiqa.create_metric(base, device=device, **kwargs)
            )

        metric = get_cached_metric("topiq", key, factory)
        ensure_on_device(metric, device)
        return (metric,)


class TOPIQFRScore(LazyScoreMixin):
    """TOPIQ full-reference score."""

    CATEGORY = "pyiqa/TOPIQ"
    DESCRIPTION = (
        "TOPIQ full-reference score of target against reference. ~0-1, higher "
        "is better. Order-sensitive: target and reference are not "
        "interchangeable. Lazy model load: the model is fetched only after the image(s) exist."
    )
    LAZY_INPUTS = ("target", "reference", "model")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "memory_management": MEMORY_MANAGEMENT_INPUT,
                "model": ("PYIQA_TOPIQ", dict(LAZY_MODEL_OPT)),
                "target": ("IMAGE", lazy_image_opt(TARGET_TOOLTIP)),
                "reference": (
                    "IMAGE",
                    lazy_image_opt(reference_tooltip(ORDER_ASYMMETRIC)),
                ),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("score",)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_TOOLTIPS = (
        "Per-pair TOPIQ quality of the target (the produced/distorted image "
        "being judged) against its reference (the pristine original), ~0-1, "
        "higher is better. Order-sensitive: swapping target and reference "
        "changes the value.",
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
        if model.mode != "FR":
            raise ValueError(
                f"TOPIQ variant '{model.label}' is NO-reference — use the "
                f"TOPIQ NR Score node for it."
            )
        dev = ensure_on_device(model, default_device())
        x = comfy_image_to_pyiqa(target, dev)
        y = comfy_image_to_pyiqa(reference, dev)
        out = model.metric(x, y)
        return ([float(v) for v in out.flatten().tolist()],)


class TOPIQNRScore(LazyScoreMixin):
    """TOPIQ no-reference score."""

    CATEGORY = "pyiqa/TOPIQ"
    DESCRIPTION = (
        "TOPIQ no-reference quality score, single images. ~0-1, higher is "
        "better. Lazy model load: the model is fetched only after the image(s) exist."
    )
    LAZY_INPUTS = ("image", "model")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "memory_management": MEMORY_MANAGEMENT_INPUT,
                "model": ("PYIQA_TOPIQ", dict(LAZY_MODEL_OPT)),
                "image": (
                    "IMAGE",
                    lazy_image_opt(
                        "Image(s) to score; batch is scored per-image. Lazy: "
                        "evaluated before the model loads."
                    ),
                ),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("score",)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_TOOLTIPS = (
        "Per-image TOPIQ no-reference quality score, ~0-1, higher is better.",
    )
    FUNCTION = "score"

    @with_memory_management
    def score(self, model=None, image=None, **kwargs):
        if self._skipped(model, image):
            return ([],)
        if model.mode != "NR":
            raise ValueError(
                f"TOPIQ variant '{model.label}' is FULL-reference — use the "
                f"TOPIQ FR Score node (it needs a reference image) for it."
            )
        dev = ensure_on_device(model, default_device())
        x = comfy_image_to_pyiqa(image, dev)
        out = model.metric(x)
        return ([float(v) for v in out.flatten().tolist()],)
