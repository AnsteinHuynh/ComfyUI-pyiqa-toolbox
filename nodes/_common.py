"""Shared building blocks for the per-family loader/score node pairs.

Each family keeps its own thin module (variants, tooltips, exclusions) but the
loader mechanics and the FR/NR score-node bodies live here so the families stay
short and consistent.

Design notes:
- Loader output is a ``MetricBundle`` wrapping the ``pyiqa.create_metric``
  object, so the family cache, eviction, device restore after Unload Models and
  the unload hook all apply unchanged.
- Score nodes are ``OUTPUT_IS_LIST``: per-image (NR) or per-pair (FR) scores.
- ``MODE_CHECK`` makes a score node refuse a model of the wrong reference mode
  with a clear message instead of a cryptic shape error.
"""

import functools
import importlib
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
    make_room_for_metric,
    reference_tooltip,
    release_metric,
)


# Memory handling for the metric after scoring.
MEMORY_FREE = "free_after_score"
MEMORY_KEEP = "keep_loaded"

MEMORY_MANAGEMENT_INPUT = (
    [MEMORY_FREE, MEMORY_KEEP],
    {
        "default": MEMORY_FREE,
        "tooltip": (
            "What happens to this metric after it scores. 'free_after_score' "
            "parks it in system RAM and releases its VRAM, so the next "
            "generation has the whole card (it is reloaded automatically on the "
            "next score - keep the weights on a fast disk). 'keep_loaded' leaves "
            "it on the GPU: fastest to re-score, but it holds its VRAM between "
            "runs. Note: with several score nodes sharing one loader, "
            "'free_after_score' on each makes the shared model bounce GPU<->RAM "
            "between them. int8_convrot Q-ReAlign models cannot be parked in RAM "
            "at all - 'free_after_score' is a no-op for them (they stay on the "
            "GPU until evicted by the next loader that needs the VRAM)."
        ),
    },
)


def with_memory_management(fn):
    """Apply the node's ``memory_management`` choice after a score runs.

    A decorator (rather than an edit inside every score body) so the behavior
    is uniform across all score nodes: read the mode from the inputs, then
    release the metric to CPU when the mode asks for it.
    """

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        result = fn(self, *args, **kwargs)
        metric = kwargs.get("model")
        if metric is None and args:
            metric = args[0]
        mode = kwargs.get("memory_management", MEMORY_FREE)
        if metric is not None and mode == MEMORY_FREE:
            release_metric(metric)
        return result

    return wrapper


# Shared UI fragments for the lazy-evaluation pattern.
LAZY_MODEL_OPT = {
    "lazy": True,
    "tooltip": (
        "From the matching Loader node. This input (and the loader feeding it) "
        "is evaluated LAZILY - only once this node's image inputs exist - so a "
        "generation-only run never loads the metric model."
    ),
}


def lazy_image_opt(tooltip):
    return {"lazy": True, "tooltip": tooltip}


class LazyScoreMixin:
    """Defer heavy inputs until they are actually needed.

    ``LAZY_INPUTS`` lists the lazy input names in the order they should be
    requested. ``check_lazy_status`` returns at most ONE name per call so
    ComfyUI evaluates them strictly in order (it collects the returned names
    into a set, so returning several at once would lose the ordering) -
    image(s) first, then the model, which is what keeps a big metric's load
    from racing the generator.

    A node still runs if its lazy inputs are never produced (ComfyUI hands it
    ``None``); the score function must therefore tolerate missing inputs and
    return empty outputs in that case.
    """

    LAZY_INPUTS = ("model",)

    def check_lazy_status(self, **kwargs):
        for name in self.LAZY_INPUTS:
            if kwargs.get(name) is None:
                return [name]
        return []

    @staticmethod
    def _skipped(*values):
        return any(v is None for v in values)


class MetricBundle:
    """A loaded pyiqa metric plus its display label and FR order hint."""

    def __init__(self, label, metric, order_hint=None):
        self.label = label
        self.metric = metric
        self.order_hint = order_hint
        self.mode = getattr(metric, "metric_mode", None)

    @property
    def device(self):
        return self.metric.device

    @property
    def dummy_param(self):
        return self.metric.dummy_param

    def to(self, device):
        self.metric.to(device)
        return self


class FamilyLoaderBase:
    """Loader with a variant combo and an optional custom-checkpoint combo.

    Subclass attributes:
        FAMILY_KEY        cache family key
        MODEL_TYPE        ComfyUI type string emitted, e.g. "PYIQA_MUSIQ"
        VARIANT_MAP       {dropdown label: pyiqa registry metric name}
        METRIC_TOOLTIP    tooltip for the variant combo
        CUSTOM_PATTERNS   globs for custom checkpoints; () disables customs
        DEFAULT_WEIGHTS   arch module name (or tuple of names) whose default
                          URL-table basenames are excluded from customs
        METRIC_INPUT_NAME combo input name ("metric"; older families shipped
                          "model_name" - keep it there to not break saved
                          workflows)
        CUSTOM_BASE_INPUT_NAME   likewise for the custom-base combo
        CUSTOM_BASE_OPTIONS      combo options for the custom-base input
                          (None = all VARIANT_MAP keys); families whose custom
                          checkpoints load onto only some heads restrict it
    """

    CATEGORY = "pyiqa"
    MODEL_TYPE = "PYIQA_MODEL"
    VARIANT_MAP = {}
    METRIC_TOOLTIP = ""
    CUSTOM_PATTERNS = ()
    # extra default-weight basenames to hide from the custom scan, for arches
    # whose defaults cannot be derived from a default_model_urls table
    CUSTOM_WEIGHT_EXCLUDES = frozenset()
    DEFAULT_WEIGHTS = None
    METRIC_INPUT_NAME = "metric"
    CUSTOM_BASE_INPUT_NAME = "custom_base_metric"
    CUSTOM_BASE_OPTIONS = None
    DESCRIPTION = ""
    # Set True on families whose model is large enough that loading it after a
    # generation can OOM: the loader then asks ComfyUI to evict its cached
    # models (SDXL checkpoints) before allocating, matching ComfyUI's own
    # "make room for the next model" behaviour.
    FREE_COMFY_MEMORY = False

    RETURN_TYPES = ("PYIQA_MODEL",)  # replaced from MODEL_TYPE per subclass
    RETURN_NAMES = ("model",)
    OUTPUT_TOOLTIPS = (
        "The loaded metric. Wire it into the matching score node of this family.",
    )
    FUNCTION = "load"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "MODEL_TYPE" in cls.__dict__:
            cls.RETURN_TYPES = (cls.MODEL_TYPE,)

    @classmethod
    def _default_weight_basenames(cls):
        if cls.DEFAULT_WEIGHTS is None:
            return set()
        modules = cls.DEFAULT_WEIGHTS
        if isinstance(modules, str):
            modules = (modules,)
        names = set()
        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
                urls = getattr(module, "default_model_urls", None)
                if callable(urls):  # some arches expose a callable table
                    urls = urls()
                names.update(
                    os.path.basename(str(url).split("?")[0])
                    for url in (urls or {}).values()
                )
            except Exception:
                continue
        return names

    @classmethod
    def _custom_exclude_paths(cls):
        """Paths to drop from the custom scan (e.g. the resolved default
        checkpoint of a family that ships it as a plain models/pyiqa file)."""
        return ()

    @classmethod
    def _custom_options(cls):
        options = []
        for pattern in cls.CUSTOM_PATTERNS:
            options.extend(
                list_pyiqa_checkpoints(
                    pattern,
                    exclude_names=(
                        cls._default_weight_basenames() | set(cls.CUSTOM_WEIGHT_EXCLUDES)
                    ),
                    exclude_paths=cls._custom_exclude_paths(),
                )
            )
        return options

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            cls.METRIC_INPUT_NAME: (
                list(cls.VARIANT_MAP),
                {
                    "default": next(iter(cls.VARIANT_MAP)),
                    "tooltip": cls.METRIC_TOOLTIP,
                },
            )
        }
        if cls.CUSTOM_PATTERNS:
            required["custom_checkpoint"] = (
                ["(none)"] + [label for label, _ in cls._custom_options()],
                {
                    "default": "(none)",
                    "tooltip": (
                        "Custom .pth from models/pyiqa matching this family's name "
                        "patterns. When selected it loads onto the base variant below."
                    ),
                },
            )
            required[cls.CUSTOM_BASE_INPUT_NAME] = (
                cls.CUSTOM_BASE_OPTIONS or list(cls.VARIANT_MAP),
                {
                    "default": (
                        cls.CUSTOM_BASE_OPTIONS[0]
                        if cls.CUSTOM_BASE_OPTIONS
                        else next(iter(cls.VARIANT_MAP))
                    ),
                    "tooltip": (
                        "Which architecture a custom checkpoint loads onto; a file name "
                        "cannot say which variant it was trained for."
                    ),
                },
            )
        return {"required": required}

    def load(
        self,
        metric=None,
        custom_checkpoint="(none)",
        custom_base_metric=None,
        **kwargs,
    ):
        # ComfyUI passes inputs by name; older families renamed the combos
        # ("model_name" / "custom_base_variant") - accept both spellings while
        # keeping positional calls (tests, direct use) working
        if metric is None:
            metric = kwargs.get(self.METRIC_INPUT_NAME)
        custom_base = (
            custom_base_metric
            if custom_base_metric is not None
            else kwargs.get(self.CUSTOM_BASE_INPUT_NAME)
        )
        if metric is None:
            raise TypeError(f"missing required input {self.METRIC_INPUT_NAME}")
        device = default_device()
        if custom_checkpoint != "(none)":
            base = self.VARIANT_MAP[custom_base or metric]
            path = dict(self._custom_options())[custom_checkpoint]
            key = (path, base, str(device))
            label = custom_checkpoint
        else:
            base = self.VARIANT_MAP[metric]
            path = None
            key = (base, str(device))
            label = metric

        def factory():
            if self.FREE_COMFY_MEMORY:
                make_room_for_metric(device)
            extra = {"pretrained_model_path": path} if path else {}
            return MetricBundle(
                label, pyiqa.create_metric(base, device=device, **extra)
            )

        bundle = get_cached_metric(self.FAMILY_KEY, key, factory)
        ensure_on_device(bundle, device)
        return (bundle,)


class NRScoreBase(LazyScoreMixin):
    """No-reference score node: one image in, per-image score out."""

    CATEGORY = "pyiqa"
    MODEL_TYPE = "PYIQA_MODEL"
    DESCRIPTION = ""
    MODE_CHECK = "NR"
    LAZY_INPUTS = ("image", "model")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (cls.MODEL_TYPE, dict(LAZY_MODEL_OPT)),
                "image": (
                    "IMAGE",
                    lazy_image_opt(
                        "Image(s) to score; batch is scored per-image. Lazy: the "
                        "generator/decoder feeding this runs before the metric "
                        "model is loaded."
                    ),
                ),
                "memory_management": MEMORY_MANAGEMENT_INPUT,
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("score",)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_TOOLTIPS = (
        "Per-image quality score for each input image. Direction and range "
        "follow the loaded metric (see this node's description) - e.g. a "
        "CLIP-IQA/MUSIQ score is higher-better, a lower-better metric is "
        "flagged as such.",
    )
    FUNCTION = "score"

    @with_memory_management
    def score(self, model=None, image=None, **kwargs):
        if self._skipped(model, image):
            return ([],)
        if getattr(model, "mode", None) not in (None, self.MODE_CHECK):
            raise ValueError(
                f"Model '{getattr(model, 'label', '?')}' is a "
                f"{model.mode}-reference model; this node scores "
                f"{self.MODE_CHECK}-reference models only."
            )
        dev = ensure_on_device(model, default_device())
        x = comfy_image_to_pyiqa(image, dev)
        out = model.metric(x)
        return ([float(v) for v in out.flatten().tolist()],)


class FRScoreBase(LazyScoreMixin):
    """Full-reference score node: target + reference in, per-pair score out."""

    CATEGORY = "pyiqa"
    MODEL_TYPE = "PYIQA_MODEL"
    DESCRIPTION = ""
    MODE_CHECK = "FR"
    ORDER_HINT = ORDER_SYMMETRIC
    LAZY_INPUTS = ("target", "reference", "model")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (cls.MODEL_TYPE, dict(LAZY_MODEL_OPT)),
                "target": ("IMAGE", lazy_image_opt(TARGET_TOOLTIP)),
                "reference": (
                    "IMAGE",
                    lazy_image_opt(reference_tooltip(cls.ORDER_HINT)),
                ),
                "memory_management": MEMORY_MANAGEMENT_INPUT,
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("score",)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_TOOLTIPS = (
        "Per-pair score: how close the target image is to its reference. The "
        "target is the produced/distorted image being judged, the reference is "
        "its pristine original; one score per pair. Higher- or lower-better "
        "depends on the loaded metric (see this node's description).",
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
        if getattr(model, "mode", None) not in (None, self.MODE_CHECK):
            raise ValueError(
                f"Model '{getattr(model, 'label', '?')}' is a "
                f"{model.mode}-reference model; this node scores "
                f"{self.MODE_CHECK}-reference models only."
            )
        dev = ensure_on_device(model, default_device())
        x = comfy_image_to_pyiqa(target, dev)
        y = comfy_image_to_pyiqa(reference, dev)
        out = model.metric(x, y)
        return ([float(v) for v in out.flatten().tolist()],)
