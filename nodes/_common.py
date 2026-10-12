"""Shared building blocks for the pack's loader/score nodes.

Most of the node surface is one generic loader plus generic FR/NR score nodes;
this module holds their mechanics, and the few specialised families (FGResQ,
Q-ReAlign, FID) reuse the same bases so they stay short and consistent.

Design notes:
- A variant is a *variant of an architecture*: a registry entry or a custom
  checkpoint (see ``utility/metric_catalog.py``). The loader exposes the
  architecture and the variant; a custom checkpoint is simply one of the
  variant's entries, labeled with the registry variant it loads onto.
- Loader output is a ``MetricBundle`` wrapping the ``pyiqa.create_metric``
  object, so the metric cache, eviction, device restore after Unload Models and
  the unload hook all apply unchanged. It carries the resolved variant, which
  is what lets a score node say *which* model it was handed and refuse the
  wrong family by name.
- Score nodes are ``OUTPUT_IS_LIST``: per-image (NR) or per-pair (FR) scores.
  They adapt the input to the model's own requirements (``Resolved.policy``,
  single-image arches) instead of asking the user to know them.
- ``MODE_CHECK``/``FAMILY`` make a score node refuse a model it cannot score
  with a message naming the node that would have worked, rather than failing
  with a shape or attribute error. ``SCORE_NODES`` is filled in from the
  registered classes, so those messages cannot name a node that is gone.
"""

import functools
import inspect
import logging

import pyiqa

from ..utility import metric_catalog
from ..utility.pyiqa_utils import (
    ORDER_SYMMETRIC,
    TARGET_TOOLTIP,
    comfy_image_to_pyiqa,
    default_device,
    drop_metric,
    ensure_on_device,
    load_state_dict,
    prepare_image,
    prepare_pair,
    reference_tooltip,
    revive_metric,
)

logger = logging.getLogger(__name__)

# The model socket type. Every loader emits a `PYIQA_MODEL`, and the generic
# "(Down)load pyiqa Model" emits ANY_MODEL_TYPE; family score nodes accept the union
# of the two (see ModelValidationMixin._model_input_type).
ANY_MODEL_TYPE = "PYIQA_ANY"

# (family, mode) -> display label of the score node that handles it, filled in
# by ``ModelValidationMixin.__init_subclass__`` from the nodes that are really
# registered. Refusal messages go through ``score_node_for`` so they can only
# ever name a node that exists - the generic nodes register under "any" and are
# the fallback.
SCORE_NODES = {}


def score_node_for(family, mode):
    """Display label of the node that scores ``family`` models in ``mode``."""
    return SCORE_NODES.get((family or "any", mode)) or SCORE_NODES.get(("any", mode))

# Memory handling for the metric after scoring. Two modes, one law each:
MEMORY_KEEP = "keep loaded"
MEMORY_DROP = "unload model (VRAM + RAM)"

MEMORY_MANAGEMENT_INPUT = (
    [MEMORY_DROP, MEMORY_KEEP],
    {
        "default": MEMORY_DROP,
        "tooltip": (
            "What happens to this metric after it scores. "
            "'unload model (VRAM + RAM)': the metric is fully unloaded - "
            "gone from VRAM and RAM, only the file on disk remains; the next "
            "score reloads it (seconds for small models, ~30s for Q-ReAlign "
            "Pro). Re-queueing just works: a previously-unloaded model handed "
            "back by ComfyUI's output cache is rebuilt transparently. "
            "'keep loaded': stays on the GPU, fastest to re-score, holds its "
            "VRAM between runs. With several score nodes sharing one loader, "
            "unload only on the LAST one - earlier nodes should 'keep loaded' "
            "or you pay a reload between them."
        ),
    },
)


def with_memory_management(fn):
    """Apply the node's ``memory_management`` choice around a score.

    A decorator (rather than an edit inside every score body) so the behavior
    is uniform across all score nodes. Before the score: a fully-unloaded
    (gutted) model handed over by ComfyUI's output cache is revived through
    its cached recipe. After the score: unload or nothing, per the mode.
    INPUT_IS_LIST nodes (FID) receive their model and widgets list-wrapped;
    both are unwrapped here before use.
    """

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        bound = inspect.signature(fn).bind(self, *args, **kwargs)
        extra = bound.arguments.get("kwargs", {})
        model = bound.arguments.get("model")
        if isinstance(model, list):  # INPUT_IS_LIST node: model arrives wrapped
            model = model[0] if model else None
            bound.arguments["model"] = model
        if model is not None:
            revived = revive_metric(model)
            if revived is not model:
                bound.arguments["model"] = revived
        result = fn(*bound.args, **bound.kwargs)
        mode = bound.arguments.get("memory_management") or extra.get(
            "memory_management", MEMORY_DROP
        )
        if isinstance(mode, list):  # INPUT_IS_LIST node: widget arrives wrapped
            mode = mode[0] if mode else MEMORY_DROP
        acted = bound.arguments.get("model") or model
        if acted is not None and mode != MEMORY_KEEP:
            drop_metric(acted)
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
    """A loaded pyiqa metric plus the variant it was resolved from."""

    def __init__(self, resolved, metric, family=""):
        self.resolved = resolved
        self.metric = metric
        self.family = family or metric_catalog.family_of(resolved.arch)
        self.order_hint = None

    @property
    def label(self):
        return self.resolved.label

    @property
    def mode(self):
        return self.resolved.mode

    @property
    def model_name(self):
        """The pyiqa registry name the metric was created from."""
        return self.resolved.base

    @property
    def device(self):
        return self.metric.device

    @property
    def dummy_param(self):
        return self.metric.dummy_param

    def to(self, device):
        self.metric.to(device)
        return self

    def for_mode(self, mode):
        """The object that scores ``mode``.

        Normally the wrapped metric itself. A family whose ONE architecture
        serves several modes (FGResQ: an NR metric and a pair metric over the
        same net) hands back the right one, instead of the pair default every
        caller would otherwise get.
        """
        resolver = getattr(self.metric, "for_mode", None)
        return resolver(mode) if callable(resolver) else self.metric

    def __call__(self, *args, **kwargs):
        """Let older score-node bodies call the bundle as the metric itself."""
        return self.metric(*args, **kwargs)

    def __getattr__(self, name):
        """Forward family extras to the wrapped object (e.g. FGResQ's ``.pair``).

        The generic loader wraps whatever an architecture's builder returns, so
        a score node that needs a family-specific handle keeps working no matter
        which loader produced the model.
        """
        if name.startswith("_") or name in ("metric", "resolved", "family"):
            raise AttributeError(name)
        return getattr(self.metric, name)


def metric_for_mode(model, mode):
    """The callable that scores ``mode`` from a loaded model.

    Loaders hand score nodes a :class:`MetricBundle`; one architecture can serve
    several modes (FGResQ), so the bundle is asked which metric to use. Anything
    else (a raw pyiqa metric) is used as-is, as before.
    """
    resolver = getattr(model, "for_mode", None)
    return resolver(mode) if callable(resolver) else getattr(model, "metric", model)


# Architecture label -> builder that owns how that architecture is created.
# Families register here when pyiqa's plain ``create_metric`` is not the whole
# story: Q-ReAlign (dtype, int8_convrot, snapshot directories) and FGResQ (main
# plus degradation weights, and the transformers>=5.6 key remap). Resolved
# through a hook rather than reimplemented so "(Down)load pyiqa Model" gets the exact
# same model as the family's own loader.
_ARCH_BUILDERS = {}


def register_arch_builder(arch_label, builder):
    """builder(resolved, device) -> metric object (see MetricBundle)."""
    _ARCH_BUILDERS[arch_label] = builder


# Architecture label -> pre-load gate. Run by ``run_arch_gate`` before anything
# is created, so an architecture with a hard environment or capacity requirement
# is refused (or prepared for) in one place instead of in every loader: Q-Align's
# quantized variants silently fall back to a 15 GiB fp16 load when bitsandbytes
# is unusable (see nodes/arch_gates.py). Every metric built through
# ``build_metric`` - i.e. everything "(Down)load pyiqa Model" reaches, Q-Align among
# them - passes the gate; a bespoke loader that builds its own architecture
# outside this path would have to call ``run_arch_gate`` itself.
_ARCH_PRELOAD = {}


def register_arch_preload(arch_label, hook):
    """hook(resolved, device) -> None, called before the metric is built.

    The hook validates and may prepare (free ComfyUI's memory); it never builds
    anything, so construction stays on one code path (see ``build_metric``).
    Raise from it to refuse the load with the fix in the message.
    """
    _ARCH_PRELOAD[arch_label] = hook


def run_arch_gate(resolved, device):
    """Run the pre-load gate of ``resolved``'s architecture, if it has one.

    The single place gates run, so "was the gate applied?" has one answer.
    """
    gate = _ARCH_PRELOAD.get(resolved.arch)
    if gate is not None:
        gate(resolved, device)


def build_metric(resolved, device):
    """Create the pyiqa metric for a resolved variant."""
    run_arch_gate(resolved, device)
    builder = _ARCH_BUILDERS.get(resolved.arch)
    if builder is not None:
        return builder(resolved, device)
    if resolved.kind == "safetensors":
        return _build_from_safetensors(resolved, device)
    extra = {"pretrained_model_path": resolved.path} if resolved.path else {}
    try:
        return pyiqa.create_metric(resolved.base, device=device, **extra)
    except Exception as exc:
        raise RuntimeError(_custom_load_hint(resolved, exc)) from exc


def _build_from_safetensors(resolved, device):
    """Load a .safetensors custom onto its base architecture.

    pyiqa's ``pretrained_model_path`` is torch.load-only, so a safetensors
    checkpoint has to be read here. The base is built first (its own default
    weights come from the pyiqa cache, and a fine-tune of a base is normally
    used by someone who already has it), then the file's tensors are loaded
    over it. ``strict=False`` mirrors the leniency of pyiqa's own per-arch
    loaders, with the key overlap checked afterwards: a file whose keys match
    nothing is a wrong base or an HF-format checkpoint, and says so.
    """
    from pyiqa.archs.arch_util import clean_state_dict

    metric = pyiqa.create_metric(resolved.base, device=device)
    state = load_state_dict(resolved.path)
    if isinstance(state, dict) and isinstance(state.get("params"), dict):
        state = state["params"]  # pyiqa's own weight_keys='params' wrapper
    net = getattr(metric, "net", metric)
    missing, unexpected = net.load_state_dict(clean_state_dict(state), strict=False)
    matched = len(state) - len(unexpected)
    if matched <= 0:
        raise RuntimeError(
            f"'{resolved.display}' does not fit {resolved.base} "
            f"({resolved.arch}): none of its {len(state)} tensors match this "
            f"architecture. Check that the file is a {resolved.arch} "
            f"checkpoint for '{resolved.base}', not a HuggingFace-format "
            "model - and if it is a fine-tune of a different variant, pick "
            "that variant (or name the file after it) so the right "
            "architecture is built."
        )
    if missing or unexpected:
        logger.info(
            "(pyiqa-toolbox) %s loaded onto %s: %d/%d tensors used "
            "(%d architecture keys left at their defaults, %d unused)",
            resolved.display,
            resolved.base,
            matched,
            len(state),
            len(missing),
            len(unexpected),
        )
    return metric


def _custom_load_hint(resolved, exc):
    """Turn a per-arch checkpoint failure into something actionable."""
    if not resolved.path:
        return f"could not create pyiqa metric '{resolved.base}': {exc}"
    return (
        f"could not load '{resolved.display}' as {resolved.base} "
        f"({resolved.arch}): {exc}\n"
        "pyiqa's checkpoint formats are per-architecture (most read a plain "
        "state dict, some expect a {'params': ...} wrapper, A-FINE expects its "
        "own multi-part dict, the classic NR metrics read .mat files), so a "
        "file that is really a different architecture or variant fails here. "
        "Pick the variant this checkpoint was trained for, or convert it to "
        "the format that architecture expects."
    )


class ModelLoaderBase:
    """Shared contract of the pack's loader nodes.

    A loader emits ``(model, model_name, variant_label)`` and caches its metric
    under ``FAMILY_KEY`` (metrics sharing a key evict each other). Subclasses
    bring their own combos: the pack has one generic loader plus the few
    families whose weights need more than a variant name, so there is no single
    combo shape worth sharing.

    Subclass attributes:
        MODEL_TYPE   ComfyUI type of the model output socket (also sets
                     RETURN_TYPES)
        FAMILY       catalog family key; "any"/"" for the generic loader
        FAMILY_KEY   cache key
        DESCRIPTION  node description
    """

    CATEGORY = "pyiqa"
    MODEL_TYPE = "PYIQA_MODEL"
    FAMILY = ""
    FAMILY_KEY = ""
    DESCRIPTION = ""

    RETURN_TYPES = ("PYIQA_MODEL", "STRING", "STRING")
    RETURN_NAMES = ("model", "model_name", "variant_label")
    OUTPUT_TOOLTIPS = (
        "The loaded metric. Wire it into pyiqa FR Score / pyiqa NR Score, or "
        "into the matching family's score node, which re-checks that the model "
        "is one it can score.",
        "The pyiqa registry name the metric was created from, e.g. 'topiq_fr' "
        "- stable, so it is what you want for filenames, notes or logging.",
        "The variant as shown in the dropdown, e.g. 'NR · SPAQ (ResNet50) - "
        "NR, higher better, ~0-~1'.",
    )
    FUNCTION = "load"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "MODEL_TYPE" in cls.__dict__:
            cls.RETURN_TYPES = (cls.MODEL_TYPE, "STRING", "STRING")

    @property
    def _family_key(self):
        return self.FAMILY_KEY or self.FAMILY or self.MODEL_TYPE


class ModelValidationMixin:
    """Refuse a model this node cannot score, naming what would work."""

    MODE_CHECK = None
    FAMILY = ""
    SCORE_LABEL = "This score node"
    # Nodes that DO consume a multi-value metric (FGResQ's pair API returns
    # quality0, quality1, rank, rank_prob) set this; everyone else refuses such a
    # model instead of crashing on a 4-tuple.
    HANDLES_MULTI_OUTPUT = False
    # Kinds from metric_catalog.SPECIAL_INPUTS this node can score. Empty by
    # default: a metric that needs two image sets, or one set, is not a per-image
    # metric and no generic score node can run it. CLIP Score sets ("caption",)
    # because a caption is exactly the extra input its own node provides.
    HANDLES_SPECIAL_INPUTS = ()
    # Family nodes whose ONE architecture serves several pyiqa metrics over a
    # shared net (FGResQ: an NR metric and a pair metric, same weights - the
    # probe-verified ``bundle.nr.net is bundle.pair.net``) set this. The bundle
    # resolves the metric for the mode the node asks for (``MetricBundle
    # .for_mode``), so which of the architecture's variants named the net
    # constrains neither the mode nor the output shape: the node gets the metric
    # it asks for either way, and refusing one of two interchangeable entries
    # would be an arbitrary rule. The generic FR/NR score nodes leave it off on
    # purpose - there the entry's mode and shape ARE the guard (an "fgresq"
    # bundle must not be scored by "pyiqa FR Score", whose contract is one value
    # per pair).
    METRIC_FROM_BUNDLE = False

    def __init_subclass__(cls, **kwargs):
        """Register this node so refusal messages can name a node that exists.

        Filled automatically from the classes that are really here: a message
        can only ever suggest ``pyiqa FR Score`` or ``FGResQ NR Score`` while
        that node is registered, so removing a node can never leave a stale
        suggestion pointing at it.
        """
        super().__init_subclass__(**kwargs)
        label = cls.__dict__.get("SCORE_LABEL")
        if label and cls.MODE_CHECK in ("FR", "NR"):
            SCORE_NODES[(cls.FAMILY or "any", cls.MODE_CHECK)] = label

    @classmethod
    def _model_input_type(cls):
        """The model socket type: the family's own type, plus the generic one.

        ComfyUI accepts comma-separated unions on both sides (its front end
        splits them, and ``comfy_execution.validation`` compares the sets), so
        "PYIQA_LPIPS,PYIQA_ANY" lets "(Down)load pyiqa Model" feed this node while a
        *wrong* dedicated loader is still refused while you are wiring, not at
        queue time. Anything the generic loader feeds in is re-checked at run
        time below, which is where the message you actually read comes from.
        """
        if cls.MODEL_TYPE == ANY_MODEL_TYPE:
            return ANY_MODEL_TYPE
        return f"{cls.MODEL_TYPE},{ANY_MODEL_TYPE}"

    def _require_compatible(self, model):
        resolved = getattr(model, "resolved", None)
        label = getattr(resolved, "label", None) or getattr(model, "label", "?")
        mode = getattr(model, "mode", None)
        if resolved is not None:
            # set-level and caption metrics first: pyiqa calls them all "NR"
            # because none takes a reference image, so a per-image score node
            # would otherwise happily try to score a FID model
            special = metric_catalog.special_input(resolved.base)
            if special is not None and special[0] not in self.HANDLES_SPECIAL_INPUTS:
                raise ValueError(
                    f"Model '{label}' is not scored per image: {special[1]}."
                )
        # the bundle carries the model's own family (the generic loader sets it
        # from the chosen architecture; "any" when no family owns that
        # architecture), so a family node can tell whether it can score this
        if resolved is not None:
            family = getattr(model, "family", None) or metric_catalog.family_of(
                resolved.arch
            )
        else:
            family = getattr(model, "family", None) or "any"
        if (
            mode in ("FR", "NR")
            and mode != self.MODE_CHECK
            and not self.METRIC_FROM_BUNDLE
        ):
            # mode is "?" when pyiqa's metric table could not be read; there is
            # nothing to check against then, so the node runs and says nothing
            # rather than refusing every model (see metric_catalog
            # _fallback_architectures)
            target = score_node_for(family, mode)
            if target:
                hint = f" Use {target} for it."
            else:
                article = "an" if mode[:1] in "AEIOU" else "a"
                hint = f" Load {article} {mode}-reference variant instead."
            raise ValueError(
                f"Model '{label}' is a {mode}-reference model; this node scores "
                f"{self.MODE_CHECK}-reference models only.{hint}"
            )
        if resolved is None:
            return
        accepted = (self.FAMILY,) if self.FAMILY else ()
        if accepted and family not in accepted:
            wanted = " / ".join(
                metric_catalog.FAMILY_NAMES.get(f, f) for f in accepted
            )
            got = metric_catalog.FAMILY_NAMES.get(family, family)
            # name the node that can score THIS model, which is decided by the
            # model's own mode when pyiqa's table knows it. Using this node's
            # mode instead pointed an FR model at "pyiqa NR Score" from the
            # FGResQ NR node - the one node that certainly cannot score it.
            model_mode = mode if mode in ("FR", "NR") else self.MODE_CHECK
            target = score_node_for(family, model_mode)
            if target:
                hint = f"Use {target} for it."
            else:
                hint = (
                    f"'{got}' has no {model_mode}-reference node in this "
                    f"pack - use pyiqa {model_mode} Score."
                )
            raise ValueError(
                f"{self.SCORE_LABEL} scores {wanted} models only, but "
                f"'{resolved.base}' is {resolved.arch} ({got}) - {label}. {hint}"
            )
        if (
            resolved.multi_output
            and not self.HANDLES_MULTI_OUTPUT
            and not self.METRIC_FROM_BUNDLE
        ):
            raise ValueError(
                f"{self.SCORE_LABEL} scores one value per image, and "
                f"'{resolved.base}' is not that: "
                f"{metric_catalog.SPECIAL_METRICS.get(resolved.base, 'it returns several values')}."
            )

    @staticmethod
    def _policy(model):
        resolved = getattr(model, "resolved", None)
        return getattr(resolved, "policy", ()) if resolved else ()

    @staticmethod
    def _single_image(model):
        resolved = getattr(model, "resolved", None)
        return bool(getattr(resolved, "single_image", False)) if resolved else False


class NRScoreBase(ModelValidationMixin, LazyScoreMixin):
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
                "model": (cls._model_input_type(), dict(LAZY_MODEL_OPT)),
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
        self._require_compatible(model)
        dev = ensure_on_device(model, default_device())
        metric = metric_for_mode(model, self.MODE_CHECK)
        x = comfy_image_to_pyiqa(image, dev)
        x = prepare_image(x, self._policy(model))
        if self._single_image(model):
            # this arch asserts batch 1 (it preprocesses per image), so score
            # one image at a time rather than failing on a batch
            return (
                [
                    float(metric(x[i : i + 1]).reshape(-1)[0])
                    for i in range(x.shape[0])
                ],
            )
        out = metric(x)
        return ([float(v) for v in out.flatten().tolist()],)


class FRScoreBase(ModelValidationMixin, LazyScoreMixin):
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
                "model": (cls._model_input_type(), dict(LAZY_MODEL_OPT)),
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
        self._require_compatible(model)
        if target.shape[0] != reference.shape[0]:
            raise ValueError(
                f"target and reference must have the same batch size, got "
                f"{target.shape[0]} and {reference.shape[0]}"
            )
        dev = ensure_on_device(model, default_device())
        metric = metric_for_mode(model, self.MODE_CHECK)
        x = comfy_image_to_pyiqa(target, dev)
        y = comfy_image_to_pyiqa(reference, dev)
        x, y = prepare_pair(x, y, self._policy(model))
        if self._single_image(model):
            # this arch asserts batch 1 (e.g. AFINE's normalize squeezes dim 0),
            # so score one pair at a time rather than failing on a batch
            return (
                [
                    float(metric(x[i : i + 1], y[i : i + 1]).reshape(-1)[0])
                    for i in range(x.shape[0])
                ],
            )
        out = metric(x, y)
        return ([float(v) for v in out.flatten().tolist()],)
