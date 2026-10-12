"""FGResQ nodes (fine-grained IQA, AAAI 2026, Sheng et al.).

One architecture, two pyiqa registry entries with identical weights:
- 'fgresq'      metric_mode NR: one image -> sigmoid quality score in [0, 1]
- 'fgresq_pair' metric_mode FR: two images -> (score0, score1, rank, rank_prob),
  rank encoded 0=image2_better, 1=image1_better, 2=similar_quality.

The loader loads the weights once through the 'fgresq_pair' wrapper and shares
that single arch object into the 'fgresq' NR wrapper (the registry entries have
identical metric_opts, so this is exactly the model pyiqa would build either
way, without a second copy in VRAM).

Weights are looked up in the ComfyUI ``models/pyiqa`` folder first
(``FGResQ.pth`` and ``FGResQ_Degradation.pth``), falling back to pyiqa's
auto-download when absent. Custom fine-tunes live in ``models/pyiqa/FGResQ/``
and appear as variants of this loader.

This module also owns how the FGResQ architecture is *built*, for every node in
the pack (``register_arch_builder``): the main plus degradation weights and the
transformers>=5.6 key remap are this family's business, so "(Down)load pyiqa Model"
with architecture=FGResQ produces the same bundle, usable by both FGResQ score
nodes.
"""

import logging
import warnings

import pyiqa

from ..utility.pyiqa_utils import (
    ORDER_ASYMMETRIC,
    TARGET_TOOLTIP,
    comfy_image_to_pyiqa,
    default_device,
    ensure_on_device,
    find_pyiqa_model,
    get_cached_metric,
    load_state_dict,
    reference_tooltip,
)

from ._common import (
    with_memory_management,
    LAZY_MODEL_OPT,
    LazyScoreMixin,
    MEMORY_MANAGEMENT_INPUT,
    ModelValidationMixin,
    lazy_image_opt,
    metric_for_mode,
    register_arch_builder,
)

logger = logging.getLogger(__name__)

RANK_VERDICTS = {
    0: "reference_better",
    1: "target_better",
    2: "similar",
}


def _strip_vision_model_prefix(state_dict):
    """Drop the transformers-4 CLIPVisionModel ``vision_model.`` key segment.

    The published FGResQ checkpoints were saved with transformers 4, where
    CLIPVisionModel nested everything under ``vision_model.``. transformers
    >= ~5.6 removed that wrapper, so the keys no longer match the module tree
    and pyiqa (as of 0.1.15.post28 / upstream main) only warns — the
    fine-tuned backbone silently fails to load and the model runs on the stock
    HF CLIP. Stripping the segment restores the intended load.
    """
    out = {}
    for k, v in state_dict.items():
        if ".vision_model." in k:
            k = k.replace(".vision_model.", ".", 1)
        elif k.startswith("vision_model."):
            k = k[len("vision_model."):]
        out[k] = v
    return out


def _fix_fgresq_transformers5_weights(net, main_path=None, deg_path=None, main_state=None):
    """Load FGResQ's fine-tuned weights, remapping the transformers-4 key layout.

    ``main_state`` (an already-read state dict) replaces ``main_path`` — that is
    how a ``.safetensors`` custom is loaded, since pyiqa's own checkpoint reader
    is torch.load-only.
    """
    # Imported here rather than at module level: ``pyiqa.archs.fgresq_arch``
    # imports transformers, so a top-level import let one missing/older
    # transformers take down every node in the pack (it registered zero) instead
    # of failing only when an FGResQ model is actually built.
    from pyiqa.archs.fgresq_arch import default_model_urls, load_checkpoint
    from pyiqa.utils.download_util import load_file_from_url

    needs_remap = not hasattr(net.clip_model, "vision_model")  # transformers >= 5.6
    if main_state is None and not needs_remap:
        return  # transformers <= 5.5.4 layout: the arch loaded it as-is

    if main_state is None:
        main_src = main_path or load_file_from_url(default_model_urls["fgresq"])
        main_sd = load_checkpoint(main_src)
    else:
        main_sd = main_state
    if needs_remap:
        main_sd = _strip_vision_model_prefix(main_sd)
    missing, unexpected = net.load_state_dict(main_sd, strict=False)
    if unexpected:
        raise RuntimeError(f"FGResQ main weights still mismatch: {unexpected[:5]}")
    leftover = [k for k in missing if k not in ("default_mean", "default_std")]
    if leftover:
        raise RuntimeError(f"FGResQ main weights still missing: {leftover[:5]}")

    deg_src = deg_path or load_file_from_url(default_model_urls["degradation"])
    deg_sd = _strip_vision_model_prefix(load_checkpoint(deg_src))
    clip_sd = net._extract_vision_state_dict(deg_sd)
    missing, unexpected = net.task_cls_clip.load_state_dict(clip_sd, strict=False)
    # The degradation checkpoint is a full CLIP checkpoint; the text branch and
    # logit_scale are expected to be unused (per pyiqa's own loader docstring),
    # and visual_projection no longer exists on transformers>=5.6's
    # CLIPVisionModel (FGResQ reads the pooler output, not the projection).
    leftover_unexpected = [
        k
        for k in unexpected
        if not (
            k.startswith("text_")
            or k in ("logit_scale", "visual_projection.weight")
        )
    ]
    if missing or leftover_unexpected:
        raise RuntimeError(
            f"FGResQ degradation weights still mismatch: "
            f"missing={missing[:3]} unexpected={leftover_unexpected[:3]}"
        )
    logger.info(
        "(pyiqa-toolbox) FGResQ weights remapped for transformers>=5.6 "
        "(vision_model. prefix stripped); fine-tuned backbone loaded."
    )


class FGResQBundle:
    """The single loaded FGResQ arch wrapped in both pyiqa metric modes."""

    def __init__(self, nr, pair):
        self.nr = nr
        self.pair = pair

    @property
    def device(self):
        return self.nr.device

    @property
    def dummy_param(self):
        # lets ensure_on_device treat the bundle like an InferenceModel
        return self.nr.dummy_param

    @property
    def metric(self):
        # MetricBundle contract: drop_metric unwraps ``.metric`` on wrappers
        # that are not nn.Modules, so unload must find a real module here -
        # the pair wrapper carries the (shared) arch's own parameters
        return self.pair

    def to(self, device):
        # the two wrappers share one net; .to on both moves it once plus each
        # wrapper's dummy param
        self.nr.to(device)
        self.pair.to(device)
        return self

    def for_mode(self, mode):
        """The metric that scores ``mode``: "NR" -> single image, else the pair.

        One net serves both pyiqa metrics, so a score node has to ask for the
        one it needs. Without this the generic loader's NR variant was handed to
        pyiqa NR Score as the pair wrapper, which asserted "Please specify
        reference image for Full Reference metric".
        """
        return self.nr if mode == "NR" else self.pair

    def __call__(self, *args, **kwargs):
        """Behave like a plain pyiqa metric: the pair wrapper.

        Kept for direct/legacy calls. Score nodes go through ``for_mode``, and
        the pair metric needs BOTH images - it is not a single-image fallback.
        """
        return self.pair(*args, **kwargs)


def _build_pair_bundle(main_path, deg_path, device, main_state=None):
    """Build the shared FGResQ arch: both metric modes over ONE net."""
    kwargs = {}
    if main_path is not None:
        kwargs["pretrained_model_path"] = main_path
    if deg_path is not None:
        kwargs["degradation_model_path"] = deg_path
    if kwargs:
        kwargs["pretrained"] = False  # paths above replace the url load
    elif main_state is not None:
        kwargs["pretrained"] = False  # the state dict below replaces the url load
    # pyiqa's own strict=False load warns about the vision_model key
    # mismatch on transformers>=5.6; _fix... reloads with remapped keys
    # right after and logs its own confirmation, so hide pyiqa's warning.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="FGResQ .* loaded with", category=RuntimeWarning
        )
        pair = pyiqa.create_metric("fgresq_pair", device=device, **kwargs)
        _fix_fgresq_transformers5_weights(
            pair.net, main_path, deg_path, main_state=main_state
        )
        # A second net object would just be discarded for the shared
        # arch below; skip its weight loading entirely.
        nr = pyiqa.create_metric("fgresq", device=device, pretrained=False)
    nr.net = pair.net
    return FGResQBundle(nr, pair)


def _build_fgresq_bundle(path, kind, device):
    """Custom weights file (or the default) -> the shared FGResQ bundle.

    ``.safetensors`` is read here because pyiqa's checkpoint reader is
    torch.load-only. The degradation branch always uses its own weights file.
    """
    main_path = None if kind == "safetensors" else (path or find_pyiqa_model("FGResQ.pth"))
    main_state = None
    if kind == "safetensors":
        main_state = load_state_dict(path)
        if isinstance(main_state, dict) and isinstance(main_state.get("params"), dict):
            main_state = main_state["params"]
    deg_path = find_pyiqa_model("FGResQ_Degradation.pth")
    bundle = get_cached_metric(
        "fgresq",
        (str(device), main_path, deg_path, path),
        lambda: _build_pair_bundle(main_path, deg_path, device, main_state),
    )
    ensure_on_device(bundle, device)
    return bundle


def build_fgresq(resolved, device):
    """Build an FGResQ bundle for ANY loader in the pack ("(Down)load pyiqa Model")."""
    return _build_fgresq_bundle(resolved.path, resolved.kind, device)


register_arch_builder("FGResQ", build_fgresq)


class FGResQScore(ModelValidationMixin, LazyScoreMixin):
    """No-reference FGResQ quality score. Higher is better, range 0-1.

    ``score`` is a per-image list (OUTPUT_IS_LIST): each image's score flows on
    as its own value downstream, so scalar consumers run once per image.
    """

    CATEGORY = "pyiqa/FGResQ"
    MODEL_TYPE = "PYIQA_FGRESQ"
    FAMILY = "fgresq"
    SCORE_LABEL = "FGResQ NR Score"
    MODE_CHECK = "NR"
    # one net serves this metric and the pair metric (same weights): the bundle
    # decides which one to run, so either dropdown variant scores here (see
    # _common.METRIC_FROM_BUNDLE)
    METRIC_FROM_BUNDLE = True
    DESCRIPTION = (
        "No-reference FGResQ quality score. 0-1, higher is better. Score is "
        "per-image. Lazy model load: the model is fetched only after the image(s) exist."
    )
    LAZY_INPUTS = ("image", "model")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "memory_management": MEMORY_MANAGEMENT_INPUT,
                "model": (cls._model_input_type(), dict(LAZY_MODEL_OPT)),
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
        "Per-image FGResQ quality score, 0-1, higher is better. "
        "(pyiqa's NR fgresq metric returns a single score per image; the "
        "quality0/quality1 pair only exists in the full-reference pair mode.)",
    )
    FUNCTION = "score"

    @with_memory_management
    def score(self, model=None, image=None, **kwargs):
        if self._skipped(model, image):
            return ([],)
        self._require_compatible(model)
        # the family loader hands over the bundle, "(Down)load pyiqa Model" a
        # MetricBundle wrapping it - both resolve to the single-image metric
        metric = metric_for_mode(model, self.MODE_CHECK)
        dev = ensure_on_device(model, default_device())
        x = comfy_image_to_pyiqa(image, dev)
        out = metric(x)
        return ([float(v) for v in out],)


class FGResQPairScore(ModelValidationMixin, LazyScoreMixin):
    """Pairwise FGResQ comparison (full-reference).

    image1 = x0 (e.g. the distorted/restored candidate), image2 = x1 (the
    reference), following pyiqa's (distorted, reference) convention; images are
    paired by batch index. All outputs are per-pair lists (OUTPUT_IS_LIST), so
    a batch of N image pairs yields N of each output downstream.
    """

    CATEGORY = "pyiqa/FGResQ"
    MODEL_TYPE = "PYIQA_FGRESQ"
    FAMILY = "fgresq"
    SCORE_LABEL = "FGResQ FR Score"
    MODE_CHECK = "FR"
    HANDLES_MULTI_OUTPUT = True  # this is the node for the pair API's 4 values
    # one net serves this metric and the NR metric (same weights): the bundle
    # decides which one to run, so either dropdown variant scores here (see
    # _common.METRIC_FROM_BUNDLE)
    METRIC_FROM_BUNDLE = True
    DESCRIPTION = (
        "Pairwise FGResQ comparison: a quality score for each side plus a rank "
        "verdict per pair. pyiqa's pair API returns (quality0, quality1, rank, "
        "rank_prob) with rank encoded 0=image2_better, 1=image1_better, "
        "2=similar; this node maps x0/image1 to 'target' and x1/image2 to "
        "'reference', so 0=reference_better, 1=target_better. Order-SENSITIVE: "
        "target and reference are not interchangeable - swapping them swaps "
        "the two scores and can change the rank. The model is loaded lazily, "
        "after the image exists."
    )
    LAZY_INPUTS = ("target", "reference", "model")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "memory_management": MEMORY_MANAGEMENT_INPUT,
                "model": (cls._model_input_type(), dict(LAZY_MODEL_OPT)),
                "target": ("IMAGE", lazy_image_opt(TARGET_TOOLTIP)),
                "reference": (
                    "IMAGE",
                    lazy_image_opt(reference_tooltip(ORDER_ASYMMETRIC)),
                ),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "INT", "STRING", "FLOAT")
    RETURN_NAMES = ("target_score", "reference_score", "rank", "verdict", "rank_prob")
    OUTPUT_IS_LIST = (True, True, True, True, True)
    OUTPUT_TOOLTIPS = (
        "FGResQ quality for the TARGET image - the one being judged (the "
        "produced/restored/denoised image, or any distorted version): pyiqa's "
        "quality0, the x0 side. 0-1, higher is better.",
        "FGResQ quality for the REFERENCE image - the pristine original it is "
        "compared against: pyiqa's quality1, the x1 side. 0-1, higher is better.",
        "Pairwise verdict as pyiqa's integer code: 0 = reference is better, "
        "1 = target is better, 2 = similar. (pyiqa names these image2/image1 "
        "because it sees x0=first argument, x1=second.)",
        "The rank as words: 'target_better' (your produced/distorted image won) / "
        "'reference_better' (the pristine original won) / 'similar'.",
        "Confidence of the rank verdict: the softmax probability of the chosen "
        "class (0-1).",
    )
    FUNCTION = "score"

    @with_memory_management
    def score(self, model=None, target=None, reference=None, **kwargs):
        if self._skipped(model, target, reference):
            return ([], [], [], [], [])
        self._require_compatible(model)
        if target.shape[0] != reference.shape[0]:
            raise ValueError(
                f"target and reference must have the same batch size, got "
                f"{target.shape[0]} and {reference.shape[0]}"
            )
        pair = getattr(model, "pair", None) or getattr(model, "metric", None)
        if not callable(pair):
            raise ValueError(
                "FGResQ FR Score needs the FGResQ model: the pair verdict comes "
                "from the arch loaded by (Down)load FGResQ or by Load pyiqa "
                "Model with an FGResQ variant."
            )
        dev = ensure_on_device(model, default_device())
        x0 = comfy_image_to_pyiqa(target, dev)
        x1 = comfy_image_to_pyiqa(reference, dev)
        quality0, quality1, rank, rank_prob = pair(x0, x1)

        ranks = [int(r) for r in rank.flatten().tolist()]
        return (
            [float(v) for v in quality0.flatten().tolist()],
            [float(v) for v in quality1.flatten().tolist()],
            ranks,
            [RANK_VERDICTS[r] for r in ranks],
            [float(v) for v in rank_prob.flatten().tolist()],
        )
