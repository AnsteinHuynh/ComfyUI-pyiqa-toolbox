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
auto-download when absent.
"""

import logging
import os
import warnings

import pyiqa
from pyiqa.archs.fgresq_arch import default_model_urls, load_checkpoint
from pyiqa.utils.download_util import load_file_from_url

from ..utility.pyiqa_utils import (
    ORDER_ASYMMETRIC,
    TARGET_TOOLTIP,
    comfy_image_to_pyiqa,
    default_device,
    ensure_on_device,
    find_pyiqa_model,
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


def _fix_fgresq_transformers5_weights(net, main_path=None, deg_path=None):
    if hasattr(net.clip_model, "vision_model"):
        return  # transformers <= 5.5.4 layout: the checkpoint loads as-is

    main_src = main_path or load_file_from_url(default_model_urls["fgresq"])
    main_sd = _strip_vision_model_prefix(load_checkpoint(main_src))
    missing, unexpected = net.load_state_dict(main_sd, strict=False)
    assert not unexpected, f"FGResQ main weights still mismatch: {unexpected[:5]}"
    leftover = [k for k in missing if k not in ("default_mean", "default_std")]
    assert not leftover, f"FGResQ main weights still missing: {leftover[:5]}"

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
    assert not missing and not leftover_unexpected, (
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

    def to(self, device):
        # the two wrappers share one net; .to on both moves it once plus each
        # wrapper's dummy param
        self.nr.to(device)
        self.pair.to(device)
        return self


def _default_weight_basenames():
    """Basenames of pyiqa's OWN default FGResQ weights (FGResQ.pth and
    FGResQ_Degradation.pth) — never customs, so drop them from dropdowns."""
    return {os.path.basename(url) for url in default_model_urls.values()}


def _custom_options():
    """Custom FGResQ*.pth fine-tunes: dropdown label -> path.

    Excludes pyiqa's own default/degradation weights (by basename) and the
    currently-resolved default main checkpoint (by path), so the family's own
    files never masquerade as customs.
    """
    default_path = find_pyiqa_model("FGResQ.pth")
    return list_pyiqa_checkpoints(
        "FGResQ*.pth",
        exclude_names=_default_weight_basenames(),
        exclude_paths=[default_path] if default_path is not None else (),
    )


class FGResQLoader:
    """Loads FGResQ once; feeds both the NR and the pair score nodes."""

    CATEGORY = "pyiqa/FGResQ"
    DESCRIPTION = (
        "Loads the FGResQ model once (single arch shared by the NR and pair "
        "modes). Feeds FGResQ Score (NR) and FGResQ Pair Score (FR)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        customs = _custom_options()
        return {
            "required": {
                "checkpoint": (
                    ["(default)"] + [label for label, _ in customs],
                    {
                        "default": "(default)",
                        "tooltip": (
                            "Main FGResQ weights. '(default)' resolves FGResQ.pth from the "
                            "pyiqa folders / pyiqa's cache, else downloads the published "
                            "checkpoint. A custom .pth named FGResQ*.pth in models/pyiqa "
                            "(e.g. a fine-tune trained on a specific dataset) loads instead; "
                            "the degradation branch stays on the default weights."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("PYIQA_FGRESQ",)
    RETURN_NAMES = ("model",)
    OUTPUT_TOOLTIPS = (
        "The loaded FGResQ model (one arch shared by both modes). Wire it into "
        "FGResQ NR Score and/or FGResQ FR Score.",
    )
    FUNCTION = "load"

    def load(self, checkpoint):
        device = default_device()
        customs = dict(_custom_options())
        main_path = customs[checkpoint] if checkpoint != "(default)" else find_pyiqa_model("FGResQ.pth")
        deg_path = find_pyiqa_model("FGResQ_Degradation.pth")

        def factory():
            kwargs = {}
            if main_path is not None:
                kwargs["pretrained_model_path"] = main_path
            if deg_path is not None:
                kwargs["degradation_model_path"] = deg_path
            if kwargs:
                kwargs["pretrained"] = False  # paths above replace the url load
            # pyiqa's own strict=False load warns about the vision_model key
            # mismatch on transformers>=5.6; _fix... reloads with remapped keys
            # right after and logs its own confirmation, so hide pyiqa's warning.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="FGResQ .* loaded with", category=RuntimeWarning
                )
                pair = pyiqa.create_metric("fgresq_pair", device=device, **kwargs)
                _fix_fgresq_transformers5_weights(pair.net, main_path, deg_path)
                # A second net object would just be discarded for the shared
                # arch below; skip its weight loading entirely.
                nr = pyiqa.create_metric("fgresq", device=device, pretrained=False)
            nr.net = pair.net
            return FGResQBundle(nr, pair)

        metric = get_cached_metric(
            "fgresq", (str(device), main_path, deg_path), factory
        )
        ensure_on_device(metric, device)
        return (metric,)


class FGResQScore(LazyScoreMixin):
    """No-reference FGResQ quality score. Higher is better, range 0-1.

    ``score`` is a per-image list (OUTPUT_IS_LIST): each image's score flows on
    as its own value downstream, so scalar consumers run once per image.
    """

    CATEGORY = "pyiqa/FGResQ"
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
                "model": ("PYIQA_FGRESQ", dict(LAZY_MODEL_OPT)),
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
        dev = ensure_on_device(model, default_device())
        x = comfy_image_to_pyiqa(image, dev)
        out = model.nr(x)
        return ([float(v) for v in out],)


class FGResQPairScore(LazyScoreMixin):
    """Pairwise FGResQ comparison (full-reference).

    image1 = x0 (e.g. the distorted/restored candidate), image2 = x1 (the
    reference), following pyiqa's (distorted, reference) convention; images are
    paired by batch index. All outputs are per-pair lists (OUTPUT_IS_LIST), so
    a batch of N image pairs yields N of each output downstream.
    """

    CATEGORY = "pyiqa/FGResQ"
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
                "model": ("PYIQA_FGRESQ", dict(LAZY_MODEL_OPT)),
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
        if target.shape[0] != reference.shape[0]:
            raise ValueError(
                f"target and reference must have the same batch size, got "
                f"{target.shape[0]} and {reference.shape[0]}"
            )
        dev = ensure_on_device(model, default_device())
        x0 = comfy_image_to_pyiqa(target, dev)
        x1 = comfy_image_to_pyiqa(reference, dev)
        quality0, quality1, rank, rank_prob = model.pair(x0, x1)

        ranks = [int(r) for r in rank.flatten().tolist()]
        return (
            [float(v) for v in quality0.flatten().tolist()],
            [float(v) for v in quality1.flatten().tolist()],
            ranks,
            [RANK_VERDICTS[r] for r in ranks],
            [float(v) for v in rank_prob.flatten().tolist()],
        )
