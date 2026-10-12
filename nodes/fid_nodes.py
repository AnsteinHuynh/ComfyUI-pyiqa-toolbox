"""FID / sFID nodes (Fréchet Inception Distance) — pack-side IMAGE-batch path.

pyiqa's own FID API is FOLDER-based: it extracts Inception features over two
folders of image files, then computes the Fréchet distance between the feature
distributions. ComfyUI hands us image tensors, so this node writes the wired
batches to temporary 8-bit PNG folders and then calls pyiqa's own folder API
(``metric(dir_target, dir_reference, mode=...)``) — the same code path pyiqa's
CLI uses, so the value matches a folder-based run over the same images.

Consequences to keep in mind (documented in the tooltips):
- This is a SET-level metric: it returns a single score, not one per image. It
  answers "does my generated set look like that real set", which is why it is
  the wrong tool for comparing ONE picture against ONE picture (use a pairwise
  metric: LPIPS, DISTS, A-FINE FR, SSIM, ... via (Down)load pyiqa Model + pyiqa FR
  Score, or FGResQ FR Score).
- The Fréchet distance needs a feature mean AND covariance per side. One image
  has no covariance at all (pyiqa's own ``sqrtm`` errors out on the degenerate
  matrix), so < 2 per side is refused outright, and below ~16 the covariance has
  rank n-1 of 2048: the number is then dominated by how many images were wired
  rather than by image quality, so it is warned about and should only be
  compared across runs with similar set sizes.
- Images are quantised to 8-bit PNG, exactly like the files pyiqa would read
  from a folder. A float-precision difference vs a raw-tensor implementation is
  expected and is inherent to pyiqa's folder semantics.
"""

import logging
import os
import tempfile

import numpy as np
import pyiqa
from PIL import Image

from ..utility import metric_catalog
from ..utility.pyiqa_utils import make_room_for_metric

from ._common import (
    ANY_MODEL_TYPE,
    register_arch_builder,
    with_memory_management,
    MEMORY_MANAGEMENT_INPUT,
)

logger = logging.getLogger(__name__)

# A Fréchet distance between distributions cannot be computed from one sample:
# the covariance of a single image is degenerate (all-nan after the n-1
# normalisation, rank 0), which pyiqa's sqrtm raises on.
MIN_IMAGES_PER_SIDE = 2
# The sample covariance in InceptionV3's 2048 dimensions has rank n-1, so with
# few images the value mostly measures the sample size. Below this the node still
# scores (the user is in charge) but says so.
MEANINGFUL_IMAGES_PER_SIDE = 16

# pyiqa's FID/sFID are set-level, folder-based metrics: they cannot score an
# IMAGE batch, which is why the catalog marks them as needing a dedicated score
# node. The weights come from "(Down)load pyiqa Model" like every other metric
# (architecture "FID"), and the generic FR/NR score nodes refuse them with a
# pointer here (see metric_catalog.SPECIAL_INPUTS).
DISTANCE_METRICS = ("fid", "sfid", "fid_dinov2")


def build_fid(resolved, device):
    """Build a FID / sFID metric for ``(Down)load pyiqa Model``.

    pyiqa builds the metric outside ``comfy.model_management``, so ComfyUI does
    not know how much VRAM it holds and will not free room for it: scoring right
    after a generation would allocate InceptionV3 against a resident checkpoint.
    Freeing ComfyUI's own cache first is what the old dedicated FID loader did,
    and keeping it in a registered builder means every path that builds FID
    (i.e. the generic loader) still does it.
    """
    make_room_for_metric(device)
    return pyiqa.create_metric(resolved.base, device=device)


register_arch_builder("FID", build_fid)


class FIDScore:
    """Set-level FID / sFID between two image sets. LOWER is better.

    The node is INPUT_IS_LIST (node-level - ComfyUI has no per-input list
    mode), so EVERY input arrives list-wrapped, widgets and model included;
    ``target`` and ``reference`` each accept MULTIPLE image wires (every
    batch is concatenated into one set). Not lazy - InceptionV3 is small and
    loads in seconds, and INPUT_IS_LIST batching does not compose with lazy
    inputs.
    """

    CATEGORY = "pyiqa/Generic"
    FAMILY = "fid"
    MODEL_TYPE = ANY_MODEL_TYPE
    SCORE_LABEL = "FID Score"
    DESCRIPTION = (
        "Fréchet distance between the target and reference image SETS: one "
        "set-level score (not per-image), LOWER is better, answering 'does my "
        "generated set look like that real set'. Load the weights with "
        "(Down)load pyiqa Model -> architecture 'FID' (fid / sFID / fid_dinov2). "
        "Needs 2+ images per side (~16+ to be meaningful) - for ONE image "
        "against ONE image use a pairwise metric instead (LPIPS, DISTS, A-FINE "
        "FR, ... via (Down)load pyiqa Model + pyiqa FR Score). Batches are staged "
        "as temporary PNGs and scored through pyiqa's own folder-based FID."
    )
    # target and reference accept multiple wires; every batch is concatenated
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "memory_management": MEMORY_MANAGEMENT_INPUT,
                "model": (
                    ANY_MODEL_TYPE,
                    {
                        "tooltip": (
                            "From (Down)load pyiqa Model, architecture 'FID' "
                            "(fid, sFID or fid_dinov2). The generic pyiqa score "
                            "nodes refuse these on purpose."
                        )
                    },
                ),
                "target": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Target image SET (the generated/distorted side). "
                            "Accepts MULTIPLE image wires - every wired batch "
                            "joins one set. FID compares a feature mean AND "
                            "covariance, so it needs a set: 2+ images per side is "
                            "the floor, ~16+ the useful range. For a single image "
                            "pair use a pairwise metric instead (LPIPS, DISTS, "
                            "A-FINE FR, ... via (Down)load pyiqa Model + pyiqa FR "
                            "Score)."
                        )
                    },
                ),
                "reference": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Reference image SET (real/ground-truth side). "
                            "Accepts MULTIPLE image wires; the two sets need not "
                            "have equal counts, but each needs >= 2 images (one "
                            "image has no covariance) and ~16+ to be meaningful."
                        )
                    },
                ),
                "resize_mode": (
                    ["clean", "legacy_pytorch", "legacy_tensorflow"],
                    {
                        "default": "clean",
                        "tooltip": (
                            "pyiqa's FID resize mode: 'clean' (clean-fid channel-wise "
                            "PIL bicubic, the CLI default), 'legacy_pytorch' "
                            "(bilinear), 'legacy_tensorflow' (TF-compatible bilinear). "
                            "Keep both sides on the same mode."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("score",)
    OUTPUT_TOOLTIPS = (
        "ONE set-level score for the two image sets (not per image): the "
        "Fréchet distance between the feature distributions of the target set "
        "(your generated/processed images) and the reference set (the real "
        "images). "
        "LOWER is better, unbounded; identical sets give ~0.",
    )
    FUNCTION = "score"

    @with_memory_management
    def score(self, model=None, target=None, reference=None, resize_mode="clean", **kwargs):
        # INPUT_IS_LIST: ComfyUI wraps every input (widgets and model too) in
        # a list; unwrap the singletons, keep target/reference as batch lists
        if isinstance(model, list):
            model = model[0] if model else None
        if isinstance(resize_mode, list):
            resize_mode = resize_mode[0] if resize_mode else "clean"
        if model is None or not target or not reference:
            return ([],)
        # the other direction of the guard the generic score nodes apply: a
        # per-image metric wired in here must say which node it belongs in
        resolved = getattr(model, "resolved", None)
        base = getattr(resolved, "base", "")
        special = metric_catalog.special_input(base)
        if special is None or special[0] != "distance":
            label = getattr(resolved, "label", None) or base or "?"
            raise ValueError(
                f"FID Score needs a set-level distance metric - fid, sFID or "
                f"fid_dinov2 from (Down)load pyiqa Model, architecture 'FID' - "
                f"but the model is '{label}'. Per-image metrics belong in "
                "pyiqa FR Score or pyiqa NR Score."
            )
        # INPUT_IS_LIST: target/reference arrive as lists of IMAGE batches
        target = [t for t in target if t is not None]
        reference = [r for r in reference if r is not None]
        counts = {}
        for name, batches in (("target", target), ("reference", reference)):
            counts[name] = sum(b.shape[0] for b in batches)
            if counts[name] < MIN_IMAGES_PER_SIDE:
                raise ValueError(
                    f"{name} has {counts[name]} image(s) across "
                    f"{len(batches)} wire(s). FID compares two DISTRIBUTIONS: it "
                    "needs a feature mean and covariance per side, and one image "
                    "has no covariance at all (pyiqa's own sqrt errors out on "
                    "it). For a single image pair use a pairwise metric instead - "
                    "LPIPS, DISTS, A-FINE FR, SSIM, ... through (Down)load pyiqa Model "
                    "+ pyiqa FR Score (or FGResQ FR Score). For FID, wire a set: "
                    f"{MIN_IMAGES_PER_SIDE}+ per side is the floor, "
                    f"~{MEANINGFUL_IMAGES_PER_SIDE}+ the useful range."
                )
        thin = [
            f"{name}={counts[name]}"
            for name in ("target", "reference")
            if counts[name] < MEANINGFUL_IMAGES_PER_SIDE
        ]
        if thin:
            logger.warning(
                "(pyiqa-toolbox) FID with %s image(s) per side: the Inception "
                "feature covariance has rank n-1 (of 2048), so this number is "
                "dominated by the set size rather than by image quality - treat "
                "it as indicative and compare runs at similar set sizes",
                ", ".join(thin),
            )
        with tempfile.TemporaryDirectory(prefix="pyiqa_fid_target_") as dir_t:
            with tempfile.TemporaryDirectory(prefix="pyiqa_fid_reference_") as dir_r:
                n_t = n_r = 0
                for batch in target:
                    n_t = _write_batch(batch, dir_t, n_t)
                for batch in reference:
                    n_r = _write_batch(batch, dir_r, n_r)
                # pyiqa's own folder API, with a single-process loader so nothing
                # spawns worker processes inside ComfyUI. For fid metrics
                # InferenceModel.forward hands the paths straight to the arch and
                # injects device= itself, so we must not pass device (and calling
                # the model - not .metric - is what accepts the paths).
                out = model(
                    dir_t,
                    dir_r,
                    mode=resize_mode,
                    num_workers=0,
                    batch_size=8,
                    verbose=False,
                )
        return (float(out),)


def _write_batch(images, folder, start=0):
    """Write a ComfyUI BHWC [0,1] batch as 8-bit PNGs; returns the next index.

    ``start`` continues the numbering so several wired batches share one
    folder without overwriting each other's files.
    """
    arr = (images.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    for i in range(arr.shape[0]):
        Image.fromarray(arr[i]).save(os.path.join(folder, f"{start + i:05d}.png"))
    return start + arr.shape[0]
