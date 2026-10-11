"""FID / sFID nodes (Fréchet Inception Distance) — pack-side IMAGE-batch path.

pyiqa's own FID API is FOLDER-based: it extracts Inception features over two
folders of image files, then computes the Fréchet distance between the feature
distributions. ComfyUI hands us image tensors, so this node writes the wired
batches to temporary 8-bit PNG folders and then calls pyiqa's own folder API
(``metric(dir_target, dir_reference, mode=...)``) — the same code path pyiqa's
CLI uses, so the value matches a folder-based run over the same images.

Consequences to keep in mind (documented in the tooltips):
- This is a SET-level metric: it returns a single score, not one per image.
- The covariance estimate needs several images per side (>= ~16 to be
  meaningful); < 2 per side is rejected outright.
- Images are quantised to 8-bit PNG, exactly like the files pyiqa would read
  from a folder. A float-precision difference vs a raw-tensor implementation is
  expected and is inherent to pyiqa's folder semantics.
"""

import os
import tempfile

import numpy as np
from PIL import Image

from ._common import (
    with_memory_management,
    MEMORY_MANAGEMENT_INPUT,
    LAZY_MODEL_OPT,
    FamilyLoaderBase,
    LazyScoreMixin,
    lazy_image_opt,
)

MIN_IMAGES_PER_SIDE = 2


class FIDLoader(FamilyLoaderBase):
    FAMILY_KEY = "fid"
    MODEL_TYPE = "PYIQA_FID"
    VARIANT_MAP = {
        "fid": "fid",
        "sfid": "sfid",
    }
    METRIC_TOOLTIP = (
        "fid uses InceptionV3 pool features (2048-d) with the Fréchet distance; "
        "sfid uses the mixed_6a block (2020-d), the spatial variant from "
        "guided-diffusion. LOWER is better, unbounded. Weights auto-download on "
        "first use."
    )
    CUSTOM_PATTERNS = ()
    OUTPUT_TOOLTIPS = (
        "The loaded FID / sFID metric. Wire it into FID Score.",
    )
    CATEGORY = "pyiqa/FID"
    DESCRIPTION = (
        "Loads FID or sFID. Folder-based in pyiqa; the FID Score node feeds it "
        "IMAGE batches by staging them as temporary PNG folders."
    )


class FIDScore(LazyScoreMixin):
    """Set-level FID / sFID between two IMAGE batches. LOWER is better."""

    CATEGORY = "pyiqa/FID"
    DESCRIPTION = (
        "Fréchet distance between the target and reference image SETS. A single "
        "set-level score (not per-image), LOWER is better. Batches are staged as "
        "temporary PNGs and scored through pyiqa's own folder-based FID. The "
        "model is loaded lazily, after the image exists."
    )
    LAZY_INPUTS = ("target", "reference", "model")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "memory_management": MEMORY_MANAGEMENT_INPUT,
                "model": ("PYIQA_FID", dict(LAZY_MODEL_OPT)),
                "target": (
                    "IMAGE",
                    lazy_image_opt(
                        "Target image set (the generated/distorted side). Needs "
                        "several images - the Fréchet distance compares feature "
                        "mean and covariance, so a handful (>= ~16) is the "
                        "minimum for a meaningful number."
                    ),
                ),
                "reference": (
                    "IMAGE",
                    lazy_image_opt(
                        "Reference image set (real/ground-truth side). Same size "
                        "guidance as target; the two sets need not have equal "
                        "counts, but both must have >= 2 images."
                    ),
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
        if self._skipped(model, target, reference):
            return ([],)
        for name, batch in (("target", target), ("reference", reference)):
            if batch.shape[0] < MIN_IMAGES_PER_SIDE:
                raise ValueError(
                    f"{name} has {batch.shape[0]} image(s); FID needs at least "
                    f"{MIN_IMAGES_PER_SIDE} per side (and >= ~16 to be meaningful)."
                )
        with tempfile.TemporaryDirectory(prefix="pyiqa_fid_target_") as dir_t:
            with tempfile.TemporaryDirectory(prefix="pyiqa_fid_reference_") as dir_r:
                _write_batch(target, dir_t)
                _write_batch(reference, dir_r)
                # pyiqa's own folder API, with a single-process loader so nothing
                # spawns worker processes inside ComfyUI. Note: InferenceModel
                # injects device= itself for fid metrics, so we must not pass it.
                out = model.metric(
                    dir_t,
                    dir_r,
                    mode=resize_mode,
                    num_workers=0,
                    batch_size=8,
                    verbose=False,
                )
        return (float(out),)


def _write_batch(images, folder):
    """Write a ComfyUI BHWC [0,1] batch as 8-bit PNGs (one file per image)."""
    arr = (images.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    for i in range(arr.shape[0]):
        Image.fromarray(arr[i]).save(os.path.join(folder, f"{i:05d}.png"))
