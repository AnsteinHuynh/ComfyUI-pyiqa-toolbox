"""LAION aesthetic scorer nodes.

Registry variant (via ``pyiqa.create_metric``):
- laion_aes: the LAION "improved aesthetic predictor" (CLIP ViT-L/14
  backbone + linear head, AVA-trained) — the aesthetic scorer people already
  use for batch cherry-picking in generation workflows.

No-reference, ~1-10, HIGHER is better.
"""

from ._common import FamilyLoaderBase, NRScoreBase


class LAIONLoader(FamilyLoaderBase):
    """Loads the LAION aesthetic predictor via pyiqa.create_metric."""

    CATEGORY = "pyiqa/LAION Aesthetic"
    FAMILY_KEY = "laion"
    MODEL_TYPE = "PYIQA_LAION"
    VARIANT_MAP = {
        "laion_aes (improved aesthetic predictor)": "laion_aes",
    }
    METRIC_TOOLTIP = (
        "laion_aes: LAION's improved aesthetic predictor (CLIP ViT-L/14 + "
        "linear head, AVA-trained). No-reference, range ~1-10, HIGHER is "
        "better. The standard scorer for picking the best image out of a "
        "batch."
    )
    CUSTOM_PATTERNS = ("*aes*.pth",)
    DEFAULT_WEIGHTS = "pyiqa.archs.laion_aes_arch"
    DESCRIPTION = (
        "Loads the LAION aesthetic predictor (AVA-trained, CLIP backbone). "
        "~1-10, higher is better - score a whole batch and keep the best."
    )
    OUTPUT_TOOLTIPS = (
        "The loaded aesthetic scorer. Wire it into LAION Aesthetic Score.",
    )


class LAIONNRScore(NRScoreBase):
    """LAION aesthetic score, ~1-10, higher is better."""

    CATEGORY = "pyiqa/LAION Aesthetic"
    MODEL_TYPE = "PYIQA_LAION"
    DESCRIPTION = (
        "No-reference aesthetic score, range ~1-10, HIGHER is better "
        "(AVA-trained). Score is per-image - feed a batch and pick the "
        "highest. Lazy: the model loads only once the image exists."
    )
    OUTPUT_TOOLTIPS = (
        "Per-image aesthetic score (~1-10, higher is better).",
    )
