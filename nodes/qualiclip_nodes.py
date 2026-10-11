"""QualiCLIP nodes (AGIQA / QualiCLIP, "Quality-Aware Image-Text Alignment").

All registry variants are NO-reference, 0-1, higher is better:
- qualiclip / qualiclip+: base and CLIP-finetuned variants
- qualiclip+-clive/-flive/-spaq: variants trained on CLIVE / FLIVE / SPAQ
"""

from ._common import FamilyLoaderBase, NRScoreBase


class QualiCLIPLoader(FamilyLoaderBase):
    FAMILY_KEY = "qualiclip"
    MODEL_TYPE = "PYIQA_QUALICLIP"
    VARIANT_MAP = {
        "qualiclip": "qualiclip",
        "qualiclip+": "qualiclip+",
        "qualiclip+-clive": "qualiclip+-clive",
        "qualiclip+-flive": "qualiclip+-flive",
        "qualiclip+-spaq": "qualiclip+-spaq",
    }
    METRIC_TOOLTIP = (
        "QualiCLIP is a quality-aware CLIP alignment model; the + variants are "
        "its fine-tuned heads, and -clive/-flive/-spaq denote the training set. "
        "All NO-reference, 0-1, higher is better. Weights auto-download on first "
        "use."
    )
    CUSTOM_PATTERNS = ("*qualiclip*.pth",)
    DEFAULT_WEIGHTS = "pyiqa.archs.qualiclip_arch"
    CATEGORY = "pyiqa/QualiCLIP"
    FREE_COMFY_MEMORY = True
    DESCRIPTION = (
        "Loads a QualiCLIP scorer (quality-aware CLIP). 0-1, higher is better."
    )


class QualiCLIPNRScore(NRScoreBase):
    CATEGORY = "pyiqa/QualiCLIP"
    MODEL_TYPE = "PYIQA_QUALICLIP"
    MODE_CHECK = "NR"
    OUTPUT_TOOLTIPS = (
        "Per-image QualiCLIP score, 0-1, higher is better (1 = high quality).",
    )
    DESCRIPTION = "QualiCLIP no-reference score, single images. 0-1, higher is better."
