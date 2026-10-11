"""CLIP-IQA nodes (Wang et al., "Exploring CLIP for Assessing the Look and
Feel of Images").

All registry variants are NO-reference, range 0-1, higher is better:
- clipiqa:   the original CLIP-IQA prompt-pair scorer
- clipiqa+:  the fine-tuned full-reference-style variant (still registered NR)
- clipiqa+_vitL14_512 / clipiqa+_rn50_512: CLIP-IQA+ with an explicit backbone
  and 512px test size
"""

from ._common import FamilyLoaderBase, NRScoreBase


class CLIPIQALoader(FamilyLoaderBase):
    FAMILY_KEY = "clipiqa"
    MODEL_TYPE = "PYIQA_CLIPIQA"
    VARIANT_MAP = {
        "clipiqa": "clipiqa",
        "clipiqa+": "clipiqa+",
        "clipiqa+_vitL14_512": "clipiqa+_vitL14_512",
        "clipiqa+_rn50_512": "clipiqa+_rn50_512",
    }
    METRIC_TOOLTIP = (
        "clipiqa is the prompt-pair CLIP scorer; clipiqa+ is the fine-tuned "
        "variant; the _vitL14_512 / _rn50_512 entries pin the backbone and a "
        "512px test size. All NO-reference, 0-1, higher is better. Weights "
        "auto-download on first use."
    )
    CUSTOM_PATTERNS = ("*clipiqa*.pth",)
    DEFAULT_WEIGHTS = "pyiqa.archs.clipiqa_arch"
    CATEGORY = "pyiqa/CLIP-IQA"
    FREE_COMFY_MEMORY = True
    DESCRIPTION = (
        "Loads a CLIP-IQA scorer (CLIP-based no-reference quality). 0-1, higher "
        "is better."
    )


class CLIPIQANRScore(NRScoreBase):
    CATEGORY = "pyiqa/CLIP-IQA"
    MODEL_TYPE = "PYIQA_CLIPIQA"
    MODE_CHECK = "NR"
    OUTPUT_TOOLTIPS = (
        "Per-image CLIP-IQA score, 0-1, higher is better (1 = high quality).",
    )
    DESCRIPTION = "CLIP-IQA no-reference score, single images. 0-1, higher is better."
