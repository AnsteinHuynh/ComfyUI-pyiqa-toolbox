"""MUSIQ nodes (Multi-scale Image Quality Transformer, Ke et al.).

All registry variants are NO-reference and higher-is-better, but the score
range depends on the training set:
- musiq (KonIQ-10k), musiq-paq2piq, musiq-spaq: ~0..100
- musiq-ava (AVA aesthetic): 1..10
"""

from ._common import FamilyLoaderBase, NRScoreBase


class MUSIQLoader(FamilyLoaderBase):
    FAMILY_KEY = "musiq"
    MODEL_TYPE = "PYIQA_MUSIQ"
    VARIANT_MAP = {
        "musiq (KonIQ-10k)": "musiq",
        "musiq-ava (AVA aesthetic)": "musiq-ava",
        "musiq-paq2piq": "musiq-paq2piq",
        "musiq-spaq": "musiq-spaq",
    }
    METRIC_TOOLTIP = (
        "MUSIQ variants differ by training set. Score range depends on the "
        "variant: ~0-100 for KonIQ/PAQ2PIQ/SPAQ, 1-10 for AVA (aesthetic). "
        "Higher is better. Weights auto-download on first use."
    )
    CUSTOM_PATTERNS = ("*musiq*.pth",)
    DEFAULT_WEIGHTS = "pyiqa.archs.musiq_arch"
    CATEGORY = "pyiqa/MUSIQ"
    FREE_COMFY_MEMORY = True
    DESCRIPTION = (
        "Loads MUSIQ (multi-scale transformer IQA). All variants are "
        "no-reference; ranges differ per training set (see the variant tooltip)."
    )


class MUSIQNRScore(NRScoreBase):
    CATEGORY = "pyiqa/MUSIQ"
    MODEL_TYPE = "PYIQA_MUSIQ"
    MODE_CHECK = "NR"
    OUTPUT_TOOLTIPS = (
        "Per-image MUSIQ quality score, higher is better. Range depends on the "
        "variant: 1-10 for musiq-ava (aesthetic), ~0-100 for the others.",
    )
    DESCRIPTION = (
        "MUSIQ no-reference quality score, single images. Higher is better; "
        "range depends on the variant (AVA: 1-10, others ~0-100)."
    )
