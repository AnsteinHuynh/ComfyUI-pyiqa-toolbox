"""WaDIQaM nodes (WaDIQaM-FR / WaDIQaM-NR, Bosse et al.).

Two registry entries of the same arch:
- wadiqam_fr: FULL-reference (KADID-trained), order-SENSITIVE per measurement
- wadiqam_nr: NO-reference (KonIQ-trained)

Both report a quality score (arch-reported range roughly -1..1); pyiqa marks
neither lower_better.
"""

from ..utility.pyiqa_utils import ORDER_ASYMMETRIC

from ._common import FRScoreBase, FamilyLoaderBase, NRScoreBase


class WaDIQaMLoader(FamilyLoaderBase):
    FAMILY_KEY = "wadiqam"
    MODEL_TYPE = "PYIQA_WADIQAM"
    VARIANT_MAP = {
        "wadiqam_fr (full-reference)": "wadiqam_fr",
        "wadiqam_nr (no-reference)": "wadiqam_nr",
    }
    METRIC_TOOLTIP = (
        "WaDIQaM-FR compares target against reference (order-SENSITIVE: the two "
        "inputs are not interchangeable); WaDIQaM-NR scores single images. "
        "Weights auto-download on first use."
    )
    CUSTOM_PATTERNS = ("*wadiqam*.pth",)
    DEFAULT_WEIGHTS = "pyiqa.archs.wadiqam_arch"
    CATEGORY = "pyiqa/WaDIQaM"
    DESCRIPTION = (
        "Loads WaDIQaM (deep CNN-based IQA). wadiqam_fr is full-reference, "
        "wadiqam_nr no-reference."
    )


class WaDIQaMFRScore(FRScoreBase):
    CATEGORY = "pyiqa/WaDIQaM"
    MODEL_TYPE = "PYIQA_WADIQAM"
    ORDER_HINT = ORDER_ASYMMETRIC
    OUTPUT_TOOLTIPS = (
        "Per-pair WaDIQaM quality of the target (the produced/distorted image "
        "being judged) against its reference (the pristine original); arch "
        "range ~-1..1, higher is better. Order-sensitive: swapping target and "
        "reference changes the value.",
    )
    DESCRIPTION = (
        "WaDIQaM full-reference score of target against reference. "
        "Order-SENSITIVE: target and reference are not interchangeable."
    )


class WaDIQaMNRScore(NRScoreBase):
    CATEGORY = "pyiqa/WaDIQaM"
    MODEL_TYPE = "PYIQA_WADIQAM"
    MODE_CHECK = "NR"
    OUTPUT_TOOLTIPS = (
        "Per-image WaDIQaM no-reference quality score (arch range ~-1..1).",
    )
    DESCRIPTION = "WaDIQaM no-reference quality score, single images."
