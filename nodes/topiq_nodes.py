"""TOPIQ (CFANet) nodes — banker-training-free fine-grained IQA.

Registry variants (all via ``pyiqa.create_metric``):
- topiq_fr / topiq_fr-pipal: FULL-reference (KADID-10k / PIPAL trained)
- topiq_nr / topiq_nr-flive / topiq_nr-spaq: NO-reference (KonIQ / FLIVE / SPAQ)
- topiq_nr-face / topiq_nr_swin-face / topiq_nr-face-v1: GFIQA face models

Custom checkpoints: a ``.pth`` matching ``*topiq*`` or ``cfanet*`` dropped into
a registered ``pyiqa`` folder appears in the loader's custom dropdown; the base
variant combo selects which architecture (FR or NR) the checkpoint loads onto,
because a file name cannot say which head it was trained with.
"""

from ..utility.pyiqa_utils import ORDER_ASYMMETRIC
from ._common import FamilyLoaderBase, FRScoreBase, NRScoreBase

BUILTIN_VARIANTS = [
    "topiq_fr",
    "topiq_fr-pipal",
    "topiq_nr",
    "topiq_nr-flive",
    "topiq_nr-spaq",
    "topiq_nr-face",
    "topiq_nr_swin-face",
    "topiq_nr-face-v1",
    "topiq_iaa",
    "topiq_iaa_res50",
]


class TOPIQLoader(FamilyLoaderBase):
    """Loads a TOPIQ/CFANet scorer via pyiqa.create_metric."""

    CATEGORY = "pyiqa/TOPIQ"
    FAMILY_KEY = "topiq"
    MODEL_TYPE = "PYIQA_TOPIQ"
    VARIANT_MAP = {name: name for name in BUILTIN_VARIANTS}
    METRIC_INPUT_NAME = "model_name"  # shipped name - keep saved workflows
    CUSTOM_BASE_INPUT_NAME = "custom_base_variant"
    CUSTOM_BASE_OPTIONS = ["topiq_fr", "topiq_nr"]
    METRIC_TOOLTIP = (
        "topiq_fr (KADID-10k) and topiq_fr-pipal are FULL-reference "
        "(and order-sensitive); topiq_nr (KonIQ), -flive, -spaq are "
        "NO-reference. topiq_nr-face / -swin-face / -face-v1 are GFIQA face "
        "models (run face alignment internally; expect face crops). "
        "topiq_iaa / topiq_iaa_res50 are AESTHETIC heads (AVA-trained, "
        "1-10, higher is better) - use them with TOPIQ NR Score. Range "
        "~0-1 for quality heads, higher is better. Weights auto-download "
        "on first use."
    )
    CUSTOM_PATTERNS = ("*topiq*.pth", "cfanet*.pth")
    DEFAULT_WEIGHTS = "pyiqa.archs.topiq_arch"
    DESCRIPTION = (
        "Loads a TOPIQ (CFANet) quality model. FR variants (topiq_fr, "
        "topiq_fr-pipal) need a reference image; NR variants (topiq_nr, "
        "-flive, -spaq) score single images."
    )
    OUTPUT_TOOLTIPS = (
        "The loaded TOPIQ metric. Wire it into TOPIQ FR Score (for topiq_fr "
        "variants) or TOPIQ NR Score (for the nr variants); the score node "
        "checks the mode and tells you if you picked the wrong one.",
    )
    FREE_COMFY_MEMORY = True


class TOPIQFRScore(FRScoreBase):
    """TOPIQ full-reference score."""

    CATEGORY = "pyiqa/TOPIQ"
    MODEL_TYPE = "PYIQA_TOPIQ"
    ORDER_HINT = ORDER_ASYMMETRIC
    DESCRIPTION = (
        "TOPIQ full-reference score of target against reference. ~0-1, higher "
        "is better. Order-sensitive: target and reference are not "
        "interchangeable. Lazy model load: the model is fetched only after "
        "the image(s) exist."
    )
    OUTPUT_TOOLTIPS = (
        "Per-pair TOPIQ quality of the target (the produced/distorted image "
        "being judged) against its reference (the pristine original), ~0-1, "
        "higher is better. Order-sensitive: swapping target and reference "
        "changes the value.",
    )


class TOPIQNRScore(NRScoreBase):
    """TOPIQ no-reference score."""

    CATEGORY = "pyiqa/TOPIQ"
    MODEL_TYPE = "PYIQA_TOPIQ"
    DESCRIPTION = (
        "TOPIQ no-reference quality score, single images. ~0-1, higher is "
        "better. Lazy model load: the model is fetched only after the "
        "image(s) exist."
    )
    OUTPUT_TOOLTIPS = (
        "Per-image TOPIQ no-reference quality score, ~0-1, higher is better.",
    )
