"""DISTS family nodes (hand-crafted deep structure/texture FR distances).

Registry variants (via ``pyiqa.create_metric``, both FR, LOWER is better):
- dists: DISTS (Ding et al.) — deep image structure and texture similarity,
  range 0-1
- msswd: MS-SWD (learned multi-scale Sliced-Wasserstein distance), range 0-~10

Both measured exactly order-independent.
"""

from ..utility.pyiqa_utils import ORDER_SYMMETRIC
from ._common import FamilyLoaderBase, FRScoreBase


class DISTSLoader(FamilyLoaderBase):
    """Loads DISTS or MS-SWD via pyiqa.create_metric."""

    CATEGORY = "pyiqa/DISTS"
    FAMILY_KEY = "dists"
    MODEL_TYPE = "PYIQA_DISTS"
    VARIANT_MAP = {
        "dists": "dists",
        "msswd (MS-SWD learned)": "msswd",
    }
    METRIC_TOOLTIP = (
        "dists: deep structure+texture similarity (0-1). msswd: learned "
        "multi-scale Sliced-Wasserstein distance (0-~10). Both LOWER is better "
        "and order-independent."
    )
    CUSTOM_PATTERNS = ("*dists*.pth", "*msswd*.pth")
    DEFAULT_WEIGHTS = "pyiqa.archs.dists_arch"
    # msswd_arch has no default_model_urls table, so its auto-downloaded
    # msswd_weights.pth cannot be excluded via DEFAULT_WEIGHTS - hardcode it
    CUSTOM_WEIGHT_EXCLUDES = {"msswd_weights.pth"}
    DESCRIPTION = (
        "Loads DISTS (deep structure/texture distance, range 0-1) or MS-SWD "
        "(learned multi-scale Sliced-Wasserstein, 0-~10). Both FR, LOWER is "
        "better."
    )
    OUTPUT_TOOLTIPS = (
        "The loaded DISTS / MS-SWD metric. Wire it into DISTS FR Score.",
    )


class DISTSFrScore(FRScoreBase):
    """DISTS / MS-SWD score of target against reference. LOWER is better."""

    CATEGORY = "pyiqa/DISTS"
    MODEL_TYPE = "PYIQA_DISTS"
    ORDER_HINT = ORDER_SYMMETRIC
    DESCRIPTION = (
        "DISTS / MS-SWD distance between target and reference. LOWER is better. "
        "Order-independent. Lazy model load: the model is fetched only after "
        "the image(s) exist."
    )
    OUTPUT_TOOLTIPS = (
        "Per-pair structure/texture distance between the target (the "
        "produced/distorted image being judged) and its reference (the "
        "pristine original). LOWER is better; dists is 0-1, msswd 0-~10; "
        "identical images give 0.",
    )
