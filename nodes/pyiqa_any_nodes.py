"""Generic pyiqa loader: architecture -> variant, over the whole registry.

'(Down)load pyiqa Model' exposes every registry metric as a TWO-combo hierarchy
built from ``utility/metric_catalog.py`` (generated from pyiqa's own
``DEFAULT_CONFIGS``): ``architecture`` picks the network family (LPIPS, TOPIQ
(CFANet), Q-ReAlign, ...), ``variant`` picks that architecture's entry, labeled
with the axes a human chooses between - head (FR/NR/aesthetic), training set,
backbone - plus mode, direction and score range. Custom checkpoints from
``models/pyiqa/<architecture>/`` are variants too: a fine-tune is not a special
case, it is just another choice of weights.

The variant dropdown's values are prefixed ``"<arch> | <variant>"``; the
browser extension (``web/js/pyiqaAnyLoader.js``) filters them to the selected
architecture, and headless/API use keeps working off the same flat list.

Every registered metric stays reachable - nothing is pruned. Set-level and
text-conditioned metrics (fid / sfid / fid_dinov2 / inception_score / clipscore)
ARE listed, with labels saying so, because they load through this node like any
other metric: the per-image score nodes refuse them with a pointer instead
(``metric_catalog.SPECIAL_INPUTS``) - FID Score for the distances, CLIP Score
for ``clipscore``, and InceptionScore has no score node in this pack yet.

Outputs: ``model`` (the metric), ``model_name`` (the pyiqa registry name, e.g.
``topiq_fr``) and ``variant_label`` (the human label as shown in the dropdown).

The metric can be wired into any of this pack's score nodes: the model socket is
the union "<family>,PYIQA_ANY", so wiring is checked in the editor, and a family
score node re-checks the loaded model at run time - feeding an LPIPS model to
FGResQ FR Score fails with a message naming the node that would have worked,
instead of a shape error deep inside pyiqa.
"""

from ..utility import metric_catalog
from ..utility.pyiqa_utils import (
    default_device,
    ensure_on_device,
    get_cached_metric,
)
from ._common import (
    ANY_MODEL_TYPE,
    FRScoreBase,
    MetricBundle,
    ModelLoaderBase,
    NRScoreBase,
    build_metric,
)


class PyiqaAnyLoader(ModelLoaderBase):
    """Loads any pyiqa registry metric: architecture combo + variant combo."""

    CATEGORY = "pyiqa/Generic"
    FAMILY = "any"  # not a catalog family: the variant list is per-architecture
    FAMILY_KEY = "pyiqa_any"
    MODEL_TYPE = ANY_MODEL_TYPE
    DESCRIPTION = (
        "Loads ANY metric from pyiqa's registry: pick the architecture, then "
        "the variant. The variant label says its mode (FR/NR), direction and "
        "score range, and custom checkpoints from models/pyiqa/<architecture>/ "
        "appear there too. Wire the model into pyiqa FR Score or pyiqa NR "
        "Score - or into a family node (FGResQ, Q-ReAlign) when the metric "
        "belongs to one; those re-check the model and explain a mismatch."
    )
    OUTPUT_TOOLTIPS = (
        "The loaded pyiqa metric. Wire it into pyiqa FR Score (variants tagged "
        "FR) or pyiqa NR Score (variants tagged NR) - or into a family score "
        "node, which checks that the model is one it can score.",
        "The pyiqa registry name the metric was created from, e.g. 'topiq_fr' "
        "(for a custom checkpoint: the variant it was loaded onto, e.g. "
        "'lpips-vgg'). Stable, so it is what you want for filenames, notes or "
        "logging.",
        "The variant as shown in the dropdown, e.g. 'NR · SPAQ (ResNet50) - "
        "NR, higher better, ~0-~1'.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        # rescan the {architecture}/ folders so a checkpoint dropped since the
        # last refresh shows up in this combo too (ComfyUI re-queries the node
        # definitions on a browser refresh)
        metric_catalog.refresh_customs()
        return {
            "required": {
                "architecture": (
                    metric_catalog.ARCHES,
                    {
                        "default": metric_catalog.DEFAULT_ARCH,
                        "tooltip": (
                            "The network family - LPIPS, TOPIQ (CFANet), "
                            "ARNIQA, ... The variant dropdown beside it filters "
                            "to this architecture's entries (needs this pack's "
                            "web extension; without it the variant list shows "
                            "every architecture, prefixed by its name)."
                        ),
                    },
                ),
                "variant": (
                    metric_catalog.VARIANT_VALUES,
                    {
                        "default": metric_catalog.DEFAULT_VARIANT,
                        "tooltip": (
                            "Which weights of that architecture: its head "
                            "(FR/NR/aesthetic), training set and backbone, then "
                            "mode, direction and score range - plus any custom "
                            "checkpoint from models/pyiqa/<architecture>/, "
                            "labeled with the variant it loads onto. "
                            "Set-level and text-conditioned metrics "
                            "(fid/sfid/fid_dinov2, inception_score, clipscore) "
                            "are listed too - the per-image score nodes refuse "
                            "them and point at the node that takes them "
                            "(FID Score, CLIP Score)."
                        ),
                    },
                ),
            }
        }

    def load(self, architecture=None, variant=None, **kwargs):
        # ComfyUI passes inputs by name; keep positional calls (tests, direct
        # use) working with the catalog defaults
        arch = (
            architecture
            or kwargs.get("architecture")
            or kwargs.get("metric")
            or metric_catalog.DEFAULT_ARCH
        )
        resolved = metric_catalog.resolve(arch, variant)
        device = default_device()

        def factory():
            return MetricBundle(
                resolved,
                build_metric(resolved, device),
                # the MODEL's family (not this loader's), so a family score node
                # can tell whether this model is one it can score
                metric_catalog.family_of(resolved.arch) or self._family_key,
            )

        bundle = get_cached_metric(
            self._family_key,
            (resolved.base, resolved.path, str(device)),
            factory,
        )
        ensure_on_device(bundle, device)
        return (bundle, resolved.base, resolved.label)


class PyiqaAnyFRScore(FRScoreBase):
    """Full-reference score with any registry FR metric."""

    CATEGORY = "pyiqa/Generic"
    MODEL_TYPE = ANY_MODEL_TYPE
    SCORE_LABEL = "pyiqa FR Score"
    DESCRIPTION = (
        "Scores the target against the reference with ANY pyiqa registry FR "
        "metric loaded by (Down)load pyiqa Model - pick a variant tagged FR. "
        "Direction and range follow the metric's own registration (the "
        "variant label lists both). Lazy model load: the model is fetched "
        "only after the image(s) exist."
    )
    OUTPUT_TOOLTIPS = (
        "Per-pair score between the target (the produced/distorted image "
        "being judged) and its reference (the pristine original), computed "
        "by the selected registry metric. Higher- or lower-better depends "
        "on the metric (the variant label says which).",
    )


class PyiqaAnyNRScore(NRScoreBase):
    """No-reference score with any registry NR metric."""

    CATEGORY = "pyiqa/Generic"
    MODEL_TYPE = ANY_MODEL_TYPE
    SCORE_LABEL = "pyiqa NR Score"
    DESCRIPTION = (
        "No-reference score with ANY pyiqa registry NR metric loaded by "
        "(Down)load pyiqa Model - pick a variant tagged NR. Direction and range "
        "follow the metric's own registration (the variant label lists "
        "both). Lazy model load: the model is fetched only after the "
        "image(s) exist."
    )
    OUTPUT_TOOLTIPS = (
        "Per-image quality score computed by the selected registry metric. "
        "Higher- or lower-better depends on the metric (the variant label "
        "says which).",
    )
