"""Catalog for this pack's loader/score nodes: architectures, variants and
custom checkpoints.

One hierarchy, used by every loader node in the pack:

- **architecture** — the network family, from ``metric_opts.type`` in pyiqa's
  own machine-readable defaults (``DEFAULT_CONFIGS``). Readable labels:
  "LPIPS", "TOPIQ (CFANet)", "Q-ReAlign", ...
- **variant** — one concrete choice of WEIGHTS within that architecture: either
  a registry entry (``lpips-vgg``) or a **custom checkpoint** from the
  ``{architecture}/`` folder convention on disk. A fine-tune is not a special
  case of anything; it is just another variant.

Custom checkpoints live in the user-facing pyiqa folders as::

    models/pyiqa/LPIPS/lpips-vgg-myft.pth              (a weight file)
    models/pyiqa/TOPIQ/topiq_nr-spaq-myft.safetensors
    models/pyiqa/Q-ReAlign/Q-ReAlign-Mini-0.8B-int8-convrot/   (config.json + weights)

The FOLDER is the architecture, so nothing about a custom file's name has to be
guessed to know which network it belongs to. What a file name can still decide
is which *variant* of that architecture it replaces, and that is only a list
shortener: ``TOPIQ/topiq_nr-spaq-myft.safetensors`` matches the registry name
``topiq_nr-spaq`` and gets one dropdown entry labeled with the variant it loads
onto; a name that matches nothing (``TOPIQ/mystery.pth``) is offered once per
variant of that one architecture, so the choice is made explicitly in the node
instead of silently guessed from a file name.

Why the layout is not just a naming preference: pyiqa's own default weights
land loose at the top of its cache folders, so treating only *architecture
folders* as custom sources makes the whole "hide pyiqa's own checkpoints from
the custom scan" machinery (default-URL tables, basename excludes, backbone
excludes) unnecessary.

Every architecture-specific override the pack knows — its readable label, the
folder names it answers to, input policies (sides divisible by 32), batch-of-one
requirements, a snapshot-folder loader — is ONE ROW of ``_ARCH_RULES`` below, and
the ``Architecture`` objects the rest of the pack reads are built from those rows.
An architecture with no row is a plain registry metric, which is the common case,
so this file is the only place architecture rules live.

Registry drift: a name in ``list_models()`` that is unknown to
``DEFAULT_CONFIGS`` is listed under an "Unlisted" architecture, so a pyiqa
upgrade can never make a metric unreachable.

Reading pyiqa's table is the one place where a pyiqa upgrade could take the
whole pack down, so it is done defensively in two directions: every field is
read with ``.get`` (an entry that gained or lost a key cannot crash the import),
and if the table itself cannot be read the catalog falls back to a short
built-in metric list (``_fallback_architectures``) so the nodes still exist.
"""

import logging
import os
import time
from dataclasses import dataclass

from . import pyiqa_utils

logger = logging.getLogger(__name__)


def _read_registry_configs():
    """pyiqa's machine-readable metric table, or ``({}, reason)``.

    ``pyiqa.default_model_configs`` is a public module of the pyiqa package and
    ``DEFAULT_CONFIGS`` is the only place FR/NR mode, direction and score range
    are declared, so there is no second source to fall back to for *metadata* —
    only for the list of names (see ``_fallback_architectures``).
    """
    try:
        from pyiqa.default_model_configs import DEFAULT_CONFIGS as configs
    except Exception as exc:  # depends on the installed pyiqa version
        return {}, f"cannot import pyiqa.default_model_configs ({type(exc).__name__}: {exc})"
    if not isinstance(configs, dict) or not configs:
        return {}, "pyiqa.default_model_configs.DEFAULT_CONFIGS is empty"
    return configs, ""


_CONFIGS, REGISTRY_ERROR = _read_registry_configs()

SEPARATOR = " | "

# The three tables below are keyed by REGISTRY METRIC, not by architecture, and
# stay separate from _ARCH_RULES on purpose: they are exceptions for individual
# registry entries (a set-level metric, a variant whose label needs explaining),
# not rules of the architecture they happen to belong to.

# registry name -> (kind, why the generic pyiqa FR/NR score nodes cannot run it).
# These ARE listed by the generic loader — nothing pyiqa ships stays invisible —
# and the score nodes refuse them at run time with the pointer below.
#   "distance"  two image sets    -> FID Score
#   "one_set"   one image set     -> no node in this pack yet
#   "caption"   image + a caption -> CLIP Score
SPECIAL_INPUTS = {
    "fid": ("distance", "two image sets - use FID Score"),
    "sfid": ("distance", "two image sets - use FID Score"),
    "fid_dinov2": ("distance", "two image sets - use FID Score"),
    "inception_score": (
        "one_set",
        "one image set - this pack has no score node for a single set yet",
    ),
    "clipscore": ("caption", "needs a caption for each image - use CLIP Score"),
}

# pyiqa labels every metric in SPECIAL_INPUTS "NR" because none of them takes a
# reference IMAGE - true, and useless in a dropdown that has to say what the
# number is a distance between. The label gets the honest mode instead;
# ``Resolved.mode`` keeps pyiqa's own value, which is what the score-node guards
# reason about (they check ``special_input`` first anyway).
_MODE_OVERRIDES = {
    "fid": "set-level",
    "sfid": "set-level",
    "fid_dinov2": "set-level",
    "inception_score": "one set",
    "clipscore": "NR + caption",
}

# Listed metrics that the generic score nodes cannot run as-is: pyiqa's pair
# API returns (quality0, quality1, rank, rank_prob), which is four outputs, not
# one score. The dedicated node unpacks them.
SPECIAL_METRICS = {
    "fgresq_pair": "pair API returns 4 values (use FGResQ FR Score)",
}

# Last resort when pyiqa's config table cannot be read at all: names that have
# been in every pyiqa release this pack has supported, so the nodes stay usable
# instead of disappearing (see _fallback_architectures).
_FALLBACK_METRICS = (
    "afine_fr",
    "brisque",
    "clipiqa",
    "cw_ssim",
    "dists",
    "fsim",
    "gmsd",
    "hyperiqa",
    "ilniqe",
    "laion_aes",
    "lpips",
    "mad",
    "maniqa",
    "ms_ssim",
    "msswd",
    "musiq",
    "nima",
    "niqe",
    "nlpd",
    "paq2piq",
    "piqe",
    "psnr",
    "qalign",
    "qrealign",
    "ssim",
    "stlpips",
    "topiq_fr",
    "topiq_nr",
    "vif",
    "vsi",
)

@dataclass(frozen=True)
class _ArchRule:
    """Everything special about ONE architecture, in one row.

    Every field is an override, and the common case has none: an architecture
    with no row gets its label and its folder from pyiqa's own
    ``metric_opts.type`` and is treated as a plain registry metric. A row exists
    only where pyiqa's behaviour is not enough for the nodes, and it is the ONLY
    place that architecture's special needs are declared - the ``Architecture``
    objects the rest of the pack reads are built from it.

    Fields:
        folders        every folder name accepted on disk (see below)
        policy         input policies, e.g. "sides divisible by 32"
        single_image   model asserts a batch of one
        single_image_metrics  batch-of-one for these variants only
        snapshot_loader  registry name of a loader that takes a snapshot folder
        signpost       this architecture is listed in the dropdown but does NOT
                       load here; the text says which nodes to use instead
    """

    label: str
    folders: tuple = ()  # () means "the label itself is the only folder"
    policy: tuple = ()
    single_image: bool = False
    single_image_metrics: tuple = ()
    snapshot_loader: str = ""
    signpost: str = ""


# Architecture display names, on-disk folder names and hard model constraints,
# anchored on one representative METRIC rather than on pyiqa's
# ``metric_opts.type``.
#
# Why a metric name is the key: metric names are pyiqa's public API, while
# ``metric_opts.type`` is an internal class name. A table keyed by the type
# silently changes both the dropdown label and the folder we look for when pyiqa
# renames a class - and every checkpoint a user had already dropped into
# ``models/pyiqa/{folder}/`` would stop being found, with no error to explain it.
# Keyed by a metric name, such a rename costs nothing.
#
# The FIRST folder is the canonical one: it is what the scaffold creates and what
# the pack has always used, so never reorder or respell it. Matching is case- and
# punctuation-insensitive, so "A-FINE" already matches "AFINE" and "Q-ReAlign"
# matches "QReAlign"; list only names that normalization cannot relate.
#
# APPEND-ONLY: when a pyiqa rename makes an architecture answer to a new name,
# add that name and keep every name that was there before.
_ARCH_RULES = {
    # A-FINE asserts sides divisible by 32 (CLIP ViT-B/32 patches); its own
    # padding branch sits behind that assert and is unreachable.
    "afine_nr": _ArchRule(
        "A-FINE", policy=pyiqa_utils.POLICY_MULT32, single_image=True
    ),
    "clipiqa": _ArchRule("CLIP-IQA"),
    "compare2score": _ArchRule("Compare2Score", single_image=True),
    "cw_ssim": _ArchRule("CW-SSIM"),
    "hyperiqa": _ArchRule("HyperIQA", ("HyperIQA", "HyperNet")),
    "laion_aes": _ArchRule("LAION Aesthetic", ("LAION Aesthetic", "LAIONAes")),
    "ms_ssim": _ArchRule("MS-SSIM"),
    "msswd": _ArchRule("MS-SWD (learned)", ("MS-SWD", "MS_SWD_learned")),
    "paq2piq": _ArchRule("PaQ-2-PiQ"),
    "qalign": _ArchRule("Q-Align", single_image=True),
    "qrealign": _ArchRule(
        "Q-ReAlign",
        snapshot_loader="qrealign_snapshot",
        # The generic loader must not build Q-ReAlign: the int8_convrot builds,
        # the snapshot folders and the dtype choice live in the dedicated loader,
        # and listing pyiqa's four names here only produced a second, shorter
        # list of the same family (see the README's breaking changes).
        signpost=(
            "has its own loader and score node - use (Down)load Q-ReAlign + "
            "Q-ReAlign NR Score (int8_convrot builds, dtype and snapshot "
            "folders live there)"
        ),
    ),
    "stlpips": _ArchRule("ST-LPIPS"),
    # pyiqa calls the mean-absolute-difference metric "mad", which reads like a
    # new architecture rather than the L1-style metric it is. The label says so;
    # the FOLDER stays "MAD", because renaming a label must never move the place
    # users keep their files.
    "mad": _ArchRule("MAD (L1)", folders=("MAD",)),
    # Inception Score is a one-set metric: pyiqa ships it and this pack has no
    # score node for a single set yet, so the entry says that instead of loading
    # and failing later.
    "inception_score": _ArchRule(
        "Inception Score",
        signpost=(
            "no score node for a single image set yet - FID Score handles two "
            "sets (see METRICS.md)"
        ),
    ),
    # CLIPScore needs a caption per image; listed here, scored by CLIP Score.
    # Folder keeps pyiqa's spelling so an existing drop folder still matches.
    "clipscore": _ArchRule("CLIP Score", folders=("CLIPScore",)),
    "topiq_nr": _ArchRule(
        "TOPIQ (CFANet)",
        ("TOPIQ", "CFANet"),
        # the GFIQA face pipeline takes one face at a time
        single_image_metrics=(
            "topiq_nr-face",
            "topiq_nr-face-v1",
            "topiq_nr_swin-face",
        ),
    ),
}

# family key -> architecture labels it covers. A loader node owns exactly one
# family; the score nodes of that family accept a model from any of its
# architectures (DISTS owns the learned MS-SWD variant too).
FAMILIES = {
    "afine": ("A-FINE",),
    "clipiqa": ("CLIP-IQA",),
    "clipscore": ("CLIP Score",),
    "classic_fr": (
        "CW-SSIM",
        "FSIM",
        "GMSD",
        "MAD (L1)",
        "MS-SSIM",
        "NLPD",
        "PSNR",
        "SSIM",
        "VIF",
        "VSI",
    ),
    "dists": ("DISTS", "MS-SWD (learned)"),
    "fgresq": ("FGResQ",),
    "fid": ("FID", "Inception Score"),
    "laion": ("LAION Aesthetic",),
    "lpips": ("LPIPS", "ST-LPIPS"),
    "musiq": ("MUSIQ",),
    "qualiclip": ("QualiCLIP",),
    "qrealign": ("Q-ReAlign",),
    "topiq": ("TOPIQ (CFANet)",),
    "wadiqam": ("WaDIQaM",),
}

# Families whose loader node takes custom weights: their architecture folders
# are created on import so there is a visible place to drop a checkpoint.
SCAFFOLD_FAMILIES = (
    "afine",
    "clipiqa",
    "dists",
    "fgresq",
    "laion",
    "lpips",
    "musiq",
    "qualiclip",
    "qrealign",
    "topiq",
    "wadiqam",
)

# Human name for each family, for score-node error messages.
FAMILY_NAMES = {
    "afine": "A-FINE",
    "clipiqa": "CLIP-IQA",
    "clipscore": "CLIP Score",
    "classic_fr": "Classic FR",
    "dists": "DISTS",
    "fgresq": "FGResQ",
    "fid": "FID",
    "laion": "LAION Aesthetic",
    "lpips": "LPIPS",
    "musiq": "MUSIQ",
    "qualiclip": "QualiCLIP",
    "qrealign": "Q-ReAlign",
    "topiq": "TOPIQ",
    "wadiqam": "WaDIQaM",
}

# Input policies, batch-of-one constraints and snapshot loaders are NOT separate
# tables: they are fields of each architecture's ``_ARCH_RULES`` row, and the
# ``Architecture`` objects below are built from it. Consumers read them there
# (``arch.policy``, ``arch.single_image_for(name)``, ``arch.snapshot_loader``), so
# there is exactly one place to look for "what is special about architecture X".
# The policies themselves (POLICY_MULT32) are implemented in pyiqa_utils, next to
# the image preparation they describe.

# Dedicated-loader dropdown labels are gone with the dedicated loaders: every
# loader now shows the generic variant label, which carries the mode, direction
# and score range a bare curated name did not.

# registry name -> human axes for the variant label (multi-variant
# architectures; singletons need no axes — mode/direction/range suffice)
_VARIANT_AXES = {
    # SSIM / PSNR families: the axis is which channels are compared
    "ssim": "luminance (Y channel)",
    "ssimc": "color (RGB channels)",
    "psnr": "RGB",
    "psnry": "luminance (Y channel)",
    # LPIPS: method x backbone
    # LPIPS names its variants after the change, but pyiqa's own names are
    # lpips / lpips-vgg / lpips+ / lpips-vgg+ and people know the "+" ones as
    # "LPIPS+": spelling both means either name finds the entry.
    "lpips": "LPIPS (AlexNet)",
    "lpips-vgg": "LPIPS (VGG)",
    "lpips+": "LPIPS+ (AlexNet, semantic weighting)",
    "lpips-vgg+": "LPIPS+ (VGG, semantic weighting)",
    "stlpips": "shift-tolerant (AlexNet)",
    "stlpips-vgg": "shift-tolerant (VGG)",
    # TOPIQ (CFANet): head x training set x backbone
    "topiq_fr": "FR · KADID-10k (ResNet50)",
    "topiq_fr-pipal": "FR · PIPAL (ResNet50)",
    "topiq_nr": "NR · KonIQ-10k (ResNet50)",
    "topiq_nr-flive": "NR · FLIVE (ResNet50, 384px)",
    "topiq_nr-spaq": "NR · SPAQ (ResNet50)",
    "topiq_nr-face": "NR · faces GFIQA-cg (ResNet50, 512px)",
    "topiq_nr-face-v1": "NR · faces GFIQA (ResNet50, 512px)",
    "topiq_nr_swin-face": "NR · faces GFIQA-cg (Swin-B, 384px)",
    "topiq_iaa": "aesthetic · AVA (Swin-B)",
    "topiq_iaa_res50": "aesthetic · AVA (ResNet50, 384px)",
    # ARNIQA: which dataset the regressor was trained on
    "arniqa": "default regressor (KonIQ-10k)",
    "arniqa-clive": "CLIVE regressor",
    "arniqa-csiq": "CSIQ regressor",
    "arniqa-flive": "FLIVE regressor",
    "arniqa-kadid": "KADID-10k regressor",
    "arniqa-live": "LIVE regressor",
    "arniqa-spaq": "SPAQ regressor",
    "arniqa-tid": "TID2013 regressor",
    # QualiCLIP: base / improved (+) x training set
    "qualiclip": "base",
    "qualiclip+": "improved (+)",
    "qualiclip+-clive": "+ · CLIVE-trained",
    "qualiclip+-flive": "+ · FLIVE-trained",
    "qualiclip+-spaq": "+ · SPAQ-trained",
    # A-FINE: head / problem formulation
    "afine_fr": "FR head",
    "afine_nr": "NR head",
    "afine_all": "FR · all-problem",
    "afine_all_scale": "FR · all-problem, scaled to 0-100",
    # CLIP-IQA: revision x backbone
    "clipiqa": "original",
    "clipiqa+": "improved (+)",
    "clipiqa+_rn50_512": "+ · RN50 @ 512px",
    "clipiqa+_vitL14_512": "+ · ViT-L/14 @ 512px",
    # MUSIQ / NIMA / MANIQA / TReS / DBCNN / LIQE: training set
    "musiq": "KonIQ-10k",
    "musiq-ava": "AVA",
    "musiq-paq2piq": "PaQ-2-PiQ",
    "musiq-spaq": "SPAQ",
    "nima": "InceptionResNet-v2 · AVA",
    "nima-koniq": "InceptionResNet-v2 · KonIQ-10k",
    "nima-spaq": "InceptionResNet-v2 · SPAQ",
    "nima-vgg16-ava": "VGG-16 · AVA",
    "maniqa": "KonIQ-10k",
    "maniqa-kadid": "KADID-10k",
    "maniqa-pipal": "PIPAL",
    "tres": "KonIQ-10k",
    "tres-flive": "FLIVE",
    "liqe": "KonIQ-10k",
    "liqe_mix": "mixed datasets",
    # Q-ReAlign / Q-Align: model size / precision
    "qrealign": "Mini 0.8B (default)",
    "qrealign-mini": "Mini 0.8B",
    "qrealign-lite": "Lite 4B",
    "qrealign-pro": "Pro 9B",
    "qalign": "default precision",
    "qalign_4bit": "4-bit",
    "qalign_8bit": "8-bit",
    # reference-mode heads within one architecture
    "wadiqam_fr": "FR head",
    "wadiqam_nr": "NR head",
    "fgresq": "NR (single image) · use FGResQ NR Score",
    "fgresq_pair": "FR pair (rank + verdict)",
    # set-level / text-conditioned metrics: listed here (so nothing pyiqa ships
    # is invisible), scored by their dedicated node, refused by the generic
    # score nodes with the same pointer (see SPECIAL_INPUTS)
    "fid": "InceptionV3 features · use FID Score",
    "sfid": "InceptionV3 mixed_6a, spatial · use FID Score",
    "fid_dinov2": "DINOv2 features · use FID Score",
    "inception_score": "single image set (no score node in this pack yet)",
    "clipscore": "ViT-B/32 · use CLIP Score",
    "mad": "mean absolute difference",
    # MATLAB ports of the classic NR metrics
    "brisque": "standard",
    "brisque_matlab": "MATLAB implementation",
    "niqe": "standard",
    "niqe_matlab": "MATLAB implementation",
}

# registry name -> readable score range (DEFAULT_CONFIGS strings are compacted
# from "a, b" to "a-b"; these override the awkward/unbounded ones)
_RANGE_OVERRIDES = {
    "mad": "0-unbounded",
    "dmm": "0-unbounded",
    "pi": "0-unbounded",
    "afine_all": "up to ~2",
    "fgresq_pair": "quality 0-1 · rank 0-2",
}


def _normalize(text):
    """Folder/file matching key: lowercase, punctuation and spaces dropped."""
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def _range_text(name, cfg):
    if name in _RANGE_OVERRIDES:
        return _RANGE_OVERRIDES[name]
    return str(cfg.get("score_range", "?")).replace(", ", "-")


def _variant_label(name, cfg):
    if not cfg:
        # pyiqa's config table could not be read: say so instead of inventing a
        # direction (see _fallback_architectures)
        return "mode, direction and range unknown (pyiqa metric table unavailable)"
    axes = _VARIANT_AXES.get(name, "")
    mode = _MODE_OVERRIDES.get(name) or cfg.get("metric_mode", "?")
    direction = "lower better" if cfg.get("lower_better") else "higher better"
    tail = f"{mode}, {direction}, {_range_text(name, cfg)}"
    return f"{axes} - {tail}" if axes else tail


@dataclass(frozen=True)
class Architecture:
    """One network family: pyiqa's ``metric_opts.type`` plus our naming.

    Built from this architecture's ``_ARCH_RULES`` row (or from pyiqa's type name
    when it has none) and read by everything else in the pack, so this object is
    the runtime rule table for the architecture.
    """

    label: str
    type: str
    variants: tuple  # registry metric names, DEFAULT_CONFIGS order
    folders: tuple  # accepted on-disk folder names; first is canonical
    family: str
    policy: tuple = ()
    snapshot_loader: str = ""  # registry name of a snapshot-folder loader
    single_image: bool = False  # the arch asserts a batch of one
    single_image_metrics: tuple = ()  # ...for these variants of it only
    signpost: str = ""  # listed but not loadable here: where to go instead

    @property
    def folder_keys(self):
        """Normalized folder names that address this architecture."""
        keys = {_normalize(f) for f in self.folders}
        keys.add(_normalize(self.label))
        keys.add(_normalize(self.type))
        return frozenset(keys)

    def single_image_for(self, name):
        """True when variant ``name`` of this architecture must be scored alone.

        The score nodes chunk their batch in that case instead of failing on
        batch > 1 — a hard model requirement, not a user knob.
        """
        return self.single_image or name in self.single_image_metrics


@dataclass(frozen=True)
class Resolved:
    """A variant choice resolved to something loadable."""

    arch: str  # architecture label
    base: str  # pyiqa registry name the metric is created from
    label: str  # human variant label (no architecture prefix)
    mode: str  # "FR" / "NR"
    lower_better: bool
    score_range: str
    policy: tuple = ()
    single_image: bool = False
    path: str = ""  # custom weights (file or snapshot dir), "" for registry
    kind: str = "registry"  # registry | checkpoint | safetensors | snapshot
    display: str = ""  # what to call the weights in messages
    # True for metrics whose arch returns several values (pyiqa's pair API):
    # the plain FR score node cannot unpack those, and says so
    multi_output: bool = False

    @property
    def custom(self):
        return self.kind != "registry"


@dataclass(frozen=True)
class _Custom:
    """A discovered custom checkpoint."""

    label: str
    path: str
    kind: str
    base: str = ""  # matched registry name, "" when the name says nothing


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------


def _arch_rule(type_name, names):
    """This architecture's ``_ARCH_RULES`` row, or a plain default one.

    Matched on the metrics the architecture owns, never on ``type_name``, so a
    pyiqa version that renames a class cannot rename the folder users keep their
    checkpoints in (see ``_ARCH_RULES``).
    """
    for name in names:
        rule = _ARCH_RULES.get(name)
        if rule is not None:
            return rule
    # uncurated architecture: pyiqa's class name is both label and folder, which
    # is exactly what the old identity table spelled out 52 times
    return _ArchRule(type_name)


def _family_of_arch(label):
    for family, labels in FAMILIES.items():
        if label in labels:
            return family
    return "any"


def _match_base(variants, name):
    """Longest registry variant name contained in a custom's file name.

    Purely a list shortener: a match pairs the custom with the variant it names
    (``topiq_nr-spaq-myft`` -> ``topiq_nr-spaq``, not the shorter ``topiq_nr``),
    and no match leaves the choice to the node instead of guessing here.
    """
    key = _normalize(os.path.splitext(name)[0])
    best = ""
    for variant in variants:
        norm = _normalize(variant)
        if norm and norm in key and len(norm) > len(_normalize(best)):
            best = variant
    return best


def _customs_for(arch):
    """Custom weight files (and snapshot dirs, where loadable) for ``arch``."""
    found = []
    seen = set()
    for _folder, entry, path in pyiqa_utils.iter_user_pyiqa_dirs():
        if not os.path.isdir(path) or _normalize(entry) not in arch.folder_keys:
            continue
        for label, fpath, kind in pyiqa_utils.list_weight_files(path):
            key = os.path.normcase(os.path.abspath(fpath))
            if key in seen:
                continue
            seen.add(key)
            found.append(_Custom(label, fpath, kind, _match_base(arch.variants, label)))
        if not arch.snapshot_loader:
            # a directory is only loadable where the arch takes one; listing it
            # everywhere would advertise variants that cannot load
            continue
        for label, dpath in pyiqa_utils.list_snapshot_dirs(path):
            key = os.path.normcase(os.path.abspath(dpath))
            if key in seen:
                continue
            seen.add(key)
            found.append(_Custom(label, dpath, "snapshot", _match_base(arch.variants, label)))
    return found


def _resolved_registry(arch, name):
    cfg = _CONFIGS.get(name, {})
    return Resolved(
        arch=arch.label,
        base=name,
        label=_variant_label(name, cfg),
        mode=cfg.get("metric_mode") or "?",
        lower_better=bool(cfg.get("lower_better")),
        score_range=_range_text(name, cfg),
        policy=arch.policy,
        single_image=arch.single_image_for(name),
        multi_output=name in SPECIAL_METRICS,
    )


def _resolved_custom(arch, custom, base_name):
    cfg = _CONFIGS.get(base_name, {})
    axes = _VARIANT_AXES.get(base_name) or base_name
    if custom.kind == "snapshot":
        # the directory name IS the display name; what it loads onto is the
        # part a user cannot see from the folder name alone
        label = f"{custom.label} → {axes}"
    else:
        label = f"custom: {custom.label} → {axes}"
    return Resolved(
        arch=arch.label,
        base=base_name,
        label=label,
        mode=cfg.get("metric_mode") or "?",
        lower_better=bool(cfg.get("lower_better")),
        score_range=_range_text(base_name, cfg),
        policy=arch.policy,
        single_image=arch.single_image_for(base_name),
        path=custom.path,
        kind=custom.kind,
        display=custom.label,
        multi_output=base_name in SPECIAL_METRICS,
    )


def special_input(name):
    """``(kind, message)`` for a metric only a dedicated score node can run.

    Kind is ``"distance"`` (two image sets), ``"one_set"`` or ``"caption"``.
    ``None`` for every normal per-image metric. The generic score nodes check
    this before anything else, so a set-level model can never be handed to a
    per-image scorer just because pyiqa calls it "NR".
    """
    return SPECIAL_INPUTS.get(name)


def _signpost_entry(arch):
    """The single dropdown entry of a signpost architecture.

    A ``Resolved`` with ``kind="signpost"``: it exists so the architecture is
    visible and picking it explains where to go, and ``resolve`` refuses to hand
    it to a builder. Nothing loads, so the mode/direction/range fields are
    deliberately unknown - the label is the message, not a score description.
    """
    return Resolved(
        arch=arch.label,
        base=arch.variants[0] if arch.variants else "",
        label=arch.signpost,
        mode="?",
        lower_better=True,
        score_range="?",
        kind="signpost",
    )


def _entries_for(arch):
    """[(variant label, Resolved)] for one architecture, customs included."""
    if arch.signpost:
        return [(arch.signpost, _signpost_entry(arch))]
    entries = []
    for name in arch.variants:
        resolved = _resolved_registry(arch, name)
        entries.append((resolved.label, resolved))
    for custom in _customs_for(arch):
        # a named custom gets exactly its own entry; an unnamed one is offered
        # once per variant of THIS architecture so the base is chosen in the
        # node rather than guessed from the file name
        bases = [custom.base] if custom.base else list(arch.variants)
        for base_name in bases:
            resolved = _resolved_custom(arch, custom, base_name)
            entries.append((resolved.label, resolved))
    return entries


def _unique(entries):
    """Disambiguate duplicate variant labels for one dropdown with ``[base]``."""
    seen = {}
    out = []
    for label, resolved in entries:
        if label in seen:
            seen[label] += 1
            label = f"{label} [{resolved.base}]"
        else:
            seen[label] = 1
        out.append((label, resolved))
    return out


def _flag_special(entries):
    """Mark metrics the plain FR score node cannot unpack (kept reachable)."""
    flagged = []
    for label, resolved in entries:
        if resolved.multi_output:
            label = f"{label} ⚠ {SPECIAL_METRICS[resolved.base]}"
        flagged.append((label, resolved))
    return flagged


def _registered_names():
    """``pyiqa.list_models()``, or ``set()`` if pyiqa's own API changed."""
    try:
        import pyiqa

        return set(pyiqa.list_models())
    except Exception as exc:
        logger.warning(
            "(pyiqa-toolbox) pyiqa.list_models() failed (%s: %s); listing every "
            "metric in pyiqa's config table",
            type(exc).__name__,
            exc,
        )
        return set()


def _fallback_architectures():
    """A usable catalog when pyiqa's config table cannot be read.

    The names below have been in every pyiqa release this pack has supported.
    Nothing is known about their mode, direction or range, so the variants say
    so and the score nodes stop refusing mismatches (see ``_require_compatible``)
    rather than refusing everything. Better a working loader with honest labels
    than an empty node menu and one line in the console.
    """
    if REGISTRY_ERROR:
        logger.error(
            "(pyiqa-toolbox) cannot read pyiqa's metric table (%s). Falling back "
            "to %d built-in metric names: the nodes still work, but FR/NR mode, "
            "direction and score range are unknown until pyiqa is fixed - please "
            "report the pyiqa version in use.",
            REGISTRY_ERROR,
            len(_FALLBACK_METRICS),
        )
    return {
        "Unclassified": Architecture(
            label="Unclassified",
            type="Unclassified",
            variants=tuple(_FALLBACK_METRICS),
            folders=("Unclassified",),
            family="any",
        )
    }


def _variant_signature(name):
    """What makes two registry names the same weights: opts, mode, direction.

    pyiqa ships aliases (``qrealign`` and ``qrealign-mini`` both build
    ``{'type': 'QReAlign', 'model': 'mini'}``), and two dropdown entries for one
    set of weights looks like a bug because it is one. Direction and mode are
    part of the signature so genuinely different variants that happen to share
    ``metric_opts`` - ``fgresq`` (NR) and ``fgresq_pair`` (FR) - are kept.
    """
    cfg = _CONFIGS.get(name) or {}
    opts = tuple(
        sorted(
            (str(key), repr(value))
            for key, value in (cfg.get("metric_opts") or {}).items()
        )
    )
    return (opts, cfg.get("metric_mode"), bool(cfg.get("lower_better")))


def _build():
    if not _CONFIGS:
        return _fallback_architectures()

    registered = _registered_names()
    groups = {}
    skipped = []
    aliases = []
    for name, cfg in _CONFIGS.items():
        try:
            if registered and name not in registered:
                continue
            # .get everywhere: a pyiqa entry that lost or gained a key lands in
            # "Unclassified" instead of taking the whole node pack down
            type_name = (cfg.get("metric_opts") or {}).get("type") or "Unclassified"
            names = groups.setdefault(type_name, [])
            if name not in SPECIAL_METRICS:
                signature = _variant_signature(name)
                twin = next(
                    (n for n in names if _variant_signature(n) == signature), None
                )
                if twin is not None:
                    aliases.append((name, twin))
                    continue
            names.append(name)
        except Exception as exc:
            skipped.append(f"{name} ({type(exc).__name__})")
    if skipped:
        logger.warning(
            "(pyiqa-toolbox) %d metric entry/entries in pyiqa's config table "
            "could not be read and are not listed: %s",
            len(skipped),
            ", ".join(skipped),
        )
    if aliases:
        logger.info(
            "(pyiqa-toolbox) %d pyiqa name(s) build the same weights as another "
            "entry; listing the first only: %s",
            len(aliases),
            ", ".join(f"{alias} = {twin}" for alias, twin in aliases),
        )
    global ALIASES
    ALIASES = dict(aliases)

    architectures = {}
    for type_name, names in groups.items():
        rule = _arch_rule(type_name, names)
        architectures[rule.label] = Architecture(
            label=rule.label,
            type=type_name,
            variants=tuple(names),
            folders=rule.folders or (rule.label,),
            family=_family_of_arch(rule.label),
            policy=rule.policy,
            snapshot_loader=rule.snapshot_loader,
            single_image=rule.single_image,
            single_image_metrics=rule.single_image_metrics,
            signpost=rule.signpost,
        )

    listed = {n for names in groups.values() for n in names}
    # a collapsed alias is a KNOWN name (it builds the listed entry's weights);
    # reporting it as unknown would be a lie and would point at the wrong file
    unlisted = sorted(registered - listed - set(SPECIAL_METRICS) - set(ALIASES))
    if unlisted:
        logger.warning(
            "(pyiqa-toolbox) metric(s) %s not in DEFAULT_CONFIGS; listing them "
            "under 'Unlisted' with their raw name",
            ", ".join(repr(n) for n in unlisted),
        )
        architectures["Unlisted"] = Architecture(
            label="Unlisted",
            type="Unlisted",
            variants=tuple(unlisted),
            folders=("Unlisted",),
            family="any",
        )
    if not architectures:
        return _fallback_architectures()
    return architectures


def _safe_build():
    """``_build()`` can never take the node pack down."""
    try:
        return _build()
    except Exception as exc:
        logger.error(
            "(pyiqa-toolbox) building the architecture list failed (%s: %s); "
            "falling back to the built-in metric list",
            type(exc).__name__,
            exc,
        )
        return _fallback_architectures()


# pyiqa names that build the SAME weights as another entry (see _build): alias ->
# the entry that is listed. Kept so the "unlisted metrics" warning does not
# report a collapsed alias as an unknown metric, and so the document can say
# which name was folded away. Declared BEFORE the build, which fills it in.
ALIASES = {}

ARCHITECTURES = _safe_build()
ARCHES = sorted(ARCHITECTURES, key=str.lower)

# The custom-checkpoint half of the tables is rescanned on demand (see
# refresh_customs): a checkpoint dropped into a {architecture}/ folder must show
# up on a ComfyUI *refresh* - no restart - which is what the docs promise. The
# registry half never changes (it is pyiqa's own DEFAULT_CONFIGS).
_ARCH_ENTRIES = {}
VARIANT_VALUES = []
VARIANT_LOOKUP = {}

_REFRESH_TTL = 1.0  # seconds; keeps a burst of resolves from rescanning per call
_last_refresh = 0.0


def refresh_customs(force=False):
    """Rescan the ``{architecture}/`` folders and rebuild the derived tables.

    Cheap (a couple of ``listdir`` calls per architecture) and idempotent, so
    every public lookup calls it: that is what makes a newly dropped checkpoint
    visible without restarting ComfyUI. Registry variants are recreated too, but
    from static data.
    """
    global _ARCH_ENTRIES, VARIANT_VALUES, VARIANT_LOOKUP, _last_refresh
    now = time.monotonic()
    if not force and _ARCH_ENTRIES and now - _last_refresh < _REFRESH_TTL:
        return
    _last_refresh = now
    _ARCH_ENTRIES = {
        label: _flag_special(_unique(_entries_for(ARCHITECTURES[label])))
        for label in ARCHES
    }
    # arch-prefixed dropdown values for the generic loader. The prefix is the
    # filter key the browser extension matches on (and what keeps the flat list
    # self-describing for headless/API use, where nothing filters it).
    VARIANT_VALUES = [
        f"{arch}{SEPARATOR}{label}"
        for arch in ARCHES
        for label, _resolved in _ARCH_ENTRIES[arch]
    ]
    VARIANT_LOOKUP = {
        f"{arch}{SEPARATOR}{label}": resolved
        for arch in ARCHES
        for label, resolved in _ARCH_ENTRIES[arch]
    }


refresh_customs(force=True)


DEFAULT_ARCH = "LPIPS"
DEFAULT_VARIANT = next(
    v for v in VARIANT_VALUES if v.startswith(DEFAULT_ARCH + SEPARATOR)
)


def metric_opts(name):
    """pyiqa's own construction options for a registry metric, or ``{}``.

    Read from the same config table that supplies mode, direction and score range
    (``DEFAULT_CONFIGS[name]["metric_opts"]``, e.g. ``{"type": "QAlign",
    "dtype": "4bit"}``). The per-architecture pre-load gates need them: Q-Align's
    precision is decided there rather than in the variant label.
    """
    opts = (_CONFIGS.get(name) or {}).get("metric_opts") or {}
    return opts if isinstance(opts, dict) else {}


def family_of(arch):
    """Family key owning an architecture label ("" when unknown)."""
    arch_obj = ARCHITECTURES.get(arch)
    return arch_obj.family if arch_obj else ""


def settle_variant(value, arch):
    """Strip an optional ``"<arch> | "`` prefix from a variant value."""
    if value and SEPARATOR in value:
        prefix, bare = value.split(SEPARATOR, 1)
        if prefix in ARCHITECTURES:
            if prefix != arch:
                raise ValueError(
                    f"variant '{value}' belongs to '{prefix}', but this loader "
                    f"is set to '{arch}'. The variant dropdown filters to the "
                    "selected architecture."
                )
            return bare
    return value


def resolve(arch, variant=None):
    """(architecture label, variant value) -> Resolved.

    Tolerates a raw registry name, a bare variant label, a snapshot/custom
    label, or a missing variant (the architecture's first entry). Raises
    ValueError with a fix-it hint on anything else.
    """
    refresh_customs()
    arch_obj = ARCHITECTURES.get(arch)
    if arch_obj is None:
        raise ValueError(
            f"unknown architecture '{arch}' — choose one of: {', '.join(ARCHES)}"
        )
    if arch_obj.signpost:
        raise ValueError(f"{arch_obj.label} {arch_obj.signpost}")
    entries = dict(_ARCH_ENTRIES[arch])
    if variant is None:
        name = arch_obj.variants[0] if arch_obj.variants else ""
        return resolve(arch, name)
    bare = settle_variant(variant, arch)
    if bare in entries:
        return entries[bare]
    if bare in arch_obj.variants:  # raw registry name
        return _resolved_registry(arch_obj, bare)
    for label, resolved in _ARCH_ENTRIES[arch]:
        if resolved.base == bare and not resolved.custom:
            return resolved
    raise ValueError(
        f"unknown variant '{variant}' for architecture '{arch}' — pick a "
        "value from the variant dropdown"
    )


def family_arches(family):
    return tuple(
        label for label in FAMILIES.get(family, ()) if label in ARCHITECTURES
    )


def family_custom_dirs(family):
    """Existing ``{architecture}/`` folders of a family, in search order.

    The dedicated loaders that keep their own variant scheme (Q-ReAlign's
    snapshot models, FGResQ's two-checkpoint bundle) read their customs from
    here, so they follow the same folder convention as everything else.
    """
    found = []
    for label in FAMILIES.get(family, ()):
        arch = ARCHITECTURES.get(label)
        if arch is None:
            continue
        for _folder, entry, path in pyiqa_utils.iter_user_pyiqa_dirs():
            if os.path.isdir(path) and _normalize(entry) in arch.folder_keys:
                found.append(path)
    return found


def custom_roots():
    """The pyiqa folders the ``{architecture}/`` drop folders are created in.

    The install-local ``models/pyiqa`` always, plus every yaml-registered pyiqa
    folder that is not a download cache: pointing the pack at
    ``~/.cache/huggingface/hub`` (a documented way to reuse pyiqa's own
    downloads) must not litter that cache with empty architecture folders, and
    hand-placed fine-tunes never live there anyway.
    """
    roots = []
    if pyiqa_utils._pyiqa_folder_paths():  # outside ComfyUI there is nowhere real
        roots.append(pyiqa_utils.local_pyiqa_dir())
    seen = {os.path.normcase(os.path.abspath(r)) for r in roots}
    # note: the FOLDERS themselves, not iter_user_pyiqa_dirs()' entries
    for folder in pyiqa_utils._user_pyiqa_dirs():
        if not os.path.isdir(folder):
            continue
        norm = os.path.normcase(os.path.abspath(folder))
        if ".cache" in norm and ("huggingface" in norm or "torch" in norm):
            continue
        if norm not in seen:
            seen.add(norm)
            roots.append(folder)
    return roots


def scaffold_folders():
    """Create the ``{architecture}/`` drop folders in the user's pyiqa folders.

    Only for families with a loader that takes custom weights — a folder is an
    invitation to put a checkpoint in it. Existing folders are left alone.
    """
    created = []
    for root in custom_roots():
        for family in SCAFFOLD_FAMILIES:
            for label in family_arches(family):
                folder = os.path.join(root, ARCHITECTURES[label].folders[0])
                try:
                    os.makedirs(folder, exist_ok=True)
                    created.append(folder)
                except OSError as exc:
                    logger.debug("could not create %s: %s", folder, exc)
    return created
