"""Q-ReAlign nodes (NR quality scoring on a Qwen3.5-VL backbone).

Faithful to pyiqa: the loader wraps ``pyiqa.create_metric('qrealign', model=...)``
and the score node just calls the returned InferenceModel, which routes through
the arch's own Qwen processor. Reference: Wu et al., "Q-Align: Teaching LMMs for
Visual Scoring via Discrete Text-Defined Levels", arXiv:2312.17009.

Weights are looked up in the ComfyUI ``models/pyiqa`` folder first (each
variant as an HF-snapshot-style subfolder, e.g. ``Q-ReAlign-Pro-9B/``, in the
``Q-ReAlign/`` architecture folder or loose), falling back to the q-future
HuggingFace repos when absent. The folder is redirectable via an ``pyiqa:`` key
in extra_model_paths.yaml.

This module also owns how the Q-ReAlign architecture is *built*, for every node
in the pack: it registers a builder with ``nodes._common``, so "Load pyiqa
Model" with architecture=Q-ReAlign produces exactly what this family's loader
does (the transformers>=5 check, the built-in size variants resolved through the
local-folder/repo chain, custom snapshot folders, and the int8_convrot build
for a directory named that way).
"""

import glob
import logging
import os

import pyiqa

from ._common import (
    with_memory_management,
    LAZY_MODEL_OPT,
    LazyScoreMixin,
    MEMORY_MANAGEMENT_INPUT,
    ModelValidationMixin,
    lazy_image_opt,
    metric_for_mode,
    register_arch_builder,
)

from ..utility import metric_catalog

from ..utility.pyiqa_utils import (
    comfy_image_to_pyiqa,
    default_device,
    ensure_on_device,
    find_pyiqa_model,
    get_cached_metric,
    list_snapshot_dirs,
    local_pyiqa_dir,
    make_room_for_metric,
)

logger = logging.getLogger(__name__)

# dropdown label -> (arch model key, HF repo, models/pyiqa subfolder)
QREALIGN_VARIANTS = {
    "qrealign-mini-0.8B": ("mini", "q-future/Q-ReAlign-Mini-0.8B", "Q-ReAlign-Mini-0.8B"),
    "qrealign-lite-4B": ("lite", "q-future/Q-ReAlign-Lite-4B", "Q-ReAlign-Lite-4B"),
    "qrealign-pro-9B": ("pro", "q-future/Q-ReAlign-Pro-9B", "Q-ReAlign-Pro-9B"),
}

# dropdown label -> arch task key
TASKS = {
    "quality IQA": "quality",
    "aesthetic IAA": "aesthetic",
}


def _require_transformers_v5():
    """Q-ReAlign (Qwen3.5-VL) needs transformers>=5. Deliberately NOT pinned in
    requirements.txt - forcing that upgrade on every user for one family can
    break other packs - so check here and fail with the fix in hand."""
    import transformers

    major = int(transformers.__version__.split(".")[0])
    if major < 5:
        raise RuntimeError(
            f"Q-ReAlign needs transformers>=5.0 (found {transformers.__version__}). "
            "Install it into ComfyUI's python with: pip install -U \"transformers>=5\" "
            "- everything else in this pack works without it."
        )


def _builtin_resolved_paths():
    """Resolved local paths of the built-in variants (whatever the search chain
    finds); used to drop them from the custom-dir scan so a broad search path
    (e.g. X:/Models) that holds a built-in's folder does not list it twice."""
    paths = []
    for _, _, folder in QREALIGN_VARIANTS.values():
        resolved = find_pyiqa_model(folder)
        if resolved is not None and os.path.exists(resolved):
            paths.append(resolved)
    return paths


def _custom_options():
    """Custom snapshot models: the ``models/pyiqa/Q-ReAlign/`` folder.

    A custom model is a snapshot-style directory (``config.json`` + weights) —
    the layout Q-ReAlign's ``from_pretrained`` needs. Directories that are one
    of this pack's built-ins are skipped, so the same weights are not offered
    twice: the ``Q-ReAlign-<size>[-int8-convrot]`` download targets resolve to
    the loader's own built-in entries.
    """
    builtin_labels = {label.lower() for label in QREALIGN_VARIANTS}
    builtin_folders = {folder.lower() for _, _, folder in QREALIGN_VARIANTS.values()}
    # this pack's own int8_convrot download targets are built-ins as well (they
    # live at the pyiqa root, but a user may have copied one in here); a user's
    # OWN int8 build is a perfectly good custom and is NOT skipped
    builtin_folders |= {
        f"{folder}{INT8_SUFFIX}".lower()
        for _, _, folder in QREALIGN_VARIANTS.values()
    }
    builtin_paths = {
        os.path.normcase(os.path.abspath(p)) for p in _builtin_resolved_paths()
    }
    found = []
    for directory in metric_catalog.family_custom_dirs("qrealign"):
        for label, path in list_snapshot_dirs(directory):
            if os.path.normcase(os.path.abspath(path)) in builtin_paths:
                continue
            if label.lower() in builtin_labels or label.lower() in builtin_folders:
                continue
            found.append((label, path))
    return found


def _resolve_model_arg(label):
    """Built-ins resolve through the folder/cache chain; custom labels are
    snapshot-style dirs straight from models/pyiqa."""
    customs = dict(_custom_options())
    if label in customs:
        return customs[label]
    _, repo, folder = QREALIGN_VARIANTS[label]
    local = find_pyiqa_model(folder)
    return local if local is not None else repo


INT8_SUFFIX = "-int8-convrot"
INT8_LABEL_SUFFIX = " (int8_convrot)"
DEFAULT_INT8_REPO = "AnsteinHuynh/Q-ReAlign-int8-convrot"


def _int8_repo():
    """int8 download repo: the default release repo, env-overridable."""
    return os.environ.get("PYIQA_TOOLBOX_INT8_REPO", DEFAULT_INT8_REPO)


def _int8_variant_base(label):
    """Dropdown label -> built-in variant label if it is an int8 entry."""
    if label.endswith(INT8_LABEL_SUFFIX):
        base = label[: -len(INT8_LABEL_SUFFIX)]
        if base in QREALIGN_VARIANTS:
            return base
    return None


# pyiqa registry name <-> this family's dropdown variant (pyiqa registers the
# sizes as qrealign-mini/-lite/-pro, the HF repos carry the parameter counts)
_VARIANT_TO_REGISTRY = {
    "qrealign-mini-0.8B": "qrealign-mini",
    "qrealign-lite-4B": "qrealign-lite",
    "qrealign-pro-9B": "qrealign-pro",
}
_REGISTRY_TO_VARIANT = {name: label for label, name in _VARIANT_TO_REGISTRY.items()}


def _registry_name(label):
    """Dropdown label -> pyiqa registry name (for the model_name output)."""
    base = _int8_variant_base(label)
    if base is not None:
        return _VARIANT_TO_REGISTRY[base]
    if label in _VARIANT_TO_REGISTRY:
        return _VARIANT_TO_REGISTRY[label]
    return "qrealign"  # a custom snapshot folder: any size, arch-level name


def build_qrealign(resolved, device):
    """Build a Q-ReAlign metric for ANY loader in this pack.

    Registered as the architecture's builder, so "(Down)load pyiqa Model" with
    architecture=Q-ReAlign goes through exactly this code: the transformers>=5
    check, the built-in sizes resolved through the local-folder/repo chain, and
    custom snapshot directories from models/pyiqa/Q-ReAlign/ (through the
    int8_convrot loader when the directory name says so, else a plain
    ``from_pretrained`` on the folder).
    """
    _require_transformers_v5()
    make_room_for_metric(device)  # 0.8B..9B, int8 or not: always make room
    if resolved.kind == "snapshot":
        if resolved.path.lower().endswith(INT8_SUFFIX.lower()):
            from ..utility import qrealign_int8

            return qrealign_int8.create_int8_metric(resolved.path, device)
        return pyiqa.create_metric("qrealign", device=device, model=resolved.path)
    variant = _REGISTRY_TO_VARIANT.get(resolved.base, "qrealign-mini-0.8B")
    return pyiqa.create_metric(
        "qrealign", device=device, dtype="auto", model=_resolve_model_arg(variant)
    )


register_arch_builder("Q-ReAlign", build_qrealign)


def _int8_complete(path):
    return bool(
        path
        and os.path.isdir(path)
        and os.path.isfile(os.path.join(path, "config.json"))
        and glob.glob(os.path.join(path, "*.safetensors"))
    )


def _int8_repo_folder_size(repo, folder):
    """Human-readable size of ``folder`` in ``repo`` for the download log
    (first run can be 1.5-11.5 GiB - worth announcing). None if unknown."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo, files_metadata=True)
        total = sum(
            (s.size or 0) for s in (info.siblings or []) if s.rfilename.startswith(folder + "/")
        )
        if total > 0:
            return f"{total / 2**30:.1f} GiB, one-time"
    except Exception:
        pass
    return None


def _resolve_int8_dir(label, repo):
    """Local int8_convrot copy first; else download it from ``repo`` (a
    HuggingFace repo holding one folder per variant, produced by this pack's
    tools/quantize_qrealign_int8_convrot.py).

    The repo carries ALL variants at its top level, so the download is
    scoped with ``allow_patterns`` to this variant's folder only and lands
    as ``models/pyiqa/<variant>-int8-convrot/<files>`` - the flat layout the
    search chain and the completeness check expect. (An older unscoped
    download nested the whole repo inside that folder; such leftovers are
    flagged, never read.)"""
    if label not in QREALIGN_VARIANTS:
        raise ValueError(
            "int8_convrot is wired for the built-in variants only; quantize a "
            "custom fine-tune with tools/quantize_qrealign_int8_convrot.py and "
            "drop the folder in models/pyiqa to use it here."
        )
    _, _, folder = QREALIGN_VARIANTS[label]
    name = f"{folder}-int8-convrot"
    local = find_pyiqa_model(name)
    if _int8_complete(local):
        return local
    if local:
        # stale full-repo download (bug): variant folders NESTED inside
        # models/pyiqa/<name>/ are not the expected layout and are never
        # read - point the user at the reclaimable space
        stale = sorted(glob.glob(os.path.join(local, "Q-ReAlign-*-int8-convrot")))
        if stale:
            logger.warning(
                "(pyiqa-toolbox) %s holds leftover folders from an older "
                "full-repo download: %s - unused, delete them to reclaim "
                "the space",
                local,
                ", ".join(os.path.basename(s) for s in stale),
            )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            f"no local {name} and huggingface_hub is not installed ({exc})"
        ) from exc
    root = local_pyiqa_dir()
    size_note = _int8_repo_folder_size(repo, name)
    logger.info(
        "(pyiqa-toolbox) %s not local; downloading %s/ from %s into %s%s",
        name, name, repo, root, f" ({size_note})" if size_note else "",
    )
    snapshot_download(
        repo_id=repo,
        local_dir=root,
        allow_patterns=[f"{name}/**"],
    )
    path = os.path.join(root, name)
    if not _int8_complete(path):
        raise RuntimeError(
            f"{repo} does not contain a usable {name} folder (need config.json "
            f"+ safetensors). Run tools/quantize_qrealign_int8_convrot.py and "
            f"upload its output folder to the repo."
        )
    return path


class QReAlignLoader:
    """Loads a Q-ReAlign scorer via pyiqa.create_metric."""

    CATEGORY = "pyiqa/QReAlign"
    DESCRIPTION = (
        "Loads a Q-ReAlign scorer (Qwen3.5-VL backbone, Q-Align recipe): "
        "mini-0.8B / lite-4B / pro-9B built-ins, each also available as an "
        "int8_convrot build (a third less size/VRAM, produced by this pack's "
        "quantizer and downloadable from a HuggingFace repo), plus custom "
        "snapshot folders from models/pyiqa. Needs transformers>=5.0 - checked "
        "at load time with a clear error (not forced in requirements.txt, so "
        "other packs are not upgraded behind your back)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    [k + INT8_LABEL_SUFFIX for k in QREALIGN_VARIANTS]
                    + list(QREALIGN_VARIANTS.keys())
                    + [label for label, _ in _custom_options()],
                    {
                        "default": "qrealign-mini-0.8B",
                        "tooltip": (
                            "The '(int8_convrot)' entries (listed first) load the "
                            "quantized builds - same pick, a third less size/VRAM, "
                            "tiny score drift. The plain entries are Mini 0.8B "
                            "(fast) / Lite 4B / Pro 9B (best correlation, slowest). "
                            "Plus any custom snapshot folder in "
                            "models/pyiqa/Q-ReAlign/ (a directory with a "
                            "config.json, e.g. your own Qwen3.5-VL quality "
                            "fine-tune; the directory name is the entry name). "
                            "Built-ins come from the pyiqa "
                            "folders/HF cache; absent weights auto-download on "
                            "first use."
                        ),
                    },
                ),
                "dtype": (
                    ["auto", "bfloat16", "float16", "float32"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "torch dtype passed to from_pretrained. 'auto' follows the checkpoint "
                            "(bfloat16). Ignored for the (int8_convrot) entries (weights are "
                            "already int8 with bf16 compute)."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("PYIQA_QREALIGN", "STRING", "STRING")
    RETURN_NAMES = ("model", "model_name", "variant_label")
    OUTPUT_TOOLTIPS = (
        "The loaded Q-ReAlign scorer. Wire it into the QReAlign NR Score node.",
        "The pyiqa registry name this build corresponds to, e.g. "
        "'qrealign-mini' (or 'qrealign' for a custom snapshot folder).",
        "The variant as shown in the dropdown, e.g. 'qrealign-mini-0.8B "
        "(int8_convrot)'.",
    )
    FREE_COMFY_MEMORY = True  # pro-9B is huge; free the SDXL cache first
    FUNCTION = "load"

    def load(self, model_name, dtype):
        device = default_device()
        _require_transformers_v5()
        int8_base = _int8_variant_base(model_name)

        if int8_base is not None:
            # the download repo is not a node input on purpose - two extra
            # combos on a loader scare people off; override with the
            # PYIQA_TOOLBOX_INT8_REPO env var if you fork the weights
            int8_dir = _resolve_int8_dir(int8_base, _int8_repo())

            def factory():
                make_room_for_metric(device)
                from ..utility import qrealign_int8

                return qrealign_int8.create_int8_metric(int8_dir, device)

            metric = get_cached_metric(
                "qrealign", ("int8", int8_dir, str(device)), factory
            )
            ensure_on_device(metric, device)
            return (metric, _registry_name(model_name), model_name)

        model_arg = _resolve_model_arg(model_name)

        def factory():
            make_room_for_metric(device)
            return pyiqa.create_metric(
                "qrealign", device=device, dtype=dtype, model=model_arg
            )

        metric = get_cached_metric(
            "qrealign", (model_arg, dtype, str(device)), factory
        )
        ensure_on_device(metric, device)
        return (metric, _registry_name(model_name), model_name)


class QReAlignScore(ModelValidationMixin, LazyScoreMixin):
    """No-reference Q-ReAlign score. Higher is better, range 0-1.

    ``score`` is a per-image list (OUTPUT_IS_LIST): each image's score flows on
    as its own value downstream, so scalar consumers run once per image.
    """

    CATEGORY = "pyiqa/QReAlign"
    MODEL_TYPE = "PYIQA_QREALIGN"
    FAMILY = "qrealign"
    SCORE_LABEL = "QReAlign NR Score"
    # Q-ReAlign metrics are NR. The dedicated loader hands over pyiqa's metric
    # object itself (whose mode reads as unknown, so the check stays silent),
    # while "(Down)load pyiqa Model" hands over a bundle that DOES know its mode -
    # without this line that path was refused with "scores None-reference
    # models only".
    MODE_CHECK = "NR"
    DESCRIPTION = (
        "No-reference Q-ReAlign quality (or aesthetic) score. 0-1, higher is "
        "better. Score is per-image. Lazy: the model loads only once the image "
        "exists."
    )
    LAZY_INPUTS = ("image", "model")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "memory_management": MEMORY_MANAGEMENT_INPUT,
                "model": (cls._model_input_type(), dict(LAZY_MODEL_OPT)),
                "image": (
                    "IMAGE",
                    lazy_image_opt(
                        "Image(s) to score; batch is scored per-image. Lazy: "
                        "evaluated before the model loads."
                    ),
                ),
                "task": (
                    list(TASKS.keys()),
                    {
                        "default": "quality IQA",
                        "tooltip": "Scoring task: quality IQA or aesthetic IAA. Passed as the arch's per-call task_ override.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("score",)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_TOOLTIPS = (
        "Per-image Q-ReAlign score in 0-1 for the chosen task (quality IQA or "
        "aesthetic IAA); higher is better. Computed from the model's level-word "
        "probabilities (excellent..bad weighted 1.0..0.0).",
    )
    FUNCTION = "score"

    @with_memory_management
    def score(self, model=None, image=None, task="quality IQA", **kwargs):
        if self._skipped(model, image):
            return ([],)
        self._require_compatible(model)
        # the family loader hands over the InferenceModel itself, "Load pyiqa
        # Model" hands over a MetricBundle wrapping it
        metric = metric_for_mode(model, self.MODE_CHECK)
        dev = ensure_on_device(model, default_device())
        x = comfy_image_to_pyiqa(image, dev)
        out = metric(x, task_=TASKS[task])
        return ([float(v) for v in out],)
