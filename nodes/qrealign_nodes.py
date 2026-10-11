"""Q-ReAlign nodes (NR quality scoring on a Qwen3.5-VL backbone).

Faithful to pyiqa: the loader wraps ``pyiqa.create_metric('qrealign', model=...)``
and the score node just calls the returned InferenceModel, which routes through
the arch's own Qwen processor. Reference: Wu et al., "Q-Align: Teaching LMMs for
Visual Scoring via Discrete Text-Defined Levels", arXiv:2312.17009.

Weights are looked up in the ComfyUI ``models/pyiqa`` folder first (each
variant as an HF-snapshot-style subfolder, e.g. ``Q-ReAlign-Pro-9B/``), falling
back to the q-future HuggingFace repos when absent. The folder is redirectable
via an ``pyiqa:`` key in extra_model_paths.yaml.
"""

import glob
import logging
import os

import pyiqa

from ._common import (
    with_memory_management,
    MEMORY_MANAGEMENT_INPUT,
    LAZY_MODEL_OPT,
    LazyScoreMixin,
    lazy_image_opt,
)

from ..utility.pyiqa_utils import (
    comfy_image_to_pyiqa,
    default_device,
    ensure_on_device,
    find_pyiqa_model,
    get_cached_metric,
    list_pyiqa_checkpoints,
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
    # snapshot-style dirs (config.json present); skip names shadowing built-ins
    # by label (case-insensitive), built-ins' own resolved paths, and this
    # pack's int8_convrot folders (those are selected via the quantization
    # combo, not as customs).
    builtin_labels = {label.lower() for label in QREALIGN_VARIANTS}
    builtin_folders = {folder.lower() for _, _, folder in QREALIGN_VARIANTS.values()}
    return [
        (label, path)
        for label, path in list_pyiqa_checkpoints(
            require_config=True, exclude_paths=_builtin_resolved_paths()
        )
        if label.lower() not in builtin_labels
        and label.lower() not in builtin_folders
        and not label.lower().endswith(INT8_SUFFIX.lower())
    ]


def _resolve_model_arg(label):
    """Built-ins resolve through the folder/cache chain; custom labels are
    snapshot-style dirs straight from models/pyiqa."""
    customs = dict(_custom_options())
    if label in customs:
        return customs[label]
    _, repo, folder = QREALIGN_VARIANTS[label]
    local = find_pyiqa_model(folder)
    return local if local is not None else repo


INT8_QUANT = "int8_convrot"
INT8_SUFFIX = "-int8-convrot"
DEFAULT_INT8_REPO = "AnsteinHuynh/Q-ReAlign-int8-convrot"


def _int8_complete(path):
    return bool(
        path
        and os.path.isdir(path)
        and os.path.isfile(os.path.join(path, "config.json"))
        and glob.glob(os.path.join(path, "*.safetensors"))
    )


def _resolve_int8_dir(label, repo):
    """Local int8_convrot copy first; else download it from ``repo`` (a
    HuggingFace repo holding one folder per variant, produced by this pack's
    tools/quantize_qrealign_int8_convrot.py)."""
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
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            f"no local {name} and huggingface_hub is not installed ({exc})"
        ) from exc
    target = os.path.join(local_pyiqa_dir(), name)
    logger.info(
        "(pyiqa-toolbox) %s not local; downloading from %s into %s",
        name, repo, target,
    )
    path = snapshot_download(repo_id=repo, local_dir=target)
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
        "mini-0.8B / lite-4B / pro-9B built-ins plus custom snapshot folders "
        "from models/pyiqa. The int8_convrot option loads the ConvRot-rotated "
        "int8 quants (about a third less size/VRAM; produced by this pack's "
        "quantizer and downloadable from your HuggingFace repo)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    list(QREALIGN_VARIANTS.keys())
                    + [label for label, _ in _custom_options()],
                    {
                        "default": "qrealign-mini-0.8B",
                        "tooltip": (
                            "Mini 0.8B (fast) / Lite 4B / Pro 9B (best correlation, slowest), "
                            "plus any custom snapshot-style folder in models/pyiqa (dir with a "
                            "config.json, e.g. your own Qwen3.5-VL quality fine-tune). Built-ins "
                            "come from the pyiqa folders/HF cache; absent weights auto-download "
                            "from q-future on first use."
                        ),
                    },
                ),
                "dtype": (
                    ["auto", "bfloat16", "float16", "float32"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "torch dtype passed to from_pretrained. 'auto' follows the checkpoint "
                            "(bfloat16). Ignored when quantization is int8_convrot (weights are "
                            "already int8 with bf16 compute)."
                        ),
                    },
                ),
                "quantization": (
                    ["none (native)", INT8_QUANT],
                    {
                        "default": "none (native)",
                        "tooltip": (
                            "none (native) = the original bf16 weights. int8_convrot = ComfyUI's "
                            "native int8_tensorwise + ConvRot format (group-256 Hadamard rotation; "
                            "same recipe as Comfy-Org's official int8_convrot text encoders): about "
                            "a third less download/disk/VRAM, small score drift vs bf16. Resolved from "
                            "models/pyiqa first, else downloaded from the repo below. Note: an "
                            "int8 model always stays on the GPU (the score node's "
                            "'free_after_score' is a no-op for it)."
                        ),
                    },
                ),
                "int8_hf_repo": (
                    "STRING",
                    {
                        "default": DEFAULT_INT8_REPO,
                        "tooltip": (
                            "HuggingFace repo holding the int8_convrot variants as folders "
                            "(Q-ReAlign-Mini-0.8B-int8-convrot/ etc.), one folder per variant at "
                            "the repo root. Only used when quantization is int8_convrot and no "
                            "local copy exists."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("PYIQA_QREALIGN",)
    RETURN_NAMES = ("model",)
    OUTPUT_TOOLTIPS = (
        "The loaded Q-ReAlign scorer. Wire it into the QReAlign NR Score node.",
    )
    FREE_COMFY_MEMORY = True  # pro-9B is huge; free the SDXL cache first
    FUNCTION = "load"

    def load(self, model_name, dtype, quantization, int8_hf_repo):
        device = default_device()

        if quantization == INT8_QUANT:
            int8_dir = _resolve_int8_dir(model_name, int8_hf_repo)

            def factory():
                make_room_for_metric(device)
                from ..utility import qrealign_int8

                return qrealign_int8.create_int8_metric(int8_dir, device)

            metric = get_cached_metric(
                "qrealign", ("int8", int8_dir, str(device)), factory
            )
            ensure_on_device(metric, device)
            return (metric,)

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
        return (metric,)


class QReAlignScore(LazyScoreMixin):
    """No-reference Q-ReAlign score. Higher is better, range 0-1.

    ``score`` is a per-image list (OUTPUT_IS_LIST): each image's score flows on
    as its own value downstream, so scalar consumers run once per image.
    """

    CATEGORY = "pyiqa/QReAlign"
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
                "model": ("PYIQA_QREALIGN", dict(LAZY_MODEL_OPT)),
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
        dev = ensure_on_device(model, default_device())
        x = comfy_image_to_pyiqa(image, dev)
        out = model(x, task_=TASKS[task])
        return ([float(v) for v in out],)
