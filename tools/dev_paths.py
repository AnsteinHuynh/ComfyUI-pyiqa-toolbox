"""Paths for the pack's dev-only tools - no machine baked into the repo.

The tools in this folder move gigabytes of weights and score real image folders,
so they inevitably point at places only their author has. Those places are
resolved here and nowhere else, in this order:

1. the environment (works for anyone, and for CI):

   ======================  ====================================================
   ``PYIQA_TOOLBOX_ROOT``  the pack checkout (default: this file's parent)
   ``PYIQA_TOOLBOX_COMFY`` a ComfyUI install (for tools importing ``comfy*``)
   ``PYIQA_TOOLBOX_MODELS`` a pyiqa weights folder, e.g. ``<ComfyUI>/models/pyiqa``
   ``PYIQA_TOOLBOX_IMAGES`` a folder of images, or a glob, to score
   ``PYIQA_TOOLBOX_HF_CACHE`` HuggingFace home dir (default: ``<MODELS>/hf-cache``)
   ======================  ====================================================

2. ``dev_paths_local.py`` beside this file, if it exists. It is GITIGNORED, so a
   contributor's machine paths never enter the published repo: define ``ROOT``,
   ``COMFY``, ``MODELS``, ``IMAGES`` or ``HF_CACHE`` there to make the bare
   commands work again.

The ``require``-style helpers exit with the variable to set, instead of letting a
tool fail later with an empty image list or a checkpoint that is not there.
"""

import glob
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

try:  # machine-local defaults; gitignored, so it contributes nothing upstream
    import dev_paths_local as _local
except ImportError:  # normal for everyone but the author
    _local = None


def _value(name):
    from_env = os.environ.get("PYIQA_TOOLBOX_" + name)
    if from_env:
        return from_env
    return getattr(_local, name, None) if _local else None


ROOT = _value("ROOT") or os.path.dirname(_HERE)
COMFY = _value("COMFY")
MODELS = _value("MODELS")
IMAGES = _value("IMAGES")
HF_CACHE = _value("HF_CACHE")


def _require(value, name, hint):
    if value:
        return value
    raise SystemExit(
        f"dev_paths: {name} is not set.\n"
        f"  set PYIQA_TOOLBOX_{name} to {hint}\n"
        f"  (or put {name} = r'...' in tools/dev_paths_local.py)"
    )


def comfy_root():
    """A ComfyUI install, for tools that import ``comfy_kitchen``/``comfy``."""
    return _require(COMFY, "COMFY", "the ComfyUI folder that holds main.py")


def models_dir():
    """A pyiqa weights folder (``Q-ReAlign-Pro-9B``, ``FGResQ``, ... live here)."""
    return _require(MODELS, "MODELS", "a folder such as <ComfyUI>/models/pyiqa")


def hf_cache():
    """The HuggingFace *home* dir these tools point HF at."""
    return HF_CACHE or os.path.join(models_dir(), "hf-cache")


def hf_hub():
    """The HuggingFace hub cache that holds downloaded snapshots."""
    explicit = os.environ.get("HF_HUB_CACHE")
    if explicit:
        return explicit
    home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    return os.path.join(home, "hub")


def site_packages():
    """Where this interpreter keeps its packages (pyiqa ships test fixtures)."""
    import sysconfig

    return sysconfig.get_paths()["purelib"]


def image_paths(limit=None):
    """Sample images to score, sorted; ``IMAGES`` may be a folder or a glob."""
    source = _require(IMAGES, "IMAGES", "a folder of images, or a glob such as .../*.png")
    pattern = source if any(c in source for c in "*?[") else os.path.join(source, "*")
    paths = sorted(p for p in glob.glob(pattern) if os.path.isfile(p))
    if not paths:
        raise SystemExit(f"dev_paths: PYIQA_TOOLBOX_IMAGES={source!r} matched no files")
    return paths[:limit] if limit else paths


def bootstrap(*, comfy=False):
    """Put the pack (and optionally a ComfyUI install) on ``sys.path``."""
    if comfy:
        sys.path.insert(0, comfy_root())
    sys.path.insert(0, ROOT)
    return ROOT
