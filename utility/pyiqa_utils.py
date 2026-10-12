"""Shared helpers for the pyiqa-toolbox nodes: device selection, ComfyUI image
conversion, score formatting, and a small metric cache so loader nodes reuse one
loaded model across score nodes and re-queues instead of duplicating VRAM."""

import gc
import glob
import json
import logging
import os

import torch

logger = logging.getLogger(__name__)

# Order-sensitivity hints for FR score nodes. Measured in this pack's env by
# scoring f(a,b) vs f(b,a) on identical inputs (symmetry probe run 2026-10-10):
#   symmetric  = ssim, ssimc, ms_ssim, fsim, psnr, psnry, mad, l1, lpips,
#                lpips-vgg, stlpips, stlpips-vgg, dists, msswd, afine_fr
#   asymmetric = topiq_fr (~0.8%), wadiqam_fr (~2%), fgresq_pair (rank and
#                quality0/quality1 swap)
#   near       = lpips+, lpips-vgg+ (~0.1%, semantic weighting)
ORDER_SYMMETRIC = (
    "Order-independent: swapping target and reference yields the same score."
)
ORDER_ASYMMETRIC = (
    "Order-SENSITIVE: target and reference are NOT interchangeable - swapping "
    "them changes the score."
)
ORDER_NEAR_SYMMETRIC = (
    "Nominally symmetric: swapping target and reference changes the score only "
    "slightly (about 0.1%)."
)

TARGET_TOOLTIP = (
    "The image being judged. In an enhancement workflow this is normally the "
    "produced/restored/denoised image, or any distorted version (noisy, blurry, "
    "compressed) - the one whose quality you want to measure. pyiqa calls it `x`."
)


def reference_tooltip(order_hint):
    return (
        "The pristine original / ground-truth image the target is compared "
        "against (pyiqa's `ref`). Paired with target by batch position: target "
        f"image #i is compared to reference image #i. {order_hint}"
    )

# Paths registered by an `pyiqa:` key in extra_model_paths.yaml. Captured at
# import, then kept SEARCH-ONLY (never library-registered) — see below.
_EXTRA_SEARCH_PATHS = []
_LOCAL_PYIQA_DIR = None

try:
    import folder_paths

    # ComfyUI models folder for this pack (RMBG-style registration).
    #
    # The `pyiqa:` paths in extra_model_paths.yaml arrive BEFORE custom nodes
    # load (the yaml is parsed first), and ComfyUI's Model Library lists every
    # path registered under a model-folder key. Broad yaml entries (X:/Models,
    # the HF/torch hub caches) would therefore flood the library with every
    # unrelated model on those drives, so they are treated as SEARCH-ONLY: we
    # capture them here, then rewrite the "pyiqa" registration so only the
    # install-local models/pyiqa stays library-visible.
    _LOCAL_PYIQA_DIR = os.path.join(folder_paths.models_dir, "pyiqa")

    try:
        _registered_pyiqa = list(folder_paths.get_folder_paths("pyiqa"))
    except Exception:
        _registered_pyiqa = []

    _local_key = os.path.normcase(os.path.abspath(_LOCAL_PYIQA_DIR))
    _EXTRA_SEARCH_PATHS = [
        p
        for p in _registered_pyiqa
        if os.path.normcase(os.path.abspath(p)) != _local_key
    ]

    # Register the install-local folder (a visible place to drop weights).
    folder_paths.add_model_folder_path("pyiqa", _LOCAL_PYIQA_DIR)

    # Drop the yaml search paths from the library-visible registration, keeping
    # only the local folder. Guarded so a different/absent internal structure
    # degrades to "leave registration as-is and proceed".
    try:
        if hasattr(folder_paths, "folder_names_and_paths"):
            _entry = folder_paths.folder_names_and_paths.get("pyiqa")
            if isinstance(_entry, tuple) and len(_entry) == 2:
                folder_paths.folder_names_and_paths["pyiqa"] = (
                    [_LOCAL_PYIQA_DIR],
                    _entry[1],
                )
            # any other shape: leave the registration untouched rather than
            # rewriting it into a form ComfyUI does not expect
    except Exception:
        pass

    def _implicit_default_paths():
        """pyiqa's own caches: always relevant, never library-registered.

        Derived generically (torch hub dir, HF hub cache) and only returned
        when the directory actually exists.
        """
        paths = []
        try:
            paths.append(os.path.join(torch.hub.get_dir(), "pyiqa"))
        except Exception:
            pass
        try:
            from huggingface_hub import constants as _hf_constants

            paths.append(_hf_constants.HF_HUB_CACHE)
        except Exception:
            paths.append(
                os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
            )
        return [p for p in paths if p and os.path.isdir(p)]

    def _user_pyiqa_dirs():
        """The user-facing model folders (install-local + extra_model_paths).

        These are the ones the ``{architecture}/`` custom-checkpoint layout is
        read from; pyiqa's own caches are deliberately excluded from it (their
        loose weight files are pyiqa's defaults, not customs).
        """
        return [_LOCAL_PYIQA_DIR] + list(_EXTRA_SEARCH_PATHS)

    def _pyiqa_folder_paths():
        """Search paths in priority order: local, yaml, implicit caches.

        Deduped by normalized path; order preserved. This is what find/list
        scan — it is deliberately broader than the library-visible registration.
        """
        ordered = (
            [_LOCAL_PYIQA_DIR] + list(_EXTRA_SEARCH_PATHS) + _implicit_default_paths()
        )
        seen = set()
        out = []
        for path in ordered:
            key = os.path.normcase(os.path.abspath(path))
            if key in seen:
                continue
            seen.add(key)
            out.append(path)
        return out

    for _folder in _pyiqa_folder_paths():
        try:
            os.makedirs(_folder, exist_ok=True)
        except Exception:
            pass

except Exception:  # standalone use (smoke tests outside ComfyUI)

    def _user_pyiqa_dirs():
        return []

    def _pyiqa_folder_paths():
        return []


def _snapshot_complete(snap):
    """A snapshot is usable only when every shard in its index is present."""
    if not os.path.isfile(os.path.join(snap, "config.json")):
        return False
    index = os.path.join(snap, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index) as f:
            shards = set(json.load(f)["weight_map"].values())
        return all(os.path.isfile(os.path.join(snap, s)) for s in shards)
    return bool(
        glob.glob(os.path.join(snap, "*.safetensors"))
        or glob.glob(os.path.join(snap, "*.bin"))
    )


def local_pyiqa_dir():
    """The install-local ``models/pyiqa`` folder (created on import). Used as
    the download target for HF-hosted int8_convrot models so they persist."""
    if _LOCAL_PYIQA_DIR:
        return _LOCAL_PYIQA_DIR
    folders = _pyiqa_folder_paths()
    return folders[0] if folders else os.path.join(os.path.expanduser("~"), "pyiqa")


def find_pyiqa_model(name):
    """Resolve ``name`` against the registered ``pyiqa`` folders, else None.

    Two layouts are recognized inside every registered folder:
    - plain: ``<folder>/<name>`` (file or dir, e.g. ``FGResQ.pth`` or a
      ``Q-ReAlign-Mini-0.8B/`` snapshot-style dir);
    - HF hub cache: ``models--<org>--<name>/snapshots/<hash>/`` — complete
      snapshots only (every shard from model.safetensors.index.json present),
      so the folder can point straight at the HuggingFace cache (single-copy
      setup, nothing is duplicated or re-downloaded). Incomplete snapshots are
      treated as absent so the loader keeps looking in later folders before
      falling back to the repo id and finishing the download.

    The user-facing folders are additionally searched ONE LEVEL DOWN for the
    ``<folder>/<Architecture>/<name>`` custom layout — a model moved into an
    architecture folder must still be found, or the loader would re-download
    gigabytes of weights that are already on disk.
    """
    for folder in _user_pyiqa_dirs():
        for arch_dir in sorted(glob.glob(os.path.join(folder, "*"))):
            candidate = os.path.join(arch_dir, name)
            if os.path.exists(candidate):
                return candidate
    for folder in _pyiqa_folder_paths():
        candidate = os.path.join(folder, name)
        if os.path.exists(candidate):
            return candidate
        for hf_dir in sorted(glob.glob(os.path.join(folder, f"models--*--{name}"))):
            for snap in sorted(glob.glob(os.path.join(hf_dir, "snapshots", "*"))):
                if _snapshot_complete(snap):
                    return snap
    return None


# Custom-checkpoint containers.
#
# The extension only hints at the container, and the container is read by the
# ARCH: pyiqa's ``pretrained_model_path`` is consumed inside each arch's own
# __init__ — most call ``load_pretrained_network`` (torch.load, some of them
# expecting a ``params`` key), the classic NR metrics read MATLAB ``.mat``
# files, and AFINE expects a multi-key dict of its own. So the arch decides
# what it can read; listing every plausible container and failing a wrong pick
# with a clear message beats hiding files that would have worked.
WEIGHT_EXTS = (".pth", ".pt", ".ckpt", ".bin", ".safetensors", ".mat")
SAFETENSORS_EXT = ".safetensors"
SNAPSHOT_MARKER = "config.json"


def iter_user_pyiqa_dirs():
    """(folder, entry_name, entry_path) for every entry of the user-facing
    pyiqa folders — the ``{architecture}/`` layout's candidate folders."""
    for folder in _user_pyiqa_dirs():
        try:
            names = sorted(os.listdir(folder))
        except OSError:
            continue
        for name in names:
            yield folder, name, os.path.join(folder, name)


def list_weight_files(directory):
    """[(label, path, kind)] for the weight files directly in ``directory``.

    kind is ``"safetensors"`` (this pack reads it — pyiqa cannot) or
    ``"checkpoint"`` (handed to pyiqa's own per-arch loader untouched).
    """
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return []
    found = []
    for name in names:
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext not in WEIGHT_EXTS:
            continue
        found.append(
            (name, path, "safetensors" if ext == SAFETENSORS_EXT else "checkpoint")
        )
    return found


def list_snapshot_dirs(directory):
    """[(label, path)] for snapshot-style subfolders of ``directory``.

    A snapshot model is a directory holding a ``config.json`` plus its weights
    (the ``from_pretrained`` layout, e.g. HF downloads of Q-ReAlign builds).
    """
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return []
    found = []
    for name in names:
        path = os.path.join(directory, name)
        if os.path.isfile(os.path.join(path, SNAPSHOT_MARKER)):
            found.append((name, path))
    return found


def load_state_dict(path):
    """Read a checkpoint's tensor dict from either container.

    torch formats go through ``torch.load`` exactly as pyiqa does;
    ``.safetensors`` is read directly, because pyiqa's
    ``load_pretrained_network`` is torch.load-only and raises on one.
    """
    if path.lower().endswith(SAFETENSORS_EXT):
        from safetensors.torch import load_file

        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=False)


# Input adaptation ("auto"): what a model needs done to its images before
# scoring. A per-architecture policy tuple (see metric_catalog._ARCH_RULES and
# Architecture.policy) rather than a user choice — every requirement here is a
# property of the model, and the sizes a user could previously type (224/384/...)
# were the one option that could disagree with the model's own preprocessing.
POLICY_MULT32 = ("mult32",)


def _to_multiple_of_32(x):
    import torch.nn.functional as F

    _, _, h, w = x.shape
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    if pad_h or pad_w:
        x = F.interpolate(
            x, size=(h + pad_h, w + pad_w), mode="bicubic", align_corners=False
        )
    return x.clamp_(0, 1)


def prepare_image(x, policy=()):
    """Adapt one image batch to the loaded model's requirements."""
    if "mult32" in policy:
        x = _to_multiple_of_32(x)
    return x


def prepare_pair(x, y, policy=()):
    """Adapt a (target, reference) pair to the loaded model's requirements.

    Full-reference metrics compare the two images pixel-wise, so a pair of
    different sizes is resized to the target's geometry — the target is the
    image being judged, its framing is the one that must not move.
    """
    import torch.nn.functional as F

    x = prepare_image(x, policy)
    y = prepare_image(y, policy)
    if x.shape[2:] != y.shape[2:]:
        logger.info(
            "(pyiqa-toolbox) reference %s resized to the target's %s "
            "(full-reference metrics compare pixel-wise)",
            tuple(y.shape[2:]),
            tuple(x.shape[2:]),
        )
        y = F.interpolate(y, size=x.shape[2:], mode="bicubic", align_corners=False)
        y = y.clamp_(0, 1)
    return x, y


def default_device():
    try:
        from comfy import model_management

        return model_management.get_torch_device()
    except Exception:  # standalone use (smoke tests outside ComfyUI)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return torch.device("cuda")
        return torch.device("cpu")


def comfy_image_to_pyiqa(image, device):
    """ComfyUI IMAGE (B, H, W, C) float in [0, 1] -> pyiqa input (B, C, H, W).

    pyiqa's InferenceModel asserts 4D input in [0, 1]; it applies each metric's
    own preprocess internally (FGResQ: resize 256 -> crop 224 -> CLIP norm;
    QReAlign: its Qwen3.5-VL processor), so we only move axes/dtype/device.
    """
    return (
        image.permute(0, 3, 1, 2)
        .contiguous()
        .to(device=device, dtype=torch.float32)
    )


# (family, key) -> (family, metric, factory). The factory is kept so a metric
# that was fully unloaded (gut-and-drop) can be rebuilt transparently when a
# cached node output turns out to be a hollow shell. Loading a new metric of
# the same family evicts the previous one (e.g. switching qrealign mini -> pro
# frees the 0.8B model); other families are kept so qrealign + fgresq can be
# wired together.
_METRIC_CACHE = {}


def get_cached_metric(family, key, factory):
    # namespaced by (family, key): a bare key from one family colliding with
    # an equal key from another would otherwise silently hand back the WRONG
    # family's metric
    entry = (family, key)
    hit = _METRIC_CACHE.get(entry)
    if hit is not None:
        # a gutted entry is a tombstone from a full unload - rebuild it
        if not getattr(hit[1], "_pyiqa_gutted", False):
            return hit[1]
        del _METRIC_CACHE[entry]
    # switch to another variant of the same family: gut the outgoing live
    # metric (its memory frees) but KEEP its entry as a tombstone - ComfyUI's
    # output cache may still hand the old object to a score node later, and
    # the tombstone is what lets that reference revive. Tombstones are empty
    # shells, so a family may accumulate several at no cost.
    stale = [
        k
        for k, (f, m, _) in _METRIC_CACHE.items()
        if f == family and k != entry and not getattr(m, "_pyiqa_gutted", False)
    ]
    for k in stale:
        drop_metric(_METRIC_CACHE[k][1])
    if stale and torch.cuda.is_available():
        torch.cuda.empty_cache()
    metric = factory()
    _METRIC_CACHE[entry] = (family, metric, factory)
    return metric


def revive_metric(model):
    """Rebuild a fully-unloaded (gutted) metric; identity pass-through else.

    ComfyUI caches node outputs between prompts, so a score node can be handed
    the loader's previous output - a hollow shell after a full unload. The
    gutted shell carries its (family, key) recipe, so it finds the cache entry
    even after other revivals replaced the entry's metric with a newer object
    (the entry then hands back the LIVE metric instead of rebuilding).
    """
    if model is None or not getattr(model, "_pyiqa_gutted", False):
        return model
    recipe = getattr(model, "_pyiqa_recipe", None)
    if recipe is not None:
        hit = _METRIC_CACHE.get(tuple(recipe))
        if hit is not None and not getattr(hit[1], "_pyiqa_gutted", False):
            return hit[1]  # someone already revived this family/key
        if hit is not None:
            family, key, factory = recipe[0], recipe[1], hit[2]
            del _METRIC_CACHE[tuple(recipe)]
            logger.info(
                "(pyiqa-toolbox) fully-unloaded metric rebuilt for the next "
                "score (reload from disk)"
            )
            return get_cached_metric(family, key, factory)
    # fallback for gutted shells without a recipe: identity lookup
    for key, (family, m, factory) in list(_METRIC_CACHE.items()):
        if m is model:
            del _METRIC_CACHE[key]
            logger.info(
                "(pyiqa-toolbox) fully-unloaded metric rebuilt for the next "
                "score (reload from disk)"
            )
            return get_cached_metric(family, key, factory)
    raise RuntimeError(
        "(pyiqa-toolbox) this model was fully unloaded and no loader recipe "
        "remains for it - re-run the workflow with the loader node re-executed"
    )


def _wrapper_chain(metric):
    """``metric`` and every ``.metric`` it wraps, outermost first.

    A loaded metric is a chain: the generic loader's ``MetricBundle`` around a
    family bundle (FGResQ's shared net) around pyiqa's ``InferenceModel``. The
    walk stops at the first object that can be gutted - one exposing
    ``named_modules``, either because it IS an ``nn.Module`` or because it
    forwards to one (a ``MetricBundle`` around a plain pyiqa metric forwards, so
    that chain is one object long; FGResQ's bundle does not, so its chain has all
    three levels). That object is also the only one holding memory.
    """
    chain = []
    seen = set()
    obj = metric
    while obj is not None and id(obj) not in seen:
        chain.append(obj)
        seen.add(id(obj))
        if callable(getattr(obj, "named_modules", None)):
            break
        obj = getattr(obj, "metric", None)
    return chain


def _gut(metric):
    """Release the weights behind ``metric``, in place.

    Only the innermost object in the wrapper chain owns modules, so that is what
    is emptied: every parameter becomes an empty meta tensor and every buffer
    becomes ``None``. The shells stay alive for anyone still referencing them
    (ComfyUI's output cache) but hold no VRAM and no RAM.
    """
    chain = _wrapper_chain(metric)
    target = chain[-1] if chain else None
    if not callable(getattr(target, "named_modules", None)):
        raise TypeError(f"{type(metric).__name__} wraps no module to unload")
    for _, module in target.named_modules():
        for attr in list(module._parameters):
            module._parameters[attr] = torch.nn.Parameter(
                torch.empty(0, device="meta"), requires_grad=False
            )
        for attr in list(module._buffers):
            module._buffers[attr] = None


def _mark_chain(chain):
    """Tombstone every wrapper in ``chain`` and stamp each one's own recipe.

    EVERY level needs both marks, not just the innermost object: ComfyUI's output
    cache can hand any of them to the next score node, and a shell without
    ``_pyiqa_gutted`` is used as-is instead of being revived (that was a real
    crash - after Edit > Unload Models, FGResQ's shared pair net reached a score
    node hollowed out, ``self.default_mean is None``). The recipe is per object
    because two cache entries can hold two different wrappers of ONE net, and a
    revived shell has to find the entry that would rebuild the thing it is.
    """
    for obj in chain:
        obj._pyiqa_gutted = True
        if getattr(obj, "_pyiqa_recipe", None) is None:
            for key, (_family, m, _factory) in _METRIC_CACHE.items():
                if m is obj:
                    obj._pyiqa_recipe = key
                    break


def drop_metric(metric):
    """Fully unload one metric: gone from VRAM and from system RAM.

    The weights are gutted in place (so references held by ComfyUI's output
    cache cost nothing), every wrapper becomes a rebuild tombstone, and the CUDA
    cache is flushed. The next score reloads from disk.
    """
    if metric is None:
        return False
    chain = _wrapper_chain(metric)
    unloaded = False
    if not any(getattr(obj, "_pyiqa_gutted", False) for obj in chain):
        try:
            _gut(metric)
        except Exception as exc:  # never break a score over cleanup
            logger.debug("drop_metric gut failed: %s", exc)
            return False
        unloaded = True
    # the net may already have been gutted through another cache entry that
    # wraps the same weights (FGResQ's bundle is cached under its family key AND
    # wrapped by the generic loader), so this runs even then: the wrapper in
    # front of us still has to be revivable
    _mark_chain(chain)
    if unloaded:
        logger.info(
            "(pyiqa-toolbox) metric fully unloaded (VRAM + RAM freed; "
            "reloads from disk on the next score)"
        )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True


def _metric_marker(metric):
    """A parameter/buffer whose device tells us where the metric really lives."""
    marker = getattr(metric, "dummy_param", None)
    if marker is None:
        try:
            marker = next(metric.parameters())
        except (StopIteration, AttributeError):
            marker = None
    return marker


def _metric_size_gib(metric):
    try:
        return sum(p.numel() * p.element_size() for p in metric.parameters()) / 2**30
    except Exception:
        return 0.0


def _device_matches(real, target):
    real = torch.device(real)
    target = torch.device(target)
    if real.type != target.type:
        return False
    if target.index is None or real.index is None:
        return True
    return real.index == target.index


def unload_metric_cache():
    """React to ComfyUI's Edit > Unload Models: release every cached metric.

    The pack has no CPU-parking path (it proved buggy: pyiqa keeps plain
    tensor attributes that ``.to()`` does not move, and int8 weights corrupt
    outright), so "unload" here means a FULL unload - metrics are dropped
    from VRAM and RAM and reload from disk on the next score.
    """
    for _key, (_family, metric, _factory) in list(_METRIC_CACHE.items()):
        if not getattr(metric, "_pyiqa_gutted", False):
            drop_metric(metric)


def evict_metric_cache(fully=True):
    """The Unload Pyiqa Models node's action: release every cached metric.

    fully is kept for call-signature compatibility; there is only one release
    semantic now (full unload - VRAM and RAM).
    """
    unload_metric_cache()


def ensure_on_device(metric, device=None):
    """Return the metric's true compute device, moving it if needed.

    pyiqa's ``InferenceModel.device`` is set once at construction and is NOT
    updated by ``.to()``, so a caller handing us ``model.device`` can be handing
    us a stale value; and ``InferenceModel.forward`` computes on
    ``self.dummy_param.device``. To avoid silently running an 18 GB metric on
    the CPU, a CPU hint is ignored whenever a GPU is actually available - we
    always target the real compute device. With parking removed, metrics are
    normally already where they should be; this stays as a safety net.
    """
    if device is None:
        device = default_device()
    marker = _metric_marker(metric)
    real = marker.device if marker is not None else None
    target = torch.device(device)
    if real is not None and not _device_matches(real, target):
        logger.info(
            "(pyiqa-toolbox) moving metric to %s (found on %s)", target, real
        )
        metric.to(target)
        real = target
    return real if real is not None else target


def make_room_for_metric(device=None):
    """Ask ComfyUI to evict its cached models so a pyiqa metric can load.

    pyiqa metrics are built outside ``comfy.model_management``, so ComfyUI has
    no idea how much VRAM they hold and will not free anything for them. When a
    scoring node runs right after a generation, the SDXL checkpoint is still
    resident and the metric load can OOM. This does what ComfyUI does when
    loading one of its own models: unload cached models (respecting in-use
    refcounts) to make room. No-op when ComfyUI is not importable (e.g. tests).

    Returns True if a ComfyUI memory manager was asked to free, else False.
    """
    try:
        import comfy.model_management as mm
    except Exception:
        return False
    free_memory = getattr(mm, "free_memory", None)
    if free_memory is None:
        return False
    try:
        target = device
        if target is None:
            get_device = getattr(mm, "get_torch_device", None)
            target = get_device() if get_device is not None else None
        # A requirement far larger than any real free memory makes ComfyUI
        # unload every cached model it can (this is the "make room" call).
        free_memory(1e32, target)
        return True
    except Exception as exc:  # never let the relief break a load
        logger.debug("make_room_for_metric: %s", exc)
        return False
