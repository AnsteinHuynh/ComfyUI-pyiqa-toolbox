"""Shared helpers for the pyiqa-toolbox nodes: device selection, ComfyUI image
conversion, score formatting, and a small metric cache so loader nodes reuse one
loaded model across score nodes and re-queues instead of duplicating VRAM."""

import dataclasses
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
    """
    for folder in _pyiqa_folder_paths():
        candidate = os.path.join(folder, name)
        if os.path.exists(candidate):
            return candidate
        for hf_dir in sorted(glob.glob(os.path.join(folder, f"models--*--{name}"))):
            for snap in sorted(glob.glob(os.path.join(hf_dir, "snapshots", "*"))):
                if _snapshot_complete(snap):
                    return snap
    return None


def list_pyiqa_checkpoints(
    pattern="*.pth", require_config=False, exclude_paths=(), exclude_names=()
):
    """List custom checkpoints in the search ``pyiqa`` folders.

    ``require_config=True`` returns snapshot-style *directories* (any subfolder
    containing a config.json — HF-snapshot-like models such as custom
    Q-ReAlign fine-tunes); otherwise returns *files* matching ``pattern``
    (e.g. ``"FGResQ*.pth"``), as ``(label, full_path)`` pairs in folder order.
    Family loaders use the family name as the pattern so each dropdown only
    lists checkpoints named for that family.

    ``exclude_paths`` drops entries resolving to one of the given paths (a
    built-in that a broad search path already exposes, so the custom scan does
    not list the same model twice); ``exclude_names`` drops entries by basename
    (pyiqa's own default-family weight files, which are not customs). Both are
    compared with ``os.path.normcase(os.path.abspath(...))``.
    """
    excluded_paths = {os.path.normcase(os.path.abspath(p)) for p in exclude_paths}
    excluded_names = {os.path.normcase(n) for n in exclude_names}
    found = []
    for folder in _pyiqa_folder_paths():
        if require_config:
            try:
                entries = sorted(os.listdir(folder))
            except OSError:
                continue
            for name in entries:
                candidate = os.path.join(folder, name)
                if not os.path.isfile(os.path.join(candidate, "config.json")):
                    continue
                if os.path.normcase(os.path.abspath(candidate)) in excluded_paths:
                    continue
                if os.path.normcase(name) in excluded_names:
                    continue
                found.append((name, candidate))
        else:
            for path in sorted(glob.glob(os.path.join(folder, pattern))):
                if os.path.normcase(os.path.abspath(path)) in excluded_paths:
                    continue
                label = os.path.basename(path)
                if os.path.normcase(label) in excluded_names:
                    continue
                found.append((label, path))
    return found


RESOLUTION_MODES = [
    "auto (up to multiple of 32)",
    "native",
    "224",
    "256",
    "384",
    "512",
    "768",
]


def apply_resolution(x, resolution):
    """Smart resolution handling for arches with size requirements.

    - "auto (up to multiple of 32)": bicubic-resize up to the next multiple of
      32 per side — the graceful path AFINE's own forward intends (its padding
      code is unreachable behind an assert that fires first). Never downsizes.
    - "native": pass through untouched.
    - "224".."768": bicubic resize to that exact square size.
    """
    import torch.nn.functional as F

    if resolution == "native":
        return x
    if resolution.startswith("auto"):
        _, _, h, w = x.shape
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h or pad_w:
            x = F.interpolate(
                x, size=(h + pad_h, w + pad_w), mode="bicubic", align_corners=False
            )
        return x.clamp_(0, 1)
    size = int(resolution)
    if x.shape[2] != size or x.shape[3] != size:
        x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False)
    return x.clamp_(0, 1)


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
    stale = [k for k, (f, _, _) in _METRIC_CACHE.items() if f == family and k != entry]
    for k in stale:
        # park the outgoing metric before dropping it: ComfyUI's node output
        # cache may still hold a reference, so a bare del would leave it on
        # the GPU (release_metric itself collapses to a full drop for int8)
        release_metric(_METRIC_CACHE[k][1])
        del _METRIC_CACHE[k]
    if stale and torch.cuda.is_available():
        torch.cuda.empty_cache()
    metric = factory()
    _METRIC_CACHE[entry] = (family, metric, factory)
    return metric


def revive_metric(model):
    """Rebuild a fully-unloaded (gutted) metric; identity pass-through else.

    ComfyUI caches node outputs between prompts, so a score node can be handed
    the loader's previous output - a hollow shell after a full unload. The
    decorator revives through this before scoring.
    """
    if model is None or not getattr(model, "_pyiqa_gutted", False):
        return model
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


def _gut(target):
    """Replace every parameter and buffer with empty meta tensors, in place.

    The object stays alive for anyone still referencing it (ComfyUI's output
    cache), but holds no VRAM and no RAM - only the recipe to rebuild it.
    """
    for _, module in target.named_modules():
        for attr in list(module._parameters):
            module._parameters[attr] = torch.nn.Parameter(
                torch.empty(0, device="meta"), requires_grad=False
            )
        for attr in list(module._buffers):
            module._buffers[attr] = None
    target._pyiqa_gutted = True


def drop_metric(metric):
    """Fully unload one metric: gone from VRAM and from system RAM.

    The weights are gutted in place (so references held by ComfyUI's output
    cache cost nothing), the cache entry becomes a rebuild tombstone, and the
    CUDA cache is flushed. The next score reloads from disk.
    """
    if metric is None:
        return False
    target = (
        metric if isinstance(metric, torch.nn.Module) else getattr(metric, "metric", metric)
    )
    already = getattr(metric, "_pyiqa_gutted", False) or getattr(
        target, "_pyiqa_gutted", False
    )
    if not already:
        try:
            _gut(target)
        except Exception as exc:  # never break a score over cleanup
            logger.debug("drop_metric gut failed: %s", exc)
            return False
        metric._pyiqa_gutted = True
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


def _sync_quantized_scales(metric, device):
    """Move ComfyUI QuantizedTensor scales after a device move.

    ``nn.Module.to`` moves each QuantizedTensor's ``_qdata`` (via the flatten
    protocol) but NOT the scale tensor inside its ``_params`` dataclass - the
    module never sees it. A cuda int8 weight with a cpu scale then computes
    silently WRONG values. Duck-typed so pyiqa_utils needs no comfy import.
    """
    try:
        for _, param in metric.named_parameters():
            data = param.data
            params = getattr(data, "_params", None)
            scale = getattr(params, "scale", None) if params is not None else None
            if isinstance(scale, torch.Tensor) and not _device_matches(
                scale.device, device
            ):
                new_params = dataclasses.replace(params, scale=scale.to(device))
                param.data = data._copy_with(params=new_params, clone_params=False)
    except Exception as exc:  # never let bookkeeping break inference
        logger.debug("_sync_quantized_scales: %s", exc)


def _has_quantized_params(metric):
    """True if the metric holds ComfyUI QuantizedTensor weights (int8_convrot).

    Those cannot live on the CPU: comfy_kitchen's int8/dequant kernels are
    CUDA-only, and ``.to('cpu')`` leaves them in a state that computes silently
    wrong values. Such metrics must stay on the GPU until they are evicted
    (variant switch) or the process ends.
    """
    try:
        for _, param in metric.named_parameters():
            if getattr(param.data, "_qdata", None) is not None:
                return True
    except Exception:
        pass
    return False


def offload_metric_cache():
    """React to ComfyUI's Edit > Unload Models: move cached metrics to CPU.

    Matches ComfyUI's own unload semantics (models are offloaded to system
    RAM, not deleted), so the metrics stay warm and are moved back onto the
    GPU by ``ensure_on_device`` on the next score/loader call. int8 metrics
    cannot be parked (see ``_has_quantized_params``), so they are fully
    unloaded instead - the consistent collapse rule everywhere in this pack.
    """
    moved = []
    for key, (family, metric, _factory) in list(_METRIC_CACHE.items()):
        if getattr(metric, "_pyiqa_gutted", False):
            continue  # tombstone from a previous full unload
        if _has_quantized_params(metric):
            logger.info(
                "(pyiqa-toolbox) %s is int8_convrot and cannot be parked in "
                "RAM; fully unloading it instead", family
            )
            drop_metric(metric)
            continue
        try:
            metric.to(torch.device("cpu"))
            _sync_quantized_scales(metric, torch.device("cpu"))
            moved.append(f"{family} {_metric_size_gib(metric):.1f}GiB")
        except Exception as exc:  # never let cleanup break the unload action
            logger.debug("offload_metric_cache: %s failed: %s", family, exc)
    if moved:
        logger.info(
            "(pyiqa-toolbox) metrics parked in system RAM: %s", ", ".join(moved)
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def evict_metric_cache(fully=True):
    """The Unload Pyiqa Models node's action: release every cached metric.

    fully=True drops each metric from VRAM and RAM (reload from disk next
    score); fully=False parks them in system RAM (int8 collapses to a drop).
    """
    if not fully:
        return offload_metric_cache()
    for _key, (_family, metric, _factory) in list(_METRIC_CACHE.items()):
        if not getattr(metric, "_pyiqa_gutted", False):
            drop_metric(metric)


def ensure_on_device(metric, device=None):
    """Move ``metric`` onto the intended compute device if a previous offload
    (or a stale device attribute) left it elsewhere.

    pyiqa's ``InferenceModel.device`` is set once at construction and is NOT
    updated by ``.to()``, so a caller handing us ``model.device`` can be handing
    us a stale value; and ``InferenceModel.forward`` computes on
    ``self.dummy_param.device``. To avoid silently running an 18 GB metric on
    the CPU, a CPU hint is ignored whenever a GPU is actually available - we
    always target the real compute device. Returns the metric's true device.
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
        _sync_quantized_scales(metric, target)
        real = target
    return real if real is not None else target


def release_metric(metric):
    """Park one metric in system RAM and hand its VRAM back to the allocator.

    Used by the score nodes' ``free VRAM (park in RAM)`` mode so the next
    generation gets full VRAM; the metric is reloaded lazily (``ensure_on_device``)
    on the next score. int8_convrot metrics cannot be parked - CPU residency
    silently corrupts their values - so they collapse to a FULL unload
    (``drop_metric``): VRAM and RAM freed, reload from disk next score.
    Returns True if the metric is no longer on the GPU.
    """
    if _has_quantized_params(metric):
        logger.info(
            "(pyiqa-toolbox) int8_convrot metric cannot be parked in RAM; "
            "fully unloading it instead (reloads from disk next score)"
        )
        return drop_metric(metric)
    try:
        metric.to(torch.device("cpu"))
        _sync_quantized_scales(metric, torch.device("cpu"))
    except Exception as exc:
        logger.debug("release_metric: %s", exc)
        return False
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True


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
