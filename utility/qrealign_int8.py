"""Load a Q-ReAlign int8_convrot checkpoint (this pack's quantizer output, also
readable by ComfyUI's MixedPrecisionOps) into the pyiqa ``qrealign`` metric.

The checkpoint stores, per quantized Linear: an int8 ``<name>.weight`` + float
``<name>.weight_scale`` + a ``<name>.comfy_quant`` JSON marker, plus every other
weight unchanged (bf16). Loading rebuilds the Qwen3.5-VL model and swaps each
quantized weight in as a ``QuantizedTensor`` - torch dispatch then routes
``F.linear`` through comfy_kitchen's int8 matmul, applying the ConvRot
activation rotation automatically, with zero changes to the model's forward.

pyiqa's arch constructs the HF model itself via ``from_pretrained``, so we build
ours first and hand it across by temporarily patching that one classmethod.
"""

import json
import logging
import os
import threading

import torch
from safetensors import safe_open

logger = logging.getLogger(__name__)

LAYOUT = "TensorWiseINT8Layout"
INT8_SUFFIX = "-int8-convrot"
QUANT_MARKER = ".comfy_quant"

# the from_pretrained handover below is process-global; serialized so a
# concurrent transformers load in another thread can't receive our model
_HANDOVER_LOCK = threading.Lock()


def _comfy_quant_ops():
    """Import comfy.quant_ops lazily (only exists inside a ComfyUI install)."""
    from comfy.quant_ops import QuantizedTensor, get_layout_class

    return QuantizedTensor, get_layout_class


def _read_marker(reader, key):
    blob = reader.get_tensor(key)
    return json.loads(bytes(blob.tolist()).decode("utf-8"))


def _build_model(config, allow_meta=True):
    """Model skeleton. Meta first (fast); falls back to CPU init when the arch
    keeps non-persistent buffers (rotary inv_freq etc.) that meta cannot run."""
    from transformers import AutoModelForImageTextToText

    if allow_meta:
        try:
            with torch.device("meta"):
                return AutoModelForImageTextToText.from_config(config), True
        except Exception as exc:  # pragma: no cover - arch-specific
            logger.debug("meta init failed (%s); falling back to CPU init", exc)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        return AutoModelForImageTextToText.from_config(config).to(
            torch.device("cpu")
        ), False
    finally:
        torch.set_default_dtype(previous)


def _load_weights(model, safetensors_path, device, meta_build):
    QuantizedTensor, get_layout_class = _comfy_quant_ops()
    layout_cls = get_layout_class(LAYOUT)

    with safe_open(safetensors_path, framework="pt", device="cpu") as reader:
        keys = set(reader.keys())
        qconf = {
            k[: -len(QUANT_MARKER)]: _read_marker(reader, k)
            for k in keys
            if k.endswith(QUANT_MARKER)
        }
        # full name -> (module, store, attr); replacing through _parameters /
        # _buffers instead of `param.data =` because torch 2.12 rejects .data
        # assignment onto meta tensors ("incompatible tensor type")
        slots = {}
        for mname, module in model.named_modules():
            for store in ("_parameters", "_buffers"):
                for attr in list(getattr(module, store)):
                    full = f"{mname}.{attr}" if mname else attr
                    slots[full] = (module, store, attr)

        for full, (module, store, attr) in slots.items():
            # markers are keyed by module name (no ".weight"); the tensor keys
            # carry it - compare on the stripped form
            marker_name = full[: -len(".weight")] if full.endswith(".weight") else full
            if marker_name in qconf:
                conf = qconf[marker_name]
                qdata = reader.get_tensor(full).to(device)
                scale = reader.get_tensor(f"{full}_scale").to(device).float()
                qt = QuantizedTensor(
                    qdata,
                    LAYOUT,
                    layout_cls.Params(
                        scale=scale,
                        orig_dtype=torch.bfloat16,
                        orig_shape=tuple(module._parameters[attr].shape),
                        is_weight=True,
                        convrot=bool(conf.get("convrot", False)),
                        convrot_groupsize=int(conf.get("convrot_groupsize", 256)),
                    ),
                )
                module._parameters[attr] = torch.nn.Parameter(
                    qt, requires_grad=False
                )
            elif full in keys:
                tensor = reader.get_tensor(full).to(device)
                if store == "_parameters":
                    module._parameters[attr] = torch.nn.Parameter(
                        tensor.detach(), requires_grad=False
                    )
                else:
                    module._buffers[attr] = tensor

    if meta_build:
        # non-persistent buffers (rotary tables) are meta on a meta skeleton and
        # are recomputed by the CPU-initialized build instead
        if any(b.is_meta for _, b in model.named_buffers()):
            logger.info(
                "(pyiqa-toolbox) int8 model has meta buffers; rebuilding with "
                "CPU init (one-off)"
            )
            return None

    model.tie_weights()
    leftover = [
        n
        for n, p in model.named_parameters()
        if p.is_meta or (not isinstance(p, torch.Tensor))
    ]
    leftover = [n for n in leftover if model.get_parameter(n).is_meta]
    if leftover:
        raise RuntimeError(
            f"int8 checkpoint is missing weights for: {leftover[:5]} "
            f"({len(leftover)} total) - quantize from the matching variant"
        )
    return model


def load_int8_model(int8_dir, device):
    """Build the Qwen3.5-VL model with int8_convrot weights from ``int8_dir``."""
    from pyiqa.archs.qrealign.qrealign_compat import ensure_qwen3_5

    ensure_qwen3_5()
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(int8_dir)
    st_file = _find_safetensors(int8_dir)
    model = None
    for allow_meta in (True, False):
        candidate, meta_build = _build_model(config, allow_meta=allow_meta)
        model = _load_weights(candidate, st_file, device, meta_build)
        if model is not None:
            return model.eval()
    raise RuntimeError("could not build the int8 model")  # pragma: no cover


def _find_safetensors(folder):
    import glob

    candidates = glob.glob(os.path.join(folder, "*.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"no safetensors in {folder}")
    return candidates[0]


def create_int8_metric(int8_dir, device):
    """pyiqa ``qrealign`` InferenceModel backed by the int8_convrot weights."""
    import pyiqa
    from transformers import AutoModelForImageTextToText

    model = load_int8_model(int8_dir, device)
    original = AutoModelForImageTextToText.from_pretrained

    def _handed_over(path, dtype=None, **kwargs):
        return model

    with _HANDOVER_LOCK:
        AutoModelForImageTextToText.from_pretrained = staticmethod(_handed_over)
        try:
            # dtype is irrelevant (weights are already in place); the arch still
            # loads its processor from int8_dir, which carries the config files.
            return pyiqa.create_metric("qrealign", device=device, model=int8_dir)
        finally:
            AutoModelForImageTextToText.from_pretrained = original
