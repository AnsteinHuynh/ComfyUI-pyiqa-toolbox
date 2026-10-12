"""Per-architecture pre-load gates — no nodes live here.

Some architectures need something checked or prepared before their weights are
built, and the check belongs to *the architecture*, not to whichever loader
asked for it. Gates run in one place, ``_common.run_arch_gate``, which
``_common.build_metric`` calls before it creates anything: that covers every
metric "(Down)load pyiqa Model" can reach, Q-Align among them. (The two architectures
that build their own weights outside that path - FGResQ's checkpoint surgery and
Q-ReAlign's int8/snapshot builds - carry their requirements inside their own
builders, so no gate is registered for them.)

A gate may refuse the load (raising with the fix in the message) or prepare for
it (freeing ComfyUI's memory). It never builds the metric itself.

One row per architecture, appending only:
    register_arch_preload("<architecture label>", fn)
"""

import importlib.util
import logging

from ..utility import metric_catalog
from ..utility.pyiqa_utils import make_room_for_metric

from ._common import register_arch_preload

logger = logging.getLogger(__name__)

# Q-Align is pyiqa's ``q-future/one-align``: an mPLUG-Owl2 model of 8.20e9
# parameters (measured from the checkpoint's own config — hidden_size 4096 x 32
# layers, plus the visual tower). At fp16 that is exactly 2 bytes per parameter,
# ~15 GiB of weights: the one registry metric that can exhaust a consumer GPU on
# its own. The pre-load gate frees ComfyUI's cache first and says how big it will
# be; the quantized sizes are approximate on purpose (bitsandbytes adds per-block
# scales on top of the 1- and 0.5-byte weights).
_QALIGN_PARAMS = 8.20e9
_QALIGN_FP16_GIB = _QALIGN_PARAMS * 2 / 2**30
_QALIGN_SIZE = {
    "fp16": f"~{_QALIGN_FP16_GIB:.1f} GiB (fp16)",
    "8bit": "~8 GiB (int8 weights plus per-block scales)",
    "4bit": "~4-5 GiB (4-bit weights plus per-block scales)",
}
# bitsandbytes quantizes; accelerate is what it loads the quantized model with
_QALIGN_QUANT_DEPS = ("bitsandbytes", "accelerate")


def qalign_dtype(resolved):
    """The precision pyiqa will build this Q-Align variant with.

    ``qalign`` is fp16, ``qalign_8bit``/``qalign_4bit`` carry the precision in
    pyiqa's own metric_opts (not in our variant label).
    """
    return str(metric_catalog.metric_opts(resolved.base).get("dtype") or "fp16")


def check_qalign_deps(dtype, find_spec=importlib.util.find_spec):
    """Refuse a quantized Q-Align variant that cannot actually quantize.

    pyiqa's own behaviour here is to warn and carry on: without bitsandbytes it
    loads the full fp16 model, i.e. a user who picked 4-bit to fit their card
    silently gets ~15 GiB instead. Each pack mode is meant to do exactly what its
    label says, so this is an error naming the fix, not a warning.

    ``find_spec`` is a parameter so the check is testable without uninstalling
    anything.
    """
    if dtype not in ("8bit", "4bit"):
        return
    missing = [m for m in _QALIGN_QUANT_DEPS if find_spec(m) is None]
    if not missing:
        return
    raise RuntimeError(
        f"Q-Align {dtype} weights need {' and '.join(missing)}, which is not "
        f"installed. pyiqa would only warn and load the full fp16 model instead "
        f"(~{_QALIGN_FP16_GIB:.1f} GiB of weights). Install it "
        f"(pip install {' '.join(missing)}) or pick the fp16 Q-Align variant "
        "deliberately."
    )


def gate_qalign(resolved, device):
    """Pre-load gate for the Q-Align architecture (see the module docstring)."""
    dtype = qalign_dtype(resolved)
    check_qalign_deps(dtype)
    make_room_for_metric(device)
    logger.info(
        "(pyiqa-toolbox) Q-Align (%s) is one-align, %.1fB parameters: %s of "
        "weights, downloaded once on first use",
        resolved.label,
        _QALIGN_PARAMS / 1e9,
        _QALIGN_SIZE.get(dtype, _QALIGN_SIZE["fp16"]),
    )


register_arch_preload("Q-Align", gate_qalign)
