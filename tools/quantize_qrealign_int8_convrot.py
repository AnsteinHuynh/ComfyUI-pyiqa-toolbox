"""Quantize a Q-ReAlign checkpoint to ComfyUI's native int8_tensorwise + ConvRot format.

Produces a standalone model folder (config/processor/tokenizer files + one
safetensors of quantized weights) that is ready to upload to HuggingFace and is
consumed by this pack's QReAlign loader (`quantization: int8_convrot`) and,
because it follows ComfyUI's documented checkpoint format (QUANTIZATION.md:
per-layer `comfy_quant` markers + a `_quantization_metadata` safetensors header),
by ComfyUI's own MixedPrecisionOps loader as well.

Recipe (identical to Comfy-Org's official int8_convrot text encoders):
  - every nn.Linear with in_features % 256 == 0 (except the keep list, default
    lm_head) is rotated with the group-256 regular Hadamard (ConvRot) and
    int8-quantized with per-row scales, using comfy_kitchen's own kernels via
    comfy.quant_ops.QuantizedTensor - the same code path ComfyUI uses at runtime;
  - everything else (embeddings, norms, lm_head, biases) is copied unchanged.

Run inside the ComfyUI python env (the one with comfy_kitchen + torch cu130):

  python tools/quantize_qrealign_int8_convrot.py --src <qrealign folder> \
      --out <out folder> [--comfy-root <comfyui root>] [--keep lm_head] [--device cuda]

Example:
  python tools/quantize_qrealign_int8_convrot.py \
      --src I:/MyApps/ComfyUi/models/pyiqa/Q-ReAlign-Pro-9B \
      --out I:/MyApps/ComfyUi/models/pyiqa/Q-ReAlign-Pro-9B-int8-convrot
"""

import argparse
import json
import os
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

DEFAULT_COMFY_ROOT = r"C:\MyApps\ComfyUIRoot\20250726-ComfyUI\ComfyUI"
GROUP_SIZE = 256
QUANT_FORMAT = "int8_tensorwise"
COPY_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)


def _import_comfy(comfy_root):
    """Import comfy.quant_ops with a clean argv (comfy.cli_args parses sys.argv)."""
    if not os.path.isdir(os.path.join(comfy_root, "comfy")):
        raise SystemExit(f"no ComfyUI at {comfy_root} (use --comfy-root)")
    saved_argv, sys.argv, saved_path = sys.argv, ["quantizer"], list(sys.path)
    sys.path.insert(0, comfy_root)
    try:
        from comfy.quant_ops import QuantizedTensor  # noqa: F401
    finally:
        sys.argv, sys.path = saved_argv, saved_path
    from comfy.quant_ops import QuantizedTensor

    return QuantizedTensor


def _enumerate_linears(src, keep):
    """Meta-build the model and list its nn.Linear layers: name -> (out, in)."""
    from pyiqa.archs.qrealign.qrealign_compat import ensure_qwen3_5

    ensure_qwen3_5()
    from transformers import AutoConfig, AutoModelForImageTextToText

    config = AutoConfig.from_pretrained(src)
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(config)
    linears = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linears[name] = (module.out_features, module.in_features)
    kept = {n for n in linears if any(pat in n for pat in keep)}
    return linears, kept


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="Q-ReAlign model folder (or HF repo)")
    ap.add_argument("--out", required=True, help="output folder for the int8_convrot model")
    ap.add_argument("--comfy-root", default=DEFAULT_COMFY_ROOT)
    ap.add_argument(
        "--keep", default="lm_head",
        help="comma-separated substrings; matching Linears stay in bf16",
    )
    ap.add_argument("--device", default="cuda", help="device used for the quantize kernels")
    ap.add_argument("--filename", default="model.int8_convrot.safetensors")
    args = ap.parse_args()

    QuantizedTensor = _import_comfy(args.comfy_root)
    keep = [p.strip() for p in args.keep.split(",") if p.strip()]
    device = torch.device(args.device)

    linears, kept = _enumerate_linears(args.src, keep)
    print(f"{len(linears)} Linear layers; keeping {len(kept)} in bf16: {sorted(kept)[:8]}...")

    shards = [f for f in os.listdir(args.src) if f.endswith(".safetensors")]
    readers = {
        f: safe_open(os.path.join(args.src, f), framework="pt", device="cpu")
        for f in shards
    }
    key_to_file = {}
    for f, r in readers.items():
        for k in r.keys():
            key_to_file[k] = f

    out_tensors = {}
    layers_meta = {}
    n_quant = n_kept = 0
    t0 = time.time()
    for key in sorted(key_to_file):
        reader = readers[key_to_file[key]]
        tensor = reader.get_tensor(key)
        name = key[: -len(".weight")] if key.endswith(".weight") else None
        is_eligible = (
            name is not None
            and name in linears
            and name not in kept
            and linears[name][1] % GROUP_SIZE == 0
            and tensor.ndim == 2
        )
        if is_eligible:
            w = tensor.to(device=device, dtype=torch.bfloat16, non_blocking=False)
            qt = QuantizedTensor.from_float(
                w,
                "TensorWiseINT8Layout",
                is_weight=True,
                per_channel=True,
                convrot=True,
                convrot_groupsize=GROUP_SIZE,
            )
            for k, v in qt.state_dict(key).items():
                out_tensors[k] = v.detach().to("cpu", copy=True)
            conf = {
                "format": QUANT_FORMAT,
                "convrot": True,
                "convrot_groupsize": GROUP_SIZE,
            }
            out_tensors[f"{name}.comfy_quant"] = torch.tensor(
                list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8
            )
            layers_meta[name] = conf
            n_quant += 1
            del w, qt
        else:
            out_tensors[key] = tensor.contiguous()
            n_kept += 1
        if (n_quant + n_kept) % 100 == 0:
            done = n_quant + n_kept
            print(f"  {done}/{len(key_to_file)} tensors "
                  f"({n_quant} quantized), {time.time()-t0:.0f}s", flush=True)

    os.makedirs(args.out, exist_ok=True)
    metadata = {
        "_quantization_metadata": json.dumps(
            {"format_version": "1.0", "layers": layers_meta},
            separators=(",", ":"),
        )
    }
    out_path = os.path.join(args.out, args.filename)
    save_file(out_tensors, out_path, metadata=metadata)
    total = sum(v.numel() * v.element_size() for v in out_tensors.values())
    print(f"wrote {out_path} ({total/2**30:.2f} GiB, {n_quant} quantized / {n_kept} kept)")

    for f in COPY_FILES:
        p = os.path.join(args.src, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(args.out, f))
    with open(os.path.join(args.out, "quantization_info.json"), "w") as fh:
        json.dump(
            {
                "recipe": "int8_tensorwise + ConvRot (group-256 regular Hadamard), "
                          "per-row weight scales, comfy_kitchen kernels",
                "format": "ComfyUI quantized checkpoint v1.0 (QUANTIZATION.md)",
                "source": os.path.abspath(args.src),
                "group_size": GROUP_SIZE,
                "kept_bf16": sorted(kept),
                "quantized_layers": n_quant,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            fh,
            indent=2,
        )
    print(f"done -> {args.out} (upload this folder to HuggingFace as-is)")


if __name__ == "__main__":
    main()
