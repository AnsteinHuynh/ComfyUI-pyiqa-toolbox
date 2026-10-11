"""Validate an int8_convrot Q-ReAlign checkpoint against its bf16 reference.

Usage:
    python scratch/validate_int8_variant.py <variant> [int8_folder]

Examples:
    python scratch/validate_int8_variant.py Mini-0.8B            # canonical folder
    python scratch/validate_int8_variant.py Pro-9B ...-int8-convrot-v2

For each: int8 score, bf16 reference score, |delta|; for Mini also the
release_metric round-trip hazard check (int8 must stay GPU-resident and
bit-identical across a release attempt). One model in VRAM at a time.

bf16 references: Mini/Lite from the HF hub snapshot (complete, cached),
Pro from the local folder.
"""

import glob
import logging
import os
import sys

import torch

TOOLBOX = r"C:\Users\kerok\apps\ComfyUI-pyiqa-toolbox"
COMFY = r"C:\MyApps\ComfyUIRoot\20250726-ComfyUI\ComfyUI"
PYIQA_DIR = r"I:\MyApps\ComfyUi\models\pyiqa"

sys.path.insert(0, TOOLBOX)
sys.path.insert(0, COMFY)

logging.basicConfig(level=logging.WARNING)

import pyiqa  # noqa: E402

from utility import pyiqa_utils, qrealign_int8  # noqa: E402

DEV = "cuda"

BF16_SRC = {
    "Mini-0.8B": (
        "C:/Users/kerok/.cache/huggingface/hub/models--q-future--Q-ReAlign-Mini-0.8B"
        "/snapshots/fe1f45a7574c9e9d908875af9f7e90cb946aa19f"
    ),
    "Lite-4B": (
        "C:/Users/kerok/.cache/huggingface/hub/models--q-future--Q-ReAlign-Lite-4B"
        "/snapshots/06add46a6bc4a7dfcf0a59a2fb1ed15a6e3d30b7"
    ),
    "Pro-9B": os.path.join(PYIQA_DIR, "Q-ReAlign-Pro-9B"),
}


def load_image_tensors(n=8):
    from PIL import Image
    import numpy as np

    cands = sorted(glob.glob(r"I:\MyApps\sd-train\Flickr2K\Flickr2K_valid\*.png"))[:n]
    imgs = []
    for p in cands:
        im = Image.open(p).convert("RGB").resize((512, 512))
        imgs.append(
            torch.from_numpy(np.asarray(im))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .clone()
            / 255.0
        )
    return torch.cat(imgs).to(DEV)


def score(metric, img):
    with torch.no_grad():
        return metric(img).flatten().tolist()


def main():
    variant = sys.argv[1]
    int8_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        PYIQA_DIR, f"Q-ReAlign-{variant}-int8-convrot"
    )
    img = load_image_tensors()
    pyiqa_utils._METRIC_CACHE.clear()

    m8 = qrealign_int8.create_int8_metric(int8_dir, DEV)
    s8 = score(m8, img)

    released = pyiqa_utils.release_metric(m8)  # must be a no-op for int8
    on_gpu = next(m8.parameters()).device.type == "cuda"
    s8b = score(m8, img)
    tag = f"[{variant}]"
    print(f"{tag} int8  = {[round(v, 6) for v in s8]}")
    print(
        f"{tag} release skip ok: {released is False and on_gpu and s8 == s8b}"
    )
    del m8
    torch.cuda.empty_cache()

    bf = pyiqa.create_metric(
        "qrealign", device=DEV, dtype="bfloat16", model=BF16_SRC[variant]
    )
    sbf = score(bf, img)
    deltas = [abs(a - b) for a, b in zip(s8, sbf)]
    print(f"{tag} bf16  = {[round(v, 6) for v in sbf]}")
    print(
        f"{tag} |delta| mean={sum(deltas)/len(deltas):.6f} "
        f"max={max(deltas):.6f} (n={len(deltas)})"
    )
    del bf
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
