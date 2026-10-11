"""Validate the int8_convrot Q-ReAlign checkpoints on the GPU.

1. Round-trip hazard check (Mini): score -> release_metric (must be SKIPPED for
   int8) -> score again -> identical values.
2. bf16 vs int8 score deltas for Lite and Pro (Mini was already validated).

Run with the ComfyUI env python. One variant in VRAM at a time.
"""

import glob
import logging
import os
import sys

import torch

import os as _os

def _p(key, default):
    """env override PYIQA_TOOLBOX_<KEY>, else the dev machine default."""
    return _os.environ.get("PYIQA_TOOLBOX_" + key, default)

TOOLBOX = _p("ROOT", r"C:\Users\kerok\apps\ComfyUI-pyiqa-toolbox")
COMFY = _p("COMFY", r"C:\MyApps\ComfyUIRoot\20250726-ComfyUI\ComfyUI")
PYIQA_DIR = _p("MODELS", r"I:\MyApps\ComfyUi\models\pyiqa")

sys.path.insert(0, TOOLBOX)
sys.path.insert(0, COMFY)
os.environ.setdefault("HF_HOME", r"I:\MyApps\ComfyUi\models\pyiqa\hf-cache")

logging.basicConfig(level=logging.INFO)

import pyiqa  # noqa: E402

from utility import pyiqa_utils, qrealign_int8  # noqa: E402

DEV = "cuda"


def load_image_tensor():
    cands = sorted(
        glob.glob(r"I:\MyApps\sd-train\Flickr2K\Flickr2K_valid\*.png")
    )
    from PIL import Image

    img = Image.open(cands[0]).convert("RGB").resize((512, 512))
    t = torch.from_numpy(
        __import__("numpy").asarray(img)
    ).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return t.to(DEV), cands[0]


def score(metric, img):
    with torch.no_grad():
        return metric(img).flatten().tolist()


def free(*objs):
    for o in objs:
        del o
    torch.cuda.empty_cache()


def main():
    img, img_path = load_image_tensor()
    print(f"image: {img_path}")
    pyiqa_utils._METRIC_CACHE.clear()

    # ---- 1. round-trip hazard check on Mini ----
    mini8 = qrealign_int8.create_int8_metric(
        os.path.join(PYIQA_DIR, "Q-ReAlign-Mini-0.8B-int8-convrot"), DEV
    )
    s1 = score(mini8, img)
    released = pyiqa_utils.release_metric(mini8)
    on_gpu = next(mini8.parameters()).device.type == "cuda"
    s2 = score(mini8, img)
    print(f"[mini8] release_metric returned {released} (want False), "
          f"still on cuda: {on_gpu}")
    print(f"[mini8] before release: {s1}")
    print(f"[mini8] after release : {s2}")
    assert released is False and on_gpu, "int8 metric was parked or moved!"
    assert s1 == s2, "scores changed across release/ensure - corruption!"
    print("[mini8] round-trip OK (identical scores)")
    free(mini8)

    # ---- 2. Lite: bf16 vs int8 ----
    lite8 = qrealign_int8.create_int8_metric(
        os.path.join(PYIQA_DIR, "Q-ReAlign-Lite-4B-int8-convrot"), DEV
    )
    s_lite8 = score(lite8, img)
    free(lite8)
    lite_bf = pyiqa.create_metric(
        "qrealign", device=DEV, dtype="bfloat16",
        model="q-future/Q-ReAlign-Lite-4B",
    )
    s_lite_bf = score(lite_bf, img)
    free(lite_bf)
    d = abs(s_lite8[0] - s_lite_bf[0])
    print(f"[lite] bf16={s_lite_bf[0]:.6f} int8={s_lite8[0]:.6f} |delta|={d:.6f}")

    # ---- 3. Pro: bf16 vs int8 ----
    pro8 = qrealign_int8.create_int8_metric(
        os.path.join(PYIQA_DIR, "Q-ReAlign-Pro-9B-int8-convrot"), DEV
    )
    s_pro8 = score(pro8, img)
    free(pro8)
    pro_bf = pyiqa.create_metric(
        "qrealign", device=DEV, dtype="bfloat16",
        model=os.path.join(PYIQA_DIR, "Q-ReAlign-Pro-9B"),
    )
    s_pro_bf = score(pro_bf, img)
    free(pro_bf)
    d = abs(s_pro8[0] - s_pro_bf[0])
    print(f"[pro ] bf16={s_pro_bf[0]:.6f} int8={s_pro8[0]:.6f} |delta|={d:.6f}")

    print("ALL CHECKS DONE")


if __name__ == "__main__":
    main()
