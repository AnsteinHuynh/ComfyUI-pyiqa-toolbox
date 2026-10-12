"""Faithfulness check vs the pyiqa docs benchmark (Results calibration).

Runs OUR NODES (loader + score, the ComfyUI image conversion included) on the
5 distorted/reference pairs shipped in site-packages/ResultsCalibra, and
compares against the "(ours)" column of
https://iqa-pytorch.readthedocs.io/en/latest/benchmark.html#results-calibration.
"""
import importlib.util
import os
import sys

import numpy as np
import torch
from PIL import Image

PKG = os.environ.get(
    "PYIQA_TOOLBOX_ROOT", r"C:\Users\kerok\apps\ComfyUI-pyiqa-toolbox"
)
FIX = r"C:\Users\kerok\miniconda3\envs\20260504comfyui\Lib\site-packages\ResultsCalibra"

spec = importlib.util.spec_from_file_location(
    "pyiqa_toolbox_calibra", os.path.join(PKG, "__init__.py"),
    submodule_search_locations=[PKG],
)
pack = importlib.util.module_from_spec(spec)
sys.modules["pyiqa_toolbox_calibra"] = pack
spec.loader.exec_module(pack)


def load(folder):
    imgs = []
    for name in ["I03.bmp", "I04.bmp", "I06.bmp", "I08.bmp", "I19.bmp"]:
        im = Image.open(os.path.join(folder, name)).convert("RGB")
        imgs.append(
            torch.from_numpy(np.asarray(im)).permute(2, 0, 1).unsqueeze(0).float().clone() / 255.0
        )
    return torch.cat(imgs)  # (5, C, H, W) pyiqa layout


def comfy(t):  # pyiqa layout -> ComfyUI IMAGE layout
    return t.permute(0, 2, 3, 1).contiguous()


dist = load(os.path.join(FIX, "dist_dir"))
ref = load(os.path.join(FIX, "ref_dir"))

# docs table "(ours)" values, order I03, I04, I06, I08, I19
EXPECT_FR = {
    "ssim": [0.6997, 0.9978, 0.9989, 0.9671, 0.6522],
    "ms_ssim": [0.6707, 0.9996, 0.9998, 0.9567, 0.8418],
    "fsim": [0.6891, 0.9702, 0.9927, 0.9575, 0.822],
    "psnr": [21.1136, 20.9872, 27.0139, 23.3002, 21.6186],
    "mad": [195.2796, 80.8379, 30.3918, 84.3542, 202.2371],
    "gmsd": [0.2203, 0.0005, 0.0004, 0.1346, 0.205],
    "vif": [0.0172, 0.9891, 0.9924, 0.9104, 0.175],
    "vsi": [0.9244, 0.9497, 0.9877, 0.9541, 0.9348],
    "cw_ssim": [0.2763, 0.9996, 1.0, 0.9068, 0.8658],
}
EXPECT_NR = {
    "laion_aes": [3.7204, 5.5917, 5.0756, 4.6551, 3.0973],
    "musiq": [12.4785, 75.7772, 73.7454, 75.4609, 38.0257],
}

fails = []


def check(name, ours, expect, tol_rel=2e-3):
    ok = True
    for o, e in zip(ours, expect):
        if e == 0:
            if abs(o - e) > 1e-4:
                ok = False
        elif abs(o - e) / max(abs(e), 1e-6) > tol_rel:
            ok = False
    status = "OK " if ok else "DIFF"
    if not ok:
        fails.append(name)
    print(f"{status} {name:10s} ours={[round(v,4) for v in ours]}")
    if not ok:
        print(f"     {'':10s} docs={[round(v,4) for v in expect]}")


cd, cr = comfy(dist), comfy(ref)
for metric, expect in EXPECT_FR.items():
    (m,) = pack.NODE_CLASS_MAPPINGS["ClassicFRLoader"]().load(metric)
    (s,) = pack.NODE_CLASS_MAPPINGS["ClassicFRScore"]().score(m, cd, cr)
    check(metric, s, expect)

# dists via its family
(dm,) = pack.NODE_CLASS_MAPPINGS["DISTSLoader"]().load("dists", "(none)", "dists")
(s,) = pack.NODE_CLASS_MAPPINGS["DISTSFrScore"]().score(dm, cd, cr)
check("dists", s, [0.4742, 0.1424, 0.0683, 0.0287, 0.3123])

# lpips (AlexNet)
(lm,) = pack.NODE_CLASS_MAPPINGS["LPIPSLoader"]().load("lpips (AlexNet)", "(none)", "lpips (AlexNet)")
(s,) = pack.NODE_CLASS_MAPPINGS["LPIPSFrScore"]().score(lm, cd, cr)
check("lpips", s, [0.7236, 0.2572, 0.0508, 0.0521, 0.4253])

# NR metrics on the DISTORTED images
for metric, expect in EXPECT_NR.items():
    loader, score = ("LAIONLoader", "LAIONNRScore") if metric == "laion_aes" else ("MUSIQLoader", "MUSIQLoaderNRScore" if False else "MUSIQNRScore")
    label = (
        "laion_aes (improved aesthetic predictor)"
        if metric == "laion_aes" else "musiq (KonIQ-10k)"
    )
    (m,) = pack.NODE_CLASS_MAPPINGS[loader]().load(label, "(none)", None)
    (s,) = pack.NODE_CLASS_MAPPINGS[score]().score(m, cd)
    check(metric, s, expect)

print("\nFAILS:", fails if fails else "none - all nodes match the docs table")
