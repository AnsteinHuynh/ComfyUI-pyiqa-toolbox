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

# This machine's paths come from dev_paths (PYIQA_TOOLBOX_ROOT, or the gitignored
# tools/dev_paths_local.py); the fixture itself ships in site-packages, so that
# half needs no configuration at all.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dev_paths  # noqa: E402

PKG = dev_paths.ROOT
FIX = os.path.join(dev_paths.site_packages(), "ResultsCalibra")

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


def check(name, ours, expect, tol_rel=2e-3, atol=1e-4):
    """Compare against the docs column with a numpy-style atol + rtol.

    The docs table rounds to 4 decimals, so a metric whose value is near zero
    (gmsd: 0.0005) must be judged on the absolute difference - a pure relative
    tolerance calls an exact match a failure there.
    """
    ok = True
    worst = 0.0
    for o, e in zip(ours, expect):
        diff = abs(o - e)
        worst = max(worst, diff)
        if diff > atol + tol_rel * abs(e):
            ok = False
    status = "OK " if ok else "DIFF"
    if not ok:
        fails.append(name)
    print(f"{status} {name:10s} ours={[round(v,4) for v in ours]}")
    if not ok:
        print(f"     {'':10s} docs={[round(v,4) for v in expect]} (worst diff {worst:.2e})")


def load_by_registry(name):
    """``(bundle, mode)`` for a pyiqa registry metric, via its architecture.

    Drives the pack the way a user does - ``(Down)load pyiqa Model`` then a generic
    score node - instead of naming a node per metric, so this tool follows the
    node surface instead of pinning it.
    """
    catalog = sys.modules["pyiqa_toolbox_calibra.utility.metric_catalog"]
    for value in catalog.VARIANT_VALUES:
        arch, _, _label = value.partition(catalog.SEPARATOR)
        if catalog.ARCHITECTURES[arch].signpost:
            # listed in the dropdown so users find the dedicated loader, but
            # there is nothing here to load (and resolve refuses it on purpose)
            continue
        resolved = catalog.resolve(arch, value)
        if resolved.base == name and not resolved.custom:
            bundle = pack.NODE_CLASS_MAPPINGS["PyiqaAnyLoader"]().load(arch, value)[0]
            return bundle, resolved.mode
    raise SystemExit(f"metric {name!r} is not in the catalog any more")


cd, cr = comfy(dist), comfy(ref)
for metric, expect in EXPECT_FR.items():
    bundle, mode = load_by_registry(metric)
    assert mode == "FR", (metric, mode)
    (s,) = pack.NODE_CLASS_MAPPINGS["PyiqaAnyFRScore"]().score(bundle, cd, cr)
    check(metric, s, expect)

# dists via the generic loader too
bundle, _mode = load_by_registry("dists")
(s,) = pack.NODE_CLASS_MAPPINGS["PyiqaAnyFRScore"]().score(bundle, cd, cr)
check("dists", s, [0.4742, 0.1424, 0.0683, 0.0287, 0.3123])

# lpips (AlexNet)
bundle, _mode = load_by_registry("lpips")
(s,) = pack.NODE_CLASS_MAPPINGS["PyiqaAnyFRScore"]().score(bundle, cd, cr)
check("lpips", s, [0.7236, 0.2572, 0.0508, 0.0521, 0.4253])

# NR metrics on the DISTORTED images
for metric, expect in EXPECT_NR.items():
    bundle, mode = load_by_registry(metric)
    assert mode == "NR", (metric, mode)
    (s,) = pack.NODE_CLASS_MAPPINGS["PyiqaAnyNRScore"]().score(bundle, cd)
    check(metric, s, expect)

print("\nFAILS:", fails if fails else "none - all nodes match the docs table")
