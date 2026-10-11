"""GPU verification of the new memory semantics for int8_convrot.

Scenario the user hit: score an int8 model with the park mode -> VRAM must go
DOWN (collapse to full unload), and the next score (handed the hollow shell,
as ComfyUI's output cache would) must reproduce identical values.
"""
import glob
import os
import sys
import time

import torch

sys.path.insert(0, os.environ.get(
    "PYIQA_TOOLBOX_ROOT", r"C:\Users\kerok\apps\ComfyUI-pyiqa-toolbox"
))
sys.path.insert(0, os.environ.get(
    "PYIQA_TOOLBOX_COMFY", r"C:\MyApps\ComfyUIRoot\20250726-ComfyUI\ComfyUI"
))

import logging

logging.basicConfig(level=logging.INFO)

from utility import pyiqa_utils  # noqa: E402
from utility import qrealign_int8  # noqa: E402
from PIL import Image  # noqa: E402
import numpy as np  # noqa: E402

PYIQA_DIR = os.environ.get(
    "PYIQA_TOOLBOX_MODELS", r"I:\MyApps\ComfyUi\models\pyiqa"
)
DEV = "cuda"


def vram():
    return torch.cuda.memory_allocated() / 2**30


def imgs(n=4):
    paths = sorted(glob.glob(r"I:\MyApps\sd-train\Flickr2K\Flickr2K_valid\*.png"))[:n]
    return torch.cat([
        torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((512, 512))))
        .permute(2, 0, 1).unsqueeze(0).float().clone() / 255.0
        for p in paths
    ]).to(DEV)


def score(m, x):
    with torch.no_grad():
        return m(x).flatten().tolist()


image = imgs()

# factory mirrors the loader node's int8 path
int8_dir = os.path.join(PYIQA_DIR, "Q-ReAlign-Mini-0.8B-int8-convrot")


def factory():
    return qrealign_int8.create_int8_metric(int8_dir, DEV)


m = pyiqa_utils.get_cached_metric("qrealign", ("int8", int8_dir, DEV), factory)
s1 = score(m, image)
v_loaded = vram()
print(f"loaded:  VRAM alloc {v_loaded:.2f} GiB, scores {s1}")

# the unload mode on an int8 metric -> full drop
released = pyiqa_utils.drop_metric(m)
gutted = getattr(m, "_pyiqa_gutted", False)
v_after = vram()
print(f"park:    release_metric={released}, gutted={gutted}, "
      f"VRAM alloc {v_after:.2f} GiB (was {v_loaded:.2f})")
assert gutted and v_after < 0.05 * max(v_loaded, 0.1), "VRAM did not free!"

# next score with the SAME hollow object (what ComfyUI's output cache hands
# back) -> revival must rebuild and reproduce the values
t0 = time.perf_counter()
m2 = pyiqa_utils.revive_metric(m)
s2 = score(m2, image)
dt = time.perf_counter() - t0
print(f"revive:  {dt:.1f}s, scores {s2}")
assert s1 == s2, f"revived scores differ: {s1} vs {s2}"
print("INT8 MEMORY SEMANTICS OK (park->drop, revival exact)")
