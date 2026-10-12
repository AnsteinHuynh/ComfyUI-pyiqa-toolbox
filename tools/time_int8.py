"""Time int8 vs bf16 Q-ReAlign scoring and count kitchen kernel dispatches."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # dev_paths sibling
import dev_paths  # noqa: E402

dev_paths.bootstrap(comfy=True)  # pack root, plus a ComfyUI install (comfy_kitchen)
PYIQA_DIR = dev_paths.models_dir()

import pyiqa  # noqa: E402
from utility import qrealign_int8  # noqa: E402

DEV = "cuda"

# instrument: count fast-kernel calls vs dequantize fallback
import comfy_kitchen  # noqa: E402

CALLS = {"int8_linear": 0}
_real = torch.ops.comfy_kitchen.int8_linear


def _counting(*a, **kw):
    CALLS["int8_linear"] += 1
    return _real(*a, **kw)


torch.ops.comfy_kitchen.int8_linear = _counting


def load_imgs(n=8):
    from PIL import Image
    import numpy as np

    paths = dev_paths.image_paths(n)
    return torch.cat([
        torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((512, 512))))
        .permute(2, 0, 1).unsqueeze(0).float().clone() / 255.0
        for p in paths
    ]).to(DEV)


def timeit(metric, img, n=3):
    with torch.no_grad():
        metric(img)  # warmup
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            metric(img)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n


img = load_imgs()

m8 = qrealign_int8.create_int8_metric(
    os.path.join(PYIQA_DIR, "Q-ReAlign-Mini-0.8B-int8-convrot"), DEV
)
CALLS["int8_linear"] = 0
t8 = timeit(m8, img)
c8 = CALLS["int8_linear"]
del m8
torch.cuda.empty_cache()

mbf = pyiqa.create_metric(
    "qrealign", device=DEV, dtype="bfloat16",
    model="q-future/Q-ReAlign-Mini-0.8B",
)
tbf = timeit(mbf, img)
del mbf
torch.cuda.empty_cache()

print(f"int8: {t8:.2f} s/iter, kitchen int8_linear dispatched {c8} times")
print(f"bf16: {tbf:.2f} s/iter")
print(f"ratio int8/bf16 = {t8 / tbf:.1f}x")
