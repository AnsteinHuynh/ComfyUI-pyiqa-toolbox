---
language: en
tags:
- image-quality-assessment
- iq
- iqa
- pyiqa
- comfyui
- quantization
- int8
- convrot
- qrealign
license: apache-2.0
base_model: q-future/Q-ReAlign-Pro-9B
base_model_relation: quantized
---

# Q-ReAlign int8_convrot (Mini / Lite / Pro)

Int8 builds of the Q-ReAlign quality/aesthetic scorers — **a third less disk
and VRAM** than bf16, with scores that barely move: Pro's mean deviation from
bf16 is **0.0008** on real photos.

How? A **fidelity-first** recipe: the small-but-influential layers stay bf16,
and only the big, int8-safe projections get quantized. Prefer more compression
over more fidelity? Re-quantize with a smaller keep list (`--keep lm_head`)
using the tool under Provenance.

These are quantized copies of the upstream weights ([Q-Align / Q-ReAlign](https://github.com/Q-Future),
Apache-2.0, all credit to its authors), built with ComfyUI's own quantization
stack — the same int8_convrot format as Comfy-Org's official text encoders.

## Variants

| Folder | bf16 → int8 | Quantized Linears | \|Δ\| vs bf16 (mean / max, 8 photos) |
|---|---|---|---|
| `Q-ReAlign-Pro-9B-int8-convrot/` | 17.5 → 11.55 GiB | 173 / 359 | 0.0008 / 0.0040 |
| `Q-ReAlign-Lite-4B-int8-convrot/` | 9.7 → 6.28 GiB | 269 / 347 | 0.0085 / 0.0208 |
| `Q-ReAlign-Mini-0.8B-int8-convrot/` | 2.1 → 1.56 GiB | 175 / 237 | 0.0031 / 0.0081 |

Per-image deltas were measured with
[`ComfyUI-pyiqa-toolbox`'s validation script](https://github.com/AnsteinHuynh/ComfyUI-pyiqa-toolbox)
(`scratch/validate_int8_variant.py`); no ranking flips were observed on the
probe set. Scores stay on the same 0–1 scale as bf16.

## The recipe

Per-layer int8 (per-channel scales) with **ConvRot** — a group-256 regular
Hadamard rotation applied to weights and activations so int8 matmuls run on
tensor cores without an accuracy cliff. This is the same format as Comfy-Org's
official `int8_convrot` text encoders: each quantized Linear stores an int8
`<name>.weight`, a float `<name>.weight_scale`, and a `<name>.comfy_quant` JSON
marker, plus `_quantization_metadata` in the safetensors header. At runtime the
int8 matmul and the activation rotation are inserted by torch dispatch through
`comfy.quant_ops` / `comfy_kitchen` — the model's forward is untouched.

Kept in bf16 (the "keep bundle" — small layers with outsized influence):

- `lm_head` (the scoring head)
- the **visual merger** (`model.visual.merger.*`) — everything the language
  model sees funnels through these two layers
- the linear-attention **a/b gate projections** (`in_proj_a` / `in_proj_b`)
- `k_proj` / `v_proj` (small GQA projections)
- the **entire first and last decoder blocks**

Everything else (the big MLP and attention projections — where the size win
lives) is quantized. Note the 256-wide rotation requires `in_features % 256 == 0`,
so Pro's vision tower (hidden 1152) is bf16 by construction — likely why Pro
tracks bf16 closest; Mini/Lite vision towers (768/1024) are quantized.

## Using in ComfyUI

Install [ComfyUI-pyiqa-toolbox](https://github.com/AnsteinHuynh/ComfyUI-pyiqa-toolbox),
add a **QReAlign Loader** node, and pick the `(int8_convrot)` entry for your
variant in `model_name` — e.g. `qrealign-pro-9B (int8_convrot)`. That is the
whole setup; the loader has no other quantization inputs. If no local copy
exists under your registered `models/pyiqa` folders, the loader downloads the
matching folder from this repo automatically (default repo id
`AnsteinHuynh/Q-ReAlign-int8-convrot`; override with the
`PYIQA_TOOLBOX_INT8_REPO` env var). Score with the **QReAlign NR Score**
node (task: quality IQA or aesthetic IAA).

**Memory note:** ComfyUI's int8 kernels are CUDA-only, so int8 models are
GPU-resident — the toolbox's `free_after_score` and unload options are no-ops
for them (the model keeps its VRAM until a later loader evicts it). Use the
plain (bf16) entry instead if you need full eviction.

## Using outside ComfyUI

Any environment with `comfy` + `comfy_kitchen` importable can load these
checkpoints; see `utility/qrealign_int8.py` in the toolbox for a standalone
loader (builds the Qwen3.5-VL model, swaps each quantized Linear in as a
`QuantizedTensor`, then hands the model to `pyiqa.create_metric("qrealign")`).
Requires the same major versions as your ComfyUI install (torch cu13x,
`comfy_kitchen`, `transformers>=5.0`).

## Provenance

- Upstream models: Q-Align / Q-ReAlign by the Q-Future team
  (arXiv:2312.17009 and follow-ups).
- Quantized with `tools/quantize_qrealign_int8_convrot.py` from
  ComfyUI-pyiqa-toolbox, using ComfyUI's own kernels — no calibration data,
  no fine-tuning, weights-only PTQ.
- Quantization date: 2026-10-10.
