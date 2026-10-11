# ComfyUI-pyiqa-toolbox

ComfyUI nodes wrapping [pyiqa (IQA-PyTorch)](https://github.com/chaofengc/IQA-Pytorch)
quality metrics. Every score is produced by the metric object pyiqa itself
builds (`pyiqa.create_metric`) — the nodes only convert ComfyUI's image format
and format the outputs, so results match the library exactly.

Pattern: **one loader node per architecture family, score nodes that consume the
loaded model** (a model is loaded once and shared, re-queues reuse it, and
Edit > Unload Models offloads it — see Memory management).

## Nodes

FR nodes take `target` (pyiqa's x, the distorted/candidate side) and
`reference` (pyiqa's ref); NR nodes take `image`. All loader nodes emit a model
for the matching score node(s). Sizes are per-image unless noted. Every score
node loads its inputs lazily — see *Lazy evaluation* below.

| Family / node | Inputs | Outputs | Direction / range | Order |
|---|---|---|---|---|
| **QReAlign** Loader | model_name (mini-0.8B / lite-4B / pro-9B, each also as an int8_convrot entry, + customs), dtype | model | — | — |
| QReAlign NR Score | model, image, task (quality IQA / aesthetic IAA) | score | 0-1, higher better | NR |
| **FGResQ** Loader | checkpoint ((default) + customs) | model | — | — |
| FGResQ NR Score | model, image | score | 0-1, higher better | NR |
| FGResQ FR Score | model, target, reference | target_score, reference_score, rank (INT 0=reference better / 1=target better / 2=similar), verdict (STRING), rank_prob | 0-1 | **order-sensitive** |
| **TOPIQ** Loader | model_name (topiq_fr/-pipal, topiq_nr/-flive/-spaq, + 3 face variants, + customs) | model | — | — |
| TOPIQ FR Score | model, target, reference | score | ~0-1, higher better | **order-sensitive** |
| TOPIQ NR Score | model, image | score | ~0-1, higher better | NR |
| **A-FINE** Loader | metric (afine_fr/afine_nr + customs) | model | — | — |
| A-FINE FR Score | model, target, reference, resolution | score | **lower better**, ~-2..2 | order-independent |
| A-FINE NR Score | model, image, resolution | score | **lower better**, ~-2..2 | NR |
| **WaDIQaM** Loader | metric (wadiqam_fr/nr + customs) | model | — | — |
| WaDIQaM FR Score | model, target, reference | score | ~-1..1 | **order-sensitive** |
| WaDIQaM NR Score | model, image | score | ~-1..1 | NR |
| **MUSIQ** Loader | metric (musiq, -ava, -paq2piq, -spaq + customs) | model | — | — |
| MUSIQ NR Score | model, image | score | higher better; AVA 1-10, others ~0-100 | NR |
| **CLIP-IQA** Loader | metric (clipiqa, clipiqa+, +vitL14_512/rn50_512 + customs) | model | — | — |
| CLIP-IQA NR Score | model, image | score | 0-1, higher better | NR |
| **QualiCLIP** Loader | metric (qualiclip, qualiclip+, -clive/-flive/-spaq + customs) | model | — | — |
| QualiCLIP NR Score | model, image | score | 0-1, higher better | NR |
| **LPIPS** Loader | metric (lpips/-vgg, lpips+/-vgg+, stlpips/-vgg + customs) | model | — | — |
| LPIPS FR Score | model, target, reference | score | **lower better**, 0-1 | lpips/stlpips exact; lpips+ ~0.1% |
| **DISTS** Loader | metric (dists, msswd + customs) | model | — | — |
| DISTS FR Score | model, target, reference | score | **lower better**; dists 0-1, msswd 0-~10 | order-independent |
| **Classic FR** Loader | metric (ssim, ssimc, ms_ssim, fsim, psnr, psnry, mad) | model | — | — |
| Classic FR Score | model, target, reference | score | higher better except mad | order-independent |
| L1 Score | target, reference | score | mean abs error, lower better | order-independent |
| **FID** Loader | metric (fid, sfid) | model | — | — |
| FID Score | model, target (IMAGE set), reference (IMAGE set), resize_mode | score (one value) | **lower better**, unbounded | set-level |
| Unload Pyiqa Models | unload_pyiqa_models, empty_cuda_cache, passthrough (any) | passthrough | releases cached metrics to RAM | — |

Nodes are grouped by family in the Add-Node menu (`pyiqa/<Family>`), one loader
per family with its score node(s) beside it, plus `pyiqa/Utilities`. Display
names all follow `<Family> Loader` / `<Family> <mode> Score`. Every score node
also takes a `memory_management` combo (see below).

**Order sensitivity in the table above is MEASURED**, not assumed: each FR
metric was scored as f(a,b) vs f(b,a) on identical inputs. Order-sensitive:
FGResQ FR (its rank label and the two per-side scores swap), TOPIQ FR (~0.8%),
WaDIQaM FR (~2%). Nominal-but-tiny (~0.1%): lpips+, lpips-vgg+. Everything
else, including A-FINE FR despite its "you cannot switch dis/ref" docstring, is
exactly order-independent.

**FR inputs are `target` + `reference`** (pyiqa's argument order). Score
outputs are per-image/per-pair lists (`OUTPUT_IS_LIST`), so each value flows
downstream as its own datum and scalar consumers run once per image.

**Lazy evaluation (every score node).** ComfyUI runs any node whose outputs are
not consumed by another node — i.e. every score node is executed, regardless of
where it sits on the canvas. That is why a loader (e.g. QReAlign Pro 9B) used to
load *before* the SDXL checkpoint and OOM a generation-plus-scoring workflow: the
loader is a dependency of a terminal score node. Every score node now uses
ComfyUI's
[lazy evaluation](https://docs.comfy.org/custom-nodes/backend/lazy_evaluation):

- its image inputs and its `model` input are marked `lazy`, and
  `check_lazy_status` requests **one at a time, in order** — image(s) first, then
  the model. The loader feeding `model` is not evaluated until the image exists,
  so a big metric loads *after* the generator/decoder (and after any
  memory-cleanup node the image passes through), not before it.

ComfyUI collects the names returned by `check_lazy_status` into a **set**, so
requesting image and model together would lose the ordering and could still load
the model early; the mixin therefore returns exactly one name per call. If a lazy
input is never produced the node still runs with it as `None`, and the score
function returns empty outputs — the batch-size and FR/NR mode checks are skipped
in that case, so an unloaded node can never raise.

**Memory relief before a big load.** pyiqa metrics are built outside
`comfy.model_management`, so ComfyUI does not know how much VRAM they hold and
will not free memory for them. Loaders for the large families (QReAlign, TOPIQ,
A-FINE, MUSIQ, CLIP-IQA, QualiCLIP, FID) therefore call ComfyUI's own
`free_memory` before allocating — the same eviction ComfyUI performs when
loading one of its own models — so scoring right after a generation does not OOM
against a still-resident SDXL checkpoint. ComfyUI reloads its models on demand
on the next run.

**Releasing VRAM after a score** — `memory_management` (every score node).
Because our metric is invisible to ComfyUI, ComfyUI will never free *its* VRAM;
without help it sits on the card and the next generation can OOM against it.
Each score node therefore takes a `memory_management` combo:

- `free_after_score` (default) — after scoring, park the metric in system RAM,
  `gc.collect()`, and `torch.cuda.empty_cache()`. The next generation gets the
  whole card; the metric reloads automatically (lazily) on the next score — keep
  its weights on a fast disk.
- `keep_loaded` — leave the metric on the GPU. Fastest to re-score, but it holds
  its VRAM between runs.

Caveat: when several score nodes share one loader (e.g. a QReAlign model feeding
a quality and an aesthetic node), `free_after_score` on each makes the shared
model bounce GPU↔RAM between them. Use `keep_loaded` on the earlier nodes in that
case, or put a single `Unload Pyiqa Models` node at the end instead.

**int8_convrot Q-ReAlign models always stay on the GPU** — ComfyUI's int8
`QuantizedTensor` weights cannot round-trip through system RAM (the int8 kernels
are CUDA-only; a CPU detour silently corrupts the values), so `free_after_score`
and Unload Models are no-ops for an int8 model: it keeps its VRAM until another
loader needs the space (which evicts it, reloading from disk next score). Use
Use the plain (bf16) entry instead if you need full eviction.

## int8_convrot Q-ReAlign (a third less size and VRAM)

The QReAlign Loader's `model_name` dropdown lists each variant twice — e.g.
`qrealign-mini-0.8B` and `qrealign-mini-0.8B (int8_convrot)`. The int8 entries
load a quantized copy of the model in ComfyUI's **native int8_tensorwise +
ConvRot** format — the same recipe as Comfy-Org's official int8_convrot text
encoders (per-channel int8 weights with a group-256 Hadamard rotation applied
to weights *and* activations, so int8 matmuls run on tensor cores without an
accuracy cliff). The rotation and the int8 matmul are handled entirely by
ComfyUI's tensor-subclass dispatch (`comfy.quant_ops` + `comfy_kitchen`) — the
model's forward is untouched. That is the whole UI: no extra combos.

- **Where:** resolved from the registered `pyiqa` folders as
  `Q-ReAlign-<variant>-int8-convrot/`, else downloaded from the release repo
  (default `AnsteinHuynh/Q-ReAlign-int8-convrot`, one variant folder per repo
  root; override with the `PYIQA_TOOLBOX_INT8_REPO` env var) via
  `snapshot_download`.
- **Sizes on this machine** (`I:/MyApps/ComfyUi/models/pyiqa/`): Pro 9B
  17.5 → 11.55 GiB, Lite 4B 9.7 → 6.28 GiB, Mini 0.8B 2.1 → 1.56 GiB.
  Kept in bf16: `lm_head`, the visual merger, the linear-attention a/b gate
  projections, `k_proj`/`v_proj`, and the entire first + last decoder blocks
  (the "keep bundle" — small layers with outsized influence; the big MLP
  projections carry the size win).
- **Accuracy (8 real photos, per-image |Δ| vs bf16):** Pro mean 0.0008 / max
  0.0040 — near-lossless; Mini mean 0.0031 / max 0.0081; Lite mean 0.0085 /
  max 0.0208. No ranking flips on the probe set. Note Pro's vision tower is
  bf16 by construction (hidden 1152 not divisible by the 256 rotation group),
  which is likely why it tracks bf16 so closely; Mini/Lite vision towers
  (768/1024) ARE quantized. Protocol: `scratch/validate_int8_variant.py`.
- **Making your own:** `tools/quantize_qrealign_int8_convrot.py` quantizes any
  Q-ReAlign variant (or custom fine-tune) into this format:
  `python tools/quantize_qrealign_int8_convrot.py --src <bf16 snapshot> --out
  <dest folder>`. It meta-builds the model, quantizes every eligible Linear
  (ConvRot requires hidden dims divisible by the group size; Pro/Lite/Mini all
  qualify) and writes the ComfyUI-standard checkpoint: int8 `<name>.weight` +
  float `<name>.weight_scale` + a `<name>.comfy_quant` marker per layer, plus
  `_quantization_metadata` in the safetensors header.
- **Uploading to HuggingFace** (so other machines auto-download from the loader):
  ```bash
  huggingface-cli upload AnsteinHuynh/Q-ReAlign-int8-convrot \
      "I:/MyApps/ComfyUi/models/pyiqa/Q-ReAlign-Mini-0.8B-int8-convrot" \
      Q-ReAlign-Mini-0.8B-int8-convrot --repo-type model
  # repeat the last two arguments per variant (Lite-4B, Pro-9B)
  ```
  The folder name at the repo root must match `Q-ReAlign-<variant>-int8-convrot`
  — that is the name the loader looks for after download.

**Unload Pyiqa Models** (`pyiqa/Utilities`) — a passthrough node that releases
the whole metric cache on demand (default `unload_pyiqa_models` +
`empty_cuda_cache`, both True). Wire any value through it; the value is returned
unchanged, so it can sit on the image wire before the score nodes or at the end
of a pipeline. It touches only this pack's metrics — ComfyUI's own models are
left to ComfyUI, and Edit > Unload Models still offloads everything (including
ours).

**Resolution input (A-FINE):** AFINE needs sides divisible by 32 (CLIP
ViT-B/32 patches). `auto (up to multiple of 32)` (default) bicubic-resizes up to
the next multiple — the graceful handling AFINE's own forward intends but
cannot reach — with the overshoot clamped back to [0,1]; `native` and fixed
224-768 are also offered. AFINE is scored one image at a time internally: its
forward's squeeze/unsqueeze assume batch size 1 and corrupt larger batches.

**FID is set-level.** pyiqa's FID/sFID are folder-based (Inception features over
two folders, then a Fréchet distance). The node takes IMAGE batches, stages them
as temporary 8-bit PNG folders and calls pyiqa's own folder API — the same code
path as the CLI, so values match a folder-based run; images are 8-bit, as files
are. It returns ONE score for the two sets (not per image) and needs >= 2 images
per side (>= ~16 to be meaningful). `resize_mode` is pyiqa's own
clean / legacy_pytorch / legacy_tensorflow.

References: Q-Align (arXiv:2312.17009) / Q-ReAlign; FGResQ (AAAI 2026,
<https://github.com/sxfly99/FGResQ>); TOPIQ (AAAI 2023, CFANet); A-FINE; MUSIQ
(ICCV 2021); CLIP-IQA (AAAI 2023); QualiCLIP (2023); WaDIQaM (Bosse et al.);
LPIPS (CVPR 2018) / ST-LPIPS; DISTS (ICLR 2021); FID / clean-FID.

## Custom checkpoints (models/pyiqa)

Drop custom weights into any registered `pyiqa` folder (they appear in loader
dropdowns after a ComfyUI refresh — no restart):

- **QReAlign customs** — any subfolder with a `config.json` (HF-snapshot-style,
  e.g. your own Qwen3.5-VL quality fine-tune trained on dataset X):
  `models/pyiqa/MyModel/`.
- **Arch checkpoint customs** — `.pth` files named for the family, passed as the
  arch's `pretrained_model_path`: `FGResQ*.pth` (degradation branch stays
  default), `*topiq*` or `cfanet*.pth`, `*afine*.pth`. TOPIQ/A-FINE loaders
  ask which head (FR/NR) the checkpoint loads onto via `custom_base_variant`,
  since a file name can't say which head it was trained with.

## Weights: where they live (single copy)

The loaders resolve weights through the registered ComfyUI `pyiqa` model
folder(s). Each registered folder is checked for two layouts:

- plain — `<folder>/Q-ReAlign-Mini-0.8B/` snapshot-style dirs, `FGResQ.pth`,
  `FGResQ_Degradation.pth`;
- HF hub cache — `models--q-future--Q-ReAlign-*/snapshots/<hash>/` (complete
  snapshots only; incomplete ones fall back to finishing the download).

Point the folder at any weight stores you already have via
`extra_model_paths.yaml` (nothing is duplicated or re-downloaded), e.g.
reusing the HuggingFace and torch-hub caches pyiqa already downloads into:

```yaml
    pyiqa: |
        C:/Users/<you>/.cache/huggingface/hub
        C:/Users/<you>/.cache/torch/hub/pyiqa
```

Without a yaml entry, `<ComfyUI>/models/pyiqa` is used as the folder. Every
registered `pyiqa` folder — yaml-redirected or fallback — is **created if
missing** at startup, so `pyiqa: models/pyiqa/` under a `base_path` makes that
folder materialize and become the drop-off place for weights (snapshot-style
`Q-ReAlign-*/` dirs, `FGResQ.pth`, `FGResQ_Degradation.pth`). Absent weights
auto-download through pyiqa/HuggingFace on first use.

## Memory management

Loaded metrics are cached per family and react to ComfyUI's
**Edit > Unload Models** and **Edit > Unload Models and Execution Cache**:
both funnel through `comfy.model_management.unload_all_models()`, which this
pack wraps (this ComfyUI version has no callback API) — cached metrics are
offloaded to system RAM (ComfyUI's own unload semantics: kept warm, VRAM
freed) before the original unload runs. Score nodes and loaders detect a
CPU-resident model and move it back onto the GPU automatically, so a workflow
re-run after an unload just works. Switching a loader's variant also evicts
the previous model of that family from VRAM.

## Notes

- **transformers ≥ ~5.6 fix (FGResQ):** the published FGResQ checkpoints were
  saved with transformers 4 and carry a `vision_model.` key prefix that
  transformers ≥ ~5.6 CLIPVisionModel no longer has — pyiqa only warns, so the
  fine-tuned backbone silently fails to load and you score with the stock HF
  CLIP. The FGResQ Loader detects this layout and reloads both checkpoints with
  the prefix stripped (verified clean; scores change accordingly). No-op on
  transformers ≤ 5.5.4 where the checkpoint loads as-is.
- Weights auto-download on first use (pyiqa release/HF mirror for FGResQ;
  `q-future/Q-ReAlign-*` HF repos for Q-ReAlign). Q-ReAlign additionally needs
  `transformers>=5.0` — deliberately **not** in requirements.txt (forcing that
  upgrade on every install can break other packs); the QReAlign Loader checks
  the version at load time and raises with the exact pip command to fix it.
- `target` and `reference` must share a batch size in the pair node.
- Future families slot in as new `nodes/*_nodes.py` files with their own
  loader, subclassing the shared bases in `nodes/_common.py` (~40 lines of
  variants + tooltips per family).
- `l1` is not in pyiqa's metric registry, but pyiqa DOES ship
  `pyiqa.losses.losses.L1Loss` — the L1 Score node uses that class directly
  (pyiqa implements L1 as a loss; the registry's nearest metric is `mad`).

## Dev layout

`nodes/` (node modules), `utility/` (shared helpers), `web/` and
`example_workflows/` (reserved). Deployed to ComfyUI via directory junctions
into each install's `custom_nodes`.
