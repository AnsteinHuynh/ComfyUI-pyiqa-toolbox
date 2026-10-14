# Metric reference

Every metric this pack can score with - through `(Down)load pyiqa Model`, or through a dedicated loader where one exists - what it actually is, and which way its number points. Generated from the pack's own catalog by `tools/gen_metrics_doc.py`, so it always matches the code.

This file describes pack 0.2.1 and pyiqa 0.1.15.post28: 100 weights choices you can score per image, across 50 architectures (32 full-reference, 68 no-reference), plus 5 set-level or text-conditioned metrics. Your own checkpoints add more entries at runtime (see *Custom checkpoints* in the README).

pyiqa also registers `qrealign-mini` (same weights as `qrealign`), so it is folded into that entry rather than offered twice.

## How to read an entry

The variant dropdown shows one line per weights choice, and every line has the same four parts:

```
<architecture> | <variant> - <mode>, <direction>, <range>
```

| part | meaning |
|---|---|
| **architecture** | the network family (chosen in the first dropdown) |
| **variant** | which weights within that family |
| **mode** | FR (needs a reference image) or NR (does not) |
| **direction** | whether a higher or a lower number is better |
| **range** | the scale pyiqa expects the number to fall in |

The **pyiqa name** column is the name pyiqa itself uses, so you can look the metric up in pyiqa's documentation and compare against its published benchmark numbers. Numbers are **not** comparable across metrics: each has its own scale and direction - pick one and compare runs with it.

## What should I use?

- **Compare my result against the original** - PSNR, SSIM, MS-SSIM, CW-SSIM, FSIM, VIF, GMSD - instant, nothing to download, but they mainly measure pixel and structural fidelity.
- **Compare two images the way people see them** - LPIPS, LPIPS+ (better agreement with human judgement), DISTS, A-FINE FR, PieAPP - all want aligned crops.
- **One image against one image, with a verdict** - `FGResQ FR Score`, which also returns each side's quality, a rank and a similarity verdict.
- **Score an image on its own** - MUSIQ, TOPIQ, CLIP-IQA, ARNIQA, MANIQA, HyperIQA, TReS, LIQE (learned); NIQE, BRISQUE, PIQE, ILNIQE, NRQM (classic statistical).
- **Judge aesthetics rather than quality** - NIMA (AVA), LAION Aesthetic, TOPIQ IAA, MUSIQ-AVA.
- **Quality the way a large vision-language model scores it** - Q-Align (8.2B, ~15 GiB) and Q-ReAlign (0.8B-9B): best agreement with human ratings, slowest.
- **How well an image matches a caption** - CLIP Score - see *Set-level and text-conditioned metrics*.
- **Whether a whole batch looks like another whole batch** - FID / sFID - see *Set-level and text-conditioned metrics*.

### If you want a starting point rather than an opinion

pyiqa publishes a benchmark of its own metrics ([https://iqa-pytorch.readthedocs.io/en/latest/benchmark.html](https://iqa-pytorch.readthedocs.io/en/latest/benchmark.html)). Ranked by average SRCC - how well a metric's ordering agrees with human ratings - the leaders are:

| kind | metric | avg SRCC | avg PLCC |
|---|---|---|---|
| FR | TOPIQ-FR | 0.946 | 0.933 |
| FR | FSIM | 0.908 | 0.837 |
| FR | MS-SSIM | 0.876 | 0.778 |
| FR | WaDIQaM-FR | 0.875 | 0.872 |
| NR | Q-Align (4-bit) | 0.885 | 0.899 |
| NR | MUSIQ (PaQ-2-PiQ) | 0.766 | 0.790 |
| NR | TOPIQ-NR | 0.728 | 0.767 |
| NR | CLIP-IQA+ (RN50, 512px) | 0.716 | 0.738 |
| aesthetic | Q-Align | 0.822 | 0.819 |
| aesthetic | TOPIQ IAA (Swin-B) | 0.791 | 0.790 |
| aesthetic | NIMA (InceptionResNet-v2) | 0.713 | 0.717 |
| aesthetic | LAION Aesthetic | 0.665 | 0.666 |

pyiqa's [performance evaluation protocol](https://github.com/chaofengc/IQA-PyTorch/tree/main?tab=readme-ov-file#performance-evaluation-protocol) trains/uses one setting for everything and evaluates cross-dataset: full-reference metrics on KADID-10k, no-reference on KonIQ-10k, PLCC reported without correction, full image as a single input. Averages above cover only the columns that table lists for each metric, so coverage varies (Q-Align appears for 3 of the 6 NR sets and does poorly on FLIVE). Metrics added to pyiqa after that table was generated - ARNIQA, LIQE, FGResQ, Q-ReAlign - are simply absent from it: a gap in the table, not a verdict on them.

So: **TOPIQ-FR** for full-reference, **Q-Align** (then **MUSIQ**, **TOPIQ-NR**, **CLIP-IQA+**) for no-reference, **Q-Align** again for aesthetics. Benchmark rank is about agreeing with average human opinion on those datasets; it is not the same question as "does this look right to me", which is why the list above is a starting point and not a rule.

## Full-reference metrics

These need the original wired in as the reference. Use `pyiqa FR Score`, which refuses a no-reference metric and names the node that works instead.

| dropdown entry | pyiqa name | better | range |
|---|---|---|---|
| A-FINE | FR head | `afine_fr` | lower better | -2-2 |
| A-FINE | FR · all-problem, scaled to 0-100 | `afine_all_scale` | lower better | 0-100 |
| A-FINE | FR · all-problem | `afine_all` | lower better | up to ~2 |
| AHIQ | `ahiq` | higher better | ~0-~1 |
| CKDN | `ckdn` | higher better | 0-1 |
| CW-SSIM | `cw_ssim` | higher better | 0-1 |
| DeepDC | `deepdc` | lower better | 0-1 |
| DISTS | `dists` | lower better | 0-1 |
| DMM | `dmm` | higher better | 0-unbounded |
| FGResQ | FR pair (rank + verdict) | `fgresq_pair` | higher better | quality 0-1 · rank 0-2 |
| FSIM | `fsim` | higher better | 0-1 |
| GMSD | `gmsd` | lower better | 0-~1 |
| LPIPS | LPIPS (AlexNet) | `lpips` | lower better | 0-1 |
| LPIPS | LPIPS (VGG) | `lpips-vgg` | lower better | 0-1 |
| LPIPS | LPIPS+ (AlexNet, semantic weighting) | `lpips+` | lower better | 0-1 |
| LPIPS | LPIPS+ (VGG, semantic weighting) | `lpips-vgg+` | lower better | 0-1 |
| MAD (L1) | mean absolute difference | `mad` | lower better | 0-unbounded |
| MS-SSIM | `ms_ssim` | higher better | 0-1 |
| MS-SWD (learned) | `msswd` | lower better | 0-~10 |
| NLPD | `nlpd` | lower better | 0-1 |
| PieAPP | `pieapp` | lower better | ~0-~5 |
| PSNR | RGB | `psnr` | higher better | ~0-~40 |
| PSNR | luminance (Y channel) | `psnry` | higher better | ~0-~60 |
| SSIM | luminance (Y channel) | `ssim` | higher better | 0-1 |
| SSIM | color (RGB channels) | `ssimc` | higher better | 0-1 |
| ST-LPIPS | shift-tolerant (AlexNet) | `stlpips` | lower better | 0-1 |
| ST-LPIPS | shift-tolerant (VGG) | `stlpips-vgg` | lower better | 0-1 |
| TOPIQ (CFANet) | FR · KADID-10k (ResNet50) | `topiq_fr` | higher better | ~0-~1 |
| TOPIQ (CFANet) | FR · PIPAL (ResNet50) | `topiq_fr-pipal` | higher better | ~0-~1 |
| VIF | `vif` | higher better | 0-~1 |
| VSI | `vsi` | higher better | 0-~1 |
| WaDIQaM | FR head | `wadiqam_fr` | higher better | ~-1-~0.1 |

## No-reference metrics

These score one image on its own. Use `pyiqa NR Score`, which scores each image of a batch separately.

| dropdown entry | pyiqa name | better | range |
|---|---|---|---|
| A-FINE | NR head | `afine_nr` | lower better | -2-2 |
| ARNIQA | default regressor (KonIQ-10k) | `arniqa` | higher better | 0-1 |
| ARNIQA | LIVE regressor | `arniqa-live` | higher better | 0-1 |
| ARNIQA | CSIQ regressor | `arniqa-csiq` | higher better | 0-1 |
| ARNIQA | TID2013 regressor | `arniqa-tid` | higher better | 0-1 |
| ARNIQA | KADID-10k regressor | `arniqa-kadid` | higher better | 0-1 |
| ARNIQA | CLIVE regressor | `arniqa-clive` | higher better | 0-1 |
| ARNIQA | FLIVE regressor | `arniqa-flive` | higher better | 0-1 |
| ARNIQA | SPAQ regressor | `arniqa-spaq` | higher better | 0-1 |
| BRISQUE | standard | `brisque` | lower better | ~0-~150 |
| BRISQUE | MATLAB implementation | `brisque_matlab` | lower better | ~0-~100 |
| CLIP-IQA | original | `clipiqa` | higher better | 0-1 |
| CLIP-IQA | improved (+) | `clipiqa+` | higher better | 0-1 |
| CLIP-IQA | + · ViT-L/14 @ 512px | `clipiqa+_vitL14_512` | higher better | 0-1 |
| CLIP-IQA | + · RN50 @ 512px | `clipiqa+_rn50_512` | higher better | 0-1 |
| CNNIQA | `cnniqa` | higher better | ~0-~1 |
| Compare2Score | `compare2score` | higher better | 0-100 |
| DBCNN | `dbcnn` | higher better | ~0-~1 |
| Entropy | `entropy` | higher better | 0-8 |
| FGResQ | NR (single image) · use FGResQ NR Score | `fgresq` | higher better | 0-1 |
| HyperIQA | `hyperiqa` | higher better | ~0-~1 |
| ILNIQE | `ilniqe` | lower better | ~0-~100 |
| LAION Aesthetic | `laion_aes` | higher better | ~1-~10 |
| LIQE | KonIQ-10k | `liqe` | higher better | 1-5 |
| LIQE | mixed datasets | `liqe_mix` | higher better | 1-5 |
| MACLIP | `maclip` | higher better | 0-1 |
| MANIQA | KonIQ-10k | `maniqa` | higher better | ~0-~1 |
| MANIQA | PIPAL | `maniqa-pipal` | higher better | ~0-~1 |
| MANIQA | KADID-10k | `maniqa-kadid` | higher better | ~0-~1 |
| MetaIQA | `metaiqa` | higher better | 0-1 |
| MUSIQ | KonIQ-10k | `musiq` | higher better | ~0-~100 |
| MUSIQ | AVA | `musiq-ava` | higher better | 1-10 |
| MUSIQ | PaQ-2-PiQ | `musiq-paq2piq` | higher better | ~0-~100 |
| MUSIQ | SPAQ | `musiq-spaq` | higher better | ~0-~100 |
| NIMA | InceptionResNet-v2 · AVA | `nima` | higher better | 0-10 |
| NIMA | InceptionResNet-v2 · KonIQ-10k | `nima-koniq` | higher better | ~0-~1 |
| NIMA | InceptionResNet-v2 · SPAQ | `nima-spaq` | higher better | ~0-~1 |
| NIMA | VGG-16 · AVA | `nima-vgg16-ava` | higher better | 0-10 |
| NIQE | standard | `niqe` | lower better | ~0-~100 |
| NIQE | MATLAB implementation | `niqe_matlab` | lower better | ~0-~100 |
| NRQM | `nrqm` | higher better | ~0-~10 |
| PaQ-2-PiQ | `paq2piq` | higher better | ~0-~100 |
| PI | `pi` | lower better | 0-unbounded |
| PIQE | `piqe` | lower better | 0-100 |
| Q-Align | default precision | `qalign` | higher better | 1-5 |
| Q-Align | 8-bit | `qalign_8bit` | higher better | 1-5 |
| Q-Align | 4-bit | `qalign_4bit` | higher better | 1-5 |
| Q-ReAlign | Mini 0.8B (default) (dedicated loader) | `qrealign` | higher better | 0-1 |
| Q-ReAlign | Lite 4B (dedicated loader) | `qrealign-lite` | higher better | 0-1 |
| Q-ReAlign | Pro 9B (dedicated loader) | `qrealign-pro` | higher better | 0-1 |
| QualiCLIP | base | `qualiclip` | higher better | 0-1 |
| QualiCLIP | improved (+) | `qualiclip+` | higher better | 0-1 |
| QualiCLIP | + · CLIVE-trained | `qualiclip+-clive` | higher better | 0-1 |
| QualiCLIP | + · FLIVE-trained | `qualiclip+-flive` | higher better | 0-1 |
| QualiCLIP | + · SPAQ-trained | `qualiclip+-spaq` | higher better | 0-1 |
| TOPIQ (CFANet) | NR · KonIQ-10k (ResNet50) | `topiq_nr` | higher better | ~0-~1 |
| TOPIQ (CFANet) | NR · FLIVE (ResNet50, 384px) | `topiq_nr-flive` | higher better | ~0-~1 |
| TOPIQ (CFANet) | NR · SPAQ (ResNet50) | `topiq_nr-spaq` | higher better | ~0-~1 |
| TOPIQ (CFANet) | NR · faces GFIQA-cg (ResNet50, 512px) | `topiq_nr-face` | higher better | ~0-~1 |
| TOPIQ (CFANet) | NR · faces GFIQA-cg (Swin-B, 384px) | `topiq_nr_swin-face` | higher better | ~0-~1 |
| TOPIQ (CFANet) | NR · faces GFIQA (ResNet50, 512px) | `topiq_nr-face-v1` | higher better | ~0-~1 |
| TOPIQ (CFANet) | aesthetic · AVA (Swin-B) | `topiq_iaa` | higher better | 1-10 |
| TOPIQ (CFANet) | aesthetic · AVA (ResNet50, 384px) | `topiq_iaa_res50` | higher better | 1-10 |
| TReS | KonIQ-10k | `tres` | higher better | ~0-~100 |
| TReS | FLIVE | `tres-flive` | higher better | ~0-~100 |
| UNIQUE | `unique` | higher better | ~-3-~3 |
| URanker | `uranker` | higher better | ~-1-~2 |
| WaDIQaM | NR head | `wadiqam_nr` | higher better | ~-1-~0.1 |

## Set-level and text-conditioned metrics

These cannot be scored per image, so they are not in the tables above: one number covers a whole image set, or the score needs a caption. pyiqa labels them all "NR" because none of them takes a reference image, which is why the loader labels them by what they need instead.

| pyiqa name | what it needs | how to reach it |
|---|---|---|
| `clipscore` | one image plus a caption | `(Down)load pyiqa Model` -> architecture **CLIP Score**, then `CLIP Score` |
| `fid` | two image sets | `(Down)load pyiqa Model` -> architecture **FID**, then `FID Score` |
| `sfid` | two image sets | `(Down)load pyiqa Model` -> architecture **FID**, then `FID Score` |
| `fid_dinov2` | two image sets | `(Down)load pyiqa Model` -> architecture **FID**, then `FID Score` |
| `inception_score` | one image set | listed in the loader, but no score node yet (it is a signpost: picking it explains instead of loading) |

## Notes per architecture

- **A-FINE** - Sides must be divisible by 32; this pack pads them for you. The `all` entries are trained across problem types, the NR/FR heads on one each, and one variant is rescaled to 0-100.
- **ARNIQA** - The variants are the same network calibrated on different datasets - pick the one matching your content.
- **CLIP Score** - Image-to-text match, not image quality: the same caption can score a sharp photo and a blurry one identically if both show the described content.
- **CLIP-IQA** - `+` is pyiqa's improved, retrained version (also available with a ViT-L/14 or RN50 backbone at 512px).
- **DISTS** - Structural + texture similarity; more robust than PSNR/SSIM to texture resampling.
- **FGResQ** - One net, two pyiqa metrics: the NR entry names the single-image metric, the FR pair entry the rank+verdict API - and either entry loads the SAME weights and drives both `FGResQ NR Score` and `FGResQ FR Score` (each node asks the bundle for the metric it needs). Custom fine-tunes go in `models/pyiqa/FGResQ/` and keep the default degradation branch.
- **FID** - Set-level: distance between the feature distributions of two image SETS, so it never scores a single pair. 2+ images per side is the floor, ~16+ is where the number starts to mean something (below that it mostly measures how many images you wired).
- **LPIPS** - `+` is pyiqa's semantic-weighting variant (LPIPS+); ST-LPIPS is the shift-tolerant retraining.
- **MAD (L1)** - Mean absolute difference - the closest thing pyiqa has to an L1 loss. Values sit on pyiqa's own image scale, not on 0-1.
- **MANIQA** - Variants are calibrated on different datasets.
- **MUSIQ** - Variants are calibrated on different datasets; the AVA entry is aesthetic (1-10), the others are quality.
- **NIMA** - The AVA entries are aesthetic (0-10); the KonIQ-10k and SPAQ entries are quality (0-1).
- **PSNR** - psnr is RGB, psnry is the Y (luminance) channel only - the usual video-quality convention.
- **Q-Align** - A ~15.3 GiB fp16 model. The 8-bit and 4-bit variants need bitsandbytes and accelerate; the pack refuses a quantized pick when those are missing instead of silently loading fp16.
- **Q-ReAlign** - **Not reachable through `(Down)load pyiqa Model`**: that entry is a signpost because the int8_convrot builds, the dtype choice and the snapshot folders live in the dedicated loader (`(Down)load Q-ReAlign`), and the IQA/IAA task is chosen on `Q-ReAlign NR Score`. Mini 0.8B / Lite 4B / Pro 9B, better correlation at bigger sizes.
- **QualiCLIP** - `+` is the improved version; the `-clive` / `-flive` / `-spaq` entries are fine-tuned on those datasets.
- **SSIM** - ssim reads the Y (luminance) channel, ssimc the RGB channels.
- **ST-LPIPS** - Shift-tolerant retraining of LPIPS: use it when small translations or resampling should not count against you.
- **TOPIQ (CFANet)** - Variants differ by training dataset and backbone. The face (GFIQA) entries expect one face at a time; the aesthetic (IAA) entries score 1-10.
- **TReS** - Variants are calibrated on different datasets.

## Glossary

- **FR / NR** - **FR** (full-reference) compares your image against a reference; **NR** (no-reference) scores an image on its own. One architecture can offer both (A-FINE, TOPIQ, WaDIQaM, FGResQ).
- **SRCC** - Spearman rank-order correlation against human opinion scores: does the metric order images the way people did? The headline number in most IQA papers (1.0 is perfect, higher is better).
- **PLCC** - Pearson correlation after a logistic fit: like SRCC, but it also rewards a *linear* score-to-opinion relationship. pyiqa reports it without correction.
- **KRCC** - Kendall rank correlation - like SRCC, but built from pairwise comparisons, so it is stricter and its numbers are lower.
- **MOS** - Mean opinion score: the average human rating a dataset ships with, which the correlations above are measured against.
- **higher / lower better** - Which direction is a better score. The score node states it too, so you never have to remember.
- **range** - The scale to expect, taken from pyiqa's own calibration table. It is orientation, not a limit - some metrics are unbounded.
- **set-level** - Computed over a whole image SET (feature mean and covariance) and returning one number for two sets, as FID and sFID do. It cannot score a single pair.
- **single image** - That variant asserts a batch of one; the pack feeds it one image at a time.
- **`+`** - pyiqa's improved or retrained version of that metric (LPIPS+, CLIP-IQA+, QualiCLIP+).
- **KonIQ-10k** - Large no-reference IQA dataset (10k real photos with subjective quality scores) - the most common training set here.
- **SPAQ** - Smartphone photography quality dataset.
- **AVA** - Aesthetic Visual Analysis dataset: aesthetic ratings, not technical quality.
- **KADID-10k** - 10k distorted images with subjective scores, often used for full-reference training.
- **PIPAL** - Perceptual image quality dataset focused on GAN and restoration distortions.
- **LIVE / CSIQ / TID2013 / CLIVE / FLIVE** - Older IQA benchmarks; CLIVE and FLIVE are 'in the wild' photo sets.
- **GFIQA** - Face image quality dataset; the `-cg` and `-v1` entries are different training splits of it.
- **AlexNet / VGG / ResNet50 / Swin-B / ViT-L/14 / RN50 / InceptionResNet-v2** - The backbone network the metric is built on - usually bigger means slower and more accurate.
- **mixed_6a** - An Inception-v3 layer; sFID takes its spatial features from there instead of global features.
- **Y channel** - Luminance only, colour ignored - the classic SSIM/PSNR convention.
- **convrot** - This pack's int8 quantization of Q-ReAlign (with a rotation step): about a third of the size and VRAM, tiny score drift.

## How this file is made

`tools/gen_metrics_doc.py` reads the pack's catalog (which reads pyiqa's own `DEFAULT_CONFIGS`) and writes this file, so a label, mode, direction, range or registry name cannot drift from the code. Run it with `--check` to verify this file is current - the pack's smoke test does exactly that, so a catalog change that forgets this file fails the suite.
