"""Regenerate METRICS.md (the user-facing metric reference) from the catalog.

The catalog is the source of truth for what the loader can show, so the table in
METRICS.md is generated from it instead of being typed by hand: a label, mode,
direction, range or registry name can never drift between the two.

Run with the ComfyUI env python, from the pack root:
  python tools/gen_metrics_doc.py            # rewrite METRICS.md
  python tools/gen_metrics_doc.py --check    # exit 1 if METRICS.md is stale

The scratch/doc-only prose (glossary, notes, orientation) lives in this file,
next to the table it explains. The smoke test runs the --check above, so a
catalog change that forgets this file fails the suite.
"""

import importlib.util
import os
import sys
import tempfile
import types

# pack root = one level up from this file (tools/)
PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(PKG_DIR, "METRICS.md")

# --------------------------------------------------------------------------
# prose: only what the pack itself knows (behaviour, sizes, gotchas). Metric
# mechanics stay out - the pyiqa name column is the lookup key for papers.
# --------------------------------------------------------------------------
NOTES = {
    "A-FINE": "Sides must be divisible by 32; this pack pads them for you. The "
              "`all` entries are trained across problem types, the NR/FR heads "
              "on one each, and one variant is rescaled to 0-100.",
    "ARNIQA": "The variants are the same network calibrated on different "
              "datasets - pick the one matching your content.",
    "CLIP-IQA": "`+` is pyiqa's improved, retrained version (also available "
                "with a ViT-L/14 or RN50 backbone at 512px).",
    "DISTS": "Structural + texture similarity; more robust than PSNR/SSIM to "
             "texture resampling.",
    "FGResQ": "One net, two pyiqa metrics: the NR entry names the single-image "
              "metric, the FR pair entry the rank+verdict API - and either entry "
              "loads the SAME weights and drives both `FGResQ NR Score` and "
              "`FGResQ FR Score` (each node asks the bundle for the metric it "
              "needs). Custom fine-tunes go in `models/pyiqa/FGResQ/` and keep "
              "the default degradation branch.",
    "LPIPS": "`+` is pyiqa's semantic-weighting variant (LPIPS+); ST-LPIPS is "
             "the shift-tolerant retraining.",
    "MAD (L1)": "Mean absolute difference - the closest thing pyiqa has to an "
                "L1 loss. Values sit on pyiqa's own image scale, not on 0-1.",
    "FID": "Set-level: distance between the feature distributions of two image "
           "SETS, so it never scores a single pair. 2+ images per side is the "
           "floor, ~16+ is where the number starts to mean something (below "
           "that it mostly measures how many images you wired).",
    "CLIP Score": "Image-to-text match, not image quality: the same caption can "
                  "score a sharp photo and a blurry one identically if both show "
                  "the described content.",
    "MANIQA": "Variants are calibrated on different datasets.",
    "MUSIQ": "Variants are calibrated on different datasets; the AVA entry is "
             "aesthetic (1-10), the others are quality.",
    "NIMA": "The AVA entries are aesthetic (0-10); the KonIQ-10k and SPAQ "
            "entries are quality (0-1).",
    "PSNR": "psnr is RGB, psnry is the Y (luminance) channel only - the usual "
            "video-quality convention.",
    "Q-Align": "A ~15.3 GiB fp16 model. The 8-bit and 4-bit variants need "
               "bitsandbytes and accelerate; the pack refuses a quantized pick "
               "when those are missing instead of silently loading fp16.",
    "Q-ReAlign": "**Not reachable through `(Down)load pyiqa Model`**: that entry "
                 "is a signpost because the int8_convrot builds, the dtype "
                 "choice and the snapshot folders live in the dedicated loader "
                 "(`(Down)load Q-ReAlign`), and the IQA/IAA task is chosen on "
                 "`Q-ReAlign NR Score`. Mini 0.8B / Lite 4B / Pro 9B, better "
                 "correlation at bigger sizes.",
    "QualiCLIP": "`+` is the improved version; the `-clive` / `-flive` / "
                 "`-spaq` entries are fine-tuned on those datasets.",
    "SSIM": "ssim reads the Y (luminance) channel, ssimc the RGB channels.",
    "ST-LPIPS": "Shift-tolerant retraining of LPIPS: use it when small "
                "translations or resampling should not count against you.",
    "TOPIQ (CFANet)": "Variants differ by training dataset and backbone. The "
                      "face (GFIQA) entries expect one face at a time; the "
                      "aesthetic (IAA) entries score 1-10.",
    "TReS": "Variants are calibrated on different datasets.",
}

# pyiqa's own benchmark, ranked by average SRCC over the test sets that table
# covers (numbers transcribed from the benchmark page below, pyiqa 0.1.16 docs:
# columns are PLCC/SRCC/KRCC per dataset; "avg" here is the mean SRCC and mean
# PLCC of the columns that metric appears in). Kept as data on purpose: the
# document is generated offline, so it quotes the published table rather than
# pretending to recompute it.
BENCHMARK_SOURCE = "https://iqa-pytorch.readthedocs.io/en/latest/benchmark.html"
BENCHMARK_PROTOCOL = (
    "pyiqa's [performance evaluation protocol]"
    "(https://github.com/chaofengc/IQA-PyTorch/tree/main?tab=readme-ov-file"
    "#performance-evaluation-protocol) trains/uses one setting for everything "
    "and evaluates cross-dataset: "
    "full-reference metrics on KADID-10k, no-reference on KonIQ-10k, PLCC "
    "reported without correction, full image as a single input. Averages above "
    "cover only the columns that table lists for each metric, so coverage "
    "varies (Q-Align appears for 3 of the 6 NR sets and does poorly on FLIVE). "
    "Metrics added to pyiqa after that table was generated - ARNIQA, LIQE, "
    "FGResQ, Q-ReAlign - are simply absent from it: a gap in the table, not a "
    "verdict on them."
)
BENCHMARKS = [
    ("FR", "TOPIQ-FR", "0.946", "0.933"),
    ("FR", "FSIM", "0.908", "0.837"),
    ("FR", "MS-SSIM", "0.876", "0.778"),
    ("FR", "WaDIQaM-FR", "0.875", "0.872"),
    ("NR", "Q-Align (4-bit)", "0.885", "0.899"),
    ("NR", "MUSIQ (PaQ-2-PiQ)", "0.766", "0.790"),
    ("NR", "TOPIQ-NR", "0.728", "0.767"),
    ("NR", "CLIP-IQA+ (RN50, 512px)", "0.716", "0.738"),
    ("aesthetic", "Q-Align", "0.822", "0.819"),
    ("aesthetic", "TOPIQ IAA (Swin-B)", "0.791", "0.790"),
    ("aesthetic", "NIMA (InceptionResNet-v2)", "0.713", "0.717"),
    ("aesthetic", "LAION Aesthetic", "0.665", "0.666"),
]

GLOSSARY = [
    ("FR / NR", "**FR** (full-reference) compares your image against a "
                "reference; **NR** (no-reference) scores an image on its own. "
                "One architecture can offer both (A-FINE, TOPIQ, WaDIQaM, "
                "FGResQ)."),
    ("SRCC", "Spearman rank-order correlation against human opinion scores: "
             "does the metric order images the way people did? The headline "
             "number in most IQA papers (1.0 is perfect, higher is better)."),
    ("PLCC", "Pearson correlation after a logistic fit: like SRCC, but it also "
             "rewards a *linear* score-to-opinion relationship. pyiqa reports "
             "it without correction."),
    ("KRCC", "Kendall rank correlation - like SRCC, but built from pairwise "
             "comparisons, so it is stricter and its numbers are lower."),
    ("MOS", "Mean opinion score: the average human rating a dataset ships "
            "with, which the correlations above are measured against."),
    ("higher / lower better", "Which direction is a better score. The score "
                              "node states it too, so you never have to "
                              "remember."),
    ("range", "The scale to expect, taken from pyiqa's own calibration table. "
              "It is orientation, not a limit - some metrics are unbounded."),
    ("set-level", "Computed over a whole image SET (feature mean and "
                  "covariance) and returning one number for two sets, as FID "
                  "and sFID do. It cannot score a single pair."),
    ("single image", "That variant asserts a batch of one; the pack feeds it "
                     "one image at a time."),
    ("`+`", "pyiqa's improved or retrained version of that metric (LPIPS+, "
            "CLIP-IQA+, QualiCLIP+)."),
    ("KonIQ-10k", "Large no-reference IQA dataset (10k real photos with "
                  "subjective quality scores) - the most common training set "
                  "here."),
    ("SPAQ", "Smartphone photography quality dataset."),
    ("AVA", "Aesthetic Visual Analysis dataset: aesthetic ratings, not "
            "technical quality."),
    ("KADID-10k", "10k distorted images with subjective scores, often used for "
                  "full-reference training."),
    ("PIPAL", "Perceptual image quality dataset focused on GAN and restoration "
              "distortions."),
    ("LIVE / CSIQ / TID2013 / CLIVE / FLIVE", "Older IQA benchmarks; CLIVE and "
                                              "FLIVE are 'in the wild' photo "
                                              "sets."),
    ("GFIQA", "Face image quality dataset; the `-cg` and `-v1` entries are "
              "different training splits of it."),
    ("AlexNet / VGG / ResNet50 / Swin-B / ViT-L/14 / RN50 / InceptionResNet-v2",
     "The backbone network the metric is built on - usually bigger means slower "
     "and more accurate."),
    ("mixed_6a", "An Inception-v3 layer; sFID takes its spatial features from "
                 "there instead of global features."),
    ("Y channel", "Luminance only, colour ignored - the classic SSIM/PSNR "
                  "convention."),
    ("convrot", "This pack's int8 quantization of Q-ReAlign (with a rotation "
                "step): about a third of the size and VRAM, tiny score drift."),
]

ORIENTATION = [
    ("Compare my result against the original",
     "PSNR, SSIM, MS-SSIM, CW-SSIM, FSIM, VIF, GMSD - instant, nothing to "
     "download, but they mainly measure pixel and structural fidelity."),
    ("Compare two images the way people see them",
     "LPIPS, LPIPS+ (better agreement with human judgement), DISTS, A-FINE FR, "
     "PieAPP - all want aligned crops."),
    ("One image against one image, with a verdict",
     "`FGResQ FR Score`, which also returns each side's quality, a rank and a "
     "similarity verdict."),
    ("Score an image on its own",
     "MUSIQ, TOPIQ, CLIP-IQA, ARNIQA, MANIQA, HyperIQA, TReS, LIQE (learned); "
     "NIQE, BRISQUE, PIQE, ILNIQE, NRQM (classic statistical)."),
    ("Judge aesthetics rather than quality",
     "NIMA (AVA), LAION Aesthetic, TOPIQ IAA, MUSIQ-AVA."),
    ("Quality the way a large vision-language model scores it",
     "Q-Align (8.2B, ~15 GiB) and Q-ReAlign (0.8B-9B): best agreement with "
     "human ratings, slowest."),
    ("How well an image matches a caption",
     "CLIP Score - see *Set-level and text-conditioned metrics*."),
    ("Whether a whole batch looks like another whole batch",
     "FID / sFID - see *Set-level and text-conditioned metrics*."),
]

# how a metric that cannot be scored per image is reached. Keyed by the kind in
# metric_catalog.SPECIAL_INPUTS, so adding one there shows up here (as "?" only
# if the kind is new, which is meant to be noticed).
SET_LEVEL_HOW = {
    "distance": "`(Down)load pyiqa Model` -> architecture **FID**, then "
                "`FID Score`",
    "one_set": "listed in the loader, but no score node yet (it is a signpost: "
               "picking it explains instead of loading)",
    "caption": "`(Down)load pyiqa Model` -> architecture **CLIP Score**, then "
               "`CLIP Score`",
}


def load_catalog():
    """Import the pack's catalog, with a throwaway model folder.

    Only a temp folder is registered (when ComfyUI's folder_paths is not already
    present), so generating the document cannot scaffold the custom-checkpoint
    folders into anyone's real models directory. Registry labels do not depend
    on what is on disk, so the document is machine-independent.
    """
    if "folder_paths" not in sys.modules:
        models_dir = os.path.join(tempfile.gettempdir(), "pyiqa_toolbox_docgen")
        folder_names_and_paths = {}

        def add_model_folder_path(name, path):
            paths, _exts = folder_names_and_paths.setdefault(name, ([], set()))
            if path not in paths:
                paths.append(path)

        fp = types.ModuleType("folder_paths")
        fp.models_dir = models_dir
        fp.folder_names_and_paths = folder_names_and_paths
        fp.add_model_folder_path = add_model_folder_path
        fp.get_folder_paths = lambda name: list(
            folder_names_and_paths.get(name, ([], set()))[0]
        )
        add_model_folder_path("pyiqa", os.path.join(models_dir, "pyiqa"))
        sys.modules["folder_paths"] = fp

    name = "pyiqa_toolbox_docgen"
    spec = importlib.util.spec_from_file_location(
        name,
        os.path.join(PKG_DIR, "__init__.py"),
        submodule_search_locations=[PKG_DIR],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod, sys.modules[name + ".utility.metric_catalog"]


def collect(catalog):
    """One dict per registry variant, in dropdown order.

    ``catalog.ARCHES`` is already sorted the way the loader's architecture combo
    is, so the document lists architectures in the order the UI does. The
    variant's readable text comes from the catalog's own axis table, and the
    registry variant is built directly (not through ``resolve``) because a
    signposted architecture refuses to resolve by design - the document still
    has to describe the weights its dedicated loader offers.
    """
    rows = []
    for arch in catalog.ARCHES:
        entry = catalog.ARCHITECTURES[arch]
        for variant in entry.variants:
            r = catalog._resolved_registry(entry, variant)
            axes = catalog._VARIANT_AXES.get(variant, "")
            label = f"{entry.label} | {axes}" if axes else entry.label
            if entry.signpost:
                label += " (dedicated loader)"
            rows.append(
                {
                    "entry": label,
                    "name": r.base,
                    "mode": r.mode,
                    "better": "lower better" if r.lower_better else "higher better",
                    "range": r.score_range,
                }
            )
    return rows


def _table(rows):
    out = [
        "| dropdown entry | pyiqa name | better | range |",
        "|---|---|---|---|",
    ]
    out += [
        f"| {r['entry']} | `{r['name']}` | {r['better']} | {r['range']} |"
        for r in rows
    ]
    return out


def render(mod, catalog, rows):
    import pyiqa

    # metrics that are not per-image live in their own section: pyiqa calls them
    # all "NR", which would put FID in a table headed "no-reference"
    special_names = set(catalog.SPECIAL_INPUTS)
    per_image = [r for r in rows if r["name"] not in special_names]
    fr = [r for r in per_image if r["mode"] == "FR"]
    nr = [r for r in per_image if r["mode"] == "NR"]
    special = [r for r in rows if r["name"] in special_names]
    arch_count = len({r["entry"].split(" | ")[0] for r in per_image})
    # the startup line's own version helper, so both agree by construction
    pack_version = getattr(mod, "_pack_version", lambda: "?")()
    versions = [f"pack {pack_version}", f"pyiqa {getattr(pyiqa, '__version__', '?')}"]

    doc = [
        "# Metric reference",
        "",
        "Every metric this pack can score with - through `(Down)load pyiqa "
        "Model`, or through a dedicated loader where one exists - what it "
        "actually is, and which way its number points. Generated from the "
        "pack's own catalog by `tools/gen_metrics_doc.py`, so it always matches "
        "the code.",
        "",
        f"This file describes {' and '.join(versions)}: {len(fr) + len(nr)} "
        f"weights choices you can score per image, across {arch_count} "
        f"architectures ({len(fr)} full-reference, {len(nr)} no-reference), plus "
        f"{len(special)} set-level or text-conditioned metrics. Your own "
        "checkpoints add more entries at runtime (see *Custom checkpoints* in "
        "the README).",
        "",
    ]
    if getattr(catalog, "ALIASES", None):
        # pyiqa names that build the same weights as another entry: saying so is
        # the only way a reader can tell a folded duplicate from a missing metric
        names = sorted(catalog.ALIASES.items())
        pairs = ", ".join(f"`{alias}` (same weights as `{twin}`)" for alias, twin in names)
        tail = (
            "so it is folded into that entry rather than offered twice."
            if len(names) == 1
            else "so they are folded into those entries rather than offered twice."
        )
        doc += [f"pyiqa also registers {pairs}, {tail}", ""]
    doc += [
        "## How to read an entry",
        "",
        "The variant dropdown shows one line per weights choice, and every line "
        "has the same four parts:",
        "",
        "```",
        "<architecture> | <variant> - <mode>, <direction>, <range>",
        "```",
        "",
        "| part | meaning |",
        "|---|---|",
        "| **architecture** | the network family (chosen in the first dropdown) |",
        "| **variant** | which weights within that family |",
        "| **mode** | FR (needs a reference image) or NR (does not) |",
        "| **direction** | whether a higher or a lower number is better |",
        "| **range** | the scale pyiqa expects the number to fall in |",
        "",
        "The **pyiqa name** column is the name pyiqa itself uses, so you can "
        "look the metric up in pyiqa's documentation and compare against its "
        "published benchmark numbers. Numbers are **not** comparable across "
        "metrics: each has its own scale and direction - pick one and compare "
        "runs with it.",
        "",
        "## What should I use?",
        "",
    ]
    doc += [f"- **{what}** - {which}" for what, which in ORIENTATION]
    doc += [
        "",
        "### If you want a starting point rather than an opinion",
        "",
        f"pyiqa publishes a benchmark of its own metrics ([{BENCHMARK_SOURCE}]"
        f"({BENCHMARK_SOURCE})). Ranked by average SRCC - how well a metric's "
        "ordering agrees with human ratings - the leaders are:",
        "",
        "| kind | metric | avg SRCC | avg PLCC |",
        "|---|---|---|---|",
    ]
    doc += [
        f"| {kind} | {name} | {srcc} | {plcc} |" for kind, name, srcc, plcc in BENCHMARKS
    ]
    doc += [
        "",
        BENCHMARK_PROTOCOL,
        "",
        "So: **TOPIQ-FR** for full-reference, **Q-Align** (then **MUSIQ**, "
        "**TOPIQ-NR**, **CLIP-IQA+**) for no-reference, **Q-Align** again for "
        "aesthetics. Benchmark rank is about agreeing with average human "
        "opinion on those datasets; it is not the same question as \"does this "
        "look right to me\", which is why the list above is a starting point "
        "and not a rule.",
        "",
        "## Full-reference metrics",
        "",
        "These need the original wired in as the reference. Use `pyiqa FR "
        "Score`, which refuses a no-reference metric and names the node that "
        "works instead.",
        "",
        *_table(fr),
        "",
        "## No-reference metrics",
        "",
        "These score one image on its own. Use `pyiqa NR Score`, which scores "
        "each image of a batch separately.",
        "",
        *_table(nr),
        "",
        "## Set-level and text-conditioned metrics",
        "",
        "These cannot be scored per image, so they are not in the tables above: "
        "one number covers a whole image set, or the score needs a caption. "
        "pyiqa labels them all \"NR\" because none of them takes a reference "
        "image, which is why the loader labels them by what they need instead.",
        "",
        "| pyiqa name | what it needs | how to reach it |",
        "|---|---|---|",
    ]
    for r in special:
        kind, _message = catalog.SPECIAL_INPUTS[r["name"]]
        needs = {
            "distance": "two image sets",
            "one_set": "one image set",
            "caption": "one image plus a caption",
        }.get(kind, "?")
        doc.append(f"| `{r['name']}` | {needs} | {SET_LEVEL_HOW.get(kind, '?')} |")
    doc += [
        "",
        "## Notes per architecture",
        "",
    ]
    doc += [f"- **{arch}** - {NOTES[arch]}" for arch in sorted(NOTES)]
    doc += [
        "",
        "## Glossary",
        "",
    ]
    doc += [f"- **{term}** - {meaning}" for term, meaning in GLOSSARY]
    doc += [
        "",
        "## How this file is made",
        "",
        "`tools/gen_metrics_doc.py` reads the pack's catalog (which reads "
        "pyiqa's own `DEFAULT_CONFIGS`) and writes this file, so a label, mode, "
        "direction, range or registry name cannot drift from the code. Run it "
        "with `--check` to verify this file is current - the pack's smoke test "
        "does exactly that, so a catalog change that forgets this file fails "
        "the suite.",
        "",
    ]
    return "\n".join(doc)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    _mod, catalog = load_catalog()
    text = render(_mod, catalog, collect(catalog))
    if "--check" in argv:
        current = ""
        if os.path.exists(OUT_PATH):
            with open(OUT_PATH, encoding="utf-8") as fh:
                current = fh.read()
        if current != text:
            print(f"{os.path.relpath(OUT_PATH, PKG_DIR)} is stale - run "
                  "tools/gen_metrics_doc.py")
            return 1
        print("METRICS.md is current")
        return 0
    with open(OUT_PATH, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(f"wrote {os.path.relpath(OUT_PATH, PKG_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
