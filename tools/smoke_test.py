"""Standalone smoke test: loads the pack the way ComfyUI does (hyphenated folder
as a package) and exercises every node with synthetic images.

Run with the ComfyUI env python, from the pack root:
  cd <pack checkout>
  <comfyui env>/python.exe tools/smoke_test.py

The pack's real model folders are registered the way an extra_model_paths.yaml
`pyiqa:` section would (see inject_fake_folder_paths), so assertions about
weights that only exist on this machine stay conditional: they skip themselves
when the folder/file is absent.

Side effect, by design: importing the pack scaffolds the 13 `{architecture}/`
custom-checkpoint folders in every registered non-cache pyiqa folder (the
install-local models/pyiqa, the yaml-registered folder named by
`PYIQA_TOOLBOX_MODELS` / `tools/dev_paths_local.py`, and the throwaway trees
under %TEMP%). Broad "models" directories are always temp stand-ins, never a real
shared folder: a test run must not leave 13 drop folders in a drive-root models
directory. The real pyiqa folder is the one place that gets them for real,
because that is what happens on a user's machine.
"""

import importlib.util
import logging
import os
import shutil
import sys
import tempfile
import types

import torch

# dev_paths holds this machine's paths (env var or the gitignored local file):
# nothing machine-specific is baked into the suite itself.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dev_paths  # noqa: E402

# pack root = one level up from this file (tools/)
PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def inject_fake_folder_paths():
    """Mimic ComfyUI's folder_paths so the models/pyiqa lookup is exercised
    standalone. Broad yaml-style search dirs (HF hub cache, pyiqa's torch-hub
    cache, X:/Models) are registered BEFORE the pack imports, exactly as an
    extra_model_paths.yaml `pyiqa:` section would be, plus a not-yet-existing
    yaml-sim folder the pack must create.
    """
    if "folder_paths" in sys.modules:
        return None
    import tempfile

    tmp = tempfile.gettempdir()
    models_dir = os.path.join(tmp, "pyiqa_toolbox_smoke_models")
    local_models_pyiqa = os.path.join(models_dir, "pyiqa")
    # personal search dirs (override: PYIQA_TOOLBOX_SMOKE_PATHS="dir;dir").
    # Absent dirs are dropped, so the test runs on any machine - the
    # machine-specific assertions below skip themselves when they must.
    env_paths = os.environ.get("PYIQA_TOOLBOX_SMOKE_PATHS")
    if env_paths:
        pyiqa_dirs = [d for d in env_paths.split(";") if d]
    else:
        # dev_paths resolves the machine-local places (PYIQA_TOOLBOX_MODELS or the
        # gitignored tools/dev_paths_local.py); these fallbacks are portable.
        pyiqa_dirs = [
            p
            for p in (
                dev_paths.MODELS,
                dev_paths.hf_hub(),
                os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "pyiqa"),
            )
            if p
        ]
    pyiqa_dirs = [d for d in pyiqa_dirs if os.path.isdir(d)]
    # A user may point extra_model_paths at a broad "models" folder, which the
    # pack then searches and scaffolds like any other pyiqa folder. A throwaway
    # tree stands in for that: scaffolding into a real shared models directory
    # would leave 13 drop folders behind on the machine running the test.
    broad_dir = os.path.join(tmp, "pyiqa_toolbox_smoke_broad_models")
    os.makedirs(broad_dir, exist_ok=True)
    pyiqa_dirs.append(broad_dir)
    yaml_sim = os.path.join(
        tmp, "pyiqa_toolbox_smoke_yaml_sim", "models", "pyiqa"
    )

    # Real ComfyUI stores {name: (paths_list, extensions_set)} and
    # get_folder_paths returns the paths list — mirror that exactly so the
    # pack's library-registration rewrite is exercised.
    folder_names_and_paths = {}

    def add_model_folder_path(name, path):
        paths, _exts = folder_names_and_paths.setdefault(name, ([], set()))
        if path not in paths:
            paths.append(path)

    def get_folder_paths(name):
        return list(folder_names_and_paths.get(name, ([], set()))[0])

    fp = types.ModuleType("folder_paths")
    fp.models_dir = models_dir
    fp.folder_names_and_paths = folder_names_and_paths
    fp.add_model_folder_path = add_model_folder_path
    fp.get_folder_paths = get_folder_paths

    for d in pyiqa_dirs:
        add_model_folder_path("pyiqa", d)
    add_model_folder_path("pyiqa", yaml_sim)
    sys.modules["folder_paths"] = fp
    return {
        "yaml_sim": yaml_sim,
        "local": local_models_pyiqa,
        "yaml_dirs": pyiqa_dirs,
    }


def load_pack():
    name = "pyiqa_toolbox_smoke"
    spec = importlib.util.spec_from_file_location(
        name,
        os.path.join(PKG_DIR, "__init__.py"),
        submodule_search_locations=[PKG_DIR],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def fake_image(batch, h, w, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand((batch, h, w, 3), generator=g)


def inject_fake_comfy():
    """Mimic enough of ComfyUI to exercise the unload hook + device helpers."""
    if "comfy" in sys.modules:
        return
    comfy = types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")

    def get_torch_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def unload_all_models():
        print("[fake comfy] unload_all_models (original) called")

    mm.get_torch_device = get_torch_device
    mm.unload_all_models = unload_all_models
    comfy.model_management = mm
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = mm


def _norm(path):
    return os.path.normcase(os.path.abspath(path))


def _expected_scaffold_folders(metric_catalog):
    """The ``{architecture}/`` folder names scaffold_folders() must create."""
    return sorted(
        metric_catalog.ARCHITECTURES[label].folders[0]
        for family in metric_catalog.SCAFFOLD_FAMILIES
        for label in metric_catalog.family_arches(family)
    )


def _rebuild_catalog_tables(metric_catalog):
    """Rescan the custom-checkpoint folders and rebind the derived tables.

    Thin wrapper over the catalog's own public entry point (called by every
    lookup with a 1 s TTL); ``force=True`` makes a synthetic tree visible
    immediately and puts the real tree back afterwards.
    """
    metric_catalog.refresh_customs(force=True)


def _expect_value_error(call, *needles, exc_type=ValueError):
    """Run ``call``; assert it raised ``exc_type`` whose text contains ``needles``."""
    try:
        call()
    except exc_type as exc:
        message = str(exc)
        for needle in needles:
            assert needle in message, f"{needle!r} not in error {message!r}"
        return message
    raise AssertionError(
        f"expected {exc_type.__name__} mentioning {needles}, nothing was raised"
    )


# The 10 registered node types, in NODE_CONFIG order, with the display name each
# one must have. The surface is deliberately small: one generic loader, its two
# generic score nodes, and the four families whose weights need more than a
# variant name (FGResQ, Q-ReAlign, FID, CLIP Score) plus the unload utility.
NODE_SURFACE = (
    ("PyiqaAnyLoader", "(Down)load pyiqa Model"),
    ("PyiqaAnyFRScore", "pyiqa FR Score"),
    ("PyiqaAnyNRScore", "pyiqa NR Score"),
    ("CLIPScore", "CLIP Score"),
    ("FGResQScore", "FGResQ NR Score"),
    ("FGResQPairScore", "FGResQ FR Score"),
    ("QReAlignLoader", "(Down)load Q-ReAlign"),
    ("QReAlignScore", "QReAlign NR Score"),
    ("FIDScore", "FID Score"),
    ("UnloadPyiqaModels", "Unload Pyiqa Models"),
)

# Names the 10-node surface dropped. Their work did not go away: the generic
# loader + the generic score nodes cover the metrics, and FID/FGResQ weights come
# from "(Down)load pyiqa Model" with the architecture's own variants. "L1Score"'s
# metric is architecture "MAD (L1)" in the catalog.
REMOVED_NODES = ("L1Score", "FIDLoader", "FGResQLoader")

# The 23 node types the 12-node cull removed: their nine metric families are
# reached through "(Down)load pyiqa Model" + "pyiqa FR Score" / "pyiqa NR Score".
# Every one of these names must be gone from the registry AND from every message.
CULLED_NODES = (
    "AFINELoader AFINEFRScore AFINENRScore CLIPIQALoader CLIPIQANRScore "
    "DISTSLoader DISTSFrScore LAIONLoader LAIONNRScore LPIPSLoader LPIPSFrScore "
    "MUSIQLoader MUSIQNRScore QualiCLIPLoader QualiCLIPNRScore TOPIQLoader "
    "TOPIQFRScore TOPIQNRScore WaDIQaMLoader WaDIQaMFRScore WaDIQaMNRScore "
    "ClassicFRLoader ClassicFRScore"
).split()

# metric_catalog went through the same cull: the per-family keyed tables are
# gone (the architecture's ``variants`` and ``VARIANT_LOOKUP`` are the source).
GONE_CATALOG_API = (
    "FAMILY_LOOKUP",
    "family_variant_values",
    "default_family_variant",
    "resolve_family",
    "FAMILY_VARIANT_LABELS",
    "FAMILY_DEFAULTS",
)


def _variants_for(metric_catalog, base, mode=None):
    """[(arch label, "<arch> | <label>" value, Resolved)] for a registry metric."""
    found = []
    for value in metric_catalog.VARIANT_VALUES:
        resolved = metric_catalog.VARIANT_LOOKUP[value]
        if resolved.base != base or resolved.custom:
            continue
        if mode is not None and resolved.mode != mode:
            continue
        found.append((resolved.arch, value, resolved))
    return found


def _variant_value(metric_catalog, base, mode=None):
    """The arch-prefixed variant dropdown value of a registry metric."""
    found = _variants_for(metric_catalog, base, mode)
    assert found, f"no generic-combo variant resolves to {base!r} (mode {mode})"
    return found[0][1]


def _dropdown_values(metric_catalog, arch):
    """The generic loader's values for one architecture (signposts included)."""
    prefix = arch + metric_catalog.SEPARATOR
    return [
        value for value in metric_catalog.VARIANT_VALUES if value.startswith(prefix)
    ]


def _value_for_base(metric_catalog, base):
    """The dropdown value that loads registry metric ``base``.

    A signpost architecture's only entry is the entry for its registry variant,
    so that one is returned when there is no plain registry entry to prefer.
    """
    found = [
        value
        for value, resolved in metric_catalog.VARIANT_LOOKUP.items()
        if resolved.base == base
    ]
    assert found, f"no dropdown entry loads {base!r}"
    registry = [v for v in found if not metric_catalog.VARIANT_LOOKUP[v].custom]
    return (registry or found)[0]


def _load_generic(pack, metric_catalog, base, mode=None):
    """Load a registry metric the way a user does now: ``Load pyiqa Model``.

    Returns ``(bundle, arch, variant_value)``; the registry name the node
    reports must still be the pyiqa name, which is what makes the old
    per-family assertions (SSIM identity 1.0, MAD 0.0, ...) portable.
    """
    arch, value, _resolved = _variants_for(metric_catalog, base, mode)[0]
    bundle, name, _label = pack.NODE_CLASS_MAPPINGS["PyiqaAnyLoader"]().load(
        arch, value
    )
    assert name == base, f"{base}: Load pyiqa Model reported model_name {name!r}"
    return bundle, arch, value


def main():
    fp_info = inject_fake_folder_paths()
    yaml_sim = fp_info["yaml_sim"]
    local_models_pyiqa = fp_info["local"]
    inject_fake_comfy()
    pack = load_pack()
    assert pack.NODE_CLASS_MAPPINGS, "no nodes registered"
    print("registered:", sorted(pack.NODE_CLASS_MAPPINGS))
    assert os.path.isdir(yaml_sim), f"yaml-simulated folder was not created: {yaml_sim}"
    print(f"yaml-simulated folder created: {yaml_sim}")

    import pyiqa_toolbox_smoke.utility.pyiqa_utils as utils
    import pyiqa_toolbox_smoke.nodes._common as common
    from pyiqa_toolbox_smoke.utility import metric_catalog
    import folder_paths as fp_mod

    print("\n-- Step 1: names survive a pyiqa rename; a broken table degrades --")
    # Architecture names and folders are matched from a table keyed by METRIC
    # name, not by pyiqa's internal class name, so a pyiqa release that renames a
    # class cannot rename (or hide) the folder a user keeps checkpoints in.
    # These are the folder names the pack accepted before that change.
    accepted_before = {
        "A-FINE": ["A-FINE", "AFINE"],
        "CLIP-IQA": ["CLIP-IQA", "CLIPIQA"],
        "CW-SSIM": ["CW-SSIM", "CW_SSIM"],
        "HyperIQA": ["HyperIQA", "HyperNet"],
        "LAION Aesthetic": ["LAION Aesthetic", "LAIONAes"],
        "MS-SSIM": ["MS-SSIM", "MS_SSIM"],
        "MS-SWD (learned)": ["MS-SWD", "MS_SWD_learned"],
        "PaQ-2-PiQ": ["PaQ-2-PiQ", "PAQ2PIQ"],
        "Q-Align": ["Q-Align", "QAlign"],
        "Q-ReAlign": ["Q-ReAlign", "QReAlign"],
        "ST-LPIPS": ["ST-LPIPS", "STLPIPS"],
        "TOPIQ (CFANet)": ["TOPIQ", "CFANet"],
    }
    for label, names in accepted_before.items():
        arch = metric_catalog.ARCHITECTURES[label]
        for name in names:
            assert metric_catalog._normalize(name) in arch.folder_keys, (
                f"folder {name!r} no longer addresses {label!r}: every name the "
                "pack ever accepted must keep working"
            )
    # folders[0] is what scaffold_folders() creates: an empty tuple (or a
    # reordered one) would quietly create a second set of drop folders
    assert all(arch.folders for arch in metric_catalog.ARCHITECTURES.values()), (
        "every architecture needs a canonical folder"
    )
    assert metric_catalog.ARCHITECTURES["MS-SWD (learned)"].folders[0] == "MS-SWD"
    print("every folder name that ever worked still addresses its architecture")

    renamed = metric_catalog._arch_rule("CFANetV3", ["topiq_nr", "topiq_fr"])
    assert renamed.label == "TOPIQ (CFANet)", renamed.label
    assert "topiq" in metric_catalog._normalize(renamed.folders[0]), renamed.folders
    assert renamed.single_image_metrics, "the rule travels with its own overrides"
    print(f"a renamed pyiqa class keeps its label, folder and rules: {renamed.label}")

    # An unreadable pyiqa metric table must degrade, not delete the nodes.
    saved_configs, saved_error = metric_catalog._CONFIGS, metric_catalog.REGISTRY_ERROR
    metric_catalog._CONFIGS, metric_catalog.REGISTRY_ERROR = {}, "smoke test"
    try:
        fallback = metric_catalog._safe_build()
        assert len(fallback) == 1, fallback
        only = next(iter(fallback.values()))
        assert only.variants, "the fallback catalog must still list metrics"
        fb_resolved = metric_catalog._resolved_registry(only, only.variants[0])
        assert fb_resolved.mode == "?", fb_resolved.mode
        assert "unknown" in fb_resolved.label, fb_resolved.label
        fb_bundle = common.MetricBundle(fb_resolved, lambda x: x, "any")
        for node_name in ("PyiqaAnyFRScore", "PyiqaAnyNRScore"):
            pack.NODE_CLASS_MAPPINGS[node_name]()._require_compatible(fb_bundle)
        print(
            f"unreadable metric table -> {len(only.variants)} built-in metrics, "
            "unknown mode accepted by both score nodes"
        )
    finally:
        metric_catalog._CONFIGS, metric_catalog.REGISTRY_ERROR = saved_configs, saved_error

    print("\n-- Step 2: an alpha channel in the input tensor is dropped --")
    # pyiqa asserts 1-or-3-channel input, and RGBA tensors do reach the image
    # sockets (Qwen Image 2.1's decoder emits a 4-channel image). The conversion
    # every score node shares must score the RGB channels instead of crashing
    # (regression: first user-reported bug, 2026-10-13).
    rgba = torch.rand(2, 8, 8, 4)
    conv = utils.comfy_image_to_pyiqa(rgba, torch.device("cpu"))
    assert conv.shape == (2, 3, 8, 8), conv.shape
    torch.testing.assert_close(conv, rgba[..., :3].permute(0, 3, 1, 2))
    rgb = utils.comfy_image_to_pyiqa(torch.rand(1, 8, 8, 3), torch.device("cpu"))
    assert rgb.shape == (1, 3, 8, 8), rgb.shape
    gray = utils.comfy_image_to_pyiqa(torch.rand(1, 8, 8, 1), torch.device("cpu"))
    assert gray.shape == (1, 1, 8, 8), gray.shape
    print("4-channel tensors score their RGB channels; RGB and gray pass through")

    _expect_value_error(
        lambda: pack.NODE_CLASS_MAPPINGS["PyiqaAnyNRScore"]()._require_compatible(
            common.MetricBundle(metric_catalog.resolve("SSIM"), lambda x: x, "classic_fr")
        ),
        "FR-reference model",
    )
    print("a known FR/NR mismatch is still refused once the table is back")

    # The startup line: the pack's version comes from pyproject.toml, so it
    # cannot drift from what the registry published.
    import tomllib

    with open(os.path.join(PKG_DIR, "pyproject.toml"), "rb") as fh:
        pyproject_version = tomllib.load(fh)["project"]["version"]
    assert pack._pack_version() == pyproject_version, (
        pack._pack_version(),
        pyproject_version,
    )
    captured = []

    class _Capture(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    pack_logger = logging.getLogger("pyiqa_toolbox_smoke")
    handler = _Capture()
    previous_level = pack_logger.level
    pack_logger.setLevel(logging.INFO)  # info() is dropped at the default level
    pack_logger.addHandler(handler)
    try:
        pack._log_versions()
    finally:
        pack_logger.removeHandler(handler)
        pack_logger.setLevel(previous_level)
    line = next((m for m in captured if "architectures" in m), "")
    assert pyproject_version in line and "pyiqa" in line, captured
    print("startup line:", line)

    print("\n-- Step 3: the 10-node surface; refusal messages name survivors --")
    # No weights are loaded: the accept/refuse matrix drives the guard (which
    # runs before any model work) with bundles built from the catalog, so this
    # whole section is milliseconds and cannot be the reason the suite is slow.
    nodes = pack.NODE_CLASS_MAPPINGS
    names = pack.NODE_DISPLAY_NAME_MAPPINGS
    # the surface itself: 10 classes in NODE_CONFIG order, each with its display
    # name. Registration is what the UI shows, so a rename or a re-added node is
    # caught here rather than by a missing attribute somewhere later.
    assert len(nodes) == len(NODE_SURFACE) == 10, (
        f"the pack is a 10-node surface, got {len(nodes)}: {sorted(nodes)}"
    )
    assert list(nodes) == [node_id for node_id, _label in NODE_SURFACE], (
        f"registered classes (NODE_CONFIG order) changed: {list(nodes)}"
    )
    assert [names[node_id] for node_id, _label in NODE_SURFACE] == [
        label for _node_id, label in NODE_SURFACE
    ], names
    assert names["PyiqaAnyLoader"] == "(Down)load pyiqa Model", names["PyiqaAnyLoader"]
    for gone in REMOVED_NODES:
        assert gone not in nodes, f"{gone} is not part of the 10-node surface"
        assert gone not in names, f"{gone} still has a display name: {names.get(gone)}"
    still = sorted(n for n in CULLED_NODES if n in nodes)
    assert not still, f"culled node types still registered: {still}"
    print(f"registered: {len(nodes)} nodes in NODE_CONFIG order; "
          f"all {len(CULLED_NODES)} earlier culled types and "
          f"{len(REMOVED_NODES)} more ({' '.join(REMOVED_NODES)}) gone")
    # the same cull in the catalog's API: these names must not survive anywhere
    survivors = [api for api in GONE_CATALOG_API if hasattr(metric_catalog, api)]
    assert not survivors, f"removed metric_catalog API still present: {survivors}"
    assert hasattr(metric_catalog, "VARIANT_LOOKUP") and hasattr(
        metric_catalog, "ARCHITECTURES"
    ), "the architecture/variant tables are the catalog's public surface now"
    assert not hasattr(metric_catalog, "EXCLUDED_METRICS"), (
        "EXCLUDED_METRICS is gone: the metrics it hid are listed now "
        "(metric_catalog.SPECIAL_INPUTS says who scores them)"
    )
    assert hasattr(metric_catalog, "SPECIAL_INPUTS") and callable(
        metric_catalog.special_input
    ), "SPECIAL_INPUTS + special_input() are the catalog's special-metric surface"
    assert hasattr(common, "ModelLoaderBase"), "FamilyLoaderBase -> ModelLoaderBase"
    assert not hasattr(common, "FamilyLoaderBase"), (
        "FamilyLoaderBase was renamed to ModelLoaderBase"
    )
    print(f"metric_catalog: {len(GONE_CATALOG_API)} removed family-table APIs gone; "
          "ModelLoaderBase is the loader base")

    # SCORE_NODES is filled from the registered classes, so a refusal can only
    # ever name a node that exists - pin the exact map, then check every entry
    # against the display names. CLIP Score registers under its own family: it is
    # the node that can score a clipscore model (it brings the caption).
    assert common.SCORE_NODES == {
        ("any", "FR"): "pyiqa FR Score",
        ("any", "NR"): "pyiqa NR Score",
        ("clipscore", "NR"): "CLIP Score",
        ("fgresq", "FR"): "FGResQ FR Score",
        ("fgresq", "NR"): "FGResQ NR Score",
        ("qrealign", "NR"): "QReAlign NR Score",
    }, common.SCORE_NODES
    for (family, mode), label in sorted(common.SCORE_NODES.items()):
        assert label in names.values(), f"SCORE_NODES names an unregistered node: {label}"
    # ...and the lookup any message goes through never returns a culled label
    for family in sorted(set(metric_catalog.FAMILIES) | {"any", "", "no-such-family"}):
        for mode in ("FR", "NR"):
            assert common.score_node_for(family, mode) in names.values(), (family, mode)
    assert not any(
        culled in label
        for culled in CULLED_NODES
        for label in common.SCORE_NODES.values()
    ), common.SCORE_NODES
    print("SCORE_NODES:", dict(sorted(common.SCORE_NODES.items())))

    print("accept/refuse matrix per score node (catalog-built bundles, no weights):")
    # Every registry variant the catalog knows, INCLUDING the ones behind a
    # signpost architecture: its dropdown entry is a message, so the variant
    # objects are built from the architecture's own row here.
    candidates = [
        resolved
        for resolved in metric_catalog.VARIANT_LOOKUP.values()
        if not resolved.custom
    ]
    for label in metric_catalog.ARCHES:
        arch = metric_catalog.ARCHITECTURES[label]
        if arch.signpost:
            candidates += [
                metric_catalog._resolved_registry(arch, name)
                for name in arch.variants
            ]
    for (family, mode), label in sorted(common.SCORE_NODES.items()):
        node = next(
            cls()
            for cls in nodes.values()
            if getattr(cls, "SCORE_LABEL", None) == label
        )
        # a variant of that family and mode; FGResQ's only FR variant IS the
        # pair metric (multi-output by design, and that node handles it)
        variant = None
        for resolved in candidates:
            if resolved.mode != mode or (
                family != "any" and metric_catalog.family_of(resolved.arch) != family
            ):
                continue
            if not resolved.multi_output:
                variant = resolved
                break
            if variant is None:
                variant = resolved
        assert variant, f"no {mode} variant in the catalog for family {family}"
        good = common.MetricBundle(variant, lambda x: x, family)
        node._require_compatible(good)  # must not raise
        accepted = f"{variant.arch} | {variant.base}"
        print(f"   {label:<16} accepts {accepted[:62]}")
        # the opposite mode must be refused, naming a node that is registered
        other = "NR" if mode == "FR" else "FR"
        bad_variant = None
        for resolved in candidates:
            if resolved.multi_output or resolved.mode != other:
                continue
            if family != "any" and metric_catalog.family_of(resolved.arch) != family:
                continue
            bad_variant = resolved
            break
        if bad_variant is None:
            print(f"      no {other} variant in family {family!r} - nothing to refuse")
            continue
        bad = common.MetricBundle(bad_variant, lambda x: x, family)
        if getattr(node, "METRIC_FROM_BUNDLE", False):
            # this family's ONE net serves both modes (FGResQ): the score node
            # asks the bundle for the mode it needs (``for_mode``), so the other
            # variant of the SAME family is legitimately accepted - the mode
            # comparison is off by design, and the family check is what still
            # refuses a foreign model (asserted in the FGResQ section below)
            node._require_compatible(bad)  # must not raise
            print(f"      accepts {other} too (METRIC_FROM_BUNDLE: one net, "
                  "both modes)")
            continue
        message = _expect_value_error(
            lambda: node._require_compatible(bad),
            f"{other}-reference model",
            f"{mode}-reference models only",
        )
        assert any(v in message for v in names.values()), message
        print(f"      refuses {other}: ...{message[-62:]}")

    # the Q-ReAlign bug fix: MODE_CHECK was None, which refused EVERY model. A
    # catalog bundle is now accepted as NR, while the dedicated loader's raw
    # pyiqa metric (no ``resolved``, mode reads as unknown) stays silent. The
    # architecture's own Resolved is built from its rule row: the Q-ReAlign
    # *dropdown* entry is a signpost, so resolve("Q-ReAlign") refuses by design.
    q_node = nodes["QReAlignScore"]()
    assert q_node.MODE_CHECK == "NR", q_node.MODE_CHECK
    q_resolved = metric_catalog._resolved_registry(
        metric_catalog.ARCHITECTURES["Q-ReAlign"], "qrealign"
    )
    q_node._require_compatible(
        common.MetricBundle(q_resolved, lambda x: x, "qrealign")
    )
    q_node._require_compatible(object())  # the raw-metric path: no resolved/mode
    _expect_value_error(
        lambda: q_node._require_compatible(
            common.MetricBundle(metric_catalog.resolve("BRISQUE"), lambda x: x, "any")
        ),
        "QReAlign NR Score",
        "Q-ReAlign models only",
        "pyiqa NR Score",
    )
    print(f"QReAlign NR Score: MODE_CHECK={q_node.MODE_CHECK!r}, accepts a "
          f"Q-ReAlign bundle (mode {q_resolved.mode}), a raw metric, and refuses "
          "a foreign NR model by name")

    print("\n-- signposts: listed in the combo, never loaded from it --")
    # A signpost architecture gets exactly ONE dropdown entry and its label IS
    # the message: picking it explains where to go instead of loading weights
    # this loader cannot build (Q-ReAlign needs its int8/snapshot/dtype loader)
    # or that no score node can run (Inception Score is a one-set metric).
    # resolve() refuses the entry, so nothing can build a model from it.
    for arch_label, needles in (
        ("Q-ReAlign", ("Q-ReAlign", "(Down)load Q-ReAlign")),
        (
            "Inception Score",
            ("Inception Score", "no score node for a single image set"),
        ),
    ):
        arch = metric_catalog.ARCHITECTURES[arch_label]
        assert arch.signpost, f"{arch_label} must stay a signpost"
        values = _dropdown_values(metric_catalog, arch_label)
        assert len(values) == 1, f"{arch_label} must have one entry: {values}"
        assert values[0] == (
            f"{arch_label}{metric_catalog.SEPARATOR}{arch.signpost}"
        ), values[0]
        entry = metric_catalog.VARIANT_LOOKUP[values[0]]
        assert entry.kind == "signpost", entry
        assert entry.label == arch.signpost, (entry.label, arch.signpost)
        message = _expect_value_error(
            lambda: metric_catalog.resolve(arch_label, values[0]), *needles
        )
        assert message.startswith(arch_label), message
        print(f"  {arch_label}: resolving its entry -> ValueError: "
              f"{message[:60]}...")
    # ...and a signpost architecture whose family takes no custom weights gets no
    # drop folder: the SCAFFOLD_FAMILIES rule decides, not the signpost itself
    # (Q-ReAlign's own loader reads snapshot folders, so its folder IS created)
    scaffolded = _expected_scaffold_folders(metric_catalog)
    assert "Q-ReAlign" in scaffolded, scaffolded
    assert "Inception Score" not in scaffolded, (
        "Inception Score takes no custom weights (family 'fid' has no scaffold "
        f"folder): {scaffolded}"
    )
    print(f"  {len(scaffolded)} scaffolded architecture folders; "
          "'Inception Score' is not one of them")

    print("\n-- alias collapse: pyiqa names that build the same weights --")
    # qrealign-mini and qrealign build identical metric_opts, so they are ONE
    # entry. The folded name is not lost: it stays a known metric (it never
    # triggers the "unlisted metrics" warning) and its weights load as qrealign.
    assert metric_catalog.ALIASES == {"qrealign-mini": "qrealign"}, (
        metric_catalog.ALIASES
    )
    assert metric_catalog.ARCHITECTURES["Q-ReAlign"].variants == (
        "qrealign", "qrealign-lite", "qrealign-pro"
    ), metric_catalog.ARCHITECTURES["Q-ReAlign"].variants
    offending = [v for v in metric_catalog.VARIANT_VALUES if "qrealign-mini" in v]
    assert not offending, f"qrealign-mini must be folded into qrealign: {offending}"
    assert "Unlisted" not in metric_catalog.ARCHITECTURES, "no 'Unlisted' architecture"
    assert not any(label == "Unlisted" for label in metric_catalog.ARCHES)
    print(f"  ALIASES={metric_catalog.ALIASES}; Q-ReAlign registry variants="
          f"{metric_catalog.ARCHITECTURES['Q-ReAlign'].variants}; no 'Unlisted' arch")

    print("\n-- special inputs: listed in the combo, refused per image --")
    # The metrics that cannot be scored per image used to be hidden from the
    # dropdown. They are listed now, each with the honest mode label (set-level /
    # one set / NR + caption) and a pointer, and the per-image score nodes refuse
    # them at run time instead of failing somewhere inside pyiqa.
    for base, arch_label, kind, label_needle in (
        ("fid", "FID", "distance", "use FID Score"),
        ("sfid", "FID", "distance", "use FID Score"),
        ("fid_dinov2", "FID", "distance", "use FID Score"),
        ("clipscore", "CLIP Score", "caption", "use CLIP Score"),
        ("inception_score", "Inception Score", "one_set", "single image set"),
    ):
        value = _value_for_base(metric_catalog, base)
        resolved = metric_catalog.VARIANT_LOOKUP[value]
        assert resolved.arch == arch_label, (base, resolved.arch)
        special = metric_catalog.special_input(base)
        assert special is not None and special[0] == kind, (base, special)
        assert label_needle in resolved.label, (base, resolved.label)
    assert set(metric_catalog.SPECIAL_INPUTS) == {
        "fid", "sfid", "fid_dinov2", "inception_score", "clipscore"
    }, metric_catalog.SPECIAL_INPUTS
    print(f"  listed under their own architectures: "
          f"{sorted(metric_catalog.SPECIAL_INPUTS)}")
    fid_bundle = common.MetricBundle(
        metric_catalog.resolve("FID", _value_for_base(metric_catalog, "fid")),
        lambda x: x,
        "fid",
    )
    clip_bundle = common.MetricBundle(
        metric_catalog.resolve(
            "CLIP Score", _value_for_base(metric_catalog, "clipscore")
        ),
        lambda x: x,
        "clipscore",
    )
    for node_id in ("PyiqaAnyFRScore", "PyiqaAnyNRScore"):
        node = nodes[node_id]()
        for model, pointer in (
            (fid_bundle, "use FID Score"),
            (clip_bundle, "use CLIP Score"),
        ):
            message = _expect_value_error(
                lambda: node._require_compatible(model),
                "is not scored per image",
                pointer,
            )
        print(f"  {node_id}: refuses a FID and a clipscore model -> "
              f"...{message[-58:]}")
    # CLIP Score is the node that supplies the caption, so it must accept the
    # clipscore model its own loader produces - and still refuse a set-level one
    nodes["CLIPScore"]()._require_compatible(clip_bundle)  # must not raise
    _expect_value_error(
        lambda: nodes["CLIPScore"]()._require_compatible(fid_bundle),
        "is not scored per image",
        "use FID Score",
    )
    print("  CLIP Score accepts its clipscore model and refuses a set-level one")

    print("\n-- FID Score refuses a per-image model --")
    # The other direction of the same guard: a per-image metric wired into FID
    # Score names the node that belongs there. The model is a catalog bundle (no
    # weights); target/reference are real 2-image batches so the refusal happens
    # at the model check, before anything is staged as PNGs.
    lp_bundle = common.MetricBundle(
        metric_catalog.resolve("LPIPS"), lambda x: x, "lpips"
    )
    probe = fake_image(2, 64, 64, seed=21)
    message = _expect_value_error(
        lambda: nodes["FIDScore"]().score(
            model=[lp_bundle],
            target=[probe],
            reference=[probe],
            resize_mode=["clean"],
            memory_management=[common.MEMORY_KEEP],
        ),
        "FID Score needs a set-level distance metric",
        "(Down)load pyiqa Model",
        "pyiqa FR Score",
    )
    print(f"  LPIPS model -> ValueError: {message[:64]}...")

    print("\n-- labels: MAD (L1), the LPIPS axes, set-level modes --")
    # "mad" is pyiqa's mean-absolute-difference metric, the L1-style metric the
    # removed L1Score node computed. The label says what it is; the FOLDER stays
    # "MAD", because renaming a label must never move a user's drop folder.
    mad = metric_catalog.ARCHITECTURES["MAD (L1)"]
    assert mad.folders == ("MAD",), mad.folders
    assert "mad" in mad.variants, mad.variants
    mad_labels = [
        metric_catalog.VARIANT_LOOKUP[v].label
        for v in _dropdown_values(metric_catalog, "MAD (L1)")
    ]
    assert any("mean absolute difference" in label for label in mad_labels), mad_labels
    for arch_label, axes in (
        (
            "LPIPS",
            (
                "LPIPS (AlexNet)",
                "LPIPS (VGG)",
                "LPIPS+ (AlexNet, semantic weighting)",
                "LPIPS+ (VGG, semantic weighting)",
            ),
        ),
        ("ST-LPIPS", ("shift-tolerant (AlexNet)", "shift-tolerant (VGG)")),
    ):
        labels = [
            metric_catalog.VARIANT_LOOKUP[v].label
            for v in _dropdown_values(metric_catalog, arch_label)
        ]
        for want in axes:
            assert any(label.startswith(want + " -") for label in labels), (
                arch_label, want, labels
            )
    # set-level metrics say what they need instead of pyiqa's own "NR"
    for value in _dropdown_values(metric_catalog, "FID"):
        label = metric_catalog.VARIANT_LOOKUP[value].label
        assert "set-level" in label and ", NR," not in label, label
    print(f"  MAD (L1) [folder {mad.folders[0]}] = {mad_labels[0]}")
    print("  LPIPS/ST-LPIPS axes intact; FID entries say 'set-level', not 'NR'")

    print("\n-- CLIP Score node: caption rules + per-image forwarding --")
    clip_cls = nodes["CLIPScore"]
    caption_input = clip_cls.INPUT_TYPES()["required"]["caption"]
    assert caption_input[0] == "STRING", caption_input
    assert caption_input[1].get("multiline") is True, caption_input
    from pyiqa_toolbox_smoke.nodes.clip_nodes import expand_captions

    # one caption line is broadcast to the whole batch
    assert expand_captions("a photo of a cat", 4) == ["a photo of a cat"] * 4
    # exactly one line per image: paired in order (lines are stripped)
    assert expand_captions("cat\n dog \nbird", 3) == ["cat", "dog", "bird"]
    # any other count is refused, and the message says both counts
    message = _expect_value_error(
        lambda: expand_captions("a\nb\nc", 2), "3 captions", "2 image(s)"
    )
    # a trailing newline (normal in a text box) is not a caption
    assert expand_captions("a\nb\n", 2) == ["a", "b"]
    for blank in ("", "   \n  "):
        _expect_value_error(
            lambda: expand_captions(blank, 2), "needs a caption"
        )
    print(f"  expand_captions: 1 line -> N, N lines -> N in order, bad counts -> "
          f"ValueError ({message[:40]}...)")
    # the score body forwards one caption per image and never downloads CLIP's
    # ViT-B/32: the model is a recording stub behind a real clipscore Resolved
    clip_resolved = metric_catalog.resolve(
        "CLIP Score", _value_for_base(metric_catalog, "clipscore")
    )
    calls = []

    class _RecordingClipScore:
        """Records (shape, captions) per call; returns one score per image."""

        def __call__(self, x, captions):
            calls.append((tuple(x.shape), list(captions)))
            return torch.zeros(x.shape[0])

    clip_model = common.MetricBundle(
        clip_resolved, _RecordingClipScore(), "clipscore"
    )
    clip_node = clip_cls()
    batch = fake_image(2, 64, 64, seed=31)
    (scores,) = clip_node.score(
        model=clip_model,
        image=batch,
        caption="one caption",
        memory_management=common.MEMORY_KEEP,
    )
    assert len(calls) == 1 and calls[0][0][0] == 2, calls
    assert calls[0][1] == ["one caption", "one caption"], calls
    assert scores == [0.0, 0.0], scores
    calls.clear()
    (scores,) = clip_node.score(
        model=clip_model,
        image=batch,
        caption="first\nsecond",
        memory_management=common.MEMORY_KEEP,
    )
    assert len(calls) == 1, calls
    assert calls[0][1] == ["first", "second"], calls
    assert scores == [0.0, 0.0], scores
    print("  score(): 2 images + 1 caption -> one call with the caption twice; "
          "2 captions -> paired per image (no downloads)")

    print("\n-- Step 4: one rule table per architecture; the Q-Align gate --")
    # Nothing is downloaded: the gate is exercised as a function, and the
    # build_metric hook is proven with a temporary sentinel builder for an
    # architecture that costs nothing to "build".
    import pyiqa_toolbox_smoke.nodes.arch_gates as arch_gates

    rules = metric_catalog._ARCH_RULES
    by_label = {arch.label: arch for arch in metric_catalog.ARCHITECTURES.values()}
    checked = []
    for name, rule in rules.items():
        if rule.label not in by_label:  # this pyiqa does not ship that arch
            continue
        arch = by_label[rule.label]
        assert arch.folders == (rule.folders or (rule.label,)), arch.folders
        assert arch.policy == rule.policy, (rule.label, arch.policy)
        assert arch.single_image == rule.single_image, rule.label
        assert arch.single_image_metrics == rule.single_image_metrics, rule.label
        assert arch.snapshot_loader == rule.snapshot_loader, rule.label
        checked.append(rule.label)
    assert len(checked) >= 12, checked
    # the five tables these rules replaced must not come back as a second source
    for gone in (
        "INPUT_POLICIES",
        "SINGLE_IMAGE_ARCHES",
        "SINGLE_IMAGE_METRICS",
        "SNAPSHOT_LOADERS",
        "_ARCH_NAMES",
    ):
        assert not hasattr(metric_catalog, gone), f"{gone} is a second source of truth"
    print(f"rule table: {len(checked)} architectures carry overrides, and every one "
          "of them reaches its Architecture object")
    assert metric_catalog.ARCHITECTURES["A-FINE"].single_image_for("afine_fr")
    assert metric_catalog.ARCHITECTURES["A-FINE"].policy, "A-FINE needs the mult32 policy"
    assert not metric_catalog.ARCHITECTURES["TOPIQ (CFANet)"].single_image
    assert metric_catalog.ARCHITECTURES["TOPIQ (CFANet)"].single_image_for("topiq_nr-face")
    print("A-FINE mult32 + batch-1, TOPIQ face heads batch-1, Q-ReAlign snapshot loader:",
          by_label["Q-ReAlign"].snapshot_loader)

    # the gate reads pyiqa's own construction options (where Q-Align's precision
    # lives - the variant label does not carry it)
    assert metric_catalog.metric_opts("qalign_4bit").get("dtype") == "4bit", (
        metric_catalog.metric_opts("qalign_4bit")
    )
    assert "dtype" not in metric_catalog.metric_opts("ssim")
    assert common._ARCH_PRELOAD.get("Q-Align") is arch_gates.gate_qalign, (
        "the Q-Align gate must be registered where build_metric looks"
    )
    assert arch_gates.qalign_dtype(metric_catalog.resolve("Q-Align")) == "fp16"
    assert arch_gates.qalign_dtype(
        metric_catalog.resolve("Q-Align", "qalign_4bit")
    ) == "4bit"
    arch_gates.check_qalign_deps("fp16")  # nothing to check
    arch_gates.check_qalign_deps("4bit")  # bitsandbytes IS installed here
    # ...and a missing quantizer is refused with the fix, not silently loaded at
    # 15 GiB (pyiqa itself only warns and falls back to fp16)
    msg = _expect_value_error(
        lambda: arch_gates.check_qalign_deps("4bit", find_spec=lambda name: None),
        "bitsandbytes",
        "accelerate",
        "fp16",
        "pip install",
        exc_type=RuntimeError,
    )
    print(f"Q-Align 4-bit without bitsandbytes -> RuntimeError: {msg[:66]}...")

    # the gate runs before the build (and says how big the model is)
    captured = []

    class _GateCapture(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    gate_logger = logging.getLogger("pyiqa_toolbox_smoke.nodes.arch_gates")
    gate_handler = _GateCapture()
    gate_level = gate_logger.level
    gate_logger.setLevel(logging.INFO)  # info() is dropped at the default level
    gate_logger.addHandler(gate_handler)
    try:
        arch_gates.gate_qalign(metric_catalog.resolve("Q-Align", "qalign_4bit"), "cpu")
    finally:
        gate_logger.removeHandler(gate_handler)
        gate_logger.setLevel(gate_level)
    size_line = next((m for m in captured if "one-align" in m), "")
    assert "8.2B parameters" in size_line and "GiB" in size_line, captured
    print("Q-Align gate says:", size_line)

    # and build_metric actually consults it, whichever loader asked for the model
    sentinel = object()
    seen = []
    saved_gate = common._ARCH_PRELOAD.pop("Q-Align")  # keep Q-Align out of this
    saved_builder = common._ARCH_BUILDERS.get("SSIM")
    try:
        common.register_arch_preload("SSIM", lambda resolved, device: seen.append(resolved.base))
        common._ARCH_BUILDERS["SSIM"] = lambda resolved, device: sentinel
        assert common.build_metric(metric_catalog.resolve("SSIM"), "cpu") is sentinel
        assert seen == ["ssim"], seen
    finally:
        common._ARCH_PRELOAD.pop("SSIM", None)
        if saved_builder is None:
            common._ARCH_BUILDERS.pop("SSIM", None)
        else:
            common._ARCH_BUILDERS["SSIM"] = saved_builder
        common.register_arch_preload("Q-Align", saved_gate)
    print("build_metric runs the gate of the architecture it builds, then builds "
          "exactly as before")

    print("\n-- BUG 1: library registration vs search paths --")
    registered = fp_mod.get_folder_paths("pyiqa")
    print("registered (library-visible) pyiqa folders:", registered)
    assert registered == [local_models_pyiqa], (
        f"only the local models/pyiqa must stay registered, got {registered}"
    )
    assert not any(_norm(p) == _norm(yaml_sim) for p in registered), (
        "yaml-sim path must NOT be library-registered"
    )
    search = utils._pyiqa_folder_paths()
    print("search paths (_pyiqa_folder_paths):")
    for p in search:
        print("   ", p)
    assert _norm(local_models_pyiqa) in {_norm(p) for p in search}
    assert _norm(yaml_sim) in {_norm(p) for p in search}, (
        "yaml-sim path must still be SEARCHED even though not registered"
    )
    assert len(search) == len({_norm(p) for p in search}), "search paths must dedupe"
    print("_EXTRA_SEARCH_PATHS:", utils._EXTRA_SEARCH_PATHS)
    print("user-facing custom-checkpoint roots (_user_pyiqa_dirs):",
          utils._user_pyiqa_dirs())

    print("\n-- custom-checkpoint folders (scaffolding, per custom root) --")
    # Custom checkpoints live in {architecture}/ folders, so the pack creates
    # them where a user can drop a file. custom_roots() covers the install-local
    # models/pyiqa plus every yaml-registered pyiqa folder that is NOT a
    # download cache; the cache folders must stay free of drop folders.
    expected_arch_folders = _expected_scaffold_folders(metric_catalog)
    roots = metric_catalog.custom_roots()
    print(f"custom_roots() -> {len(roots)} root(s):")
    for root in roots:
        print("   ", root)
    assert roots, "custom_roots() must not be empty while pyiqa folders are registered"
    assert roots[0] == local_models_pyiqa and os.path.isabs(roots[0]), (
        f"custom_roots() must start at the install-local models/pyiqa, got {roots[0]!r}"
    )
    # every root must be a REAL folder: a bare entry name or a weight file name
    # would make scaffold_folders() create <CWD>/<entry>/<Architecture>/ trees
    for root in roots:
        assert os.path.isabs(root), f"custom_roots() returned a relative path: {root!r}"
        assert os.path.isdir(root), f"custom_roots() returned a non-folder: {root!r}"
    # exactly the registered user folders, minus the download caches
    registered_user = [local_models_pyiqa, *fp_info["yaml_dirs"], yaml_sim]
    expected_roots = [
        folder
        for folder in registered_user
        if os.path.isdir(folder)
        and not (
            ".cache" in _norm(folder)
            and ("huggingface" in _norm(folder) or "torch" in _norm(folder))
        )
    ]
    assert [_norm(r) for r in roots] == [_norm(r) for r in expected_roots], (
        f"custom_roots() = {roots}, expected the registered non-cache folders "
        f"{expected_roots}"
    )
    local_dirs = {
        name
        for name in os.listdir(local_models_pyiqa)
        if os.path.isdir(os.path.join(local_models_pyiqa, name))
    }
    missing = [f for f in expected_arch_folders if f not in local_dirs]
    assert not missing, (
        f"scaffold_folders() did not create {missing} under {local_models_pyiqa}"
    )
    created = metric_catalog.scaffold_folders()  # idempotent, returns every path
    created_keys = {_norm(p) for p in created}
    for root in roots:  # per root - deliberately not a fixed grand total
        if not os.path.isdir(root):
            continue  # a non-directory root cannot hold drop folders
        for folder in expected_arch_folders:
            assert _norm(os.path.join(root, folder)) in created_keys, (
                f"scaffold_folders() skipped {os.path.join(root, folder)}"
            )
    print(f"{len(expected_arch_folders)} architecture folders per custom root x "
          f"{len(roots)} roots = {len(created)} created paths")
    # the REAL machine folder must be one of them (this is the yaml-registered
    # models/pyiqa an extra_model_paths.yaml points the pack at)
    real_models = dev_paths.MODELS
    if real_models and os.path.isdir(real_models):
        assert _norm(real_models) in {_norm(r) for r in roots}, (
            f"the real yaml-registered folder {real_models} must be a custom root"
        )
        real_dirs = {
            name
            for name in os.listdir(real_models)
            if os.path.isdir(os.path.join(real_models, name))
        }
        missing_real = [f for f in expected_arch_folders if f not in real_dirs]
        assert not missing_real, (
            f"scaffold_folders() did not create {missing_real} under {real_models}"
        )
        print(f"real models/pyiqa is a custom root and holds the "
              f"{len(expected_arch_folders)} architecture drop folders: {real_models}")
    else:
        print("NOTE: PYIQA_TOOLBOX_MODELS absent -> real-root assertions vacuous "
              "(set it, or tools/dev_paths_local.py, to run them)")
    for cache in (
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
        os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "pyiqa"),
    ):
        if not os.path.isdir(cache):
            print(f"NOTE: download cache {cache} absent -> exclusion assertion vacuous")
            continue
        assert not any(_norm(r).startswith(_norm(cache)) for r in roots), (
            f"custom_roots() must skip the download cache {cache}"
        )
        leaked = [
            f for f in expected_arch_folders if os.path.isdir(os.path.join(cache, f))
        ]
        assert not leaked, f"scaffold folders leaked into the download cache {cache}: {leaked}"
    print("download caches (HF hub, torch hub/pyiqa) got no architecture folders")

    print("resolved Q-ReAlign-Mini-0.8B ->", utils.find_pyiqa_model("Q-ReAlign-Mini-0.8B"))
    print("resolved Q-ReAlign-Pro-9B    ->", utils.find_pyiqa_model("Q-ReAlign-Pro-9B"))
    print("resolved FGResQ.pth          ->", utils.find_pyiqa_model("FGResQ.pth"))

    # find_pyiqa_model also searches ONE LEVEL DOWN (<folder>/<Architecture>/
    # <name>): a model moved into its architecture folder must still be found, or
    # the loader re-downloads gigabytes that are already on disk.
    probe_name = "one-level-down-probe-model"
    probe_dir = os.path.join(yaml_sim, "probe-arch-folder")
    os.makedirs(probe_dir, exist_ok=True)
    probe_path = os.path.join(probe_dir, probe_name)
    with open(probe_path, "w") as fh:
        fh.write("stub")
    try:
        found = utils.find_pyiqa_model(probe_name)
        assert found is not None and _norm(found) == _norm(probe_path), (
            f"one-level-down lookup failed: got {found}, expected {probe_path}"
        )
        print("one-level-down lookup:", found)
        assert utils.find_pyiqa_model("no-such-model-anywhere") is None
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)

    import comfy.model_management as fake_mm

    hooked = getattr(fake_mm.unload_all_models, "_pyiqa_toolbox_hook", False)
    print("unload_all_models hooked:", hooked)
    assert hooked, "unload hook did not install"

    print("\n-- BUG 2: loader dropdown lists --")
    q_opts = pack.NODE_CLASS_MAPPINGS["QReAlignLoader"].INPUT_TYPES()["required"][
        "model_name"
    ][0]
    q_inputs = set(
        pack.NODE_CLASS_MAPPINGS["QReAlignLoader"].INPUT_TYPES()["required"]
    )
    assert q_inputs == {"model_name", "dtype"}, (
        f"QReAlignLoader must stay a plain model_name+dtype loader, got {q_inputs}"
    )
    print("QReAlign model_name options:", q_opts)
    from pyiqa_toolbox_smoke.nodes.qrealign_nodes import QREALIGN_VARIANTS

    builtin_labels = list(QREALIGN_VARIANTS)
    int8_labels = [k + " (int8_convrot)" for k in builtin_labels]
    # the int8_convrot entries must come FIRST, then the plain variants
    assert q_opts[:3] == int8_labels, f"int8 entries must come first, got {q_opts[:3]}"
    pro_path = utils.find_pyiqa_model("Q-ReAlign-Pro-9B")
    builtin_paths = {
        _norm(utils.find_pyiqa_model(folder))
        for _, _, folder in QREALIGN_VARIANTS.values()
        if utils.find_pyiqa_model(folder)
    }
    # Q-ReAlign customs are the snapshot dirs of models/pyiqa/Q-ReAlign/ (the
    # folder IS the architecture now; the old require_config glob scan is gone)
    q_customs = [
        (label, path)
        for directory in metric_catalog.family_custom_dirs("qrealign")
        for label, path in utils.list_snapshot_dirs(directory)
    ]
    print("Q-ReAlign custom snapshot dirs:", [label for label, _ in q_customs])
    # no option label may be a custom resolving to a built-in's own folder
    for label, path in q_customs:
        if label in q_opts:
            assert _norm(path) not in builtin_paths, (
                f"option {label} duplicates built-in path {path}"
            )
    pro_here = bool(real_models) and os.path.exists(
        os.path.join(real_models, "Q-ReAlign-Pro-9B")
    )
    if pro_path and pro_here:
        # the 3 built-ins follow the 3 int8 entries; customs (if any) come last
        assert q_opts[3:6] == builtin_labels, (
            f"plain built-ins must follow the int8 entries, got {q_opts[3:6]}"
        )
        if not q_customs:
            assert len(q_opts) == 6, (
                f"with Q-ReAlign-Pro-9B present in models/pyiqa and no custom "
                f"snapshot folders the list must be the 3 int8 entries + 3 "
                f"built-ins, got {q_opts}"
            )
        assert _norm(pro_path).startswith(_norm(real_models)), (
            f"Pro-9B should resolve from the yaml-registered models/pyiqa folder "
            f"({real_models}), got {pro_path}"
        )
    else:
        print("NOTE: <models>/Q-ReAlign-Pro-9B absent -> exact-6 assertion vacuous")

    # FGResQ has no loader of its own any more, so its "(default)" entry went
    # with it: the generic combo offers the architecture's two registry variants
    # (plus anything dropped in models/pyiqa/FGResQ/), and pyiqa's own default
    # weights are not offered at all - they land LOOSE in pyiqa's cache folders,
    # which the folder rule never scans (see the section below)
    fg_values = _dropdown_values(metric_catalog, "FGResQ")
    assert "(default)" not in fg_values, fg_values
    assert {metric_catalog.VARIANT_LOOKUP[v].base for v in fg_values} == {
        "fgresq", "fgresq_pair"
    }, fg_values
    for bad in ("FGResQ.pth", "FGResQ_Degradation.pth"):
        assert not any(
            bad in metric_catalog.VARIANT_LOOKUP[v].label for v in fg_values
        ), f"{bad} must not appear as a custom entry"
    print(f"FGResQ combo entries: {len(fg_values)} "
          f"(no '(default)': the architecture's registry variants)")

    print("\n-- custom checkpoint folder rule (synthetic tree) --")
    # The old "pyiqa's own defaults must never show up as customs" machinery is
    # gone because the layout made it unnecessary: pyiqa's defaults land LOOSE
    # in its cache folders, which are not scanned for customs at all. What
    # decides now is the FOLDER: {user pyiqa root}/<Architecture>/.
    tree = tempfile.mkdtemp(prefix="pyiqa_folder_rule_")

    def write_probe(rel, data=b"stub"):
        path = os.path.join(tree, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(data)
        return path

    named = write_probe(os.path.join("TOPIQ", "topiq_nr-spaq-myft.safetensors"))
    unnamed = write_probe(os.path.join("TOPIQ", "topiq-finetune.pth"))
    notes = write_probe(os.path.join("TOPIQ", "notes.txt"))
    lpips_ft = write_probe(os.path.join("LPIPS", "lpips-vgg-ft.pt"))
    loose = write_probe("loose.pth")
    topiq_snapshot = os.path.join(tree, "TOPIQ", "Some-Snapshot")
    write_probe(os.path.join("TOPIQ", "Some-Snapshot", "config.json"), b"{}")
    write_probe(os.path.join("TOPIQ", "Some-Snapshot", "model.safetensors"))
    qrealign_snapshot = os.path.join(tree, "Q-ReAlign", "My-Tune")
    write_probe(os.path.join("Q-ReAlign", "My-Tune", "config.json"), b"{}")
    write_probe(os.path.join("Q-ReAlign", "My-Tune", "model.safetensors"))

    real_user_dirs = utils._user_pyiqa_dirs
    utils._user_pyiqa_dirs = lambda: [tree]
    try:
        _rebuild_catalog_tables(metric_catalog)

        def entries_for_path(entries, path):
            return [(label, r) for label, r in entries if r.path == path]

        topiq_entries = metric_catalog._ARCH_ENTRIES["TOPIQ (CFANet)"]
        topiq_arch = metric_catalog.ARCHITECTURES["TOPIQ (CFANet)"]

        # a file named after a registry variant -> exactly one entry, loading
        # onto that variant (the file name only shortens the list, nothing more)
        chosen = entries_for_path(topiq_entries, named)
        assert len(chosen) == 1, f"named custom got {len(chosen)} entries: {chosen}"
        label, resolved = chosen[0]
        assert resolved.base == "topiq_nr-spaq", resolved
        assert resolved.kind == "safetensors", resolved
        assert resolved.custom and resolved.arch == "TOPIQ (CFANet)", resolved
        via_resolve = metric_catalog.resolve("TOPIQ (CFANet)", label)
        assert (
            via_resolve.base,
            via_resolve.kind,
            via_resolve.path,
        ) == ("topiq_nr-spaq", "safetensors", named), via_resolve
        print(f"TOPIQ: 'topiq_nr-spaq-myft.safetensors' -> 1 entry, "
              f"base=topiq_nr-spaq kind=safetensors")

        # a name matching no variant -> offered once per variant of THIS
        # architecture, so the base is chosen in the node, never guessed
        unnamed_entries = entries_for_path(topiq_entries, unnamed)
        variants = list(topiq_arch.variants)
        assert len(unnamed_entries) == len(variants) >= 2, unnamed_entries
        assert {r.base for _l, r in unnamed_entries} == set(variants), unnamed_entries
        assert all(r.kind == "checkpoint" for _l, r in unnamed_entries)
        print(f"TOPIQ: 'topiq-finetune.pth' -> {len(unnamed_entries)} entries "
              f"(one per TOPIQ variant: {len(variants)})")

        # same rule for LPIPS (the /LPIPS/ folder is the architecture)
        lpips_entries = entries_for_path(
            metric_catalog._ARCH_ENTRIES["LPIPS"], lpips_ft
        )
        assert len(lpips_entries) == 1, lpips_entries
        assert lpips_entries[0][1].base == "lpips-vgg", lpips_entries[0][1]
        assert lpips_entries[0][1].kind == "checkpoint", lpips_entries[0][1]

        all_resolved = [
            r
            for entries in metric_catalog._ARCH_ENTRIES.values()
            for _label, r in entries
        ]
        # non-weight files are never customs
        assert notes not in {r.path for r in all_resolved}, "notes.txt leaked in"
        # a weight file loose in the pyiqa root is NOT a custom any more
        assert loose not in {r.path for r in all_resolved}, (
            "a loose weight file in the pyiqa root leaked in as a custom"
        )
        # a directory loader must exist for an arch, else a config.json folder is
        # not advertised there (TOPIQ has none)
        assert topiq_snapshot not in {r.path for r in all_resolved}, (
            "TOPIQ snapshot advertised although TOPIQ has no directory loader"
        )
        assert not any(r.kind == "snapshot" for _l, r in topiq_entries)
        # Q-ReAlign is a signpost in the generic combo: its single entry is the
        # message, so the architecture's custom snapshot folders are NOT combo
        # variants any more. The dedicated loader is what discovers them - which
        # is exactly why the architecture is signposted.
        qr_entries = entries_for_path(
            metric_catalog._ARCH_ENTRIES["Q-ReAlign"], qrealign_snapshot
        )
        assert not qr_entries, (
            f"a signpost architecture must not list custom variants: {qr_entries}"
        )
        import pyiqa_toolbox_smoke.nodes.qrealign_nodes as qr_nodes

        qr_customs = dict(qr_nodes._custom_options())
        assert "My-Tune" in qr_customs, qr_customs
        assert _norm(qr_customs["My-Tune"]) == _norm(qrealign_snapshot), qr_customs
        print("Q-ReAlign: 'My-Tune' snapshot -> offered by (Down)load Q-ReAlign "
              "(the generic combo shows its signpost); TOPIQ snapshot (no "
              "directory loader) -> not offered")

        # input adaptation is a property of the model, not a node widget
        afine = metric_catalog.resolve("A-FINE", "afine_fr")
        assert afine.policy == utils.POLICY_MULT32 == ("mult32",), afine.policy
        assert afine.single_image is True, afine
        assert metric_catalog.resolve("A-FINE", "afine_nr").policy == ("mult32",)
        assert metric_catalog.resolve("TOPIQ (CFANet)", "topiq_nr-face").single_image is True
        assert metric_catalog.resolve("LPIPS", "lpips").single_image is False
        assert metric_catalog.resolve("LPIPS", "lpips").policy == ()
        print(f"A-FINE policy={afine.policy} single_image={afine.single_image}; "
              "TOPIQ face heads single-image; LPIPS needs no adaptation")
    finally:
        utils._user_pyiqa_dirs = real_user_dirs
        _rebuild_catalog_tables(metric_catalog)
        # live refresh: with the real dirs back, the synthetic checkpoint is
        # gone from the tables again - a dropped file disappears with no restart
        assert named not in {
            r.path
            for entries in metric_catalog._ARCH_ENTRIES.values()
            for _l, r in entries
        }, "a removed custom checkpoint survived a rescan"
        print("live refresh: dropping the file took the entry away again "
              "(no restart)")
        shutil.rmtree(tree, ignore_errors=True)

    image = fake_image(2, 256, 256, seed=1)
    image2 = fake_image(2, 256, 256, seed=2)

    print("\n-- FGResQ ((Down)load pyiqa Model, architecture FGResQ) --")
    # The dedicated FGResQLoader (and its "(default)" + custom combo) is gone.
    # fgresq_nodes registers the architecture's builder, so the generic loader
    # returns exactly the bundle the old loader did. ONE net serves both pyiqa
    # metrics, so that ONE bundle scores BOTH family nodes: each asks the bundle
    # for its mode through ``for_mode``, which is what METRIC_FROM_BUNDLE tells
    # the mode guard to do instead of comparing the variant's own mode (the
    # generic pyiqa FR/NR nodes leave it off - see the asymmetry asserted below).
    assert nodes["FGResQScore"].METRIC_FROM_BUNDLE is True
    assert nodes["FGResQPairScore"].METRIC_FROM_BUNDLE is True
    for node_id in ("PyiqaAnyFRScore", "PyiqaAnyNRScore", "CLIPScore", "FIDScore"):
        assert getattr(nodes[node_id], "METRIC_FROM_BUNDLE", False) is False, node_id
    fg_nr_variant = _variant_value(metric_catalog, "fgresq", "NR")
    bundle, fg_name, fg_label = pack.NODE_CLASS_MAPPINGS["PyiqaAnyLoader"]().load(
        "FGResQ", fg_nr_variant
    )
    assert fg_name == "fgresq", fg_name
    assert "use FGResQ NR Score" in fg_label, fg_label
    assert bundle.nr.net is bundle.pair.net, "NR/pair wrappers must share the arch"
    # the loader caches one metric per (base, path, device): the same variant,
    # asked for by name the way ComfyUI calls it, is the same object
    again = pack.NODE_CLASS_MAPPINGS["PyiqaAnyLoader"]().load(
        architecture="FGResQ", variant=fg_nr_variant
    )
    assert again[0] is bundle and again[1] == "fgresq", again[1:]
    (scores,) = pack.NODE_CLASS_MAPPINGS["FGResQScore"]().score(
        bundle, image, memory_management=common.MEMORY_KEEP
    )
    assert len(scores) == 2
    print(f"NR    scores={[round(s, 4) for s in scores]}")
    # the SAME bundle feeds the pair node (no second load): the pair API's five
    # per-pair outputs come from the shared net
    out = pack.NODE_CLASS_MAPPINGS["FGResQPairScore"]().score(
        bundle, image, image2, memory_management=common.MEMORY_KEEP
    )
    assert len(out[0]) == len(out[2]) == len(out[4]) == 2
    print(f"PAIR  scores0={[round(s, 4) for s in out[0]]} "
          f"scores1={[round(s, 4) for s in out[1]]}")
    print(f"      ranks={out[2]} verdicts={out[3]} rank_probs={[round(p, 4) for p in out[4]]}")
    # ...while the GENERIC FR node must not take it: its contract is one value
    # per pair, and this variant is the per-image metric (the hint names the
    # FGResQ node that does take it)
    _expect_value_error(
        lambda: pack.NODE_CLASS_MAPPINGS["PyiqaAnyFRScore"]().score(
            bundle, image, image2
        ),
        "NR-reference model",
        "FGResQ NR Score",
    )
    # ONE FGResQ architecture serves two pyiqa metrics (an NR one and a pair
    # one), so the model a score node is handed must be able to name the metric
    # it needs: "(Down)load pyiqa Model" + its NR variant + pyiqa NR Score used
    # to be handed the PAIR metric and died with pyiqa's "Please specify
    # reference image for Full Reference metric" (regression). Asked for BEFORE
    # the other variant below: loading a second variant of the family evicts (and
    # tombstones) this cache entry, so a later load would legitimately rebuild.
    gen_bundle = pack.NODE_CLASS_MAPPINGS["PyiqaAnyLoader"]().load(
        "FGResQ", fg_nr_variant
    )[0]
    assert gen_bundle is bundle, "the generic loader must hit its own cache"
    assert gen_bundle.for_mode("NR") is gen_bundle.metric.nr
    assert gen_bundle.for_mode("FR") is gen_bundle.metric.pair
    (gen_scores,) = pack.NODE_CLASS_MAPPINGS["PyiqaAnyNRScore"]().score(
        gen_bundle, image, memory_management=common.MEMORY_KEEP
    )
    assert gen_scores == scores, (gen_scores, scores)
    print(f"      generic NR path via pyiqa NR Score: {[round(s, 4) for s in gen_scores]}"
          " (same object, same numbers)")
    # the other variant of the same architecture is the same weights: it feeds
    # FGResQ NR Score too (the bundle resolves the mode), and the generic FR node
    # still refuses it - fgresq_pair is the 4-value pair API
    pair_bundle, pair_name, pair_label = pack.NODE_CLASS_MAPPINGS[
        "PyiqaAnyLoader"
    ]().load("FGResQ", _variant_value(metric_catalog, "fgresq_pair", "FR"))
    assert pair_name == "fgresq_pair", pair_name
    assert "FR pair" in pair_label, pair_label
    (pair_nr_scores,) = pack.NODE_CLASS_MAPPINGS["FGResQScore"]().score(
        pair_bundle, image, memory_management=common.MEMORY_KEEP
    )
    assert len(pair_nr_scores) == 2, pair_nr_scores
    assert max(abs(a - b) for a, b in zip(pair_nr_scores, scores)) < 1e-6, (
        pair_nr_scores, scores
    )
    _expect_value_error(
        lambda: pack.NODE_CLASS_MAPPINGS["PyiqaAnyFRScore"]().score(
            pair_bundle, image, image2
        ),
        "fgresq_pair",
        "FGResQ FR Score",
    )
    print("      fgresq_pair feeds FGResQ NR Score too (same weights); pyiqa FR "
          "Score refuses it (4-value pair API)")
    # ...and a model of ANOTHER family is refused by BOTH FGResQ nodes, by family
    lp_bundle = common.MetricBundle(
        metric_catalog.resolve("LPIPS"), lambda x: x, "lpips"
    )
    _expect_value_error(
        lambda: pack.NODE_CLASS_MAPPINGS["FGResQScore"]().score(lp_bundle, image),
        "FGResQ models only",
        "pyiqa FR Score",
    )
    _expect_value_error(
        lambda: pack.NODE_CLASS_MAPPINGS["FGResQPairScore"]().score(
            lp_bundle, image, image2
        ),
        "FGResQ models only",
        "pyiqa FR Score",
    )
    print("      an LPIPS bundle is refused by both FGResQ nodes by family")

    print("\n-- QReAlign (mini) --")
    qmodel, q_name, q_label = pack.NODE_CLASS_MAPPINGS["QReAlignLoader"]().load(
        "qrealign-mini-0.8B", "auto"
    )
    assert q_name == "qrealign-mini" and q_label == "qrealign-mini-0.8B", (q_name, q_label)
    (qscores,) = pack.NODE_CLASS_MAPPINGS["QReAlignScore"]().score(
        qmodel, image, "quality IQA"
    )
    assert len(qscores) == 2
    print(f"NR    scores={[round(s, 4) for s in qscores]}  registry={q_name}")

    print("\n-- unload/reload cycle --")
    # pair_bundle is the LIVE model here: loading the pair variant evicted (and
    # tombstoned) the "fgresq" bundle above, so the unload below is what has to
    # tombstone this one
    assert getattr(bundle, "_pyiqa_gutted", False), (
        "loading the family's other variant must tombstone the evicted bundle"
    )
    assert not getattr(pair_bundle, "_pyiqa_gutted", False), pair_bundle
    fake_mm.unload_all_models()  # should offload the cache + call the original
    # The generic loader hands over MetricBundle -> FGResQBundle -> InferenceModel,
    # so the unload has to tombstone the WHOLE chain: gutting only the inner bundle
    # (a one-level unwrap) left this wrapper looking live, and the next score ran
    # the hollow net instead of reviving it. Regression coverage: the outer wrapper
    # is marked, and the same bundle still scores the same numbers below (the
    # decorator rebuilds it through its cached recipe).
    assert getattr(pair_bundle, "_pyiqa_gutted", False), (
        "unload_all_models did not tombstone the FGResQ WRAPPER - drop_metric has "
        "to mark the whole MetricBundle -> FGResQBundle -> net chain"
    )
    assert getattr(pair_bundle.metric, "_pyiqa_gutted", False), (
        "the wrapped FGResQBundle was not tombstoned"
    )
    (post_scores,) = pack.NODE_CLASS_MAPPINGS["FGResQScore"]().score(
        pair_bundle, image, memory_management=common.MEMORY_KEEP
    )
    assert post_scores == scores, f"score changed after offload cycle: {post_scores}"
    post_pair = pack.NODE_CLASS_MAPPINGS["FGResQPairScore"]().score(
        pair_bundle, image, image2, memory_management=common.MEMORY_KEEP
    )
    assert max(abs(a - b) for a, b in zip(post_pair[0], out[0])) < 1e-6, (
        post_pair[0], out[0]
    )
    # the bundle the variant switch evicted revives through its own recipe too
    (revived,) = pack.NODE_CLASS_MAPPINGS["FGResQScore"]().score(
        bundle, image, memory_management=common.MEMORY_KEEP
    )
    assert revived == scores, (revived, scores)
    print(f"post-unload NR scores={[round(s, 4) for s in post_scores]} (unchanged: "
          "both gutted wrappers were revived from their cached recipes)")

    print("\n-- Classic FR (Load pyiqa Model + pyiqa FR Score) --")
    # The nine classic pixel metrics used to have their own weightless loader.
    # They are one architecture each in the catalog, so the generic pair reaches
    # them: architecture "SSIM" / "MAD" / ..., variant tagged FR. Same metrics,
    # same asserted values as the old dedicated-loader section.
    for metric, expect in [
        ("ssim", 1.0), ("ms_ssim", 1.0), ("fsim", 1.0), ("mad", 0.0),
        ("gmsd", 0.0), ("vif", 1.0), ("vsi", 1.0), ("cw_ssim", 1.0),
    ]:
        m, _arch, value = _load_generic(pack, metric_catalog, metric, "FR")
        assert "FR" in value, value  # the variant really is tagged FR
        (s,) = pack.NODE_CLASS_MAPPINGS["PyiqaAnyFRScore"]().score(m, image, image)
        assert abs(s[0] - expect) < 1e-3, f"{metric}(a,a) = {s[0]}, expected ~{expect}"
        (s2,) = pack.NODE_CLASS_MAPPINGS["PyiqaAnyFRScore"]().score(m, image, image2)
        print(f"{metric:8s} identity={s[0]:.4f} noise-pair={s2[0]:.4f}")
    m, _arch, _value = _load_generic(pack, metric_catalog, "psnr", "FR")
    (p,) = pack.NODE_CLASS_MAPPINGS["PyiqaAnyFRScore"]().score(m, image, image2)
    print(f"psnr     noise-pair={p[0]:.2f} dB")

    print("\n-- L1: the node is gone, the metric is still reachable --")
    # L1Score (nodes/l1_nodes.py) was removed from the surface; pyiqa's own
    # mean-absolute-difference metric is not: it is architecture "MAD (L1)",
    # variant "mad", reached through the generic loader + pyiqa FR Score. Same
    # assertion the node carried: MAD(a, a) == 0.
    assert "L1Score" not in pack.NODE_CLASS_MAPPINGS, "L1Score is not a node any more"
    assert "L1Score" not in pack.NODE_DISPLAY_NAME_MAPPINGS, (
        "L1Score still has a display name"
    )
    assert "L1Score" not in common.SCORE_NODES.values(), common.SCORE_NODES
    assert "mad" in metric_catalog.ARCHITECTURES["MAD (L1)"].variants
    l1_model, l1_arch, l1_value = _load_generic(pack, metric_catalog, "mad", "FR")
    assert l1_arch == "MAD (L1)", (l1_arch, l1_value)
    (l_self,) = pack.NODE_CLASS_MAPPINGS["PyiqaAnyFRScore"]().score(
        l1_model, image, image
    )
    (l_noise,) = pack.NODE_CLASS_MAPPINGS["PyiqaAnyFRScore"]().score(
        l1_model, image, image2
    )
    assert abs(l_self[0]) < 1e-4, f"mad identity = {l_self[0]}, expected ~0"
    print(f"mad      [{l1_arch}] identity={l_self[0]:.4f} "
          f"noise-pair={l_noise[0]:.4f}")

    print("\n-- TOPIQ (Load pyiqa Model + the generic FR/NR score nodes) --")
    # TOPIQLoader/TOPIQFRScore/TOPIQNRScore are gone: the head (FR / NR /
    # aesthetic) is the variant, so the same three metrics come through the
    # generic pair. Checked before the NR load evicts the FR one (the generic
    # loader owns ONE cache slot, "pyiqa_any").
    tf, _tf_arch, tf_variant = _load_generic(pack, metric_catalog, "topiq_fr", "FR")
    assert tf.family == "topiq", tf.family
    (tfr,) = pack.NODE_CLASS_MAPPINGS["PyiqaAnyFRScore"]().score(tf, image, image2)
    msg = _expect_value_error(
        lambda: pack.NODE_CLASS_MAPPINGS["PyiqaAnyNRScore"]().score(tf, image),
        "FR-reference model",
        "NR-reference models only",
        "pyiqa FR Score",
    )
    print(f"mode guard ok: {msg[:70]}...")
    tn, _tn_arch, tn_variant = _load_generic(pack, metric_catalog, "topiq_nr", "NR")
    (tnr,) = pack.NODE_CLASS_MAPPINGS["PyiqaAnyNRScore"]().score(tn, image)
    print(f"topiq_fr noise-pair={tfr[0]:.4f}  topiq_nr={tnr[0]:.4f}")

    print("\n-- A-FINE (Load pyiqa Model + the generic FR/NR score nodes) --")
    # no resolution input any more: A-FINE's requirement (sides divisible by 32)
    # and its batch-1 assert are properties of the model, applied automatically
    afine = metric_catalog.resolve("A-FINE", "afine_fr")
    padded = utils.prepare_image(
        torch.zeros(1, 3, 250, 333), afine.policy
    )
    assert tuple(padded.shape[2:]) == (256, 352), tuple(padded.shape)
    print(f"policy mult32: 250x333 -> {tuple(padded.shape[2:])}")
    af, _af_arch, _af_variant = _load_generic(pack, metric_catalog, "afine_fr", "FR")
    (afr,) = pack.NODE_CLASS_MAPPINGS["PyiqaAnyFRScore"]().score(af, image, image2)
    an, _an_arch, _an_variant = _load_generic(pack, metric_catalog, "afine_nr", "NR")
    (anr,) = pack.NODE_CLASS_MAPPINGS["PyiqaAnyNRScore"]().score(an, image)
    print(f"afine_fr noise-pair={afr[0]:.4f}  afine_nr={anr[0]:.4f}")
    odd = fake_image(1, 250, 333, seed=3)
    (aodd,) = pack.NODE_CLASS_MAPPINGS["PyiqaAnyNRScore"]().score(an, odd)
    # the same image pre-padded to 256x352 must score the same - that is the
    # node adapting a 250x333 input to A-FINE's multiple-of-32 requirement (the
    # two sides interpolate on possibly different devices, hence the float
    # tolerance rather than bit-equality)
    pre = utils.prepare_image(
        odd.permute(0, 3, 1, 2).float(), afine.policy
    ).permute(0, 2, 3, 1).contiguous()
    (a_pre,) = pack.NODE_CLASS_MAPPINGS["PyiqaAnyNRScore"]().score(an, pre)
    assert abs(aodd[0] - a_pre[0]) < 1e-4, (aodd, a_pre)
    print(f"afine_nr 250x333 (auto -> 256x352)={aodd[0]:.4f} "
          f"== explicit 256x352 ({a_pre[0]:.4f})")

    print("\n-- LPIPS family (Load pyiqa Model + pyiqa FR Score) --")
    # the six weights are six variants of two architectures now: LPIPS (4) and
    # ST-LPIPS (2), still identity ~0
    for metric, arch_label in [
        ("lpips", "LPIPS"),
        ("lpips-vgg", "LPIPS"),
        ("lpips+", "LPIPS"),
        ("lpips-vgg+", "LPIPS"),
        ("stlpips", "ST-LPIPS"),
        ("stlpips-vgg", "ST-LPIPS"),
    ]:
        pm, pm_arch, _value = _load_generic(pack, metric_catalog, metric, "FR")
        assert pm_arch == arch_label, (metric, pm_arch)
        (ident,) = pack.NODE_CLASS_MAPPINGS["PyiqaAnyFRScore"]().score(pm, image, image)
        (noise,) = pack.NODE_CLASS_MAPPINGS["PyiqaAnyFRScore"]().score(pm, image, image2)
        assert abs(ident[0]) < 1e-4, f"{metric} identity = {ident[0]}, expected ~0"
        print(f"{metric:14s} [{pm_arch:8s}] identity={ident[0]:.5f} "
              f"noise-pair={noise[0]:.4f}")
    # the pre-redesign kwarg still resolves the ARCHITECTURE (LPIPSLoader's
    # "metric" combo name); the shipped names are architecture + variant
    _pm_kw, kw_name, _kw_value = pack.NODE_CLASS_MAPPINGS["PyiqaAnyLoader"]().load(
        metric="LPIPS", variant=_variant_value(metric_catalog, "lpips-vgg", "FR")
    )
    assert kw_name == "lpips-vgg", kw_name
    print("PyiqaAnyLoader(metric=...) still resolves the architecture ->", kw_name)

    print("\n-- DISTS family (Load pyiqa Model + pyiqa FR Score) --")
    for metric in ["dists", "msswd"]:
        dm, dm_arch, _value = _load_generic(pack, metric_catalog, metric, "FR")
        (ident,) = pack.NODE_CLASS_MAPPINGS["PyiqaAnyFRScore"]().score(dm, image, image)
        (noise,) = pack.NODE_CLASS_MAPPINGS["PyiqaAnyFRScore"]().score(dm, image, image2)
        assert abs(ident[0]) < 1e-4, f"{metric} identity = {ident[0]}, expected ~0"
        print(f"{metric:14s} [{dm_arch:19s}] identity={ident[0]:.5f} "
              f"noise-pair={noise[0]:.4f}")

    print("\n-- LAION Aesthetic (Load pyiqa Model + pyiqa NR Score) --")
    lm, _lm_arch, _lm_value = _load_generic(pack, metric_catalog, "laion_aes", "NR")
    (ls,) = pack.NODE_CLASS_MAPPINGS["PyiqaAnyNRScore"]().score(lm, image)
    assert 0 < ls[0] <= 12, f"laion_aes score {ls[0]} outside ~1-10 band"
    print(f"laion_aes image={ls[0]:.3f}")

    print("\n-- WaDIQaM (Load pyiqa Model + the generic FR/NR score nodes) --")
    wf, _wf_arch, _wf_value = _load_generic(pack, metric_catalog, "wadiqam_fr", "FR")
    (wfr,) = pack.NODE_CLASS_MAPPINGS["PyiqaAnyFRScore"]().score(wf, image, image2)
    wn, _wn_arch, _wn_value = _load_generic(pack, metric_catalog, "wadiqam_nr", "NR")
    (wnr,) = pack.NODE_CLASS_MAPPINGS["PyiqaAnyNRScore"]().score(wn, image)
    print(f"wadiqam_fr noise-pair={wfr[0]:.4f}  wadiqam_nr={wnr[0]:.4f}")

    print("\n-- MUSIQ / CLIP-IQA / QualiCLIP (NR, via Load pyiqa Model) --")
    nr_scores = {}
    for metric in ("musiq", "clipiqa", "qualiclip"):
        nm, _nm_arch, _nm_value = _load_generic(pack, metric_catalog, metric, "NR")
        (nrs,) = pack.NODE_CLASS_MAPPINGS["PyiqaAnyNRScore"]().score(nm, image)
        nr_scores[metric] = nrs[0]
    print("  ".join(f"{k}={v:.4f}" for k, v in nr_scores.items()))

    print("\n-- Generic loader ((Down)load pyiqa Model) --")
    any_req = pack.NODE_CLASS_MAPPINGS["PyiqaAnyLoader"].INPUT_TYPES()["required"]
    assert set(any_req) == {"architecture", "variant"}, (
        f"the generic loader is architecture+variant now, got {sorted(any_req)}"
    )
    arch_opts = any_req["architecture"][0]
    variant_opts = any_req["variant"][0]
    assert arch_opts == metric_catalog.ARCHES, "architecture combo = ARCHES"
    assert variant_opts == metric_catalog.VARIANT_VALUES, "variant combo = VARIANT_VALUES"
    assert len(arch_opts) == len(set(arch_opts)) >= 50, "arch combo: 53 unique"
    assert len(variant_opts) == len(set(variant_opts)) >= 100, "variant combo"
    # nothing pyiqa ships is hidden any more: the five metrics that cannot be
    # scored per image ARE in the combo (under their own architectures) and the
    # score nodes refuse them with a pointer - asserted above, re-checked here
    # against the live combo the node itself reports
    for base in metric_catalog.SPECIAL_INPUTS:
        value = _value_for_base(metric_catalog, base)
        assert value in variant_opts, (base, value)
    # every variant value prefixes with a real arch (the JS filter key)
    prefixes = {v.split(metric_catalog.SEPARATOR, 1)[0] for v in variant_opts}
    assert metric_catalog.SEPARATOR == " | "
    assert prefixes <= set(arch_opts), "variant prefixes must be arch labels"
    assert "LPIPS" in arch_opts and "TOPIQ (CFANet)" in arch_opts
    print(f"generic combo: {len(arch_opts)} architectures x "
          f"{len(variant_opts)} variants (fid/sfid/fid_dinov2/"
          f"inception_score/clipscore listed, refused per image)")
    ssim_variant = next(v for v in variant_opts if v.startswith("SSIM | "))
    gm, gm_name, _gm_label = pack.NODE_CLASS_MAPPINGS["PyiqaAnyLoader"]().load(
        "SSIM", ssim_variant
    )
    assert gm_name == "ssim", gm_name
    (ggs,) = pack.NODE_CLASS_MAPPINGS["PyiqaAnyFRScore"]().score(gm, image, image)
    assert abs(ggs[0] - 1.0) < 1e-3, f"generic ssim(a,a) = {ggs[0]}, expected ~1"
    # bare arch (no variant) resolves to its first variant
    gn, gn_name, _gn_label = pack.NODE_CLASS_MAPPINGS["PyiqaAnyLoader"]().load("BRISQUE")
    assert gn_name == "brisque", gn_name
    # kept loaded so the run-time guard section below reuses it for free
    (gnr,) = pack.NODE_CLASS_MAPPINGS["PyiqaAnyNRScore"]().score(
        gn, image, memory_management=common.MEMORY_KEEP
    )
    print(f"generic ssim identity={ggs[0]:.4f}  brisque={gnr[0]:.4f}")
    _expect_value_error(
        lambda: pack.NODE_CLASS_MAPPINGS["PyiqaAnyLoader"]().load("LPIPS", ssim_variant),
        "belongs to",
    )
    print("cross-arch guard ok: a variant prefixed with another arch is refused")
    _expect_value_error(
        lambda: pack.NODE_CLASS_MAPPINGS["PyiqaAnyNRScore"]().score(gm, image),
        "FR-reference model",
    )
    print("generic mode guard ok: pyiqa NR Score refuses an FR metric")

    print("\n-- run-time family/mode guards (cheap metrics only) --")
    # Every score node re-checks the model it was handed: the union socket also
    # lets the generic loader in, so the check that produces the message you
    # actually read lives at run time. Both bundles below are cache hits (the
    # generic section above loaded them), and each is used BEFORE the other
    # same-family load evicts it. Every hint must name a node that still exists.
    nr_bundle = pack.NODE_CLASS_MAPPINGS["PyiqaAnyLoader"]().load("BRISQUE")[0]
    assert nr_bundle.mode == "NR" and nr_bundle.family != "fgresq", (
        nr_bundle.mode, nr_bundle.family
    )
    msg = _expect_value_error(
        lambda: pack.NODE_CLASS_MAPPINGS["FGResQScore"]().score(nr_bundle, image),
        "FGResQ NR Score",
        "FGResQ models only",
        "brisque",
        "pyiqa NR Score",
    )
    print(f"FGResQ NR Score + foreign NR model -> ValueError names the node that "
          f"works: {msg[-70:]}")
    msg = _expect_value_error(
        lambda: pack.NODE_CLASS_MAPPINGS["PyiqaAnyFRScore"]().score(
            nr_bundle, image, image
        ),
        "NR-reference model",
        "FR-reference models only",
        "pyiqa NR Score",
    )
    print(f"pyiqa FR Score + NR model -> ValueError: {msg[:70]}...")
    ssim_bundle = pack.NODE_CLASS_MAPPINGS["PyiqaAnyLoader"]().load(
        "SSIM", ssim_variant
    )[0]
    assert ssim_bundle.mode == "FR" and ssim_bundle.family == "classic_fr", (
        ssim_bundle.mode, ssim_bundle.family
    )
    # FGResQ NR Score takes the FR/NR comparison out of the guard
    # (METRIC_FROM_BUNDLE: the shared FGResQ net serves both modes), so a model
    # of another family is refused by the FAMILY check - still by name, and the
    # hint follows the MODEL's mode (an FR metric points at pyiqa FR Score, not
    # at the NR node that certainly cannot score it)
    msg = _expect_value_error(
        lambda: pack.NODE_CLASS_MAPPINGS["FGResQScore"]().score(ssim_bundle, image),
        "FGResQ models only",
        "pyiqa FR Score",
    )
    print(f"FGResQ NR Score + FR model -> ValueError names the node that works: "
          f"{msg[:70]}...")
    # a multi-output pair metric is refused by name by the generic FR node
    pair_resolved = metric_catalog.Resolved(
        arch="FGResQ",
        base="fgresq_pair",
        label="FR pair (rank + verdict)",
        mode="FR",
        lower_better=False,
        score_range="quality 0-1 · rank 0-2",
        multi_output=True,
    )
    try:
        pack.NODE_CLASS_MAPPINGS["PyiqaAnyFRScore"]().score(
            common.MetricBundle(pair_resolved, object()), image, image2
        )
        raise AssertionError("pyiqa FR Score accepted a 4-output pair metric")
    except ValueError as exc:
        msg = str(exc)
        assert "fgresq_pair" in msg, repr(msg)
        # the message must name the node that would work, in that node's own
        # spelling (metric_catalog's SPECIAL_METRICS says "use FGResQ FR
        # Score"), so the assertion above is case-sensitive on purpose.
        assert "FGResQ FR Score" in msg, repr(msg)
    print("pyiqa FR Score + fgresq_pair -> ValueError points at FGResQ FR Score")
    # ...while the FGResQ family still offers the flagged pair entry, and only
    # through its own FR node (FAMILY_LOOKUP is gone: the architecture's
    # variants plus VARIANT_LOOKUP are the catalog's tables now)
    fg_variants = metric_catalog.ARCHITECTURES["FGResQ"].variants
    assert "fgresq_pair" in fg_variants, fg_variants
    fg_pair = {
        resolved.base
        for resolved in metric_catalog.VARIANT_LOOKUP.values()
        if resolved.multi_output and resolved.arch == "FGResQ"
    }
    assert fg_pair == {"fgresq_pair"}, fg_pair
    pair_value = next(
        value
        for value, resolved in metric_catalog.VARIANT_LOOKUP.items()
        if resolved.base == "fgresq_pair"
    )
    assert "FGResQ FR Score" in pair_value, pair_value
    print(f"FGResQ combo keeps the flagged pair variant reachable: "
          f"{pair_value[-58:]}")

    print("\n-- FID (image-batch, multi-wire; weights from the generic loader) --")
    # FID/sFID/fid_dinov2 are set-level, folder-based metrics: they cannot score
    # an IMAGE batch, so the generic score nodes refuse them (asserted in the
    # special-input section above) and this node keeps the folder path. The
    # WEIGHTS come from "(Down)load pyiqa Model" - architecture "FID" - exactly
    # like every other metric; the old FIDLoader's own two-entry combo is gone.
    fm, fid_name, fid_label = pack.NODE_CLASS_MAPPINGS["PyiqaAnyLoader"]().load(
        "FID", _variant_value(metric_catalog, "fid", "NR")
    )
    assert fid_name == "fid", fid_name
    assert "use FID Score" in fid_label, fid_label
    assert isinstance(fm, common.MetricBundle), type(fm).__name__
    assert isinstance(fm.metric, torch.nn.Module), type(fm.metric).__name__
    batch = fake_image(8, 256, 256, seed=11)
    # INPUT_IS_LIST: ComfyUI wraps EVERY input (model and widgets included)
    # in a list - call the way the executor does (regression: a per-input
    # INPUT_IS_LIST tuple used to make model arrive as a list and crash)
    fnode = pack.NODE_CLASS_MAPPINGS["FIDScore"]()
    (f_same,) = fnode.score(
        model=[fm], target=[batch], reference=[batch],
        resize_mode=["clean"], memory_management=["keep loaded"],
    )
    # two separate wires on the same side must concatenate, not overwrite
    (f_split,) = fnode.score(
        model=[fm], target=[batch[:4], batch[4:]], reference=[batch],
        resize_mode=["clean"], memory_management=["keep loaded"],
    )
    (f_diff,) = fnode.score(
        model=[fm], target=[batch], reference=[fake_image(8, 256, 256, seed=12)],
        resize_mode=["clean"], memory_management=["keep loaded"],
    )
    # plain positional calls (no list wrapping) must keep working too
    # ("keep loaded" so the metric survives the guard check just below)
    (f_plain,) = pack.NODE_CLASS_MAPPINGS["FIDScore"]().score(
        fm, [batch], [batch], "clean", memory_management="keep loaded"
    )
    assert f_plain == f_same, (f_plain, f_same)
    # Analytically FID(a,a)=0; pyiqa's frechet_distance regularises the sqrt at
    # eps=1e-6, so a tiny negative value is normal.
    assert abs(f_same) < 1e-3, f"FID(a,a) = {f_same}, expected ~0"
    assert abs(f_split - f_same) < 1e-6, (
        f"split-wire FID {f_split} != single-batch FID {f_same}"
    )
    assert f_diff > 0, f"FID of different sets should be > 0, got {f_diff}"
    print(f"fid identity(a,a)={f_same:.6f}  split-wire={f_split:.6f}  "
          f"different-sets={f_diff:.4f}")
    # before the sfid load below evicts this family's fid metric: 1 image per
    # side must be refused by the guard, not crash inside pyiqa's sqrtm. The
    # message has to say WHY (a distribution needs a covariance) and what to use
    # for a single pair instead - that is the question this node gets asked.
    msg = _expect_value_error(
        lambda: pack.NODE_CLASS_MAPPINGS["FIDScore"]().score(
            fm, [batch[:1]], [batch], "clean", memory_management="keep loaded"
        ),
        "1 image(s)",
        "DISTRIBUTIONS",
        "no covariance",
        "LPIPS",
        "pyiqa FR Score",
    )
    print(f"min-images guard: 1 image per side refused -> {msg[:70]}...")
    # ...and a set that is legal but too thin to mean much is scored WITH a
    # warning (the user is in charge, but the number is dominated by set size),
    # while a set at the threshold is silent and unchanged
    fid_logger = logging.getLogger("pyiqa_toolbox_smoke.nodes.fid_nodes")

    class _FidCapture(logging.Handler):
        """Collects this node's WARNING+ lines into ``self.sink``."""

        def __init__(self, sink):
            super().__init__()
            self.sink = sink

        def emit(self, record):
            self.sink.append(record.getMessage())

    def _fid_score_with_logs(target, reference):
        msgs = []
        handler = _FidCapture(msgs)
        level = fid_logger.level
        fid_logger.setLevel(logging.WARNING)
        fid_logger.addHandler(handler)
        try:
            out = pack.NODE_CLASS_MAPPINGS["FIDScore"]().score(
                fm, target, reference, "clean", memory_management="keep loaded"
            )
        finally:
            fid_logger.removeHandler(handler)
            fid_logger.setLevel(level)
        return out[0], msgs

    f_thin, thin_msgs = _fid_score_with_logs([batch[:4]], [batch[:4]])
    match = next((m for m in thin_msgs if "rank n-1" in m), "")
    assert "target=4" in match and "reference=4" in match, thin_msgs
    assert f_thin == f_thin, "a thin set must still produce a number, not nan"
    print(f"thin-set warning at 4 images/side (still scores {f_thin:.4f}): "
          f"{match[:60]}...")
    # 8 images per side is still below the ~16 guidance, so it warns too; what
    # matters is that the score itself is untouched by the warning
    f_quiet, quiet_msgs = _fid_score_with_logs([batch], [batch])
    assert any("rank n-1" in m for m in quiet_msgs), quiet_msgs
    assert f_quiet == f_same, (f_quiet, f_same)
    # the far side of the threshold: at the documented ~16 per side it goes quiet
    big = fake_image(16, 256, 256, seed=13)
    f_big, big_msgs = _fid_score_with_logs([big], [big])
    assert not big_msgs, big_msgs
    assert f_big == f_big, "16 images per side must score, not nan"
    print(f"16 images/side: no warning, scores {f_big:.4f} (threshold verified "
          "from both sides)")
    # the second FID architecture entry is a real metric, not just a label
    sf, sf_name, sf_label = pack.NODE_CLASS_MAPPINGS["PyiqaAnyLoader"]().load(
        "FID", _variant_value(metric_catalog, "sfid", "NR")
    )
    assert sf_name == "sfid", sf_name
    assert "use FID Score" in sf_label, sf_label
    (f_sfid_same,) = fnode.score(
        model=[sf], target=[batch], reference=[batch],
        resize_mode=["clean"], memory_management=["keep loaded"],
    )
    (f_sfid_diff,) = fnode.score(
        model=[sf], target=[batch], reference=[fake_image(8, 256, 256, seed=13)],
        resize_mode=["clean"], memory_management=["keep loaded"],
    )
    assert abs(f_sfid_same) < 1e-3, f"sFID(a,a) = {f_sfid_same}, expected ~0"
    assert f_sfid_diff > 0, f"sFID of different sets should be > 0, got {f_sfid_diff}"
    print(f"sfid identity(a,a)={f_sfid_same:.6f}  different-sets={f_sfid_diff:.4f}")

    print("\n-- node naming consistency --")
    names = pack.NODE_DISPLAY_NAME_MAPPINGS
    categories = {
        node_id: cls.CATEGORY for node_id, cls in pack.NODE_CLASS_MAPPINGS.items()
    }
    bad_cat = {k: v for k, v in categories.items() if not v.startswith("pyiqa/")}
    assert not bad_cat, f"nodes outside pyiqa/<family> categories: {bad_cat}"
    bad_name = {
        k: v
        for k, v in names.items()
        if not (v.startswith("(Down)load ") or v.endswith("Score"))
        and k != "UnloadPyiqaModels"
    }
    assert not bad_name, (
        f"display names must be '(Down)load X' / 'X Score': {bad_name}"
    )
    # the generic loader is a "(Down)load X" like every other loader now: there
    # is no bare "Load pyiqa Model" exception left
    loaders = {k: v for k, v in names.items() if k.endswith("Loader")}
    assert all(v.startswith("(Down)load ") for v in loaders.values()), (
        f"every loader follows the (Down)load convention: {loaders}"
    )
    assert names["PyiqaAnyLoader"] == "(Down)load pyiqa Model", names["PyiqaAnyLoader"]
    for node_id, display in sorted(names.items()):
        print(f"  {categories[node_id]:22s} {display}")
    # the TOPIQ heads are variants of ONE architecture now (the dedicated
    # TOPIQLoader is gone), so the face and aesthetic heads are checked by the
    # registry name they resolve to (the labels are the human axes, not the name)
    topiq_bases = set(metric_catalog.ARCHITECTURES["TOPIQ (CFANet)"].variants)
    for wanted in ("topiq_nr-face", "topiq_nr_swin-face", "topiq_iaa", "topiq_iaa_res50"):
        assert wanted in topiq_bases, (
            f"{wanted} must stay reachable through Load pyiqa Model's TOPIQ "
            f"variants; TOPIQ variants: {sorted(topiq_bases)}"
        )
    print("the generic TOPIQ combo still offers the face + aesthetic heads:",
          sorted(b for b in topiq_bases if "face" in b or "iaa" in b))

    print("\n-- loader / score-node contract --")
    from pyiqa_toolbox_smoke.nodes._common import ModelLoaderBase

    # every loader returns (model, model_name, variant_label) - the redesign's
    # single biggest API change. Classic FR's model-only loader is gone with the
    # rest of the cull, and FID/FGResQ lost their own loaders too: FUNCTION ==
    # "load" is exactly the generic loader plus Q-ReAlign (the one family whose
    # combos - dtype, snapshot models - the generic pair cannot express).
    load_nodes = {
        node_id: cls
        for node_id, cls in sorted(pack.NODE_CLASS_MAPPINGS.items())
        if cls.FUNCTION == "load"
    }
    assert set(load_nodes) == {"PyiqaAnyLoader", "QReAlignLoader"}, sorted(load_nodes)
    assert issubclass(load_nodes["PyiqaAnyLoader"], ModelLoaderBase), (
        "the generic loader is the ModelLoaderBase subclass now"
    )
    for node_id, cls in sorted(load_nodes.items()):
        assert cls.RETURN_NAMES == ("model", "model_name", "variant_label"), (
            f"{node_id}: {cls.RETURN_NAMES}"
        )
        assert len(cls.RETURN_TYPES) == 3, f"{node_id}: {cls.RETURN_TYPES}"
        assert cls.RETURN_TYPES[1] == cls.RETURN_TYPES[2] == "STRING", cls.RETURN_TYPES
    print(f"all {len(load_nodes)} loaders return (model, model_name, variant_label)")

    # the shipped input names (saved workflows keep resolving): the generic
    # loader's two-combo hierarchy plus Q-ReAlign's model_name+dtype
    # (custom_checkpoint / custom_base_metric are gone)
    combos = {
        node_id: sorted(cls.INPUT_TYPES()["required"])
        for node_id, cls in sorted(load_nodes.items())
    }
    assert combos == {
        "PyiqaAnyLoader": ["architecture", "variant"],
        "QReAlignLoader": ["dtype", "model_name"],
    }, combos
    for node_id, cls in load_nodes.items():
        for gone in ("custom_checkpoint", "custom_base_metric", "resolution"):
            assert gone not in cls.INPUT_TYPES()["required"], (node_id, gone)
    print("shipped loader combos:", combos)

    # every family still resolves to a variant - the FID family included now
    # (its three metrics are listed under architecture "FID" and scored by FID
    # Score, so no family is left without a variant to point at)
    empty_families = {
        family
        for family in metric_catalog.FAMILIES
        if not any(
            metric_catalog.ARCHITECTURES[label].variants
            for label in metric_catalog.family_arches(family)
        )
    }
    assert not empty_families, (
        f"catalog families with an EMPTY variant list: {sorted(empty_families)}"
    )
    # the generic combo's default is a real dropdown member that resolves...
    assert metric_catalog.DEFAULT_ARCH in metric_catalog.ARCHITECTURES
    assert metric_catalog.DEFAULT_VARIANT in metric_catalog.VARIANT_VALUES
    default = metric_catalog.resolve(
        metric_catalog.DEFAULT_ARCH, metric_catalog.DEFAULT_VARIANT
    )
    assert default.arch == metric_catalog.DEFAULT_ARCH, default
    # ...and every family+mode a score node owns has a variant to point at. The
    # candidates include a signpost architecture's own registry variants: its
    # dropdown entry is a message, but its dedicated loader still loads them.
    for family, mode in common.SCORE_NODES:
        if family == "any":
            continue
        arches = metric_catalog.family_arches(family)
        assert arches, (family, mode)
        assert any(
            resolved.mode == mode and resolved.arch in arches
            for resolved in candidates
        ), f"no {mode} variant for family {family}"
    assert metric_catalog.family_of("LPIPS") == metric_catalog.family_of("ST-LPIPS")
    assert metric_catalog.family_of("LPIPS") == "lpips"
    assert metric_catalog.family_of("FGResQ") == "fgresq"
    print("catalog families: every family's score node has a variant (the FID "
          "family's are listed under architecture 'FID'), the generic default "
          "resolves, LPIPS/ST-LPIPS share one family")

    # the catalog must agree with the registry it is generated from: every
    # variant names a real pyiqa metric, and its mode/direction match
    # DEFAULT_CONFIGS (a hand-edited label can never silently drift)
    import pyiqa
    from pyiqa.default_model_configs import DEFAULT_CONFIGS

    registered = set(pyiqa.list_models())
    checked_variants = 0
    for label, resolved in metric_catalog.VARIANT_LOOKUP.items():
        if resolved.kind == "signpost":
            # a signpost entry is the dropdown's message, not a weights choice:
            # the mode/direction/range fields are deliberately unknown, and the
            # architecture's own variants (checked by the matrix above) are what
            # its dedicated loader offers
            assert resolved.mode == "?", (label, resolved.mode)
            continue
        assert resolved.base in registered, (label, resolved.base)
        cfg = DEFAULT_CONFIGS.get(resolved.base)
        if cfg is None:
            continue  # a custom checkpoint loaded onto an unlisted base
        assert resolved.mode == cfg.get("metric_mode"), (label, resolved.mode)
        assert resolved.lower_better == bool(cfg.get("lower_better")), label
        assert resolved.base in metric_catalog.ARCHITECTURES[resolved.arch].variants, (
            label,
            resolved.base,
        )
        checked_variants += 1
    print(f"catalog: all {checked_variants} weights-choice variants name real "
          f"pyiqa metrics with matching mode/direction "
          f"({len(metric_catalog.VARIANT_LOOKUP) - checked_variants} message/custom "
          "entries excluded)")

    # model sockets: the generic loader's PYIQA_ANY joins the family's own type.
    # FID Score and CLIP Score take the generic socket itself now (their weights
    # come from the generic loader; the node is what makes the model usable), so
    # the four ANY nodes are exactly the generic pair + FID + CLIP.
    any_socket = ("PyiqaAnyFRScore", "PyiqaAnyNRScore", "FIDScore", "CLIPScore")
    for node_id, cls in sorted(pack.NODE_CLASS_MAPPINGS.items()):
        model_input = cls.INPUT_TYPES().get("required", {}).get("model")
        if model_input is None:
            continue
        model_type = model_input[0]
        if node_id in any_socket:
            assert model_type == "PYIQA_ANY", (node_id, model_type)
        else:
            own, _sep, generic = model_type.partition(",")
            assert generic == "PYIQA_ANY", (node_id, model_type)
            assert own.startswith("PYIQA_"), (node_id, model_type)
    # ...and a dedicated loader of another family is still NOT accepted, even
    # though every family socket also accepts the generic loader
    fg_accepts = pack.NODE_CLASS_MAPPINGS["FGResQScore"].INPUT_TYPES()["required"][
        "model"
    ][0].split(",")
    assert "PYIQA_FGRESQ" in fg_accepts and "PYIQA_ANY" in fg_accepts, fg_accepts
    assert "PYIQA_LPIPS" not in fg_accepts, fg_accepts
    assert "PYIQA_TOPIQ" not in fg_accepts, fg_accepts
    print("model sockets: family score nodes accept '<family>,PYIQA_ANY', "
          "PyiqaAny FR/NR + FID Score + CLIP Score PYIQA_ANY "
          "(FGResQScore rejects an LPIPS/TOPIQ loader socket)")

    # no node exposes a resolution widget any more: adaptation is per-model
    for node_id, cls in sorted(pack.NODE_CLASS_MAPPINGS.items()):
        for group in ("required", "optional"):
            inputs = cls.INPUT_TYPES().get(group, {})
            assert "resolution" not in inputs, (
                f"{node_id} still exposes a 'resolution' input"
            )
    print("no node exposes a 'resolution' input (input adaptation is automatic)")

    # prepare_pair must give both sides the target's geometry (pixel-wise FR)
    a = torch.zeros(1, 3, 64, 64)
    b = torch.zeros(1, 3, 128, 96)
    x, y = utils.prepare_pair(a, b)
    assert x.shape == a.shape and y.shape == x.shape, (x.shape, y.shape)
    x32, y32 = utils.prepare_pair(
        torch.zeros(1, 3, 30, 30), torch.zeros(1, 3, 31, 31), utils.POLICY_MULT32
    )
    assert tuple(x32.shape[2:]) == (32, 32) and y32.shape == x32.shape, (
        x32.shape, y32.shape
    )
    print("prepare_pair: reference resized to the target's geometry "
          "(and both padded to mult32 under the A-FINE policy)")

    print("\n-- output tooltips + FR input names --")
    # unconditional: the 3-tuple loaders document all three outputs too
    for node_id, cls in sorted(pack.NODE_CLASS_MAPPINGS.items()):
        tips = getattr(cls, "OUTPUT_TOOLTIPS", None)
        assert tips is not None, f"{node_id} has no OUTPUT_TOOLTIPS"
        assert len(tips) == len(cls.RETURN_TYPES), (
            f"{node_id}: {len(tips)} output tooltip(s) for "
            f"{len(cls.RETURN_TYPES)} output(s)"
        )
        assert all(isinstance(t, str) and t.strip() for t in tips), (
            f"{node_id} has an empty output tooltip"
        )
    print(f"all {len(pack.NODE_CLASS_MAPPINGS)} nodes: OUTPUT_TOOLTIPS == outputs, "
          f"non-empty")

    for node_id in (
        "FGResQPairScore",
        "PyiqaAnyFRScore",
        "PyiqaAnyNRScore",
        "QReAlignScore",
        "FIDScore",
    ):
        inputs = pack.NODE_CLASS_MAPPINGS[node_id].INPUT_TYPES()["required"]
        print(f"  {node_id:16s} inputs: {sorted(inputs)}  outputs:")
        cls = pack.NODE_CLASS_MAPPINGS[node_id]
        for name, tip in zip(cls.RETURN_NAMES, cls.OUTPUT_TOOLTIPS):
            print(f"      {name:16s} {tip[:70]}...")
    fg_names = pack.NODE_CLASS_MAPPINGS["FGResQPairScore"].RETURN_NAMES
    assert fg_names[:2] == ("target_score", "reference_score"), fg_names

    print("\n-- lazy evaluation wiring --")
    from pyiqa_toolbox_smoke.nodes._common import (
        MEMORY_DROP,
        MEMORY_KEEP,
        LazyScoreMixin,
        with_memory_management,
    )

    score_nodes = 0
    for node_id, cls in sorted(pack.NODE_CLASS_MAPPINGS.items()):
        if not issubclass(cls, LazyScoreMixin):
            continue  # loaders and utility nodes have no lazy score inputs
        score_nodes += 1
        inputs = cls.INPUT_TYPES()["required"]
        assert hasattr(cls, "check_lazy_status"), f"{node_id} has no check_lazy_status"
        # every LAZY_INPUTS name must be marked lazy in INPUT_TYPES
        for name in cls.LAZY_INPUTS:
            assert name in inputs, f"{node_id}: lazy input {name} missing"
            assert inputs[name][1].get("lazy") is True, (
                f"{node_id}: input {name} not marked lazy"
            )
        # no non-standard 'enabled' toggle
        assert "enabled" not in inputs, f"{node_id} still exposes 'enabled'"
        # memory_management on every node that owns a metric: exactly two modes
        # (unload / keep), unload as the default
        if "model" in inputs:
            mm = inputs.get("memory_management")
            assert mm is not None, f"{node_id} has no memory_management input"
            assert mm[1].get("default") == MEMORY_DROP, node_id
            assert mm[0][0] == MEMORY_DROP, node_id
            assert set(mm[0]) == {MEMORY_DROP, MEMORY_KEEP}, node_id
    print(f"all {score_nodes} score nodes: check_lazy_status + lazy inputs + "
          f"memory_management (no 'enabled' toggle)")

    # sequencing: one name per call, images before model, [] when all present.
    # CLIP Score replaces the removed L1Score here: one image input, one model
    # input, same two rules.
    probe = pack.NODE_CLASS_MAPPINGS["FGResQPairScore"]()
    assert probe.check_lazy_status(model=None, target=None, reference=None) == ["target"]
    assert probe.check_lazy_status(model=None, target="t", reference=None) == ["reference"]
    assert probe.check_lazy_status(model=None, target="t", reference="r") == ["model"]
    assert probe.check_lazy_status(model="m", target="t", reference="r") == []
    clip_probe = pack.NODE_CLASS_MAPPINGS["CLIPScore"]()
    assert clip_probe.check_lazy_status(model=None, image=None) == ["image"]
    assert clip_probe.check_lazy_status(model=None, image="i") == ["model"]
    assert clip_probe.check_lazy_status(model="m", image="i") == []
    print("lazy sequencing order = images first, then model")

    # a not-yet-available input makes the score a safe no-op (empty per-output lists)
    out = pack.NODE_CLASS_MAPPINGS["FGResQPairScore"]().score()
    assert out == ([], [], [], [], []), out
    out1 = pack.NODE_CLASS_MAPPINGS["PyiqaAnyFRScore"]().score()
    assert out1 == ([],), out1
    out2 = pack.NODE_CLASS_MAPPINGS["QReAlignScore"]().score()
    assert out2 == ([],), out2
    out3 = pack.NODE_CLASS_MAPPINGS["CLIPScore"]().score()
    assert out3 == ([],), out3
    print("nodes return empty outputs when lazy inputs are absent")

    # the memory_management decorator: keep_loaded leaves the metric put; any
    # other value (default) drops it. The fake _Metric cannot be gutted (no
    # nn.Module structure), so the drop path is asserted on a real metric below.
    import torch as _torch

    class _Metric:
        def __init__(self):
            self.device = _torch.device("cuda")

        def to(self, dev):
            self.device = dev
            return self

    class _Node:
        @with_memory_management
        def score(self, model=None, target=None, reference=None, **kwargs):
            return ("ran",)

    node = _Node()
    kept = _Metric()
    assert node.score(model=kept, memory_management=MEMORY_KEEP) == ("ran",)
    assert kept.device.type == "cuda", kept.device
    print("memory_management: keep_loaded leaves it on the GPU")

    # full-unload (gut-and-drop) end to end on a real metric reached through the
    # generic pair now: score, drop, then score AGAIN with the same (now hollow)
    # object - the decorator must revive it and reproduce the same numbers
    sm, _sm_arch, _sm_value = _load_generic(pack, metric_catalog, "ssim", "FR")
    snode = pack.NODE_CLASS_MAPPINGS["PyiqaAnyFRScore"]()
    (before,) = snode.score(sm, image, image)
    snode.score(sm, image, image, memory_management=MEMORY_DROP)
    gutted = getattr(sm, "_pyiqa_gutted", False)
    (after,) = snode.score(sm, image, image)  # revives through the recipe
    assert gutted, "drop mode did not gut the metric"
    assert before == after, f"revived metric scores differ: {before} vs {after}"
    print(
        "full unload: metric gutted (VRAM+RAM freed), revival reproduces "
        f"score {before[0]:.4f}"
    )

    print("\n-- custom checkpoint discovery (architecture folders) --")
    from pyiqa.archs.fgresq_arch import default_model_urls as _fgresq_urls

    fg_defaults = {os.path.basename(url) for url in _fgresq_urls.values()}
    assert {"FGResQ.pth", "FGResQ_Degradation.pth"} <= fg_defaults, fg_defaults
    fg_dirs = metric_catalog.family_custom_dirs("fgresq")
    fg_files = [
        (label, path, kind)
        for directory in fg_dirs
        for label, path, kind in utils.list_weight_files(directory)
    ]
    print(f"FGResQ folders: {[os.path.basename(d) for d in fg_dirs]} "
          f"weight files: {[label for label, _p, _k in fg_files]}")
    # FGResQ has no dedicated loader any more, so there is no "(default)" entry
    # and no basename exclusion: the generic combo offers the architecture's two
    # registry variants plus whatever the user dropped in models/pyiqa/FGResQ/.
    # pyiqa's own default weights are not entries at all - they land LOOSE in
    # pyiqa's cache folders, which the folder rule never scans (asserted below).
    fg_values = _dropdown_values(metric_catalog, "FGResQ")
    assert "(default)" not in fg_values, fg_values
    assert {metric_catalog.VARIANT_LOOKUP[v].base for v in fg_values} == {
        "fgresq", "fgresq_pair"
    }, fg_values
    fg_customs = [v for v in fg_values if metric_catalog.VARIANT_LOOKUP[v].custom]
    for label, _path, _kind in fg_files:
        assert any(
            label in metric_catalog.VARIANT_LOOKUP[v].label for v in fg_values
        ), f"a weight file in models/pyiqa/FGResQ/ is not offered: {label}"
    print(f"FGResQ combo = the architecture's 2 registry variants + "
          f"{len(fg_customs)} entr(y/ies) from models/pyiqa/FGResQ/; pyiqa's own "
          f"default weights ({', '.join(sorted(fg_defaults))}) are not entries")

    for family in (
        "afine", "clipiqa", "dists", "laion", "lpips",
        "musiq", "qualiclip", "qrealign", "topiq", "wadiqam",
    ):
        dirs = metric_catalog.family_custom_dirs(family)
        if dirs:
            # full paths: the same {architecture}/ folder name appears once per
            # custom root (install-local + every yaml-registered pyiqa folder)
            print(f"  {family:10s} arch folders: {dirs}")
    q_snapshots = [
        (label, path)
        for directory in metric_catalog.family_custom_dirs("qrealign")
        for label, path in utils.list_snapshot_dirs(directory)
    ]
    print("Q-ReAlign snapshot dirs:", [label for label, _p in q_snapshots])
    # every custom variant offered anywhere must exist and say what it is (a
    # signpost entry is `custom` too - it is a message, not a weights choice, so
    # it is not a custom variant here)
    custom_variants = [
        (label, resolved)
        for label, resolved in metric_catalog.VARIANT_LOOKUP.items()
        if resolved.custom and resolved.kind != "signpost"
    ]
    for label, resolved in custom_variants:
        assert resolved.kind in ("checkpoint", "safetensors", "snapshot"), (
            label, resolved.kind
        )
        assert os.path.exists(resolved.path), (
            f"custom variant {label!r} points at a missing path {resolved.path!r}"
        )
    signposts = [
        (label, resolved)
        for label, resolved in metric_catalog.VARIANT_LOOKUP.items()
        if resolved.kind == "signpost"
    ]
    assert all(resolved.path == "" for _label, resolved in signposts), signposts
    print(f"custom variants in the generic combo on this machine: "
          f"{len(custom_variants)}; {len(signposts)} signpost message entries")

    print("\n-- int8 download scoping (regression: full repo was downloaded) --")
    import pyiqa_toolbox_smoke.nodes.qrealign_nodes as qr_mod

    sd_calls = {}

    def _fake_snapshot_download(repo_id, local_dir, allow_patterns=None, **kw):
        sd_calls.update(
            repo_id=repo_id, local_dir=local_dir, allow_patterns=list(allow_patterns or [])
        )
        # lay out files the way the real scoped download would
        name = sd_calls["allow_patterns"][0].split("/")[0]
        folder = os.path.join(local_dir, name)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "config.json"), "w") as f:
            f.write("{}")
        with open(os.path.join(folder, "model.int8_convrot.safetensors"), "wb") as f:
            f.write(b"stub")
        return local_dir

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.snapshot_download = _fake_snapshot_download
    real_hf = sys.modules.get("huggingface_hub")
    real_find = qr_mod.find_pyiqa_model
    sys.modules["huggingface_hub"] = fake_hf
    qr_mod.find_pyiqa_model = lambda n: None  # force the download path
    try:
        d = qr_mod._resolve_int8_dir("qrealign-mini-0.8B", qr_mod._int8_repo())
    finally:
        qr_mod.find_pyiqa_model = real_find
        if real_hf is not None:
            sys.modules["huggingface_hub"] = real_hf
        else:
            del sys.modules["huggingface_hub"]
    exp = "Q-ReAlign-Mini-0.8B-int8-convrot"
    assert sd_calls["allow_patterns"] == [f"{exp}/**"], sd_calls
    assert os.path.normcase(sd_calls["local_dir"]) == os.path.normcase(
        utils.local_pyiqa_dir()
    ), sd_calls["local_dir"]
    assert os.path.normcase(os.path.basename(d)) == os.path.normcase(exp), d
    assert os.path.normcase(os.path.dirname(d)) == os.path.normcase(
        utils.local_pyiqa_dir()
    ), d
    print(f"int8 download scoped to {sd_calls['allow_patterns']} into "
          f"models/pyiqa (flat layout, one variant only)")

    print("\n-- metric cache --")
    # (torch is imported at module level; no local re-import - that would shadow
    # it for the whole function and break earlier uses)
    calls = {"n": 0}

    def _factory():
        calls["n"] += 1
        return torch.nn.Linear(2, 2)

    key = ("unit-test-cache", "k")
    first = utils.get_cached_metric("unit-test-cache", key, _factory)
    second = utils.get_cached_metric("unit-test-cache", key, _factory)
    assert calls["n"] == 1, f"factory ran {calls['n']} times, expected 1"
    assert first is second, "cache hit returned a different object"
    assert isinstance(first, torch.nn.Module) and not isinstance(first, tuple), (
        f"cache hit returned {type(first).__name__}, expected a module (not a tuple)"
    )
    assert next(first.parameters()) is not None
    print("get_cached_metric: hit returns the metric itself (regression: was a tuple)")

    print("\n-- METRICS.md is generated from the catalog --")
    # The user-facing metric reference is generated from the same catalog this
    # test just exercised, so a label / mode / range / registry change that
    # forgets to regenerate the document fails here instead of shipping stale
    # docs. gen_metrics_doc is loaded as a plain module: its module level is
    # data only (load_catalog() is what imports the pack, and we do not call it
    # - we hand it the modules this test already has).
    doc_spec = importlib.util.spec_from_file_location(
        "gen_metrics_doc", os.path.join(PKG_DIR, "tools", "gen_metrics_doc.py")
    )
    docgen = importlib.util.module_from_spec(doc_spec)
    doc_spec.loader.exec_module(docgen)
    rows = docgen.collect(metric_catalog)
    expected = docgen.render(pack, metric_catalog, rows)
    with open(os.path.join(PKG_DIR, "METRICS.md"), encoding="utf-8") as fh:
        current = fh.read()
    assert current == expected, (
        "METRICS.md is out of date - run tools/gen_metrics_doc.py"
    )
    # The document lists every REGISTRY variant - including the ones behind a
    # signpost, which the dropdown replaces with one message entry each. The
    # header's counts come from the same rows, so they are pinned here: 100
    # per-image weights choices (32 FR / 68 NR) across 50 architectures, plus the
    # 5 set-level / text-conditioned metrics.
    special_names = set(metric_catalog.SPECIAL_INPUTS)
    per_image = [row for row in rows if row["name"] not in special_names]
    special_rows = [row for row in rows if row["name"] in special_names]
    fr_rows = sum(1 for row in per_image if row["mode"] == "FR")
    nr_rows = sum(1 for row in per_image if row["mode"] == "NR")
    arch_count = len({row["entry"].split(" | ")[0] for row in per_image})
    assert (len(per_image), arch_count, fr_rows, nr_rows, len(special_rows)) == (
        100, 50, 32, 68, 5
    ), (len(per_image), arch_count, fr_rows, nr_rows, len(special_rows))
    assert len(rows) == len(per_image) + len(special_rows), len(rows)
    header_number_line = next(
        (line for line in expected.splitlines() if "weights choices" in line), ""
    )
    assert (
        f"{len(per_image)} weights choices you can score per image, across "
        f"{arch_count} architectures ({fr_rows} full-reference, {nr_rows} "
        f"no-reference), plus {len(special_rows)} set-level or text-conditioned "
        "metrics" in header_number_line
    ), header_number_line
    # ...and the doc's names are exactly the catalog's registry metrics: the
    # dropdown has 2 entries fewer because the two signpost architectures fold
    # their registry variants (Q-ReAlign's 3) into one message entry each
    documented = {row["name"] for row in rows}
    listed = {
        resolved.base
        for resolved in metric_catalog.VARIANT_LOOKUP.values()
        if not resolved.custom
    }
    signposted = {
        name
        for arch in metric_catalog.ARCHES
        if metric_catalog.ARCHITECTURES[arch].signpost
        for name in metric_catalog.ARCHITECTURES[arch].variants
    }
    assert documented == listed | signposted, (
        documented ^ (listed | signposted)
    )
    # ...and the dropdown is exactly the bookkeeping of those: one entry per
    # listed registry variant, one per signpost architecture, one per custom
    signpost_entries = sum(
        1
        for resolved in metric_catalog.VARIANT_LOOKUP.values()
        if resolved.kind == "signpost"
    )
    custom_entries = sum(
        1
        for resolved in metric_catalog.VARIANT_LOOKUP.values()
        if resolved.custom and resolved.kind != "signpost"
    )
    assert len(metric_catalog.VARIANT_VALUES) == (
        len(listed) + signpost_entries + custom_entries
    ), (
        len(metric_catalog.VARIANT_VALUES),
        len(listed),
        signpost_entries,
        custom_entries,
    )
    print(f"METRICS.md matches the catalog: {len(rows)} registry weight choices "
          f"({len(special_rows)} set-level/text), {len(expected.splitlines())} lines")

    print("\nOK")


if __name__ == "__main__":
    main()
