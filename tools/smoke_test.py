"""Standalone smoke test: loads the pack the way ComfyUI does (hyphenated folder
as a package) and exercises every node with synthetic images.

Run with the ComfyUI env python, e.g.:
  C:/Users/kerok/miniconda3/envs/20260504comfyui/python.exe smoke_test.py
"""

import importlib.util
import os
import sys
import types

import torch

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
        pyiqa_dirs = [
            "I:/MyApps/ComfyUi/models/pyiqa",
            "C:/Users/kerok/.cache/huggingface/hub",
            "C:/Users/kerok/.cache/torch/hub/pyiqa",
            "X:/Models",
        ]
    pyiqa_dirs = [d for d in pyiqa_dirs if os.path.isdir(d)]
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
    import folder_paths as fp_mod

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

    print("resolved Q-ReAlign-Mini-0.8B ->", utils.find_pyiqa_model("Q-ReAlign-Mini-0.8B"))
    print("resolved Q-ReAlign-Pro-9B    ->", utils.find_pyiqa_model("Q-ReAlign-Pro-9B"))
    print("resolved FGResQ.pth          ->", utils.find_pyiqa_model("FGResQ.pth"))

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
    assert q_opts[:3] == builtin_labels, f"built-ins must come first, got {q_opts[:3]}"
    pro_path = utils.find_pyiqa_model("Q-ReAlign-Pro-9B")
    builtin_paths = {
        _norm(utils.find_pyiqa_model(folder))
        for _, _, folder in QREALIGN_VARIANTS.values()
        if utils.find_pyiqa_model(folder)
    }
    # no option label may be a custom resolving to a built-in's own folder
    for label, path in utils.list_pyiqa_checkpoints(require_config=True):
        if label in q_opts:
            assert _norm(path) not in builtin_paths, (
                f"option {label} duplicates built-in path {path}"
            )
    pro_here = os.path.exists("I:/MyApps/ComfyUi/models/pyiqa/Q-ReAlign-Pro-9B")
    if pro_path and pro_here:
        assert len(q_opts) == 6, (
            f"with Q-ReAlign-Pro-9B present in models/pyiqa the list must be "
            f"the 3 built-ins + 3 int8 entries, got {q_opts}"
        )
        assert q_opts[3:] == [k + " (int8_convrot)" for k in builtin_labels], (
            f"int8 entries must follow the built-ins, got {q_opts[3:]}"
        )
        assert _norm(pro_path).startswith(_norm("I:/MyApps/ComfyUi/models/pyiqa")), (
            f"Pro-9B should resolve from the I: models/pyiqa folder, got {pro_path}"
        )
    else:
        print("NOTE: I:/MyApps/ComfyUi/models/pyiqa/Q-ReAlign-Pro-9B absent "
              "-> exact-6 assertion vacuous")

    fg_opts = pack.NODE_CLASS_MAPPINGS["FGResQLoader"].INPUT_TYPES()["required"][
        "checkpoint"
    ][0]
    print("FGResQ checkpoint options:", fg_opts)
    assert fg_opts[0] == "(default)", "the (default) entry must stay first"
    for bad in ("FGResQ.pth", "FGResQ_Degradation.pth"):
        assert bad not in fg_opts, f"{bad} must not appear as a custom"

    # custom combos exist ONLY when customs are on disk; when they exist,
    # pyiqa's own defaults must not appear among them
    for node_id, patterns in (
        ("TOPIQLoader", ("cfanet_",)),
        ("AFINELoader", ("afine.pth",)),
    ):
        req = pack.NODE_CLASS_MAPPINGS[node_id].INPUT_TYPES()["required"]
        opts = req.get("custom_checkpoint")
        if opts is None:
            print(f"{node_id}: no customs on disk -> custom combos hidden")
            continue
        assert opts[0][0] == "(none)", f"{node_id}: (none) must stay first"
        print(f"{node_id} custom_checkpoint options:", opts[0])
        for bad in patterns:
            assert bad not in [str(o).lower() for o in opts[0]], (
                f"{node_id}: {bad} must not appear as a custom: {opts[0]}"
            )

    image = fake_image(2, 256, 256, seed=1)
    image2 = fake_image(2, 256, 256, seed=2)

    print("\n-- FGResQ --")
    (bundle,) = pack.NODE_CLASS_MAPPINGS["FGResQLoader"]().load("(default)")
    assert bundle.nr.net is bundle.pair.net, "NR/pair wrappers must share the arch"
    (scores,) = pack.NODE_CLASS_MAPPINGS["FGResQScore"]().score(bundle, image)
    assert len(scores) == 2
    print(f"NR    scores={[round(s, 4) for s in scores]}")
    out = pack.NODE_CLASS_MAPPINGS["FGResQPairScore"]().score(bundle, image, image2)
    assert len(out[0]) == len(out[2]) == len(out[4]) == 2
    print(f"PAIR  scores0={[round(s, 4) for s in out[0]]} "
          f"scores1={[round(s, 4) for s in out[1]]}")
    print(f"      ranks={out[2]} verdicts={out[3]} rank_probs={[round(p, 4) for p in out[4]]}")

    print("\n-- QReAlign (mini) --")
    (qmodel,) = pack.NODE_CLASS_MAPPINGS["QReAlignLoader"]().load(
        "qrealign-mini-0.8B", "auto"
    )
    (qscores,) = pack.NODE_CLASS_MAPPINGS["QReAlignScore"]().score(
        qmodel, image, "quality IQA"
    )
    assert len(qscores) == 2
    print(f"NR    scores={[round(s, 4) for s in qscores]}")

    print("\n-- unload/reload cycle --")
    fake_mm.unload_all_models()  # should offload the cache + call the original
    (post_scores,) = pack.NODE_CLASS_MAPPINGS["FGResQScore"]().score(bundle, image)
    assert post_scores == scores, f"score changed after offload cycle: {post_scores}"
    print(f"post-unload NR scores={[round(s, 4) for s in post_scores]} (unchanged)")

    print("\n-- Classic FR --")
    for metric, expect in [
        ("ssim", 1.0), ("ms_ssim", 1.0), ("fsim", 1.0), ("mad", 0.0),
        ("gmsd", 0.0), ("vif", 1.0), ("vsi", 1.0), ("cw_ssim", 1.0),
    ]:
        (m,) = pack.NODE_CLASS_MAPPINGS["ClassicFRLoader"]().load(metric)
        (s,) = pack.NODE_CLASS_MAPPINGS["ClassicFRScore"]().score(m, image, image)
        assert abs(s[0] - expect) < 1e-3, f"{metric}(a,a) = {s[0]}, expected ~{expect}"
        (s2,) = pack.NODE_CLASS_MAPPINGS["ClassicFRScore"]().score(m, image, image2)
        print(f"{metric:8s} identity={s[0]:.4f} noise-pair={s2[0]:.4f}")
    (m,) = pack.NODE_CLASS_MAPPINGS["ClassicFRLoader"]().load("psnr")
    (p,) = pack.NODE_CLASS_MAPPINGS["ClassicFRScore"]().score(m, image, image2)
    print(f"psnr     noise-pair={p[0]:.2f} dB")

    print("\n-- L1 (custom node) --")
    (l_self,) = pack.NODE_CLASS_MAPPINGS["L1Score"]().score(image, image)
    (l_noise,) = pack.NODE_CLASS_MAPPINGS["L1Score"]().score(image, image2)
    assert l_self[0] == 0.0
    print(f"l1       identity={l_self[0]:.4f} noise-pair={l_noise[0]:.4f}")

    print("\n-- TOPIQ --")
    (tf,) = pack.NODE_CLASS_MAPPINGS["TOPIQLoader"]().load("topiq_fr", "(none)", "topiq_fr")
    (tfr,) = pack.NODE_CLASS_MAPPINGS["TOPIQFRScore"]().score(tf, image, image2)
    (tn,) = pack.NODE_CLASS_MAPPINGS["TOPIQLoader"]().load("topiq_nr", "(none)", "topiq_fr")
    (tnr,) = pack.NODE_CLASS_MAPPINGS["TOPIQNRScore"]().score(tn, image)
    print(f"topiq_fr noise-pair={tfr[0]:.4f}  topiq_nr={tnr[0]:.4f}")
    try:
        pack.NODE_CLASS_MAPPINGS["TOPIQNRScore"]().score(tf, image)
        raise AssertionError("NR score with FR model should have raised")
    except ValueError as e:
        print(f"mode guard ok: {str(e)[:60]}...")

    print("\n-- A-FINE --")
    auto = "auto (up to multiple of 32)"
    (af,) = pack.NODE_CLASS_MAPPINGS["AFINELoader"]().load("afine_fr", "(none)", "afine_fr")
    (afr,) = pack.NODE_CLASS_MAPPINGS["AFINEFRScore"]().score(af, image, image2, auto)
    (an,) = pack.NODE_CLASS_MAPPINGS["AFINELoader"]().load("afine_nr", "(none)", "afine_fr")
    (anr,) = pack.NODE_CLASS_MAPPINGS["AFINENRScore"]().score(an, image, auto)
    print(f"afine_fr noise-pair={afr[0]:.4f}  afine_nr={anr[0]:.4f}")
    odd = fake_image(1, 250, 333, seed=3)
    (aodd,) = pack.NODE_CLASS_MAPPINGS["AFINENRScore"]().score(an, odd, auto)
    print(f"afine_nr 250x333 (auto -> 256x352)={aodd[0]:.4f}")

    print("\n-- LPIPS family --")
    for label in [
        "lpips (AlexNet)",
        "lpips (VGG)",
        "lpips+ (AlexNet)",
        "lpips+ (VGG)",
        "stlpips (AlexNet)",
        "stlpips (VGG)",
    ]:
        (pm,) = pack.NODE_CLASS_MAPPINGS["LPIPSLoader"]().load(label, "(none)", label)
        (ident,) = pack.NODE_CLASS_MAPPINGS["LPIPSFrScore"]().score(pm, image, image)
        (noise,) = pack.NODE_CLASS_MAPPINGS["LPIPSFrScore"]().score(pm, image, image2)
        assert abs(ident[0]) < 1e-4, f"{label} identity = {ident[0]}, expected ~0"
        print(f"{label:18s} identity={ident[0]:.5f} noise-pair={noise[0]:.4f}")

    print("\n-- DISTS family --")
    for label in ["dists", "msswd (MS-SWD learned)"]:
        (dm,) = pack.NODE_CLASS_MAPPINGS["DISTSLoader"]().load(label, "(none)", label)
        (ident,) = pack.NODE_CLASS_MAPPINGS["DISTSFrScore"]().score(dm, image, image)
        (noise,) = pack.NODE_CLASS_MAPPINGS["DISTSFrScore"]().score(dm, image, image2)
        assert abs(ident[0]) < 1e-4, f"{label} identity = {ident[0]}, expected ~0"
        print(f"{label:24s} identity={ident[0]:.5f} noise-pair={noise[0]:.4f}")

    print("\n-- LAION Aesthetic --")
    (lm,) = pack.NODE_CLASS_MAPPINGS["LAIONLoader"]().load("laion_aes (improved aesthetic predictor)", "(none)", None)
    (ls,) = pack.NODE_CLASS_MAPPINGS["LAIONNRScore"]().score(lm, image)
    assert 0 < ls[0] <= 12, f"laion_aes score {ls[0]} outside ~1-10 band"
    print(f"laion_aes image={ls[0]:.3f}")

    print("\n-- WaDIQaM --")
    (wf,) = pack.NODE_CLASS_MAPPINGS["WaDIQaMLoader"]().load(
        "wadiqam_fr (full-reference)", "(none)", None
    )
    (wfr,) = pack.NODE_CLASS_MAPPINGS["WaDIQaMFRScore"]().score(wf, image, image2)
    (wn,) = pack.NODE_CLASS_MAPPINGS["WaDIQaMLoader"]().load(
        "wadiqam_nr (no-reference)", "(none)", None
    )
    (wnr,) = pack.NODE_CLASS_MAPPINGS["WaDIQaMNRScore"]().score(wn, image)
    print(f"wadiqam_fr noise-pair={wfr[0]:.4f}  wadiqam_nr={wnr[0]:.4f}")

    print("\n-- MUSIQ / CLIP-IQA / QualiCLIP (NR) --")
    (mq,) = pack.NODE_CLASS_MAPPINGS["MUSIQLoader"]().load(
        "musiq (KonIQ-10k)", "(none)", None
    )
    (mqr,) = pack.NODE_CLASS_MAPPINGS["MUSIQNRScore"]().score(mq, image)
    (ci,) = pack.NODE_CLASS_MAPPINGS["CLIPIQALoader"]().load(
        "clipiqa", "(none)", None
    )
    (cir,) = pack.NODE_CLASS_MAPPINGS["CLIPIQANRScore"]().score(ci, image)
    (qc,) = pack.NODE_CLASS_MAPPINGS["QualiCLIPLoader"]().load(
        "qualiclip", "(none)", None
    )
    (qcr,) = pack.NODE_CLASS_MAPPINGS["QualiCLIPNRScore"]().score(qc, image)
    print(f"musiq={mqr[0]:.4f}  clipiqa={cir[0]:.4f}  qualiclip={qcr[0]:.4f}")

    print("\n-- FID (image-batch, multi-wire) --")
    (fm,) = pack.NODE_CLASS_MAPPINGS["FIDLoader"]().load("fid", "(none)", None)
    batch = fake_image(8, 256, 256, seed=11)
    # INPUT_IS_LIST: the score receives LISTS of batches per side
    (f_same,) = pack.NODE_CLASS_MAPPINGS["FIDScore"]().score(
        fm, [batch], [batch], "clean"
    )
    # two separate wires on the same side must concatenate, not overwrite
    (f_split,) = pack.NODE_CLASS_MAPPINGS["FIDScore"]().score(
        fm, [batch[:4], batch[4:]], [batch], "clean"
    )
    (f_diff,) = pack.NODE_CLASS_MAPPINGS["FIDScore"]().score(
        fm, [batch], [fake_image(8, 256, 256, seed=12)], "clean"
    )
    # Analytically FID(a,a)=0; pyiqa's frechet_distance regularises the sqrt at
    # eps=1e-6, so a tiny negative value is normal.
    assert abs(f_same) < 1e-3, f"FID(a,a) = {f_same}, expected ~0"
    assert abs(f_split - f_same) < 1e-6, (
        f"split-wire FID {f_split} != single-batch FID {f_same}"
    )
    assert f_diff > 0, f"FID of different sets should be > 0, got {f_diff}"
    print(f"fid identity(a,a)={f_same:.6f}  split-wire={f_split:.6f}  "
          f"different-sets={f_diff:.4f}")
    try:
        pack.NODE_CLASS_MAPPINGS["FIDScore"]().score(
            fm, [batch[:1]], [batch], "clean"
        )
        raise AssertionError("FID with 1 image per side should have raised")
    except ValueError as e:
        print(f"min-images guard ok: {str(e)[:60]}...")

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
        if not (v.endswith("Loader") or v.endswith("Score"))
        and k != "UnloadPyiqaModels"
    }
    assert not bad_name, f"display names must end with Loader/Score: {bad_name}"
    for node_id, display in sorted(names.items()):
        print(f"  {categories[node_id]:22s} {display}")
    face_variants = pack.NODE_CLASS_MAPPINGS["TOPIQLoader"].INPUT_TYPES()[
        "required"
    ]["model_name"][0]
    assert "topiq_nr-face" in face_variants and "topiq_nr_swin-face" in face_variants
    assert "topiq_iaa" in face_variants and "topiq_iaa_res50" in face_variants

    print("\n-- output tooltips + FR input names --")
    for node_id, cls in sorted(pack.NODE_CLASS_MAPPINGS.items()):
        tips = getattr(cls, "OUTPUT_TOOLTIPS", None)
        assert tips is not None, f"{node_id} has no OUTPUT_TOOLTIPS"
        assert len(tips) == len(cls.RETURN_TYPES), (
            f"{node_id}: {len(tips)} tooltips for {len(cls.RETURN_TYPES)} outputs"
        )
        assert all(isinstance(t, str) and t.strip() for t in tips), (
            f"{node_id} has an empty output tooltip"
        )
    print(f"all {len(pack.NODE_CLASS_MAPPINGS)} nodes: OUTPUT_TOOLTIPS == outputs, non-empty")

    for node_id in ("FGResQPairScore", "AFINEFRScore", "LPIPSFrScore", "FIDScore"):
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

    # sequencing: one name per call, images before model, [] when all present
    probe = pack.NODE_CLASS_MAPPINGS["FGResQPairScore"]()
    assert probe.check_lazy_status(model=None, target=None, reference=None) == ["target"]
    assert probe.check_lazy_status(model=None, target="t", reference=None) == ["reference"]
    assert probe.check_lazy_status(model=None, target="t", reference="r") == ["model"]
    assert probe.check_lazy_status(model="m", target="t", reference="r") == []
    l1 = pack.NODE_CLASS_MAPPINGS["L1Score"]()
    assert l1.check_lazy_status(target="t", reference="r") == []
    print("lazy sequencing order = images first, then model")

    # a not-yet-available input makes the score a safe no-op (empty per-output lists)
    out = pack.NODE_CLASS_MAPPINGS["FGResQPairScore"]().score()
    assert out == ([], [], [], [], []), out
    out1 = pack.NODE_CLASS_MAPPINGS["LPIPSFrScore"]().score()
    assert out1 == ([],), out1
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

    # full-unload (gut-and-drop) end to end on a real family metric: score,
    # drop, then score AGAIN with the same (now hollow) object - the decorator
    # must revive it and reproduce the same numbers
    (sm,) = pack.NODE_CLASS_MAPPINGS["ClassicFRLoader"]().load("ssim")
    snode = pack.NODE_CLASS_MAPPINGS["ClassicFRScore"]()
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

    print("\n-- custom checkpoint discovery --")
    customs = utils.list_pyiqa_checkpoints("FGResQ*.pth")
    print("FGResQ files on disk:", [label for label, _ in customs])
    fg_dropdown = pack.NODE_CLASS_MAPPINGS["FGResQLoader"].INPUT_TYPES()[
        "required"
    ]["checkpoint"][0]
    print("FGResQ dropdown:", fg_dropdown)
    leak = [
        o
        for o in fg_dropdown
        if o in ("FGResQ.pth", "FGResQ_Degradation.pth")
    ]
    assert not leak, f"FGResQ dropdown leaks pyiqa default weights: {leak}"
    qcustoms = utils.list_pyiqa_checkpoints(require_config=True)
    print("snapshot dirs:", [label for label, _ in qcustoms])
    perc_opts = pack.NODE_CLASS_MAPPINGS["LPIPSLoader"].INPUT_TYPES()[
        "required"
    ].get("custom_checkpoint")
    print("LPIPS custom_checkpoint options:",
          perc_opts[0] if perc_opts else "(hidden - no customs on disk)")

    print("\n-- metric cache --")
    import torch
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

    print("\nOK")


if __name__ == "__main__":
    main()
