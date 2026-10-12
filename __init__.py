"""ComfyUI-pyiqa-toolbox: ComfyUI nodes wrapping the pyiqa (IQA-PyTorch)
library's metrics.

The node surface is deliberately small: one loader over pyiqa's whole registry
(``(Down)load pyiqa Model``: architecture -> variant, custom checkpoints
included) plus generic FR/NR score nodes, and then only the things that need
something the generic path cannot express - FGResQ (a checkpoint that needs
surgery, and a pair API returning four values), Q-ReAlign (int8/snapshot builds,
a scoring-task switch), FID (set-level, folder-based) and CLIP Score (an image
plus a caption). A pyiqa release that adds metrics, or a whole new architecture,
therefore needs no change here.

Registration follows the KJNodes NODE_CONFIG pattern; a missing/broken pyiqa
install degrades to a warning instead of killing ComfyUI startup.
"""

import logging
import os
import sys

logger = logging.getLogger(__name__)

NODE_CONFIG = {}

try:
    from .nodes import arch_gates  # noqa: F401  (registers pre-load gates, no nodes)
    from .nodes.clip_nodes import CLIPScore
    from .nodes.fgresq_nodes import FGResQPairScore, FGResQScore
    from .nodes.fid_nodes import FIDScore
    from .nodes.pyiqa_any_nodes import PyiqaAnyFRScore, PyiqaAnyLoader, PyiqaAnyNRScore
    from .nodes.qrealign_nodes import QReAlignLoader, QReAlignScore
    from .nodes.utility_nodes import UnloadPyiqaModels

    # Display names follow one pattern: "(Down)load {thing}" for loaders
    # (KJNodes' "(Down)load CLIPSeg" convention - they download when the weights
    # are absent) and "{thing} {mode} Score" (mode = FR / NR / none) for score
    # nodes, no parenthetical modes.
    NODE_CONFIG.update(
        {
            "PyiqaAnyLoader": {
                "class": PyiqaAnyLoader,
                "name": "(Down)load pyiqa Model",
            },
            "PyiqaAnyFRScore": {"class": PyiqaAnyFRScore, "name": "pyiqa FR Score"},
            "PyiqaAnyNRScore": {"class": PyiqaAnyNRScore, "name": "pyiqa NR Score"},
            "CLIPScore": {"class": CLIPScore, "name": "CLIP Score"},
            "FGResQScore": {"class": FGResQScore, "name": "FGResQ NR Score"},
            "FGResQPairScore": {"class": FGResQPairScore, "name": "FGResQ FR Score"},
            "QReAlignLoader": {"class": QReAlignLoader, "name": "(Down)load Q-ReAlign"},
            "QReAlignScore": {"class": QReAlignScore, "name": "QReAlign NR Score"},
            "FIDScore": {"class": FIDScore, "name": "FID Score"},
            "UnloadPyiqaModels": {"class": UnloadPyiqaModels, "name": "Unload Pyiqa Models"},
        }
    )
except Exception as e:
    # exc_info: a single typo'd import must not degrade to "zero nodes +
    # one warning line" with no diagnostic to trace
    logger.exception(f"(pyiqa-toolbox) could not import pyiqa nodes: {e}")


def _scaffold_custom_folders():
    """Create the ``models/pyiqa/<Architecture>/`` drop folders.

    Custom checkpoints are read from a per-architecture folder (a fine-tune of
    an architecture is just another variant of it), so the families whose loader
    takes custom weights get a visible place to put one. Existing folders are
    left untouched; failures are logged, never fatal.
    """
    try:
        from .utility import metric_catalog

        created = metric_catalog.scaffold_folders()
        logger.info(
            "(pyiqa-toolbox) %d custom-checkpoint folders ready under %s",
            len(created),
            " + ".join(metric_catalog.custom_roots()),
        )
    except Exception as e:
        logger.warning(f"(pyiqa-toolbox) could not prepare model folders: {e}")


def _version_of(module_name):
    """Version of an installed package, without importing it.

    ``transformers`` is only imported when a node actually needs it, so its
    version comes from the installed distribution's metadata instead - an
    import here would cost seconds at every ComfyUI start.
    """
    version = getattr(sys.modules.get(module_name), "__version__", None)
    if version:
        return str(version)
    try:
        from importlib.metadata import version as dist_version

        return dist_version(module_name)
    except Exception:
        return "not installed"


def _pack_version():
    """This pack's own version, read from pyproject.toml: one source of truth."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pyproject.toml")
    try:
        import tomllib

        with open(path, "rb") as fh:
            return str(tomllib.load(fh)["project"]["version"])
    except Exception:
        pass
    try:  # python 3.10 has no tomllib; the version line is all we need
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("version"):
                    return line.split("=", 1)[1].strip().strip("\"'")
    except Exception:
        pass
    return "unknown"


def _log_versions():
    """One startup line, so any bug report says which versions were running."""
    try:
        from .utility import metric_catalog

        catalog = (
            f"{len(metric_catalog.ARCHES)} architectures, "
            f"{len(metric_catalog.VARIANT_VALUES)} variants"
        )
        if metric_catalog.REGISTRY_ERROR:
            catalog += " (pyiqa metric table unreadable)"
        logger.info(
            "(pyiqa-toolbox) v%s | pyiqa %s | transformers %s | torch %s | %s",
            _pack_version(),
            _version_of("pyiqa"),
            _version_of("transformers"),
            _version_of("torch"),
            catalog,
        )
    except Exception as e:  # a log line must never break startup
        logger.debug(f"(pyiqa-toolbox) version line failed: {e}")


def _install_unload_hook():
    """Make Edit > Unload Models unload the pyiqa metric cache too.

    The pack's one monkey patch, kept deliberately: ComfyUI core unloads only
    the models it loaded itself, and this version offers no callback API (there
    is no unload registry in comfy.model_management), while both menu items
    funnel through comfy.model_management.unload_all_models() (main.py's
    queue-flag loop). A wrapper there is the single reliable trigger point, and
    it is written to be harmless: ComfyUI's own function always runs with the
    arguments and return value it expects, our part is wrapped in try/except so
    it can never break that unload, the marker attribute makes a module reload
    idempotent, and a ComfyUI that ever renames the function simply gets no hook
    (the pack's own Unload Pyiqa Models node and the score nodes'
    memory_management still release everything).
    """
    try:
        from comfy import model_management

        from .utility import pyiqa_utils
    except Exception:
        return
    if getattr(model_management.unload_all_models, "_pyiqa_toolbox_hook", False):
        return
    original = model_management.unload_all_models

    def unload_all_models(*args, **kwargs):
        try:
            pyiqa_utils.unload_metric_cache()
        except Exception as e:  # never break ComfyUI's own unload
            logger.warning(f"(pyiqa-toolbox) metric unload on unload failed: {e}")
        return original(*args, **kwargs)

    unload_all_models._pyiqa_toolbox_hook = True
    model_management.unload_all_models = unload_all_models
    logger.info(
        "(pyiqa-toolbox) hooked comfy.model_management.unload_all_models — "
        "Edit > Unload Models now unloads pyiqa metrics too (VRAM + RAM)"
    )


if NODE_CONFIG:
    _install_unload_hook()
    _scaffold_custom_folders()
    _log_versions()


def generate_node_mappings(node_config):
    node_class_mappings = {}
    node_display_name_mappings = {}
    for node_name, node_info in node_config.items():
        node_class_mappings[node_name] = node_info["class"]
        node_display_name_mappings[node_name] = node_info.get(
            "name", node_info["class"].__name__
        )
    return node_class_mappings, node_display_name_mappings


NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = generate_node_mappings(NODE_CONFIG)
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Client extension: "(Down)load pyiqa Model"'s variant dropdown follows the
# architecture combo (see web/js/pyiqaAnyLoader.js). Served even when the
# pyiqa import above failed - it is harmless without the node.
WEB_DIRECTORY = "./web/js"
__all__.append("WEB_DIRECTORY")
