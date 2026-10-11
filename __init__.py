"""ComfyUI-pyiqa-toolbox: ComfyUI nodes wrapping the pyiqa (IQA-PyTorch)
library's metrics, implemented per-architecture-family (loader + score nodes).

Registration follows the KJNodes NODE_CONFIG pattern; a missing/broken pyiqa
install degrades to a warning instead of killing ComfyUI startup.
"""

import logging

logger = logging.getLogger(__name__)

NODE_CONFIG = {}

try:
    from .nodes.afine_nodes import AFINELoader, AFINEFRScore, AFINENRScore
    from .nodes.classic_fr_nodes import ClassicFRLoader, ClassicFRScore, L1Score
    from .nodes.clipiqa_nodes import CLIPIQALoader, CLIPIQANRScore
    from .nodes.dists_nodes import DISTSFrScore, DISTSLoader
    from .nodes.fgresq_nodes import FGResQLoader, FGResQPairScore, FGResQScore
    from .nodes.fid_nodes import FIDLoader, FIDScore
    from .nodes.laion_nodes import LAIONLoader, LAIONNRScore
    from .nodes.lpips_nodes import LPIPSFrScore, LPIPSLoader
    from .nodes.musiq_nodes import MUSIQLoader, MUSIQNRScore
    from .nodes.qrealign_nodes import QReAlignLoader, QReAlignScore
    from .nodes.qualiclip_nodes import QualiCLIPLoader, QualiCLIPNRScore
    from .nodes.topiq_nodes import TOPIQLoader, TOPIQFRScore, TOPIQNRScore
    from .nodes.utility_nodes import UnloadPyiqaModels
    from .nodes.wadiqam_nodes import WaDIQaMLoader, WaDIQaMFRScore, WaDIQaMNRScore

    # Display names follow one pattern: "<Family> Loader" and
    # "<Family> <mode> Score" (mode = FR / NR / none), no parenthetical modes.
    NODE_CONFIG.update(
        {
            "QReAlignLoader": {"class": QReAlignLoader, "name": "QReAlign Loader"},
            "QReAlignScore": {"class": QReAlignScore, "name": "QReAlign NR Score"},
            "FGResQLoader": {"class": FGResQLoader, "name": "FGResQ Loader"},
            "FGResQScore": {"class": FGResQScore, "name": "FGResQ NR Score"},
            "FGResQPairScore": {
                "class": FGResQPairScore,
                "name": "FGResQ FR Score",
            },
            "ClassicFRLoader": {"class": ClassicFRLoader, "name": "Classic FR Loader"},
            "ClassicFRScore": {"class": ClassicFRScore, "name": "Classic FR Score"},
            "L1Score": {"class": L1Score, "name": "L1 Score"},
            "TOPIQLoader": {"class": TOPIQLoader, "name": "TOPIQ Loader"},
            "TOPIQFRScore": {"class": TOPIQFRScore, "name": "TOPIQ FR Score"},
            "TOPIQNRScore": {"class": TOPIQNRScore, "name": "TOPIQ NR Score"},
            "AFINELoader": {"class": AFINELoader, "name": "A-FINE Loader"},
            "AFINEFRScore": {"class": AFINEFRScore, "name": "A-FINE FR Score"},
            "AFINENRScore": {"class": AFINENRScore, "name": "A-FINE NR Score"},
            "LPIPSLoader": {"class": LPIPSLoader, "name": "LPIPS Loader"},
            "LPIPSFrScore": {"class": LPIPSFrScore, "name": "LPIPS FR Score"},
            "DISTSLoader": {"class": DISTSLoader, "name": "DISTS Loader"},
            "DISTSFrScore": {"class": DISTSFrScore, "name": "DISTS FR Score"},
            "WaDIQaMLoader": {"class": WaDIQaMLoader, "name": "WaDIQaM Loader"},
            "WaDIQaMFRScore": {"class": WaDIQaMFRScore, "name": "WaDIQaM FR Score"},
            "WaDIQaMNRScore": {"class": WaDIQaMNRScore, "name": "WaDIQaM NR Score"},
            "MUSIQLoader": {"class": MUSIQLoader, "name": "MUSIQ Loader"},
            "MUSIQNRScore": {"class": MUSIQNRScore, "name": "MUSIQ NR Score"},
            "CLIPIQALoader": {"class": CLIPIQALoader, "name": "CLIP-IQA Loader"},
            "CLIPIQANRScore": {"class": CLIPIQANRScore, "name": "CLIP-IQA NR Score"},
            "QualiCLIPLoader": {"class": QualiCLIPLoader, "name": "QualiCLIP Loader"},
            "QualiCLIPNRScore": {
                "class": QualiCLIPNRScore,
                "name": "QualiCLIP NR Score",
            },
            "FIDLoader": {"class": FIDLoader, "name": "FID Loader"},
            "FIDScore": {"class": FIDScore, "name": "FID Score"},
            "LAIONLoader": {"class": LAIONLoader, "name": "LAION Aesthetic Loader"},
            "LAIONNRScore": {"class": LAIONNRScore, "name": "LAION Aesthetic Score"},
            "UnloadPyiqaModels": {
                "class": UnloadPyiqaModels,
                "name": "Unload Pyiqa Models",
            },
        }
    )
except Exception as e:
    # exc_info: a single typo'd import must not degrade to "zero nodes +
    # one warning line" with no diagnostic to trace
    logger.exception(f"(pyiqa-toolbox) could not import pyiqa nodes: {e}")


def _install_unload_hook():
    """Make Edit > Unload Models (and '... and Execution Cache') offload the
    pyiqa metric cache. ComfyUI core only unloads models it loaded itself, and
    this version offers no callback API — both menu items funnel through
    comfy.model_management.unload_all_models() (main.py queue-flag loop), so a
    guarded wrapper there is the single reliable trigger point.
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
            pyiqa_utils.offload_metric_cache()
        except Exception as e:  # never break ComfyUI's own unload
            logger.warning(f"(pyiqa-toolbox) metric offload on unload failed: {e}")
        return original(*args, **kwargs)

    unload_all_models._pyiqa_toolbox_hook = True
    model_management.unload_all_models = unload_all_models
    logger.info(
        "(pyiqa-toolbox) hooked comfy.model_management.unload_all_models — "
        "Edit > Unload Models now offloads pyiqa metrics to CPU"
    )


if NODE_CONFIG:
    _install_unload_hook()


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
