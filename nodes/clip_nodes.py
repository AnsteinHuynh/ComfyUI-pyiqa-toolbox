"""CLIP Score: pyiqa's text-conditioned metric, image + caption in.

pyiqa's ``clipscore`` measures how well an image matches a caption (higher is
better, ~0-2.5). It is the one registry metric that needs an input no generic
score node can offer, which is why it has a node: the weights come from
"(Down)load pyiqa Model" like everything else, the caption comes from here.

Caption rules (see ``expand_captions``): one line is used for every image in the
batch; exactly one line per image pairs them up; anything else is an error
rather than a silent guess.
"""

from ..utility.pyiqa_utils import default_device, ensure_on_device

from ._common import (
    ANY_MODEL_TYPE,
    NRScoreBase,
    comfy_image_to_pyiqa,
    metric_for_mode,
    prepare_image,
    with_memory_management,
)


def expand_captions(text, count):
    """``text`` for ``count`` images: one caption broadcast, or exactly N.

    Blank lines are dropped first, because a trailing newline from a text box is
    normal and must not count as a caption. The strictness is on purpose: pairing
    the wrong caption with the wrong image produces a plausible number that means
    nothing, and a refusal is cheap.
    """
    lines = [line.strip() for line in str(text or "").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        raise ValueError(
            "CLIP Score needs a caption: type what the image is supposed to "
            "show (CLIP Score measures image-to-text match, not quality)."
        )
    if len(lines) == 1:
        return lines * count
    if len(lines) != count:
        raise ValueError(
            f"CLIP Score got {len(lines)} captions for {count} image(s): use one "
            "caption for the whole batch, or exactly one per line per image."
        )
    return lines


class CLIPScore(NRScoreBase):
    """Per-image CLIP Score with a caption widget."""

    CATEGORY = "pyiqa/Generic"
    MODEL_TYPE = ANY_MODEL_TYPE
    FAMILY = "clipscore"
    SCORE_LABEL = "CLIP Score"
    # clipscore IS a "special input" metric (it needs a caption); this node is
    # the one that supplies that input, so it must not refuse its own model
    HANDLES_SPECIAL_INPUTS = ("caption",)
    DESCRIPTION = (
        "How well each image matches the caption (pyiqa's CLIPScore, higher "
        "better, ~0-2.5). Load the weights with (Down)load pyiqa Model -> "
        "architecture 'CLIP Score'. One caption line applies to the whole batch; "
        "one line per image pairs them up; any other count is refused. Needs "
        "OpenAI's CLIP package (pyiqa imports it) and downloads ViT-B/32 on "
        "first use."
    )
    OUTPUT_TOOLTIPS = (
        "CLIP similarity between the image and its caption: higher means the "
        "image matches the text better. This is image-text alignment, not a "
        "quality judgement.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        types = super().INPUT_TYPES()
        types["required"]["caption"] = (
            "STRING",
            {
                "default": "",
                "multiline": True,
                "tooltip": (
                    "What the image(s) are supposed to show. One line = the same "
                    "caption for every image; one line per image = paired up in "
                    "order; any other line count is refused."
                ),
            },
        )
        return types

    @with_memory_management
    def score(self, model=None, image=None, caption="", **kwargs):
        if self._skipped(model, image):
            return ([],)
        self._require_compatible(model)
        captions = expand_captions(caption, image.shape[0])
        dev = ensure_on_device(model, default_device())
        metric = metric_for_mode(model, self.MODE_CHECK)
        x = prepare_image(comfy_image_to_pyiqa(image, dev), self._policy(model))
        # pyiqa asserts caption_list is not None and pairs it positionally, so
        # the caption list always has exactly one entry per image
        if self._single_image(model):
            return (
                [
                    float(
                        metric(x[i : i + 1], [captions[i]]).reshape(-1)[0]
                    )
                    for i in range(x.shape[0])
                ],
            )
        return (
            [
                float(value)
                for value in metric(x, captions).flatten().tolist()
            ],
        )


__all__ = ["CLIPScore", "expand_captions"]
