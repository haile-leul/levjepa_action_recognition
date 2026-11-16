"""Getting attention maps out of the LeVJEPA encoder.

The model normally uses PyTorch's fused attention (SDPA), which never builds the
attention matrix. Every attention module has a ``use_sdpa`` switch, though: turn
it off and the module computes softmax(QK^T) explicitly and returns it. We do
that for one layer at a time and catch the result with a forward hook.

Memory: one layer's attention matrix is heads x 3137 x 3137 floats per clip,
about 630 MB in float32. Use batch size 1.

What the CLS row means here: under block-causal attention the CLS token reads
from every patch of the clip while no patch reads from it, so the CLS row of
the last layer is literally "where the clip-level summary looks".
"""

from __future__ import annotations

from contextlib import contextmanager

import torch


def encoder_blocks(backbone):
    """The list of transformer blocks inside the HF model (24 for ViT-L)."""
    return backbone.model.encoder.blocks


@contextmanager
def capture_attention(backbone, layer: int = -1):
    """Context manager that records the attention matrix of one block.

        with capture_attention(backbone, layer=-1) as store:
            out = backbone(video)
        attn = store["attn"]          # (B, heads, N, N), N = 1 + T*H*W
    """
    attn_module = encoder_blocks(backbone)[layer].attn
    store: dict = {}

    def hook(_module, _inputs, output):
        store["attn"] = output[1]

    old = attn_module.use_sdpa
    attn_module.use_sdpa = False
    handle = attn_module.register_forward_hook(hook)
    try:
        yield store
    finally:
        handle.remove()
        attn_module.use_sdpa = old


@torch.no_grad()
def cls_attention(backbone, video: torch.Tensor, layer: int = -1):
    """Run the backbone and return its output plus the CLS attention map.

    Returns (out, cls_attn): out is the usual BackboneOutput, cls_attn is
    (B, heads, T, H, W), the attention from the CLS token to every patch.
    """
    with capture_attention(backbone, layer) as store:
        out = backbone(video)
    attn = store["attn"]
    b, heads = attn.shape[:2]
    cls_row = attn[:, :, 0, 1:]  # drop CLS -> CLS
    return out, cls_row.reshape(b, heads, *backbone.grid_shape)
