"""Frozen LeVJEPA video encoder.

This wraps the Hugging Face checkpoint so that the rest of the code only ever
sees two things: the CLS token and the grid of patch tokens.

    backbone = LeVJEPABackbone.from_config(load_config("configs/model.yaml"))
    out = backbone(video)       # video: (B, T, C, H, W), already normalised
    out.cls                     # (B, D)            D = 1024 for ViT-L
    out.patches                 # (B, T', H', W', D) = (B, 16, 14, 14, 1024)

The backbone is frozen for the whole project: no gradients, always in eval mode.

The remote model code is loaded with ``trust_remote_code=True``. We only rely on
it returning a sequence of tokens (a tensor, a tuple, or a ModelOutput with
``last_hidden_state``). If the checkpoint behaves differently from what is
described in docs/model_notes.md, ``_unpack_tokens`` and ``_split_tokens`` are
the only places that need to change.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

DEFAULT_MODEL_ID = "galilai-group/levjepa-videomix-large"

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class BackboneOutput:
    cls: torch.Tensor  # (B, D)
    patches: torch.Tensor  # (B, T', H', W', D)

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        return tuple(self.patches.shape[1:4])

    @property
    def embed_dim(self) -> int:
        return self.patches.shape[-1]


class LeVJEPABackbone(nn.Module):
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        num_frames: int = 16,
        image_size: int = 224,
        patch_size: int = 16,
        tubelet_size: int = 1,
        input_layout: str = "BCTHW",
        dtype: str = "float32",
        model: nn.Module | None = None,
    ):
        """
        Args:
            model_id: Hugging Face repo id of the checkpoint.
            num_frames, image_size, patch_size, tubelet_size: define the token
                grid, (num_frames / tubelet_size, image_size / patch_size, ...).
            input_layout: layout the model expects, "BCTHW" or "BTCHW". Our data
                pipeline always produces (B, T, C, H, W) and we permute here.
            dtype: weights dtype, "float32", "float16" or "bfloat16".
            model: an already built model. Only used by the tests, which pass a
                small fake model instead of downloading the real one.
        """
        super().__init__()
        if input_layout not in ("BCTHW", "BTCHW"):
            raise ValueError(f"input_layout must be BCTHW or BTCHW, got {input_layout!r}")

        self.model_id = model_id
        self.num_frames = num_frames
        self.image_size = image_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.input_layout = input_layout
        self.dtype = _DTYPES[dtype]

        if model is None:
            model = _load_pretrained(model_id)
        # Note: we deliberately do not pass or change attn_mode. The weights were
        # trained with block-causal attention, which is the default.
        self.model = model.to(self.dtype)
        self.model.eval()
        self.model.requires_grad_(False)

    @classmethod
    def from_config(cls, cfg: dict, **overrides) -> "LeVJEPABackbone":
        keys = ("model_id", "num_frames", "image_size", "patch_size",
                "tubelet_size", "input_layout", "dtype")
        kwargs = {k: cfg[k] for k in keys if k in cfg}
        kwargs.update(overrides)
        return cls(**kwargs)

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        side = self.image_size // self.patch_size
        return (self.num_frames // self.tubelet_size, side, side)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def train(self, mode: bool = True):
        # The encoder stays in eval mode no matter what, so a head's
        # model.train() call can never switch on dropout etc. in the backbone.
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward_raw(self, video: torch.Tensor):
        """Run the HF model and return whatever it returns, untouched."""
        if video.ndim != 5:
            raise ValueError(f"expected a (B, T, C, H, W) video batch, got shape {tuple(video.shape)}")
        x = video.to(device=self.device, dtype=self.dtype)
        if self.input_layout == "BCTHW":
            x = x.permute(0, 2, 1, 3, 4)
        return self.model(x)

    def parse(self, raw) -> BackboneOutput:
        """Turn a raw model output into CLS + patch grid."""
        tokens, pooled = _unpack_tokens(raw)
        return _split_tokens(tokens, pooled, self.grid_shape)

    @torch.no_grad()
    def forward(self, video: torch.Tensor) -> BackboneOutput:
        return self.parse(self.forward_raw(video))


def _load_pretrained(model_id: str) -> nn.Module:
    from transformers import AutoModel

    return AutoModel.from_pretrained(model_id, trust_remote_code=True)


def _unpack_tokens(raw) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Get (tokens, pooled) out of the raw model output.

    tokens is (B, N, D). pooled is (B, D) if the model gives one, else None.
    """
    if isinstance(raw, torch.Tensor):
        return raw, None

    # ModelOutput behaves like a dict.
    if hasattr(raw, "keys"):
        tokens = raw.get("last_hidden_state")
        if tokens is None:
            raise KeyError(
                f"model output has no 'last_hidden_state'; keys are {list(raw.keys())}"
            )
        return tokens, raw.get("pooler_output")

    if isinstance(raw, (tuple, list)):
        tokens = raw[0]
        pooled = raw[1] if len(raw) > 1 and isinstance(raw[1], torch.Tensor) and raw[1].ndim == 2 else None
        return tokens, pooled

    raise TypeError(f"don't know how to read model output of type {type(raw).__name__}")


def _split_tokens(
    tokens: torch.Tensor,
    pooled: torch.Tensor | None,
    grid_shape: tuple[int, int, int],
) -> BackboneOutput:
    """Split a (B, N, D) token sequence into CLS and a (B, T', H', W', D) grid.

    Assumes the CLS token comes first and patch tokens are ordered time-major,
    i.e. (t, h, w) flattened. Check this in sprint 3 (see docs/model_notes.md).
    """
    if tokens.ndim != 3:
        raise ValueError(f"expected tokens of shape (B, N, D), got {tuple(tokens.shape)}")

    b, n, d = tokens.shape
    t, h, w = grid_shape
    n_grid = t * h * w

    if n == n_grid + 1:
        cls, patches = tokens[:, 0], tokens[:, 1:]
    elif n == n_grid and pooled is not None:
        cls, patches = pooled, tokens
    elif n == n_grid:
        raise ValueError(
            f"model returned {n} tokens and no pooled output, so there is no CLS token to read"
        )
    else:
        raise ValueError(
            f"model returned {n} tokens, expected {n_grid + 1} (CLS + {t}x{h}x{w} patches). "
            "Check num_frames, image_size, patch_size, tubelet_size and input_layout in "
            "configs/model.yaml, and whether the model adds extra (e.g. register) tokens."
        )

    return BackboneOutput(cls=cls, patches=patches.reshape(b, t, h, w, d))
