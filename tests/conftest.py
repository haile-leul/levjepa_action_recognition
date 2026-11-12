"""A tiny stand-in for the LeVJEPA model so the tests run on CPU without
downloading 1.2 GB of weights. It has the same interface as far as our code is
concerned: video in, (B, 1 + T*H*W, D) tokens out."""

import pytest
import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPooling

NUM_FRAMES = 4
IMAGE_SIZE = 32
PATCH_SIZE = 16
EMBED_DIM = 8


class FakeVideoViT(nn.Module):
    def __init__(self, output="model_output", with_cls=True, layout="BCTHW"):
        super().__init__()
        self.output = output
        self.with_cls = with_cls
        self.layout = layout
        self.patch_embed = nn.Conv3d(3, EMBED_DIM, (1, PATCH_SIZE, PATCH_SIZE), (1, PATCH_SIZE, PATCH_SIZE))
        self.cls_token = nn.Parameter(torch.randn(1, 1, EMBED_DIM))

    def forward(self, x):
        if self.layout == "BTCHW":
            x = x.permute(0, 2, 1, 3, 4)
        assert x.shape[1] == 3, f"fake model got the wrong layout: {tuple(x.shape)}"
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, T*H*W, D), time-major
        pooled = tokens.mean(1)
        if self.with_cls:
            tokens = torch.cat([self.cls_token.expand(len(x), -1, -1), tokens], dim=1)
        if self.output == "tensor":
            return tokens
        if self.output == "tuple":
            return (tokens, pooled)
        return BaseModelOutputWithPooling(last_hidden_state=tokens, pooler_output=pooled)


@pytest.fixture
def make_backbone():
    from src.models.backbone import LeVJEPABackbone

    def _make(input_layout="BCTHW", **fake_kwargs):
        fake = FakeVideoViT(layout=input_layout, **fake_kwargs)
        return LeVJEPABackbone(
            model_id="fake",
            num_frames=NUM_FRAMES,
            image_size=IMAGE_SIZE,
            patch_size=PATCH_SIZE,
            input_layout=input_layout,
            model=fake,
        )

    return _make


@pytest.fixture
def video_batch():
    return torch.randn(2, NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE)
