"""A tiny stand-in for the LeVJEPA model so the tests run on CPU without
downloading 1.2 GB of weights. It mimics what our code relies on: video in
(B, C, T, H, W), (B, 1 + T*H*W, D) tokens out, and attention modules at
model.encoder.blocks[i].attn with a use_sdpa switch, like the real model."""

import pytest
import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPooling

NUM_FRAMES = 4
IMAGE_SIZE = 32
PATCH_SIZE = 16
EMBED_DIM = 8
NUM_HEADS = 2
DEPTH = 2


class FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = NUM_HEADS
        self.qkv = nn.Linear(EMBED_DIM, 3 * EMBED_DIM)
        self.use_sdpa = True

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, NUM_HEADS, c // NUM_HEADS).permute(2, 0, 3, 1, 4)
        attn = ((q @ k.transpose(-2, -1)) * (c // NUM_HEADS) ** -0.5).softmax(-1)
        y = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return y, (None if self.use_sdpa else attn)


class FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = FakeAttention()

    def forward(self, x):
        return x + self.attn(x)[0]


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(FakeBlock() for _ in range(DEPTH))


class FakeVideoViT(nn.Module):
    def __init__(self, output="model_output", with_cls=True, layout="BCTHW"):
        super().__init__()
        self.output = output
        self.with_cls = with_cls
        self.layout = layout
        self.patch_embed = nn.Conv3d(3, EMBED_DIM, (1, PATCH_SIZE, PATCH_SIZE), (1, PATCH_SIZE, PATCH_SIZE))
        self.cls_token = nn.Parameter(torch.randn(1, 1, EMBED_DIM))
        self.encoder = FakeEncoder()

    def forward(self, x):
        if self.layout == "BTCHW":
            x = x.permute(0, 2, 1, 3, 4)
        assert x.shape[1] == 3, f"fake model got the wrong layout: {tuple(x.shape)}"
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, T*H*W, D), time-major
        if self.with_cls:
            tokens = torch.cat([self.cls_token.expand(len(x), -1, -1), tokens], dim=1)
        for blk in self.encoder.blocks:
            tokens = blk(tokens)
        pooled = tokens[:, 1:].mean(1) if self.with_cls else tokens.mean(1)
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
