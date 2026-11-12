"""Readout heads trained on top of the frozen backbone.

Only the linear probe is here to start with. New heads go in this file as the
project moves on, and get registered in HEADS so train_probe.py can build them
by name from a config:

    sprint 5  linear      on CLS or mean-pooled patches       (done below)
    sprint 6  linear      on [CLS, mean] concatenated         (just a bigger in_dim)
              mlp         one hidden layer
    sprint 7  attentive   learned query(ies) attending over patch tokens
              temporal    small 1D conv / attention over the 16 per-frame vectors

Every head takes a features tensor and returns logits of shape (B, num_classes).
Keep heads small: the point of the project is comparing readouts, not building
a big classifier.
"""

from __future__ import annotations

from torch import nn


class LinearHead(nn.Module):
    """Linear probe. Optional BatchNorm without affine params, as in many SSL
    linear-eval protocols, so the probe is less sensitive to feature scale."""

    def __init__(self, in_dim: int, num_classes: int, norm: bool = False):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_dim, affine=False) if norm else nn.Identity()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(self.norm(x))


HEADS = {
    "linear": LinearHead,
}


def build_head(name: str, in_dim: int, num_classes: int, **kwargs) -> nn.Module:
    if name not in HEADS:
        raise KeyError(f"unknown head {name!r}, available: {sorted(HEADS)}")
    return HEADS[name](in_dim=in_dim, num_classes=num_classes, **kwargs)
