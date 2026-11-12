"""Cached backbone features.

The backbone is frozen, so we run it once per clip (scripts/extract_features.py)
and train every head on the saved features. One file per dataset split:

    features/<dataset>/<split>.pt  ->  dict with
        "cls"         (N, 1024)      CLS token
        "mean"        (N, 1024)      mean over all 16x14x14 patch tokens
        "frame_mean"  (N, 16, 1024)  mean patch token per frame (temporal position)
        "labels"      (N,)           int64, 0-based
        "paths"       list[str]      clip path relative to the dataset root
        "meta"        dict           model id, num_frames, image_size, sampling, ...

Store the feature tensors in float16 to halve the size; cast to float32 when
training. Don't save all 3136 patch tokens per clip, it's too big (about 6 MB
per clip in fp16). Spatially pooled tokens for attention pooling come later.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

FEATURE_KEYS = ("cls", "mean", "frame_mean")


def load_features(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


class FeatureDataset(Dataset):
    """Serves (features, label) pairs from a cached feature file.

    feature can be one of FEATURE_KEYS, or several joined with "+", e.g.
    "cls+mean" concatenates CLS and the mean-pooled patches along the last dim.
    """

    def __init__(self, path: str | Path, feature: str = "cls"):
        data = load_features(path)
        keys = feature.split("+")
        for k in keys:
            if k not in data:
                raise KeyError(f"{k!r} not in {path}, available: {[k for k in FEATURE_KEYS if k in data]}")
        self.x = torch.cat([data[k].float() for k in keys], dim=-1)
        self.y = data["labels"].long()
        self.paths = data.get("paths")
        self.meta = data.get("meta", {})

    @property
    def feature_dim(self) -> int:
        return self.x.shape[-1]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return self.x[i], self.y[i]
