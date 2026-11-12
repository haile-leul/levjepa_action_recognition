import csv

import pytest
import torch

from src.data.features import FeatureDataset
from src.models.heads import LinearHead, build_head
from src.utils.metrics import RESULT_COLUMNS, accuracy, count_parameters, log_result, track_compute


def test_linear_head():
    head = build_head("linear", in_dim=1024, num_classes=101)
    assert isinstance(head, LinearHead)
    assert head(torch.randn(4, 1024)).shape == (4, 101)
    assert count_parameters(head) == 1024 * 101 + 101


def test_unknown_head():
    with pytest.raises(KeyError):
        build_head("transformer-xxl", 10, 2)


def test_accuracy():
    logits = torch.tensor([[0.9, 0.1, 0.0], [0.1, 0.2, 0.7]])
    top1, top2 = accuracy(logits, torch.tensor([0, 1]), topk=(1, 2))
    assert top1 == 50.0 and top2 == 100.0


def test_track_compute():
    with track_compute("cpu") as stats:
        sum(range(1000))
    assert stats["time_s"] >= 0
    assert stats["peak_gpu_mem_mb"] is None


def test_log_result(tmp_path):
    path = tmp_path / "results.csv"
    log_result({"dataset": "ucf101", "feature": "cls", "head": "linear", "top1": 90.0}, path)
    log_result({"dataset": "ucf101", "feature": "mean", "head": "linear", "top1": 91.0}, path)
    rows = list(csv.DictReader(open(path)))
    assert len(rows) == 2
    assert list(rows[0].keys()) == RESULT_COLUMNS
    assert rows[1]["feature"] == "mean"

    with pytest.raises(KeyError):
        log_result({"acc": 1.0}, path)


def test_feature_dataset(tmp_path):
    n = 10
    path = tmp_path / "train.pt"
    torch.save({
        "cls": torch.randn(n, 1024).half(),
        "mean": torch.randn(n, 1024).half(),
        "frame_mean": torch.randn(n, 16, 1024).half(),
        "labels": torch.arange(n),
        "paths": [f"clip_{i}.avi" for i in range(n)],
    }, path)

    ds = FeatureDataset(path, "cls")
    x, y = ds[3]
    assert x.shape == (1024,) and x.dtype == torch.float32 and y == 3

    assert FeatureDataset(path, "cls+mean").feature_dim == 2048
    assert FeatureDataset(path, "frame_mean")[0][0].shape == (16, 1024)
    with pytest.raises(KeyError):
        FeatureDataset(path, "patches")
