"""Compute metrics and the shared results file.

Every run gets one row in results/results.csv. Accuracy alone is not enough for
this project: we also compare how much each readout costs.

    with track_compute() as compute:
        train(...)
    log_result({
        "dataset": "ucf101", "feature": "cls", "head": "linear",
        "lr": 1e-3, "epochs": 50, "batch_size": 256, "seed": 0,
        "top1": acc1, "top5": acc5,
        "trainable_params": count_parameters(head),
        "peak_gpu_mem_mb": compute["peak_gpu_mem_mb"],
        "train_time_s": compute["time_s"],
        "config": "configs/probes/linear_cls.yaml",
    })
"""

from __future__ import annotations

import csv
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

RESULTS_FILE = Path(__file__).resolve().parents[2] / "results" / "results.csv"

RESULT_COLUMNS = [
    "date",
    "git_commit",
    "dataset",
    "feature",
    "head",
    "lr",
    "weight_decay",
    "epochs",
    "batch_size",
    "seed",
    "top1",
    "top5",
    "trainable_params",
    "peak_gpu_mem_mb",
    "train_time_s",
    "config",
    "notes",
]


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


@contextmanager
def track_compute(device: torch.device | str | None = None):
    """Measure wall-clock time and peak GPU memory of the enclosed block.

    Yields a dict that is filled in when the block exits:
        {"time_s": float, "peak_gpu_mem_mb": float or None (no GPU)}
    """
    stats: dict = {}
    use_cuda = torch.cuda.is_available() and (device is None or torch.device(device).type == "cuda")
    if use_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    try:
        yield stats
    finally:
        if use_cuda:
            torch.cuda.synchronize(device)
        stats["time_s"] = time.perf_counter() - start
        stats["peak_gpu_mem_mb"] = (
            torch.cuda.max_memory_allocated(device) / 2**20 if use_cuda else None
        )


def accuracy(logits: torch.Tensor, labels: torch.Tensor, topk=(1, 5)) -> list[float]:
    """Top-k accuracy in percent."""
    maxk = min(max(topk), logits.shape[-1])
    pred = logits.topk(maxk, dim=-1).indices
    correct = pred.eq(labels.unsqueeze(-1))
    return [correct[:, :min(k, maxk)].any(dim=-1).float().mean().item() * 100 for k in topk]


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def log_result(row: dict, path: str | Path = RESULTS_FILE) -> None:
    """Append one run to the results CSV (header is written if the file is new)."""
    unknown = set(row) - set(RESULT_COLUMNS)
    if unknown:
        raise KeyError(f"unknown result columns {sorted(unknown)}; add them to RESULT_COLUMNS first")

    row = {"date": datetime.now().isoformat(timespec="seconds"), "git_commit": git_commit(), **row}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
