"""Plotting helpers for token maps on top of video frames."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F


def upsample_maps(maps: torch.Tensor, size: int) -> torch.Tensor:
    """(T, h, w) or (T, h, w, C) token maps -> (T, size, size[, C]), bilinear."""
    if maps.ndim == 3:
        return F.interpolate(maps[:, None].float(), size=(size, size), mode="bilinear", align_corners=False)[:, 0]
    x = maps.permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1)


def frame_grid(
    rows: list[torch.Tensor],
    row_titles: list[str],
    path: str | Path,
    frame_ids: list[int] | None = None,
    title: str | None = None,
    cmap: str = "inferno",
    overlay_on: torch.Tensor | None = None,
    alpha: float = 0.6,
):
    """Save a figure with one row per tensor and one column per frame.

    Each row is (T, S, S, 3) RGB in [0, 1], or (T, S, S) scalar maps drawn with
    cmap. Scalar rows are drawn over `overlay_on` (T, S, S, 3) if given.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = rows[0].shape[0]
    frame_ids = frame_ids if frame_ids is not None else list(range(t))
    fig, axes = plt.subplots(len(rows), len(frame_ids), figsize=(1.6 * len(frame_ids), 1.7 * len(rows)), squeeze=False)

    for r, (row, name) in enumerate(zip(rows, row_titles)):
        for c, f in enumerate(frame_ids):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            if row.ndim == 4:
                ax.imshow(row[f].clamp(0, 1).cpu().numpy())
            else:
                if overlay_on is not None:
                    ax.imshow(overlay_on[f].clamp(0, 1).cpu().numpy())
                ax.imshow(row[f].cpu().numpy(), cmap=cmap, alpha=alpha if overlay_on is not None else 1.0,
                          vmin=float(row.min()), vmax=float(row.max()))
            if r == 0:
                ax.set_title(f"t={f}", fontsize=8)
            if c == 0:
                ax.set_ylabel(name, fontsize=8)

    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
