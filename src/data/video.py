"""Reading and preprocessing video clips.

A clip goes through three steps:

    frames, fps = read_video(path)                    # (N, H, W, 3) uint8
    idx = sample_frame_indices(len(frames), 16, stride_for_fps(fps, 7.5))
    clip = preprocess(frames[idx], 224, mean, std)    # (16, 3, 224, 224) float

``load_clip`` does all three. Batched with a DataLoader this gives
(B, T, C, H, W), which is what LeVJEPABackbone expects.

LeVJEPA was trained on 16 frames sampled at roughly 7.5 fps, so one clip covers
about two seconds of video. We sample the same way by default (target_fps=7.5).
Whatever sampling we pick, it must be the same for every readout we compare.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def read_video(path: str | Path) -> tuple[torch.Tensor, float]:
    """Decode every frame of a video file.

    Returns (frames, fps): frames is (N, H, W, 3) uint8.
    """
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate or stream.guessed_rate or 0) or 25.0
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return torch.from_numpy(np.stack(frames)), fps


def stride_for_fps(video_fps: float, target_fps: float | None) -> int | None:
    """Frame stride that brings video_fps down to roughly target_fps.

    None means "no target": sample_frame_indices then spreads the frames over
    the whole video instead.
    """
    if target_fps is None:
        return None
    return max(1, round(video_fps / target_fps))


def sample_frame_indices(
    num_total: int,
    num_frames: int = 16,
    stride: int | None = None,
) -> torch.Tensor:
    """Pick which frames to keep.

    stride=None: spread num_frames evenly over the whole video.
    stride=k:    take num_frames frames, k apart, from the middle of the video.
                 If the video is too short, the window is shrunk to fit.

    Short videos (fewer than num_frames frames) get repeated frames.
    """
    if num_total < 1:
        raise ValueError("video has no frames")

    if stride is None:
        idx = np.linspace(0, num_total - 1, num_frames)
    else:
        span = (num_frames - 1) * stride
        if span > num_total - 1:
            idx = np.linspace(0, num_total - 1, num_frames)
        else:
            start = (num_total - 1 - span) // 2
            idx = start + np.arange(num_frames) * stride
    return torch.from_numpy(np.round(idx).astype(np.int64))


def resize_crop(frames: torch.Tensor, image_size: int) -> torch.Tensor:
    """(T, H, W, 3) uint8 -> (T, 3, S, S) float in [0, 1].

    Resize the short side to image_size, then centre crop to a square.
    """
    x = frames.permute(0, 3, 1, 2).float() / 255.0
    h, w = x.shape[-2:]
    scale = image_size / min(h, w)
    new_h, new_w = max(image_size, round(h * scale)), max(image_size, round(w * scale))
    x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True)

    top = (new_h - image_size) // 2
    left = (new_w - image_size) // 2
    return x[..., top:top + image_size, left:left + image_size].clamp(0, 1)


def normalize(x: torch.Tensor, mean, std) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=x.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean_t) / std_t


def preprocess(frames: torch.Tensor, image_size: int, mean, std) -> torch.Tensor:
    """(T, H, W, 3) uint8 -> (T, 3, S, S) float, normalised."""
    return normalize(resize_crop(frames, image_size), mean, std)


def load_clip(
    path: str | Path,
    num_frames: int = 16,
    image_size: int = 224,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    target_fps: float | None = 7.5,
    return_display: bool = False,
):
    """Read a video file and return one model-ready clip, (T, 3, S, S).

    With return_display=True, also return the same frames un-normalised,
    (T, S, S, 3) in [0, 1], which is what you want for plotting.
    """
    frames, fps = read_video(path)
    idx = sample_frame_indices(len(frames), num_frames, stride_for_fps(fps, target_fps))
    x = resize_crop(frames[idx], image_size)
    clip = normalize(x, mean, std)
    if return_display:
        return clip, x.permute(0, 2, 3, 1)
    return clip
