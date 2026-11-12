import numpy as np
import pytest
import torch

from src.data.video import load_clip, preprocess, read_video, sample_frame_indices


def test_uniform_sampling_covers_video():
    idx = sample_frame_indices(100, 16)
    assert len(idx) == 16
    assert idx[0] == 0 and idx[-1] == 99
    assert torch.all(idx[1:] >= idx[:-1])


def test_strided_sampling_is_centred():
    idx = sample_frame_indices(100, 16, stride=4)
    assert torch.all(idx[1:] - idx[:-1] == 4)
    assert abs(int(idx[0]) - (99 - int(idx[-1]))) <= 1  # same margin on both ends


def test_short_video_repeats_frames():
    idx = sample_frame_indices(5, 16)
    assert len(idx) == 16
    assert idx.max() == 4


def test_strided_sampling_falls_back_on_short_video():
    idx = sample_frame_indices(20, 16, stride=4)
    assert len(idx) == 16 and idx.max() <= 19


def test_preprocess_shape_and_normalisation():
    frames = torch.full((16, 240, 320, 3), 255, dtype=torch.uint8)
    x = preprocess(frames, 224, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    assert x.shape == (16, 3, 224, 224)
    torch.testing.assert_close(x, torch.ones_like(x))


@pytest.fixture
def tiny_video(tmp_path):
    av = pytest.importorskip("av")
    path = tmp_path / "clip.mp4"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=25)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(30):
            img = np.full((48, 64, 3), i * 8, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def test_read_video(tiny_video):
    frames = read_video(tiny_video)
    assert frames.shape == (30, 48, 64, 3)
    assert frames.dtype == torch.uint8


def test_load_clip(tiny_video):
    clip = load_clip(tiny_video, num_frames=16, image_size=32)
    assert clip.shape == (16, 3, 32, 32)
    assert torch.isfinite(clip).all()
