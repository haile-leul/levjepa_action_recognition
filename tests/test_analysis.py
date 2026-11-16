import torch

from conftest import IMAGE_SIZE, NUM_FRAMES, NUM_HEADS, PATCH_SIZE
from src.analysis.attention import capture_attention, cls_attention, encoder_blocks
from src.analysis.plotting import frame_grid, upsample_maps

GRID = (NUM_FRAMES, IMAGE_SIZE // PATCH_SIZE, IMAGE_SIZE // PATCH_SIZE)


def test_cls_attention_shape_and_mass(make_backbone, video_batch):
    backbone = make_backbone()
    out, attn = cls_attention(backbone, video_batch, layer=-1)
    assert attn.shape == (2, NUM_HEADS, *GRID)
    # CLS -> patches is everything except CLS -> CLS, so it sums to less than 1
    assert torch.all(attn.sum(dim=(2, 3, 4)) <= 1 + 1e-5)
    assert out.patches.shape[1:4] == GRID


def test_attention_capture_does_not_change_outputs(make_backbone, video_batch):
    backbone = make_backbone()
    ref = backbone(video_batch)
    out, _ = cls_attention(backbone, video_batch, layer=0)
    torch.testing.assert_close(out.patches, ref.patches)


def test_capture_restores_sdpa_switch(make_backbone, video_batch):
    backbone = make_backbone()
    with capture_attention(backbone, layer=1) as store:
        backbone(video_batch)
    assert store["attn"].shape[-1] == 1 + NUM_FRAMES * 4
    assert all(b.attn.use_sdpa for b in encoder_blocks(backbone))


def test_upsample_and_plot(tmp_path):
    maps = torch.rand(NUM_FRAMES, 2, 2)
    up = upsample_maps(maps, 16)
    assert up.shape == (NUM_FRAMES, 16, 16)
    rgb = upsample_maps(torch.rand(NUM_FRAMES, 2, 2, 3), 16)
    assert rgb.shape == (NUM_FRAMES, 16, 16, 3)

    frames = torch.rand(NUM_FRAMES, 16, 16, 3)
    path = tmp_path / "fig.png"
    frame_grid([frames, up], ["frames", "map"], path, overlay_on=frames)
    assert path.exists() and path.stat().st_size > 0
