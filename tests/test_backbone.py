import pytest
import torch

from conftest import EMBED_DIM, IMAGE_SIZE, NUM_FRAMES, PATCH_SIZE
from src.models.backbone import _split_tokens

GRID = (NUM_FRAMES, IMAGE_SIZE // PATCH_SIZE, IMAGE_SIZE // PATCH_SIZE)


@pytest.mark.parametrize("output", ["model_output", "tensor", "tuple"])
@pytest.mark.parametrize("layout", ["BCTHW", "BTCHW"])
def test_output_shapes(make_backbone, video_batch, output, layout):
    backbone = make_backbone(input_layout=layout, output=output)
    out = backbone(video_batch)
    assert out.cls.shape == (2, EMBED_DIM)
    assert out.patches.shape == (2, *GRID, EMBED_DIM)
    assert out.grid_shape == GRID


def test_cls_is_first_token(make_backbone, video_batch):
    backbone = make_backbone()
    raw = backbone.forward_raw(video_batch).last_hidden_state
    out = backbone.parse(raw)
    torch.testing.assert_close(out.cls, raw[:, 0])
    # patch grid is time-major: token 1 is (t=0, h=0, w=0), the last is (T-1, H-1, W-1)
    torch.testing.assert_close(out.patches[:, 0, 0, 0], raw[:, 1])
    torch.testing.assert_close(out.patches[:, -1, -1, -1], raw[:, -1])


def test_pooled_output_used_when_no_cls_token(make_backbone, video_batch):
    out = make_backbone(with_cls=False)(video_batch)
    torch.testing.assert_close(out.cls, out.patches.flatten(1, 3).mean(1))


def test_no_cls_and_no_pooled_raises():
    tokens = torch.randn(2, NUM_FRAMES * 4, EMBED_DIM)
    with pytest.raises(ValueError, match="no CLS"):
        _split_tokens(tokens, None, GRID)


def test_wrong_token_count_raises():
    tokens = torch.randn(2, 7, EMBED_DIM)
    with pytest.raises(ValueError, match="expected"):
        _split_tokens(tokens, None, GRID)


def test_backbone_is_frozen(make_backbone, video_batch):
    backbone = make_backbone()
    assert all(not p.requires_grad for p in backbone.parameters())
    backbone.train()
    assert not backbone.model.training
    assert not backbone(video_batch).cls.requires_grad


def test_rejects_non_video_input(make_backbone):
    with pytest.raises(ValueError):
        make_backbone()(torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE))
