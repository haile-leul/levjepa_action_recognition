"""Load the pretrained LeVJEPA model and look at what goes in and what comes out.

    python scripts/check_model.py                          # random input, no data needed
    python scripts/check_model.py --video a.avi b.avi      # real clips

Prints the raw output of the Hugging Face model, then the CLS token and the
patch-token grid as the rest of the repo sees them, and checks the shapes
against configs/model.yaml. Exits with an error if they don't match.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.video import load_clip  # noqa: E402
from src.models.backbone import LeVJEPABackbone  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.metrics import count_parameters  # noqa: E402


def describe(obj, name="output", indent=0):
    pad = "  " * indent
    if isinstance(obj, torch.Tensor):
        print(f"{pad}{name}: Tensor{tuple(obj.shape)} {obj.dtype}")
    elif hasattr(obj, "keys"):
        print(f"{pad}{name}: {type(obj).__name__}")
        for k in obj.keys():
            describe(obj[k], k, indent + 1)
    elif isinstance(obj, (tuple, list)):
        print(f"{pad}{name}: {type(obj).__name__} of length {len(obj)}")
        for i, v in enumerate(obj):
            describe(v, f"[{i}]", indent + 1)
    else:
        print(f"{pad}{name}: {type(obj).__name__}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/model.yaml")
    p.add_argument("--video", nargs="*", default=[], help="video files to run on (default: random input)")
    p.add_argument("--batch-size", type=int, default=2, help="batch size for random input")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default=None, help="override the dtype from the config")
    args = p.parse_args()

    cfg = load_config(args.config)
    overrides = {"dtype": args.dtype} if args.dtype else {}

    print(f"Loading {cfg['model_id']} ...")
    backbone = LeVJEPABackbone.from_config(cfg, **overrides).to(args.device)
    print(f"  parameters:           {count_parameters(backbone, trainable_only=False) / 1e6:.1f} M")
    print(f"  trainable parameters: {count_parameters(backbone)}  (should be 0, the backbone is frozen)")

    t, s = cfg["num_frames"], cfg["image_size"]
    if args.video:
        clips = [load_clip(v, t, s, cfg["mean"], cfg["std"], target_fps=cfg.get("target_fps")) for v in args.video]
        video = torch.stack(clips)
    else:
        video = torch.randn(args.batch_size, t, 3, s, s)
    print(f"\nInput (B, T, C, H, W): {tuple(video.shape)}, sent to the model as {cfg['input_layout']}")

    raw = backbone.forward_raw(video)
    print("\nRaw model output:")
    describe(raw)

    out = backbone.parse(raw)
    b, d = video.shape[0], cfg["embed_dim"]
    expected_cls = (b, d)
    expected_patches = (b, *backbone.grid_shape, d)
    ok_cls = tuple(out.cls.shape) == expected_cls
    ok_patches = tuple(out.patches.shape) == expected_patches

    print("\nParsed:")
    print(f"  cls:     {tuple(out.cls.shape)}  expected {expected_cls}  {'ok' if ok_cls else 'MISMATCH'}")
    print(f"  patches: {tuple(out.patches.shape)}  expected {expected_patches}  {'ok' if ok_patches else 'MISMATCH'}")

    cls = out.cls.float()
    mean = out.patches.float().mean(dim=(1, 2, 3))
    print("\nA few numbers to look at:")
    print(f"  all finite:              {bool(torch.isfinite(out.patches).all() and torch.isfinite(out.cls).all())}")
    print(f"  CLS norm (per clip):     {[round(x, 2) for x in cls.norm(dim=-1).tolist()]}")
    print(f"  mean-patch norm:         {[round(x, 2) for x in mean.norm(dim=-1).tolist()]}")
    print(f"  cos(CLS, mean-patch):    {[round(x, 3) for x in F.cosine_similarity(cls, mean).tolist()]}")
    if b > 1:
        sim = F.cosine_similarity(cls.unsqueeze(1), cls.unsqueeze(0), dim=-1)
        print(f"  CLS cosine sim, clip 0 vs others: {[round(x, 3) for x in sim[0, 1:].tolist()]}")

    if not (ok_cls and ok_patches):
        sys.exit("\nShapes don't match the config. See docs/model_notes.md.")


if __name__ == "__main__":
    main()
