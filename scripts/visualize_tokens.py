"""Look inside LeVJEPA: what does the CLS token read, and what do the patch tokens encode?

    python scripts/visualize_tokens.py --video path/to/clip.avi
    python scripts/visualize_tokens.py --video a.avi b.avi --layer 11 --out results/figures

For every clip this saves figures in --out (default results/figures/):

    <clip>_cls_attention.png   where the CLS token looks, frame by frame   (done, worked example)
    <clip>_pca.png             patch tokens projected to 3 colours          (TODO: patch_pca_rgb)
    <clip>_similarity.png      similarity of every patch to one query patch (TODO: query_similarity)

The two TODO functions below are yours to write. Until they are implemented the
script just skips those figures. results/examples/ shows what you should get on
two UCF101 clips (scripts/download_example_clips.py fetches them). Use a GPU if
you have one; on a laptop CPU one clip takes about 10-30 seconds.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.attention import cls_attention  # noqa: E402
from src.analysis.plotting import frame_grid, upsample_maps  # noqa: E402
from src.data.video import load_clip  # noqa: E402
from src.models.backbone import LeVJEPABackbone  # noqa: E402
from src.utils.config import load_config  # noqa: E402


# ---------------------------------------------------------------------------
# Worked example: CLS attention
# ---------------------------------------------------------------------------

def cls_attention_maps(attn: torch.Tensor, per_frame: bool = True) -> torch.Tensor:
    """(heads, T, h, w) CLS attention -> (T, h, w) map to plot.

    Averages over heads. With per_frame=True each frame is rescaled to sum to 1,
    which shows *where* in each frame the CLS looks. With per_frame=False you
    see *which frames* get the attention; try both.
    """
    a = attn.mean(0)
    if per_frame:
        a = a / a.sum(dim=(1, 2), keepdim=True)
    return a


# ---------------------------------------------------------------------------
# TODO 1: PCA of the patch tokens
# ---------------------------------------------------------------------------

def patch_pca_rgb(patches: torch.Tensor) -> torch.Tensor:
    """(T, h, w, D) patch tokens -> (T, h, w, 3) colours in [0, 1].

    Project every patch token onto the first 3 principal components (computed
    over all T*h*w tokens of the clip) and use them as R, G, B. Patches with
    similar features get similar colours. See Figure 3 of the LeVJEPA paper.

    Hints: torch.pca_lowrank or torch.linalg.svd; rescale each component to
    [0, 1] (clipping the 1% tails helps). Then try removing each frame's mean
    token before the PCA and compare the two figures. What changes, and why?
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# TODO 2: similarity to a query patch
# ---------------------------------------------------------------------------

def query_similarity(patches: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
    """(T, h, w, D) patch tokens -> (T, h, w) cosine similarity to patch (t, h, w).

    Pick a patch on the moving person/object and see whether the same thing
    lights up in the other frames. See Figure 5 of the LeVJEPA paper. As for
    the PCA, compare raw tokens with tokens whose per-frame mean is removed.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", nargs="+", required=True)
    p.add_argument("--config", default="configs/model.yaml")
    p.add_argument("--layer", type=int, default=-1, help="which block's attention to show (0-23, -1 = last)")
    p.add_argument("--query", type=int, nargs=3, default=None, metavar=("T", "H", "W"),
                   help="query patch for the similarity map (default: last frame, centre)")
    p.add_argument("--every", type=int, default=2, help="plot every n-th frame")
    p.add_argument("--out", default="results/figures")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg = load_config(args.config)
    backbone = LeVJEPABackbone.from_config(cfg).to(args.device)
    t_grid, h_grid, w_grid = backbone.grid_shape
    size = cfg["image_size"]
    frame_ids = list(range(0, t_grid, args.every))
    out_dir = Path(args.out)

    for path in args.video:
        name = Path(path).stem
        clip, frames = load_clip(path, cfg["num_frames"], size, cfg["mean"], cfg["std"],
                                 target_fps=cfg.get("target_fps"), return_display=True)
        out, attn = cls_attention(backbone, clip.unsqueeze(0), layer=args.layer)
        patches = out.patches[0].float().cpu()
        attn = attn[0].float().cpu()

        maps = cls_attention_maps(attn)
        frame_grid([frames, upsample_maps(maps, size)], ["frames", "CLS attention"],
                   out_dir / f"{name}_cls_attention.png", frame_ids, overlay_on=frames,
                   title=f"{name}: CLS attention, layer {args.layer}, head mean")
        print(f"{name}: saved CLS attention")

        try:
            rgb = patch_pca_rgb(patches)
            frame_grid([frames, upsample_maps(rgb, size)], ["frames", "patch PCA"],
                       out_dir / f"{name}_pca.png", frame_ids, title=f"{name}: PCA of patch tokens")
            print(f"{name}: saved PCA")
        except NotImplementedError:
            print(f"{name}: skipped PCA (TODO: patch_pca_rgb)")

        q = args.query or (t_grid - 1, h_grid // 2, w_grid // 2)
        try:
            sim = query_similarity(patches, *q)
            frame_grid([frames, upsample_maps(sim, size)], ["frames", f"sim to {tuple(q)}"],
                       out_dir / f"{name}_similarity.png", frame_ids, overlay_on=frames,
                       title=f"{name}: cosine similarity to patch {tuple(q)}")
            print(f"{name}: saved similarity")
        except NotImplementedError:
            print(f"{name}: skipped similarity (TODO: query_similarity)")


if __name__ == "__main__":
    main()
