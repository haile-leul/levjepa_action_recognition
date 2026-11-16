"""Run the frozen backbone once over a dataset split and cache the features.

    python scripts/extract_features.py --data configs/ucf101.yaml --split train
    python scripts/extract_features.py --data configs/ucf101.yaml --split test

Writes <feature_dir>/<split>.pt in the format described in src/data/features.py.
Record the total extraction time, we want to see it.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.backbone import LeVJEPABackbone  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="configs/model.yaml")
    p.add_argument("--data", default="configs/ucf101.yaml")
    p.add_argument("--split", choices=["train", "test"], required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None, help="only the first N clips, for quick tests")
    args = p.parse_args()

    model_cfg = load_config(args.model)
    data_cfg = load_config(args.data)
    backbone = LeVJEPABackbone.from_config(model_cfg).to(args.device)

    # TODO:
    #   1. build the dataset (src/data/ucf101.py) and a DataLoader
    #   2. for each batch: out = backbone(batch["video"])
    #        cls        = out.cls                          (B, 1024)
    #        mean       = out.patches.mean(dim=(1, 2, 3))  (B, 1024)
    #        frame_mean = out.patches.mean(dim=(2, 3))     (B, 16, 1024)
    #      move to CPU, cast to float16, append to lists
    #   3. torch.save the dict described in src/data/features.py to
    #      Path(data_cfg["feature_dir"]) / f"{args.split}.pt"
    #   4. time the whole thing (src/utils/metrics.track_compute) and print it
    raise NotImplementedError("feature extraction: to do")


if __name__ == "__main__":
    main()
