"""Train a readout head on cached features and log the result.

    python scripts/train_probe.py --config configs/probes/linear_cls.yaml
    python scripts/train_probe.py --config configs/probes/linear_mean.yaml

One run per learning rate in the config's `lrs`. Each run appends one row to
results/results.csv (see src/utils/metrics.py).

Fairness rules for every comparison: same split, same epochs, same lr grid,
same seed(s). Only the feature / head should change between runs.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.features import FeatureDataset  # noqa: E402
from src.models.heads import build_head  # noqa: E402
from src.utils.config import load_config, set_seed  # noqa: E402
from src.utils.metrics import accuracy, count_parameters, log_result, track_compute  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--data", default="configs/ucf101.yaml")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-log", action="store_true", help="don't write to results.csv (for debugging)")
    args = p.parse_args()

    cfg = load_config(args.config)
    data_cfg = load_config(args.data)
    feature_dir = Path(data_cfg["feature_dir"])

    train_set = FeatureDataset(feature_dir / "train.pt", cfg["feature"])
    test_set = FeatureDataset(feature_dir / "test.pt", cfg["feature"])

    for lr in cfg["lrs"]:
        set_seed(cfg["seed"])
        head = build_head(cfg["head"], train_set.feature_dim, data_cfg["num_classes"], **cfg.get("head_kwargs", {}))
        head = head.to(args.device)

        # TODO:
        #   - DataLoaders for train_set / test_set (shuffle train only)
        #   - optimizer from cfg (AdamW, lr, weight_decay), maybe a cosine schedule
        #   - train for cfg["epochs"] inside `with track_compute(args.device) as compute:`
        #   - evaluate on test_set with accuracy(logits, labels) -> top1, top5
        #   - unless args.no_log, log_result({... see src/utils/metrics.py ...})
        raise NotImplementedError("probe training: to do")


if __name__ == "__main__":
    main()
