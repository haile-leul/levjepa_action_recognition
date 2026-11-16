"""UCF101 dataset.

Expected layout of the dataset root (the official download, unpacked):

    UCF101/
        UCF-101/
            ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi
            ...
        ucfTrainTestlist/
            classInd.txt        "1 ApplyEyeMakeup"            (1-based class ids)
            trainlist01.txt     "ApplyEyeMakeup/v_..._c01.avi 1"
            testlist01.txt      "ApplyEyeMakeup/v_..._c01.avi" (no label, take it
                                                                from the folder name)

We use split 1 only. Each item should be one clip of shape (T, 3, S, S) built
with src.data.video.load_clip (same num_frames, image_size, mean, std and
target_fps as configs/model.yaml), plus its 0-based label.

Things to think about while writing this:
  - labels in the split files are 1-based, PyTorch wants 0-based
  - a few videos are short or slightly broken; don't let one bad file kill a
    run over 13k clips
  - decoding is the slow part, so use DataLoader workers
  - return the relative path too, so extracted features can be traced back
    to the clip they came from
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset


class UCF101(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        fold: int = 1,
        num_frames: int = 16,
        image_size: int = 224,
        mean: list[float] = (0.485, 0.456, 0.406),
        std: list[float] = (0.229, 0.224, 0.225),
        target_fps: float | None = 7.5,
    ):
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        self.root = Path(root)
        self.split = split
        self.fold = fold
        self.num_frames = num_frames
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.target_fps = target_fps

        # TODO: read classInd.txt and the split file, build
        #   self.classes:  list[str], index = 0-based label
        #   self.samples:  list[tuple[str, int]], (relative path, label)
        raise NotImplementedError("UCF101 loader: to do")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        # TODO: return {"video": (T, 3, S, S) tensor, "label": int, "path": str}
        raise NotImplementedError
