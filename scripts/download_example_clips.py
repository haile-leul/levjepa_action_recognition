"""Download the two UCF101 clips used in results/examples/, without the full dataset.

    python scripts/download_example_clips.py              # -> data/examples/
    python scripts/download_example_clips.py --out somewhere/

The clips are the original UCF101 videos:

    UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi
    UCF-101/BabyCrawling/v_BabyCrawling_g19_c02.avi

If you already have UCF101, you don't need this: use those paths directly.
Here they come from a small public UCF101 subset on the Hugging Face Hub
(sayakpaul/ucf101-subset, about 170 MB).
"""

import argparse
import shutil
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "sayakpaul/ucf101-subset"
ARCHIVE = "UCF101_subset.tar.gz"
CLIPS = {
    "v_ApplyEyeMakeup_g01_c01.avi": "UCF101_subset/val/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi",
    "v_BabyCrawling_g19_c02.avi": "UCF101_subset/test/BabyCrawling/v_BabyCrawling_g19_c02.avi",
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/examples")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    archive = hf_hub_download(REPO, ARCHIVE, repo_type="dataset")

    with tarfile.open(archive) as tar:  # despite the name, it's a plain tar
        for name, member in CLIPS.items():
            with tar.extractfile(member) as src, open(out / name, "wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"saved {out / name}")


if __name__ == "__main__":
    main()
