# Action Recognition with a Frozen LeVJEPA Encoder

**Efficient action recognition via collapse-free video pretraining with LeVJEPA**

Supervisors: Haileleul Zeyede Haile, Ian Blair

We take a pretrained [LeVJEPA](https://arxiv.org/abs/2608.27395) video model,
keep it frozen, and train only small heads on top of it for action recognition.
The question we want to answer:

> What is the most effective and compute-efficient way to read out a frozen
> LeVJEPA representation for action recognition?

The encoder gives us a CLS token and a 16 × 14 × 14 grid of patch tokens per
clip. We compare readouts of these (CLS vs mean-pooled patches, then
concatenation, a small MLP, attention pooling and temporal pooling), reporting
accuracy together with trainable parameters, GPU memory and training time.
UCF101 first, then Something-Something v2, where motion matters more.

## Sprints

We meet every two weeks. The task notes for each sprint are in
[`sprints/`](sprints/). Read the one for the current sprint; you don't need to
look ahead.

The project goes through these phases, in this order:

| Phase | What |
|---|---|
| Read | The papers behind the model (V-JEPA, LeJEPA, LeVJEPA), then the two datasets (UCF101, SSv2) |
| Understand the model | Run it, look inside it (attention, patch tokens), decide what to read out |
| Baselines | UCF101 loader, feature extraction, linear probes on CLS vs mean-pooled patches |
| Experiment | More readouts, SSv2, seeds, temporal analysis |
| Write up | Clean code, final tables, report, final talk |

Papers and links: [`docs/reading_list.md`](docs/reading_list.md).

## Repository layout

```
configs/
    model.yaml              backbone settings (checkpoint, frames, resolution, normalisation)
    ucf101.yaml             dataset path, split, frame sampling
    probes/                 one config per readout experiment
docs/
    model_notes.md          the model, how it was trained, what it means for the readout
    reading_list.md
results/
    results.csv             every training run, one row each
    figures/                figures you want us to see
scripts/
    check_env.py            check packages and GPU                       (works now)
    check_model.py          load the model and inspect its outputs       (works now)
    visualize_tokens.py     CLS attention, patch PCA, patch similarity   (partly to do)
    extract_features.py     run the backbone once, cache features        (to do)
    train_probe.py          train a head on cached features, log result  (to do)
sprints/                    task notes for each two-week sprint
src/
    models/backbone.py      frozen LeVJEPA wrapper -> CLS + patch grid
    models/heads.py         readout heads (linear for now)
    data/video.py           decode, sample 16 frames, resize, normalise
    data/ucf101.py          UCF101 dataset                               (to do)
    data/features.py        loading cached features
    analysis/               attention capture and plotting helpers
    utils/metrics.py        accuracy, param count, time/memory, results logging
tests/                      run on CPU with a tiny fake model, no download needed
```

## Setup

You need Python 3.10+ and, for the real model, a GPU (24–40 GB is enough for
feature extraction with ViT-L at batch 4–8).

```bash
git clone https://github.com/haile-leul/levjepa_action_recognition.git
cd levjepa_action_recognition
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Check that everything is in place. Neither command downloads the model:

```bash
python scripts/check_env.py    # package versions and the GPU(s) PyTorch sees
pytest                         # CPU tests, a few seconds
```

## Running the model

```bash
python scripts/check_model.py
```

This downloads the checkpoint from Hugging Face the first time (about 1.2 GB,
cached in `~/.cache/huggingface`; set `HF_HOME` to put it on shared storage),
runs it on random input and prints:

- the raw output of the Hugging Face model,
- the CLS token and patch grid as the rest of the repo sees them,
- whether the shapes match the config: `cls (B, 1024)` and `patches (B, 16, 14, 14, 1024)`.

On real clips:

```bash
python scripts/check_model.py --video path/to/clip1.avi path/to/clip2.avi
```

In your own code:

```python
from src.models.backbone import LeVJEPABackbone
from src.data.video import load_clip
from src.utils.config import load_config

cfg = load_config("configs/model.yaml")
backbone = LeVJEPABackbone.from_config(cfg).to("cuda")

clip = load_clip("clip.avi", cfg["num_frames"], cfg["image_size"], cfg["mean"], cfg["std"], cfg["target_fps"])
out = backbone(clip.unsqueeze(0))     # input (B, T, C, H, W)
out.cls                               # (1, 1024)
out.patches                           # (1, 16, 14, 14, 1024)
```

## Looking inside the model

```bash
python scripts/visualize_tokens.py --video path/to/clip.avi
```

Saves figures in `results/figures/`: where the CLS token attends in each frame
(any layer, `--layer`), and, once you have written the two TODO functions in
the script, a PCA of the patch tokens and a similarity map to one query patch.

Read [`docs/model_notes.md`](docs/model_notes.md) before interpreting them. Two
things to know about the model:

- Leave the attention mode on its default. The model was trained with
  block-causal attention and gives worse features otherwise, without any error.
- The weights are CC-BY-NC-4.0: research and coursework only.

If `from_pretrained` complains about a missing package, install it and add it to
`requirements.txt` in your PR.

## Data

Datasets and cached features are not in git. Put them on the GPU machine's
storage and set the paths in the configs:

- `configs/ucf101.yaml` → `root`: the UCF101 folder (videos + `ucfTrainTestlist/`)
- `configs/ucf101.yaml` → `feature_dir`: where extracted features are written

UCF101 is about 7 GB; an SSv2 subset will be 20–30 GB.

## Working on the repo

Work on your own branch and open pull requests; don't push to `main`. Every
training run gets logged in `results/results.csv`. The details, and the ground
rules for experiments, are in [`CONTRIBUTING.md`](CONTRIBUTING.md).

If you are stuck for more than two or three days, message us. Don't wait for
the meeting.
