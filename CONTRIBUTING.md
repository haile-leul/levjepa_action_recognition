# How we work in this repo

## Branches and pull requests

- `main` is protected. Don't push to it directly.
- One branch per piece of work, named `<your-name>/<what>`, e.g.
  `sara/ucf101-loader` or `tom/linear-baselines`.
- Open a pull request when it's ready (or a draft PR earlier if you want
  feedback). We review, you merge.
- Keep PRs small. "UCF101 loader" is a good PR; "sprints 4 to 6" is not.
- Run `pytest` before opening a PR. The tests also run on every PR on GitHub.

```bash
git checkout main && git pull
git checkout -b yourname/short-description
# ... work, commit ...
git push -u origin yourname/short-description
```

## Ground rules for experiments

1. **The backbone stays frozen.** Only the head is trained, for the whole project.
2. **Every run is logged** in `results/results.csv` with `log_result`: config,
   seed, accuracy, trainable parameters, GPU memory, training time. Failed or
   bad runs too.
3. **Fair comparison first, fancy heads second.** Same split, same number of
   epochs, same learning-rate grid, same seeds for every readout we compare.
4. **Extract features once.** Run the backbone over a dataset once, cache the
   features, and train heads on the cache.
5. **Configs, not hard-coded numbers.** If a value matters for a result, it
   lives in a file under `configs/` that is committed with the code.

## What doesn't go in git

Datasets, cached features, checkpoints and large outputs (see `.gitignore`).
Put them on the shared storage and point the configs at them.

## Code

- New readout heads go in `src/models/heads.py` and are registered in `HEADS`.
- New datasets go in `src/data/` with a config in `configs/`.
- If you add a dependency, add it to `requirements.txt` in the same PR.
- Add a small test in `tests/` for anything with shapes that are easy to get
  wrong.
