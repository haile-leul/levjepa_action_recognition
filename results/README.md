# Results

`results.csv` is the one shared results file for the whole project. Every
training run adds a row, including the ones that didn't work. Write to it with
`src.utils.metrics.log_result`, not by hand, so the columns stay consistent.

| column | meaning |
|---|---|
| `date`, `git_commit` | filled in automatically |
| `dataset` | `ucf101`, `ssv2`, ... |
| `feature` | which cached feature the head reads: `cls`, `mean`, `frame_mean`, `cls+mean`, ... |
| `head` | `linear`, `mlp`, `attentive`, `temporal`, ... |
| `lr`, `weight_decay`, `epochs`, `batch_size`, `seed` | training setup |
| `top1`, `top5` | test accuracy in % |
| `trainable_params` | parameters of the head only (the backbone is frozen) |
| `peak_gpu_mem_mb` | peak GPU memory during training |
| `train_time_s` | wall-clock training time |
| `config` | path to the config file used |
| `notes` | anything worth remembering |

If you need a new column, add it to `RESULT_COLUMNS` in `src/utils/metrics.py`
and to the header here in the same pull request.

Plots and the cleaned-up tables for the report also go in this folder.
