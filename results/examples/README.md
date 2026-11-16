# Example figures

What `scripts/visualize_tokens.py` should give you on two UCF101 clips. Use
them to check your own implementation of the two TODO functions, then go and
look at other clips and classes.

| Clip | Path inside UCF101 | What happens | Figures |
|---|---|---|---|
| Eye makeup | `UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi` | Close-up, a hand brushes makeup onto an eye | [v_ApplyEyeMakeup_g01_c01/](v_ApplyEyeMakeup_g01_c01/) |
| Baby crawling | `UCF-101/BabyCrawling/v_BabyCrawling_g19_c02.avi` | Wide shot, a baby crawls across a wooden floor | [v_BabyCrawling_g19_c02/](v_BabyCrawling_g19_c02/) |

Watch both videos before looking at the figures.

## Getting the clips

If you have UCF101, use the paths above. If not (yet), download just these two
clips (about 170 MB, from a public UCF101 subset on the Hugging Face Hub):

```bash
python scripts/download_example_clips.py      # saves them in data/examples/
```

The model itself is downloaded automatically the first time you run any script
that uses it.

Each folder has the six figures for its clip (CLS attention at two layers,
PCA and query-patch similarity, each with and without removing the per-frame
mean) and a README with the commands to reproduce them and what to look for.

## What to take from this for the readout

- The CLS token does look at the action, but which layer shows it best depends
  on the clip, and in the last layer it is often spread out and leans on the
  first frame.
- The patch tokens clearly know where the person, the hands or the baby are,
  but every token also carries "how far into the clip am I". A plain mean over
  all 3136 tokens mixes the two.
