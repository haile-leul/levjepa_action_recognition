# The model: LeVJEPA-VideoMix-Large

Checkpoint: [galilai-group/levjepa-videomix-large](https://huggingface.co/galilai-group/levjepa-videomix-large)
Paper: [LeVJEPA: Efficient & Scalable Video Pretraining without the Heuristics](https://arxiv.org/abs/2608.27395)

## What we know

- ViT-L/16, about 303M parameters, pretrained self-supervised on a mix of
  video datasets (Kinetics family and others, SSv2 included, unlabelled).
- Input: 16 frames at 224 x 224.
- Output: one CLS token plus 16 x 14 x 14 = 3136 patch tokens, all of size 1024.
  So one clip gives 3137 x 1024 numbers, about 6 MB in fp16.
- Loaded with `AutoModel.from_pretrained(..., trust_remote_code=True)`.
- **Attention mode:** the weights were trained with block-causal attention, and
  that is the default. Running it with full attention gives no error but worse
  features. Don't change `attn_mode`.
- **Licence:** the weights are CC-BY-NC-4.0. Fine for coursework and research,
  not for commercial use.
- SSv2 is in the (unlabelled) pretraining data. That's fine for frozen probing,
  V-JEPA does the same, but the report has to say it.

## How the repo uses it

`src/models/backbone.py` wraps the HF model. It takes a normalised video batch
`(B, T, C, H, W)` and returns

```
out.cls       (B, 1024)
out.patches   (B, 16, 14, 14, 1024)    # (time, height, width, dim)
```

The backbone is always frozen and in eval mode. Nothing else in the repo
touches the HF model directly.

## To confirm in sprint 3

The wrapper makes a few assumptions that we could not check without running the
real checkpoint. Check each one against the model card and the remote code
(`modeling_*.py` on the model page), and fix `configs/model.yaml` or the
backbone wrapper in a PR if any is wrong:

- [ ] Input layout: `(B, C, T, H, W)` or `(B, T, C, H, W)`? (`input_layout`)
- [ ] Pixel normalisation: ImageNet mean/std, or something else? (`mean`, `std`)
- [ ] Output: `last_hidden_state` of shape `(B, 3137, 1024)` with CLS first?
      Or is the CLS returned separately? Any register tokens?
- [ ] Patch token order: time-major `(t, h, w)`? One easy check: feed a clip
      where only the first frame is non-zero and see which tokens change.
- [ ] Expected frame rate / temporal stride the model was trained with, and
      whether our `frame_stride` choice is reasonable.
- [ ] Extra packages the remote code needs (add them to `requirements.txt`).

`python scripts/check_model.py` prints the raw output structure, which answers
most of these.
